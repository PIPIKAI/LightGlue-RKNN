import os
import sys
import urllib
import urllib.request
import time
import numpy as np
import argparse
import cv2,math
from math import ceil
from itertools import product as product
from onnx_runner import LightGlueRunner, load_image, rgb_to_grayscale, viz2d

from rknn.api import RKNN

def letterbox_resize(image, size, bg_color):
    """
    Letterbox resize: 支持灰度(1通道) / RGB(3通道) / RGBA(4通道)
    """
    if isinstance(image, str):
        image = cv2.imread(image)

    target_width, target_height = size

    # 判断输入图像通道
    if image.ndim == 2:
        h, w = image.shape
        c = 1
    else:
        h, w, c = image.shape

    # 缩放比例
    ar = min(target_width / w, target_height / h)
    new_w = int(w * ar)
    new_h = int(h * ar)

    # resize
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 创建背景画布（根据通道数来创建）
    if c == 1:
        canvas = np.ones((target_height, target_width), dtype=np.uint8) * bg_color
    else:
        canvas = np.ones((target_height, target_width, c), dtype=np.uint8) * bg_color

    # 计算偏移
    ox = (target_width - new_w) // 2
    oy = (target_height - new_h) // 2

    # 贴图
    if c == 1:
        canvas[oy:oy+new_h, ox:ox+new_w] = resized
    else:
        canvas[oy:oy+new_h, ox:ox+new_w, :] = resized

    return canvas, ar, ox, oy


superpoint_model_path = "output/rknn/superpoint_simplified_512_512_rk3588.rknn"
# lightglue_model_path = "output/rknn/lightglue_512_256_rk3588.rknn"
lightglue_model_path = "output/rknn/lightglue_simplified_512_512_rk3588.rknn"

target = "rk3588"

superpoint_model = RKNN(verbose=True)
lightglue_model = RKNN(verbose=True)
GrayScale = True

# -------------------------
# 1. 加载模型
# -------------------------
ret = superpoint_model.load_rknn(superpoint_model_path)
if ret != 0:
    print(f'Load RKNN model "{superpoint_model_path}" failed!')
    exit(ret)
print('SuperPoint loaded')

ret = lightglue_model.load_rknn(lightglue_model_path)
if ret != 0:
    print(f'Load RKNN model "{lightglue_model_path}" failed!')
    exit(ret)
print('LightGlue loaded')

# -------------------------
# 2. 初始化运行时
# -------------------------
ret = superpoint_model.init_runtime(target=target)
if ret != 0:
    print('Init SuperPoint runtime failed!')
    exit(ret)

ret = lightglue_model.init_runtime(target=target)
if ret != 0:
    print('Init LightGlue runtime failed!')
    exit(ret)
print('Runtime initialized')

# -------------------------
# 3. 图像预处理
# -------------------------
# img1 = cv2.imread('./assets/sacre_coeur1.jpg')
# img2 = cv2.imread('./assets/sacre_coeur2.jpg')
img1 = cv2.imread('./assets/DSC_0410.JPG')
img2 = cv2.imread('./assets/DSC_0411.JPG')

if GrayScale:
    img1_input = rgb_to_grayscale(img1)
    img2_input = rgb_to_grayscale(img2)
else:
    img1_input = img1
    img2_input = img2


letterbox_img1, ar1, ox1, oy1 = letterbox_resize(img1_input, (512,512), 114)
letterbox_img2, ar2, ox2, oy2 = letterbox_resize(img2_input, (512,512), 114)

# infer_img1 = (letterbox_img1[..., ::-1] / 255.0).astype(np.float16).copy()  # BGR->RGB
# infer_img2 = (letterbox_img2[..., ::-1] / 255.0).astype(np.float16).copy()  # BGR->RGB
if GrayScale:

    infer_img1 = (letterbox_img1 / 255.0).astype(np.float16).copy()
    infer_img2 = (letterbox_img2 / 255.0).astype(np.float16).copy()
else:
    infer_img1 = (letterbox_img1[..., ::-1] / 255.0).astype(np.float16).copy()  # BGR->RGB
    infer_img2 = (letterbox_img2[..., ::-1] / 255.0).astype(np.float16).copy()  # BGR->RGB
print("infer_img1 shape:" ,infer_img1.shape)
# -------------------------
# 4. SuperPoint 推理
# -------------------------
star_infer = time.perf_counter()
outputs1 = superpoint_model.inference(inputs=[infer_img1])
print(f"img1 superpoint_model 耗时: {time.perf_counter() - star_infer:.6f} 秒")

star_infer = time.perf_counter()
outputs2 = superpoint_model.inference(inputs=[infer_img2])
print(f"img2 superpoint_model 耗时: {time.perf_counter() - star_infer:.6f} 秒")

kpts0 = outputs1[0][0]   # shape (256,2)
scores0 = outputs1[1][0]
desc0 = outputs1[2][0]  # shape (256,256)

kpts1 = outputs2[0][0]
scores1 = outputs2[1][0]
desc1 = outputs2[2][0]



def normalize_keypoints(
        kpts: np.ndarray,
        h: int,
        w: int,
    ) -> np.ndarray:
        size = np.array([w, h])
        shift = size / 2
        scale = size.max() / 2
        kpts = (kpts - shift) / scale
        return kpts.astype(np.float32)
    

# -------------------------
# 5. LightGlue 推理
# -------------------------

kpts0 = normalize_keypoints(kpts0 , 512,512)
kpts1 = normalize_keypoints(kpts1 , 512,512)


print("kpts0 shape:", kpts0.shape)
print("kpts0 min/max:", kpts0.min(), kpts0.max())
print("kpts1 shape:", kpts1.shape)
print("kpts1 min/max:", kpts1.min(), kpts1.max())
print("desc0 shape:", desc0.shape)
print("desc0 min/max:", desc0.min(), desc0.max())
print("desc1 shape:", desc1.shape)
print("desc1 min/max:", desc1.min(), desc1.max())
print("scores0 range:", scores0.min(), scores0.max())
print("scores1 range:", scores1.min(), scores1.max())


def post_process( kpts0, kpts1, matches0, scales0, scales1):
    kpts0 = (kpts0 + 0.5) / scales0 - 0.5
    kpts1 = (kpts1 + 0.5) / scales1 - 0.5
    # create match indices
    valid = matches0[0] > -1
    matches = np.stack([np.where(valid)[0], matches0[0][valid]], -1)
    m_kpts0, m_kpts1 = kpts0[0][matches[..., 0]], kpts1[0][matches[..., 1]]
    return m_kpts0, m_kpts1


lg_inputs = [
kpts0.astype(np.float32),  # (1, 256, 2)
kpts1.astype(np.float32),  # (1, 256, 2)
desc0.astype(np.float32), # (1, 256, 256)
desc1.astype(np.float32)  # (1, 256, 256)
]

print("lightglue_model infer begin ")
star_infer = time.perf_counter()

lg_outputs = lightglue_model.inference(inputs=lg_inputs)
print("lightglue_model infer end !")
print(f"lightglue_model 耗时: {time.perf_counter() - star_infer:.6f} 秒")


# lg_outputs[0] -> (1,257,257) FP16
scores = lg_outputs[0][0].astype(np.float32)  # (257,257)


print("scores shape:", scores.shape)
m = scores.shape[0] - 1  # 256
n = scores.shape[1] - 1  # 256

# softmax along right-axis（每个左点分布到所有右点+unmatched）
scores = scores.copy()
scores = scores - scores.max(axis=1, keepdims=True)  # 防溢出
exp = np.exp(scores)
prob = exp / exp.sum(axis=1, keepdims=True)  # (257,257)


# left->best (包括可能指向 unmatched 列 n)
match0 = np.argmax(prob[:m, :], axis=1)      # shape (m,), 值域 0..n
mscores0 = np.max(prob[:m, :], axis=1)      # shape (m,)

# right->best 只看前 m 行（不把 unmatched 行当候选）
# 为了保证 match1 的长度为 n（索引 0..n-1），我们对前 m 行取转置再 argmax
match1 = np.argmax(prob[:m, :n].T, axis=1)  # shape (n,), 值域 0..m-1
mscores1 = np.max(prob[:m, :n].T, axis=1)  # shape (n,)


def fix_kpts(kpts):  # (1,2,1,256)
    # 原格式: B, C, 1, N
    # 你应该把 (2,1,256) 变成 (256,2)
    kpts = np.squeeze(kpts, axis=2)         # (1,2,256)
    kpts = np.transpose(kpts, (0, 2, 1))    # (1,256,2)
    return kpts.astype(np.float32)

# 阈值与 mutual check
TH = 0.01  # 自行调整
# 左侧先判断置信度
valid0_score = mscores0 > TH  # (m,)
# 右侧置信度
valid1_score = mscores1 > TH  # (n,)

# mutual consistency（只在左侧 match0 < n 的情况下检查）
valid0_mutual = np.zeros(m, dtype=bool)
valid_indices0 = np.where(match0 < n)[0]  # 只有这些索引能安全用于 match1[...] 操作
if valid_indices0.size > 0:
    # match1[ match0[valid_indices0] ]  -> 对应右点返回的左点索引
    left_votes = match1[match0[valid_indices0]]
    valid0_mutual[valid_indices0] = (left_votes == valid_indices0)

# 同理构造右侧 mutual（注意：match1 中不会出现 unmatched，因为我们没把 unmatched 行纳入）
valid1_mutual = np.zeros(n, dtype=bool)
valid_indices1 = np.where((match1 >= 0) & (match1 < m))[0]
if valid_indices1.size > 0:
    right_votes = match0[ match1[valid_indices1] ]  # 可能为 n 表示 unmatched
    # 只有当 right_votes == valid_indices1 才互检通过
    valid1_mutual[valid_indices1] = (right_votes == valid_indices1)

# 结合置信度和互检
valid0 = valid0_score & valid0_mutual
valid1 = valid1_score & valid1_mutual

# 最终 match 数组，-1 表示未匹配
final_match0 = np.where(valid0, match0, -1)  # shape (m,)
final_match1 = np.where(valid1, match1, -1)  # shape (n,)

print("LightGlue matches0 range:", np.min(np.where(final_match0>=0, final_match0, np.inf)), 
      np.max(np.where(final_match0>=0, final_match0, -np.inf)))
print("LightGlue matches1 range:", np.min(np.where(final_match1>=0, final_match1, np.inf)), 
      np.max(np.where(final_match1>=0, final_match1, -np.inf)))
print("final_match0 count:", np.sum(final_match0 >= 0))
print("final_match1 count:", np.sum(final_match1 >= 0))

# 如果需要把 final_match0/1 转为 pair 列表：
pairs = [(i, int(j)) for i, j in enumerate(final_match0) if j >= 0]
print("pairs example (left_idx, right_idx):", pairs[:20])


# 1. 先把关键点映射回原图像坐标
def denormalize_kpts_for_plot(kpts_norm, img_w, img_h, scale, offset_x, offset_y):
    """
    kpts_norm: 归一化 [-1,1] 的关键点 (N,2)
    scale/offset_x/offset_y: letterbox resize 参数
    返回: 原图像坐标 (N,2)
    """
    kpts_512 = (kpts_norm * (512/2) + 512/2)  # [-1,1] -> [0,512]
    # 反向 letterbox
    kpts_orig = (kpts_512 - offset_x) / scale
    kpts_orig = (kpts_orig - offset_y / scale)  # 对 y 方向同样缩放
    return kpts_orig

def kpts_normalized_to_original(kpts_norm, scale, offset_x, offset_y):
    """
    将 [-1,1] 归一化关键点映射回原图坐标
    kpts_norm: (N,2) 归一化 [-1,1]
    scale: letterbox 缩放比例
    offset_x, offset_y: letterbox padding
    返回: 原图像坐标 (N,2)
    """
    # 1. [-1,1] -> [0,512] (LightGlue 输入尺寸)
    kpts_512 = (kpts_norm + 1) * 512 / 2.0

    # 2. 减去 padding
    kpts_512[:,0] -= offset_x
    kpts_512[:,1] -= offset_y

    # 3. 除以缩放比例
    kpts_orig = kpts_512 / scale

    return kpts_orig.astype(np.int32)

# 转回原图坐标
kpts0_plot = kpts_normalized_to_original(
    kpts0, ar1, ox1, oy1)
kpts1_plot = kpts_normalized_to_original(
    kpts1, ar2, ox2, oy2)

# 保留有效匹配
pairs = [(i,j) for i,j in enumerate(final_match0) if j >= 0]
if len(pairs) == 0:
    print("No matches to plot!")
else:
    kpts0_match = np.array([kpts0_plot[i] for i,j in pairs])
    kpts1_match = np.array([kpts1_plot[j] for i,j in pairs])

# 4. 绘图
viz2d.plot_images([img1, img2], titles=['Image1','Image2'])
viz2d.plot_matches(kpts0_match, kpts1_match, lw=1.0, ps=6, a=0.8)
viz2d.add_text(0, f"Matches: {len(pairs)}")
viz2d.save_plot("matches_high_quality.png")
print("Saved high-quality match visualization as matches_high_quality.png")

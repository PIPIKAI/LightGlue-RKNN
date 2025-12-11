# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.

# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

from typing import Tuple

import torch
from torch import nn
from lightglue_onnx.utils import rgb_to_grayscale


def max_pool(x, nms_radius: int):
    return torch.nn.functional.max_pool2d(
        x, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius
    )


def simple_nms(scores, nms_radius: int):
    """Fast Non-maximum suppression to remove nearby points"""
    # assert nms_radius >= 0
    # scores = scores[None]
    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores, nms_radius)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float(), nms_radius) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores, nms_radius)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)[0]


def remove_borders(keypoints, scores, border: int, height: int, width: int):
    """Removes keypoints too close to the border"""
    mask_h = (keypoints[:, 1] >= border) & (keypoints[:, 1] < (height - border))
    mask_w = (keypoints[:, 2] >= border) & (keypoints[:, 2] < (width - border))
    mask = mask_h & mask_w
    print(f'@@@@@@@@@@@@@@@@ mask.shape: {mask.shape} {mask.dtype}')
    return keypoints[mask], scores[mask]


def top_k_keypoints(keypoints, scores, k: int):
    if k >= len(keypoints):
        return keypoints, scores
    scores, indices = torch.topk(scores, k, dim=0)
    return keypoints[indices], scores


# def sample_descriptors(keypoints, descriptors, s: int = 8):
#     """Interpolate descriptors at keypoint locations"""
#     b, c, h, w = descriptors.shape
#     keypoints = keypoints - s / 2 + 0.5
#     keypoints_x = torch.div(keypoints[..., 0], (w * s - s / 2 - 0.5))
#     keypoints_y = torch.div(keypoints[..., 1], (h * s - s / 2 - 0.5))
#     keypoints = torch.stack((keypoints_x, keypoints_y), dim=-1)

#     keypoints = keypoints * 2 - 1  # normalize to (-1, 1)
#     descriptors = torch.nn.functional.grid_sample(
#         descriptors, keypoints.view(b, 1, -1, 2), mode="bilinear", align_corners=True
#     )
#     descriptors = torch.nn.functional.normalize(
#         descriptors.reshape(b, c, -1), p=2, dim=1
#     )
#     return descriptors

def sample_descriptors(keypoints, descriptors, s: int = 8):
    """Interpolate descriptors at keypoint locations using manual bilinear interpolation"""
    b, c, h, w = descriptors.shape
    keypoints = keypoints - s / 2 + 0.5
    keypoints_x = torch.div(keypoints[..., 0], (w * s - s / 2 - 0.5))
    keypoints_y = torch.div(keypoints[..., 1], (h * s - s / 2 - 0.5))
    
    # 将归一化坐标 [0, 1] 转换为像素坐标 [0, w-1] 和 [0, h-1]
    x = keypoints_x * (w - 1)
    y = keypoints_y * (h - 1)
    
    # 计算双线性插值的四个邻近点
    x0 = torch.floor(x).long()
    x1 = x0 + 1
    y0 = torch.floor(y).long()
    y1 = y0 + 1
    
    # 边界裁剪
    x0 = torch.clamp(x0, 0, w - 1)
    x1 = torch.clamp(x1, 0, w - 1)
    y0 = torch.clamp(y0, 0, h - 1)
    y1 = torch.clamp(y1, 0, h - 1)
    
    # 计算插值权重
    wa = (x1.float() - x) * (y1.float() - y)
    wb = (x1.float() - x) * (y - y0.float())
    wc = (x - x0.float()) * (y1.float() - y)
    wd = (x - x0.float()) * (y - y0.float())
    
    # 对每个 batch 和每个关键点进行采样
    n_keypoints = keypoints.shape[1]
    sampled = torch.zeros(b, c, n_keypoints, device=descriptors.device, dtype=descriptors.dtype)
    
    for i in range(b):
        for j in range(n_keypoints):
            # 获取四个邻近点的描述符
            Ia = descriptors[i, :, y0[i, j], x0[i, j]]
            Ib = descriptors[i, :, y1[i, j], x0[i, j]]
            Ic = descriptors[i, :, y0[i, j], x1[i, j]]
            Id = descriptors[i, :, y1[i, j], x1[i, j]]
            
            # 双线性插值
            sampled[i, :, j] = (wa[i, j] * Ia + 
                                wb[i, j] * Ib + 
                                wc[i, j] * Ic + 
                                wd[i, j] * Id)
    
    # 归一化
    descriptors_out = torch.nn.functional.normalize(
        sampled.reshape(b, c, -1), p=2, dim=1
    )
    return descriptors_out


# 更高效的向量化版本（推荐使用）
def sample_descriptors_vectorized(keypoints, descriptors, s: int = 8):
    """Vectorized interpolation - more efficient for RKNN"""
    b, c, h, w = descriptors.shape
    # keypoints shape: (N, 2) - 注意没有batch维度
    n_keypoints = keypoints.shape[0]
    
    keypoints = keypoints - s / 2 + 0.5
    keypoints_x = torch.div(keypoints[..., 0], (w * s - s / 2 - 0.5))
    keypoints_y = torch.div(keypoints[..., 1], (h * s - s / 2 - 0.5))
    
    # 转换为像素坐标 (N,)
    x = keypoints_x * (w - 1)
    y = keypoints_y * (h - 1)
    
    # 计算邻近点 (N,)
    x0 = torch.floor(x).long()
    x1 = x0 + 1
    y0 = torch.floor(y).long()
    y1 = y0 + 1
    
    # 边界裁剪
    x0 = torch.clamp(x0, 0, w - 1)
    x1 = torch.clamp(x1, 0, w - 1)
    y0 = torch.clamp(y0, 0, h - 1)
    y1 = torch.clamp(y1, 0, h - 1)
    
    # 计算插值权重 (N,) -> (1, 1, N) for broadcasting
    wa = ((x1.float() - x) * (y1.float() - y)).view(1, 1, n_keypoints)
    wb = ((x1.float() - x) * (y - y0.float())).view(1, 1, n_keypoints)
    wc = ((x - x0.float()) * (y1.float() - y)).view(1, 1, n_keypoints)
    wd = ((x - x0.float()) * (y - y0.float())).view(1, 1, n_keypoints)
    
    # 批量索引采样 - reshape descriptors 为 (b, c, h*w)
    descriptors_flat = descriptors.reshape(b, c, h * w)
    
    # 计算索引 (N,) -> (1, 1, N) -> (b, c, N)
    idx_a = (y0 * w + x0).view(1, 1, n_keypoints).expand(b, c, n_keypoints)
    idx_b = (y1 * w + x0).view(1, 1, n_keypoints).expand(b, c, n_keypoints)
    idx_c = (y0 * w + x1).view(1, 1, n_keypoints).expand(b, c, n_keypoints)
    idx_d = (y1 * w + x1).view(1, 1, n_keypoints).expand(b, c, n_keypoints)
    
    # 使用 gather 进行采样
    Ia = torch.gather(descriptors_flat, 2, idx_a)  # (b, c, n_keypoints)
    Ib = torch.gather(descriptors_flat, 2, idx_b)
    Ic = torch.gather(descriptors_flat, 2, idx_c)
    Id = torch.gather(descriptors_flat, 2, idx_d)
    
    # 双线性插值 (b, c, N)
    sampled = wa * Ia + wb * Ib + wc * Ic + wd * Id
    
    # 归一化
    descriptors_out = torch.nn.functional.normalize(
        sampled, p=2, dim=1
    )
    return descriptors_out

def unravel_indices(
    indices: torch.LongTensor,
    shape: Tuple[int, ...],
) -> torch.LongTensor:
    r"""Converts flat indices into unraveled coordinates in a target shape.

    Args:
        indices: A tensor of (flat) indices, (*, N).
        shape: The targeted shape, (D,).

    Returns:
        The unraveled coordinates, (*, N, D).
    """
    coord = []
    for dim in reversed(shape):
        coord.append(indices % dim)
        indices = indices // dim

    coord = torch.stack(coord[::-1], dim=-1)
    return coord

def unravel_index(
    indices: torch.LongTensor,
    shape: Tuple[int, ...],
) -> Tuple[torch.LongTensor, ...]:
    r"""Converts flat indices into unraveled coordinates in a target shape.

    This is a `torch` implementation of `numpy.unravel_index`.

    Args:
        indices: A tensor of (flat) indices, (N,).
        shape: The targeted shape, (D,).

    Returns:
        A tuple of unraveled coordinate tensors of shape (D,).
    """
    coord = unravel_indices(indices, shape)
    return tuple(coord)



class SuperPoint(nn.Module):
    """SuperPoint Convolutional Detector and Descriptor

    SuperPoint: Self-Supervised Interest Point Detection and
    Description. Daniel DeTone, Tomasz Malisiewicz, and Andrew
    Rabinovich. In CVPRW, 2019. https://arxiv.org/abs/1712.07629

    """

    default_config = {
        "descriptor_dim": 256,
        "nms_radius": 4,
        "max_num_keypoints": -1,
        "detection_threshold": 0.0005,
        "remove_borders": 4,
    }

    def __init__(self, conf):
        super().__init__()
        self.config = {**self.default_config, **conf}

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256

        self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

        self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)

        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(
            c5, self.config["descriptor_dim"], kernel_size=1, stride=1, padding=0
        )

        url = "https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_v1.pth"
        self.load_state_dict(torch.hub.load_state_dict_from_url(url))

        # mk = self.config["max_num_keypoints"]
        # if mk == 0 or mk < -1:
        #     raise ValueError('"max_num_keypoints" must be positive or "-1"')

        print("Loaded SuperPoint model")

    def forward(
        self,
        image: torch.Tensor,  # (1, 1, H, W)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute keypoints, scores, descriptors for image"""
        # Shared Encoder
        image = rgb_to_grayscale(image)
        x = self.relu(self.conv1a(image))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))

        # Compute the dense keypoint scores
        cPa = self.relu(self.convPa(x))
        scores = self.convPb(cPa)
        scores = torch.nn.functional.softmax(scores, 1)[:, :-1]
        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        # scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h * 8, w * 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, 1, h * 8, w * 8)
        scores = simple_nms(scores, self.config["nms_radius"])

        # scores.shape == (B, H, W)

        # Below this, B > 1 is not supported as each image can have a different number of keypoints.

        # Extract keypoints
        # keypoints = torch.nonzero(scores > self.config["detection_threshold"])

        ############################################################################
        flat_scores = scores.view(-1)
        # total_elements = torch.prod(torch.tensor(scores.shape))
        # total_elements * 0.00025 / 10
        # total_elements_floor = torch.floor(total_elements * 0.00025 / 10)
        # top_nums = (total_elements_floor * 10).to(torch.int32)
        top_nums = self.config["top_nums"]
        # top_nums = 256
        print(f'@@@@@@@@@@@@@@@@@@@@@@@@ top_nums: {top_nums}')
        values, indices = flat_scores.topk(top_nums)
        keypoints = torch.stack(unravel_index(indices, scores.shape))
        keypoints_t = keypoints.T
        ############################################################################
        print(f'@@@@@@@@@@@@@@@@@@@@@@@@ top_nums: {top_nums}')
        print(f'@@@@@@@@@@@@@@@@@@@@@@@@ keypoints_t.shape: {keypoints_t.shape}')
        # keypoints.shape == (N, 3)

        # scores = scores[keypoints.T[0], keypoints.T[1], keypoints.T[2]]
        scores = scores[keypoints_t[0], keypoints_t[1], keypoints_t[2]]

        # scores.shape == (N,)

        # Discard keypoints near the image borders
        # keypoints, scores = remove_borders(
        #     keypoints, scores, self.config["remove_borders"], h * 8, w * 8
        # )
        #
        # To generate a shape-invariant tensor, the NonZero implementation is cut out of the model.
        if False:
            keypoints, scores = remove_borders(
                keypoints, scores, self.config["remove_borders"], h * 8, w * 8
            )

        # Keep the k keypoints with highest score
        if False:  # self.config["max_num_keypoints"] >= 0:
            keypoints, scores = top_k_keypoints(
                keypoints, scores, self.config["max_num_keypoints"]
            )

        # Convert (h, w) to (x, y)
        keypoints = torch.flip(keypoints[:, 1:], (1,))

        # keypoints.shape == (N, 2)

        # Compute the dense descriptors
        cDa = self.relu(self.convDa(x))
        descriptors = self.convDb(cDa)
        descriptors = torch.nn.functional.normalize(descriptors, p=2, dim=1)

        # Extract descriptors
        descriptors = sample_descriptors_vectorized(keypoints, descriptors, 8).permute(0, 2, 1)

        # Insert artificial batch dimension
        return (
            keypoints[None],  # (1, N, 2)
            scores[None],  # (1, N)
            descriptors,  # (1, N, desc_dim)
        )

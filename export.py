import argparse

import torch
import os
from lightglue_onnx import DISK, LightGlue, LightGlueEnd2End, SuperPoint
from lightglue_onnx.end2end import normalize_keypoints
from lightglue_onnx.ops import patch_disk_convolution_mode
from lightglue_onnx.utils import load_image, rgb_to_grayscale


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img_size",
        nargs="+",
        type=int,
        default=512,
        required=False,
        help="Sample image size for ONNX tracing. If a single integer is given, resize the longer side of the image to this value. Otherwise, please provide two integers (height width).",
    )
    
    parser.add_argument(
        "--top_nums",
        type=int,
        default=20,
        required=False
    )
    parser.add_argument(
        "--extractor_type",
        type=str,
        default="superpoint",
        required=False,
        help="Type of feature extractor. Supported extractors are 'superpoint' and 'disk'. Defaults to 'superpoint'.",
    )

    parser.add_argument(
        "--end2end",
        action="store_true",
        help="Whether to export an end-to-end pipeline instead of individual models.",
    )
    parser.add_argument(
        "--dynamic", action="store_true", help="Whether to allow dynamic image sizes."
    )
    parser.add_argument(
        "--simplify", action="store_true", help="Whether to simplify."
    )
    
    # parser.add_argument(
    #     "--rknn", action="store_true", help="conver to rknn."
    # )
    return parser.parse_args()



def simplify_model(model_path:str):
    import onnx
    from onnxsim import simplify
    model = onnx.load(model_path + ".onnx")
    model_simp, check = simplify(model=model)
    assert check, "Simplified ONNX model could not be validated"
    s_path = f"{model_path}_simplified.onnx"
    onnx.save(model_simp, s_path)
    return s_path


def exort_rknn(input_size_list , input_model_path , output_dir, exp ="" ,img_size = 512 , top_nums = 20,platform = "rk3588", verbose=True):
    from rknn.api import RKNN
    rknn = RKNN(verbose=verbose)
    print('--> Config model')
    rknn.config(
        target_platform='rk3588',
        optimization_level=3,
        disable_rules=['fuse_two_scatternd2'],  # 禁用这个融合规则
    )
    print('done')
    
    print('--> Loading model')
    ret = rknn.load_onnx(
        model=input_model_path,
        input_size_list = input_size_list
    )
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')
    
    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    # Export rknn model
    RKNN_MODEL_PATH = f'./{output_dir}/{exp}_{img_size}_{top_nums}_{platform}.rknn'
    
    print('--> Export RKNN model: {}'.format(RKNN_MODEL_PATH))
    ret = rknn.export_rknn(RKNN_MODEL_PATH)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    rknn.release()
    print('done')

    
def export_onnx(
    img_size=512,
    top_nums = 20,
    extractor_type="superpoint",
    img0_path="assets/sacre_coeur1.jpg",
    img1_path="assets/sacre_coeur2.jpg",
    end2end=False,
    dynamic=False,
    simplify=True,
    output_path=None,
    onnx_extractor_path = None,
    onnx_lightglue_path = None
):
    # Handle args

    if output_path is None:
        output_path = "output/onnx"
        
    onnx_extractor_path = f"{output_path}/{extractor_type}"
    if end2end:
        onnx_lightglue_path = f"{output_path}/{extractor_type}_lightglue_end2end"
    else:
        onnx_lightglue_path = f"{output_path}/{extractor_type}_lightglue"

    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # Sample images for tracing
    image0, scales0 = load_image(img0_path, resize=img_size)
    image1, scales1 = load_image(img1_path, resize=img_size)

    # Models
    extractor_type = extractor_type.lower()
    if extractor_type == "superpoint":
        # SuperPoint works on grayscale images.
        # image0 = rgb_to_grayscale(image0)
        # image1 = rgb_to_grayscale(image1)
        extractor = SuperPoint({"top_nums":top_nums}).eval()
        lightglue = LightGlue(extractor_type).eval()
    elif extractor_type == "disk":
        extractor = DISK().eval()
        lightglue = LightGlue(extractor_type).eval()

        patch_disk_convolution_mode(extractor)  # Fixed in PyTorch >= 2.1
    else:
        raise NotImplementedError(
            f"LightGlue has not been trained on {extractor_type} features."
        )

    # ONNX Export
    if end2end:
        pipeline = LightGlueEnd2End(extractor, lightglue).eval()
        dynamic_axes = {
            "kpts0": {1: "num_keypoints0"},
            "kpts1": {1: "num_keypoints1"},
            "matches0": {1: "num_matches0"},
            "matches1": {1: "num_matches1"},
            "mscores0": {1: "num_matches0"},
            "mscores1": {1: "num_matches1"},
        }
        if dynamic:
            dynamic_axes.update(
                {
                    "image0": {2: "height0", 3: "width0"},
                    "image1": {2: "height1", 3: "width1"},
                }
            )
        torch.onnx.export(
            pipeline,
            (image0[None], image1[None]),
            onnx_lightglue_path + ".onnx",
            input_names=["image0", "image1"],
            output_names=[
                "kpts0",
                "kpts1",
                "matches0",
                "matches1",
                "mscores0",
                "mscores1",
            ],
            opset_version=16,
            dynamic_axes=None,
            # dynamic_axes=dynamic_axes,
        )
    else:
        # Export Extractor
        dynamic_axes = {
            "keypoints": {1: "num_keypoints"},
            "scores": {1: "num_keypoints"},
            "descriptors": {1: "num_keypoints"},
        }
        if dynamic:
            dynamic_axes.update({"image": {2: "height", 3: "width"}})
        torch.onnx.export(
            extractor,
            image0[None],
            onnx_extractor_path + ".onnx",
            input_names=["image"],
            output_names=["keypoints", "scores", "descriptors"],
            opset_version=16,
            dynamic_axes=None,
            # dynamic_axes=dynamic_axes,
        )

        # Export LightGlue
        feats0, feats1 = extractor(image0[None]), extractor(image1[None])
        kpts0, scores0, desc0 = feats0
        kpts1, scores1, desc1 = feats1

        kpts0 = normalize_keypoints(kpts0, image0.shape[1], image0.shape[2])
        kpts1 = normalize_keypoints(kpts1, image1.shape[1], image1.shape[2])

        kpts0 = kpts0.unsqueeze(-1)
        kpts1 = kpts1.unsqueeze(-1)
        desc0 = desc0.unsqueeze(-1)
        desc1 = desc1.unsqueeze(-1)
        
        
        print("kpts0 shape:", kpts0.shape)
        print("desc0 shape:", desc0.shape)
        

        torch.onnx.export(
            lightglue,
            (
                kpts0,
                kpts1,
                desc0,
                desc1,
            ),
            onnx_lightglue_path + ".onnx",
            input_names=["kpts0", "kpts1", "desc0", "desc1"],
            output_names=["scores"],
            opset_version=16,
            # dynamic_axes={
            #     "kpts0": {1: "num_keypoints0"},
            #     "kpts1": {1: "num_keypoints1"},
            #     "desc0": {1: "num_keypoints0"},
            #     "desc1": {1: "num_keypoints1"},
            #     "matches0": {1: "num_matches0"},
            #     "matches1": {1: "num_matches1"},
            #     "mscores0": {1: "num_matches0"},
            #     "mscores1": {1: "num_matches1"},
            # },
            dynamic_axes=None,
        )
        if simplify :
            print(">>>>> 开始简化模型 <<<<<")
            if not end2end:
                onnx_extractor_simplify_path = simplify_model(onnx_extractor_path)
            onnx_lightglue_simplify_path = simplify_model(onnx_lightglue_path)
            
if __name__ == "__main__":
    args = parse_args()
    export_onnx(**vars(args))

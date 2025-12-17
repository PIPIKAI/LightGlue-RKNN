import argparse

import torch
import os



def exort_rknn(
    input_size_list , 
    input_model_path , 
    d_size_list,
    output_dir, 
    exp ="" ,
    img_size = 512 , 
    top_nums = 20,
    platform = "rk3588", 
    verbose=True,
    do_quantization = False
    ):
    from rknn.api import RKNN
    rknn = RKNN(verbose=verbose)
    print('--> Config model')
    rknn.config(
        target_platform= platform,
        disable_rules=['fuse_two_scatternd2'],  # 禁用这个融合规则
        # enable_flash_attention=True,
        dynamic_input= d_size_list,
        optimization_level=0
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
    ret = rknn.build(do_quantization=do_quantization)
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

   
img_size = 512
# input_model_path = "/root/codes/rknn_model_export/LightGlue-ONNX/output/onnx/superpoint_simplified.onnx"
# input_model_path = "/root/codes/rknn_model_export/LightGlue-ONNX/output/onnx/superpoint_lightglue_end2end_simplified.onnx"
input_model_path = "/root/codes/rknn_model_export/LightGlue-ONNX/output/onnx/superpoint_lightglue_simplified.onnx"
out_dir = "output/rknn"
exp = "lightglue_simplified"
top_nums = 512
platform =  "rk3588"
do_quantization = False
# input_size_list=[[1, 3, img_size, img_size]]
# input_size_list=[[1, 1, img_size, img_size],[1, 1, img_size, img_size]]
# input_size_list=[[1, top_nums, 2, 1],[1, top_nums, 2, 1],[1, top_nums, 256, 1],[1, top_nums, 256, 1]]
# input_size_list=[[1, top_nums, 2],[1, top_nums, 2],[1, top_nums, 256],[1, top_nums, 256]]
input_size_list=None
# d_size_list= [[[1, top_nums, 2, 1],[1, top_nums, 2, 1],[1, top_nums, 256, 1],[1, top_nums, 256, 1]]]
d_size_list= None


if do_quantization:
    exp += "quant"
if __name__ == "__main__":
    exort_rknn(
        input_size_list = input_size_list,
        input_model_path= input_model_path,
        d_size_list = d_size_list,
        output_dir=out_dir,
        exp =exp ,
        img_size =img_size,
        top_nums = top_nums,
        platform= platform,
        verbose= False,
        do_quantization = do_quantization
    )

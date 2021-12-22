# -*- coding: utf-8 -*-
"""
Created on 20-12-19

@author: zjs (01376022)

DBNet inference on tensorRT

@Notice: Try to export onnx model and tensorRT engine with dynamic input shape by new tensorRT API
@Notice: TensorRT 6.0.1.5 onnxparser do not support full-dims and dynamic shape, install `onnx-tensorrt 6.0-full-dims` solves it!
@Notice: TensorRT 6.0.1.5 针对full-dims和dynamic shape,提供了一批新接口,如create_cuda_engine_with_config, PluginV2等!

"""
import os
import sys
import pathlib
__dir__ = pathlib.Path(os.path.abspath(__file__))
sys.path.append(str(__dir__))
sys.path.append(str(__dir__.parent.parent))

import time
import cv2
import torch
# import onnx TODO: Notice that onnx can`t be imported with trt on some version
import numpy as np
from torch import nn
import torch.nn.functional as F

os.environ['CUDA_VISIBLE_DEVICES'] = str('0')

torch.cuda.synchronize()
import tensorrt as trt
# TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
TRT_LOGGER = trt.Logger()
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

# import pycuda.driver as cuda
# import pycuda.autoinit
# import ctypes
from data_loader import get_transforms
from models import build_model
from post_processing import get_post_processing
import deploy.common as common
torch.cuda.synchronize()


def get_engine(onnx_file_path, engine_file_path=""):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""

    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(
                network, TRT_LOGGER) as parser, builder.create_builder_config() as config:

            builder.strict_type_constraints = True
            # builder.max_workspace_size = 1 << 30  # deprecated, use config to set max_workspace_size
            # builder.fp16_mode = True   # deprecated, use config to set FP16 mode
            # builder.max_batch_size = 1  # deprecated, use EXPLICIT_BATCH

            config.set_flag(trt.BuilderFlag.FP16)
            config.max_workspace_size=common.GiB(1)

            # Parse model file
            # Try to load a previously generated graph in ONNX format:
            if not os.path.exists(onnx_file_path):
                print('ONNX file {} not found, please generate it first.'.format(
                    onnx_file_path))
                exit(0)
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                if not parser.parse(model.read()):
                    print('ERROR: Failed to parse the ONNX file.')
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    exit(0)

            # Reference: https://blog.csdn.net/weixin_43953045/article/details/103937295
            last_layer = network.get_layer(network.num_layers - 1)
            if not last_layer.get_output(0):
                network.mark_output(last_layer.get_output(0))

            print("input shape {}".format(network.get_input(0).shape))
            network.get_input(0).shape = [1, 3, -1, -1]
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            # ########################  SET DYNAMIC INPUT SHAPE #################################
            profile = builder.create_optimization_profile()
            profile.set_shape(network.get_input(0).name, (1, 3, 128, 128), (1, 3, 640, 640), (1, 3, 1920, 1920))
            config.add_optimization_profile(profile)
            engine = builder.build_engine(network, config)
            # engine = builder.build_cuda_engine(network)
            # ########################################################
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(engine.serialize())
            return engine

    # return build_engine()
    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()


def resize_image(img, short_size):
    height, width, _ = img.shape
    if height < width:
        new_height = short_size
        new_width = new_height / height * width
    else:
        new_width = short_size
        new_height = new_width / width * height
    new_height = int(round(new_height / 32) * 32)
    new_width = int(round(new_width / 32) * 32)
    resized_img = cv2.resize(img, (new_width, new_height))
    return resized_img


class DBNet:
    def __init__(self, model_path, post_p_thre=0.7, gpu_id=None, short_size=640):
        '''
        初始化pytorch模型, 转换tensorRT engine
        :param model_path: 模型地址(可以是模型的参数或者参数和计算图一起保存的文件)
        :param gpu_id: 在哪一块gpu上运行
        '''
        self.gpu_id = gpu_id

        if self.gpu_id is not None and isinstance(self.gpu_id, int) and torch.cuda.is_available():
            self.device = torch.device("cuda:%s" % self.gpu_id)
        else:
            self.device = torch.device("cpu")
        print('device:', self.device)
        self.model_path = model_path
        checkpoint = torch.load(model_path, map_location=self.device)

        config = checkpoint['config']
        config['arch']['backbone']['pretrained'] = False
        self.model = build_model(config['arch'])
        self.model.forward = self.model.forward4trt  # comment F.interpolate() of 'biliner' mode

        config['post_processing']['args']['unclip_ratio'] = 1.8
        self.post_process = get_post_processing(config['post_processing'])
        self.post_process.box_thresh = post_p_thre
        self.img_mode = config['dataset']['train']['dataset']['args']['img_mode']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.to(self.device)
        self.model.eval()

        self.short_size = short_size
        self.batch_size = 1
        self.input_names = ['inputs']
        self.output_names = ['pred_maps']
        # =============================================================================================================
        # Convert to onnx
        if not os.path.exists(model_path.replace('.pth', '_dynamic.onnx')):
            dummy_input = torch.randn(self.batch_size, 3, self.short_size, self.short_size).to(self.device)
            # dynamic_axes = {self.input_names[0]: {2:'width', 3:'height'},
            #                 self.output_names[0]: {2:'width', 3:'height'}}
            torch.onnx.export(self.model, dummy_input, model_path.replace('.pth', '_dynamic.onnx'),
                              # dynamic_axes= dynamic_axes,
                              input_names=self.input_names, output_names=self.output_names, verbose=True, opset_version=10)
            # onnx_model = onnx.load(model_path.replace('.pth', '_dynamic.onnx'))
            # onnx.checker.check_model(onnx_model)
            print('Converted to onnx model, save path {}!'.format(model_path.replace('.pth', '_dynamic.onnx')))

        # Convert to tensorRT engine
        self.engine = get_engine(model_path.replace('.pth', '_dynamic.onnx'),
                                     model_path.replace('.pth', '_dynamic.engine'))
        print('Converted to tensorRT engine, save path {}!'.format(
            model_path.replace('.pth', '_dynamic.engine')))
        self.context = self.engine.create_execution_context()
        # =============================================================================================================

        self.transform = []
        for t in config['dataset']['train']['dataset']['args']['transforms']:
            if t['type'] in ['ToTensor', 'Normalize']:
                self.transform.append(t)
        self.transform = get_transforms(self.transform)

    def predict(self, img_path: str, is_output_polygon=False, runtime='torch'):
        '''
        对传入的图像进行预测，支持图像地址,opencv 读取图片，偏慢
        :param img_path: 图像地址
        :param is_numpy:
        :param runtime: ['trt', 'torch']
        :return:
        '''
        assert os.path.exists(img_path), 'file is not exists'
        img = cv2.imread(img_path, 1 if self.img_mode != 'GRAY' else 0)
        if self.img_mode == 'RGB':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        # 改为长边resize, 短边pad, 方便batch 推理和加速
        img = resize_image(img, self.short_size)
        input_h, input_w, _ = img.shape

        # 将图片由(h,w,c)变为(1,c,h,w)
        tensor = self.transform(img)
        tensor = tensor.unsqueeze_(0)

        if runtime == 'torch' or 'both':
            tensor = tensor.to(self.device)
            batch = {'shape': [(h, w)]}
            with torch.no_grad():
                if str(self.device).__contains__('cuda'):
                    torch.cuda.synchronize(self.device)
                start = time.time()
                torch_outputs = self.model(tensor)
                preds = torch_outputs

                if str(self.device).__contains__('cuda'):
                    torch.cuda.synchronize(self.device)
                box_list, score_list = self.post_process(batch, preds, is_output_polygon=is_output_polygon)
                box_list, score_list = box_list[0], score_list[0]
                if len(box_list) > 0:
                    if is_output_polygon:
                        idx = [x.sum() > 0 for x in box_list]
                        box_list = [box_list[i] for i, v in enumerate(idx) if v]
                        score_list = [score_list[i] for i, v in enumerate(idx) if v]
                    else:
                        idx = box_list.reshape(box_list.shape[0], -1).sum(axis=1) > 0  # 去掉全为0的框
                        box_list, score_list = box_list[idx], score_list[idx]
                else:
                    box_list, score_list = [], []
                t = time.time() - start

        if runtime == 'trt' or runtime == 'both':
            if str(self.device).__contains__('cuda'):
                    torch.cuda.synchronize(self.device)
            batch = {'shape': [(h, w)]}
            # Notice: Here we only allocate device memory for speed up
            inputs, outputs, bindings, stream = common.allocate_buffersV2(self.engine, 1920, 1920)

            # Speed test: cpu(0.976s) vs gpu(0.719s)
            # ==> Set host input to the data.
            # The common.do_inference function will copy the input to the GPU before executing.
            inputs[0].host = tensor.cpu().numpy()  # for torch.Tensor
            # ==> Or set device input to the data.
            # in this mode, common.do_inference function should not copy inputs.host to inputs.device anymore.
            # c_type_pointer = ctypes.c_void_p(int(inputs[0].device))
            # x.cpu().numpy().copy_to_external(c_type_pointer)
            start = time.time()
            trt_outputs = common.do_inferenceV2(self.context, bindings=bindings, inputs=inputs, outputs=outputs,
                                              stream=stream,
                                              batch_size=self.batch_size, h_=input_h, w_=input_w)
            preds = torch.as_tensor(trt_outputs[0][:2*input_h*input_w], dtype=torch.float32, device=torch.device('cpu'))
            preds = preds.view(1, 2, input_h, input_w)
            box_list, score_list = self.post_process(batch, preds, is_output_polygon=is_output_polygon)
            t = time.time() - start
            box_list, score_list = box_list[0], score_list[0]
            if len(box_list) > 0:
                if is_output_polygon:
                    idx = [x.sum() > 0 for x in box_list]
                    box_list = [box_list[i] for i, v in enumerate(idx) if v]
                    score_list = [score_list[i] for i, v in enumerate(idx) if v]
                else:
                    idx = box_list.reshape(box_list.shape[0], -1).sum(axis=1) > 0  # 去掉全为0的框
                    box_list, score_list = box_list[idx], score_list[idx]
            else:
                box_list, score_list = [], []
        if runtime == 'both':
            print(
                "====================== Check output between tensorRT and torch =====================================")
            for i, name in enumerate(self.output_names):
                try:
                    np.testing.assert_allclose(torch_outputs[i].cpu().detach().numpy().reshape(-1), trt_outputs[i], rtol=1e-03,
                                               atol=2e-04)
                except AssertionError as e:
                    print("ouput {} mismatch {}".format(self.output_names[i], e))
                    continue
                print("ouput {} match\n".format(self.output_names[i]))

        if runtime not in ['trt', 'torch', 'both']:
            raise KeyError("support only ['torch', 'trt'] yet!")

        return preds[0, 0, :, :].detach().cpu().numpy(), box_list, score_list, t


def save_depoly(model, input, save_path):
    traced_script_model = torch.jit.trace(model, input)
    traced_script_model.save(save_path)


def init_args():
    import argparse
    parser = argparse.ArgumentParser(description='DBNet.pytorch')
    parser.add_argument('--model_path', default=r'output/DBNet_resnet18_FPN_DBHead/checkpoint/model_latest.pth',
                        type=str)
    parser.add_argument('--onnx', default="",
                        type=str)
    parser.add_argument('--input_folder', default='/media/data/zjs/BRFD/data/anomalyDetection/keypoints/val_crop_bill/10007',
                        type=str, help='img path for predict')
    parser.add_argument('--output_folder', default='./output/trt', type=str, help='img path for output')
    parser.add_argument('--thre', default=0.3, type=float, help='the thresh of post_processing')
    parser.add_argument('--polygon', action='store_true', help='output polygon or box')
    parser.add_argument('--show', action='store_true', help='show result')
    parser.add_argument('--save_result', action='store_true', help='save box and score to txt file')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    import pathlib
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    from utils.util import show_img, draw_bbox, save_result, get_file_list, crop_bbox

    args = init_args()
    print(args)
    if args.onnx:
         # Convert to tensorRT engine
        engine = get_engine(args.onnx,
                                     args.onnx.replace('.onnx', '_dynamic.engine'))
        print('Converted to tensorRT engine, save path {}!'.format(args.onnx.replace('.onnx', '_dynamic.engine')))
        
    # 初始化网络
    model = DBNet(args.model_path, post_p_thre=args.thre, gpu_id=0, short_size=640)
    img_folder = pathlib.Path(args.input_folder)
    for img_path in tqdm(get_file_list(args.input_folder, p_postfix=['.jpg'])):
        img = cv2.imread(img_path)
        preds, boxes_list, score_list, t = model.predict(img_path, is_output_polygon=args.polygon, runtime='trt')
        print('time cost: {}s'.format(t))
        crops = crop_bbox(img[:, :, ::-1], boxes_list)
        img = draw_bbox(img[:, :, ::-1], boxes_list)
        if args.show:
            show_img(preds)
            show_img(img, title=os.path.basename(img_path))
            plt.show()
        # 保存结果到路径
        os.makedirs(args.output_folder, exist_ok=True)
        img_path = pathlib.Path(img_path)
        output_path = os.path.join(args.output_folder, img_path.stem + '_result.jpg')
        pred_path = os.path.join(args.output_folder, img_path.stem + '_pred.jpg')
        cv2.imwrite(output_path, img[:, :, ::-1])
        cv2.imwrite(pred_path, preds * 255)
        for i, crop in enumerate(crops):
            cv2.imwrite(os.path.join(args.output_folder, img_path.stem + '_text_{:02d}.jpg'.format(i)), crop)
        save_result(output_path.replace('_result.jpg', '.txt'), boxes_list, score_list, args.polygon)

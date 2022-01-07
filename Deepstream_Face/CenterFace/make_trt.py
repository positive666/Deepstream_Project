import torch
from torch.autograd import Variable
import struct
import argparse
import os
import onnx
import tensorrt as trt
torch.cuda.synchronize()


TRT_LOGGER = trt.Logger()
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)



def args(known=False):
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='models/ocr-lstm.pth', type=str,help='path model') 
    parser.add_argument('--wts',  action='store_true', help='convert wts')
    parser.add_argument('--onnx',  action='store_true', help='convert onnx')
    parser.add_argument('--engine',  action='store_true', help='convert engine ')
    parser.add_argument('--output_path', default='', type=str, help=' path for onnx ')
    parser.add_argument('--dynamic',  action='store_true', help='ONNX/TF: dynamic axes')
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

def get_engine(onnx_file_path, engine_file_path=""):

    
    
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(
                network, TRT_LOGGER) as parser, builder.create_builder_config() as config:

            builder.strict_type_constraints = True
            # builder.max_workspace_size = 1 << 30  # deprecated, use config to set max_workspace_size
            # builder.fp16_mode = True   # deprecated, use config to set FP16 mode
            # builder.max_batch_size = 1  # deprecated, use EXPLICIT_BATCH

            config.set_flag(trt.BuilderFlag.FP16)
            config.max_workspace_size=1<<30

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
            network.get_input(0).shape = [1, 3 ,-1, -1]
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            # ########################  SET DYNAMIC INPUT SHAPE #################################
            profile = builder.create_optimization_profile()
            profile.set_shape(network.get_input(0).name, (1,3,32,32), (1, 3, 480, 640), (1, 3,960,960))
            config.add_optimization_profile(profile)
            engine = builder.build_engine(network, config)
            # engine = builder.build_cuda_engine(network)
            # ########################################################
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(engine.serialize())
            return engine
def build_engine(onnx_file_path, engine_file_path=""):
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.OnnxParser(
                network, TRT_LOGGER) as parser:
            builder.max_workspace_size = 1 << 30  # 1 GiB
            builder.max_batch_size = 1
            builder.fp16_mode = True
            # builder.strict_type_constraints = True

            # Parse model file
            # Try to load a previously generated CenterNet network graph in ONNX format:
            if not os.path.exists(onnx_file_path):
                print('ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(
                    onnx_file_path))
                exit(0)
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                parser.parse(model.read())

            # Reference: https://blog.csdn.net/weixin_43953045/article/details/103937295
            last_layer = network.get_layer(network.num_layers - 1)
            if not last_layer.get_output(0):
                network.mark_output(last_layer.get_output(0))
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            engine = builder.build_cuda_engine(network)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(engine.serialize())
            return engine
            
if __name__ == "__main__":
    opt = args()
    #model_path = 'models/ocr-lstm.pth'
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    print(device)
    #os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # set environment variable
    #assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {device} requested'
   


    if opt.onnx:
            input_names = ['data']
            output_names = ['prob']
            import os
               # Convert to onnx
             
            width=100
            if opt.dynamic:
                dynamic_axes= {'data':[3] 
                              } #数字0，1等是指张量的维度，表示哪个维度需要动态输入
                save_model_path=opt.output_path if opt.output_path else  opt.model_path.replace('.pth', '_dynamic.onnx')
                inputs=torch.randn(1, 1, 32, width).to(device)
            else :
                inputs=torch.randn(1,1,32,100).to(device)
                save_model_path=opt.output_path if opt.output_path else  opt.model_path.replace('.pth', '.onnx')
            #if not os.path.exists(save_model_path):
                     
        
            torch.onnx.export(model, inputs, save_model_path,
            export_params=True,
                                          dynamic_axes= dynamic_axes if opt.dynamic else None,
                                          keep_initializers_as_inputs=True,
                                          input_names=input_names, output_names=output_names, verbose=False, opset_version=11)
                        # onnx_model = onnx.load(model_path.replace('.pth', '_dynamic.onnx'))
                        # onnx.checker.check_model(onnx_model)
            print('Converted to onnx model, save path {}!'.format(save_model_path))
                        # Checks
            model_onnx = onnx.load(save_model_path)  # load onnx model
            print("successfully")
            onnx.checker.check_model(model_onnx)  # check onnx model
            # dim_proto0 = model_onnx.graph.input[0].type.tensor_type.shape.dim[0]
            # dim_proto3 = model_onnx.graph.input[0].type.tensor_type.shape.dim[3]
            # # 将该维度赋值为字符串，其维度不再为和dummy_input绑定的值
            # dim_proto0.dim_param = '1'
            # dim_proto3.dim_param = 'width'
            dim_proto_0 = model_onnx.graph.output[0].type.tensor_type.shape.dim[0]
            dim_proto_1 = model_onnx.graph.output[0].type.tensor_type.shape.dim[1]
            dim_proto_2 = model_onnx.graph.output[0].type.tensor_type.shape.dim[2]
            dim_proto_0.dim_param = '26'
            dim_proto_1.dim_param = '1'
            dim_proto_2.dim_param = '5530'
            onnx.save(model_onnx, 'infer_rec_dynamic.onnx')
    if opt.dynamic:        
        eng=get_engine(opt.model_path,'centerface_dynamice.engine')
    else:
        print("convert fixed engine file....")
        eng=build_engine(opt.model_path,'centerface.engine')
                
            
################################################################################
# Copyright (c) 2021, positive666.  All rights reserved.
################################################################################
.Instruction
CenterFace Deepstream project

# make sure your onnx and trt ,inpur dim 480*640 or 640*640
#my input 480*640 ->output 160*120
# 1.onnx->trt
cd Centerface 
wget https://github.com/Star-Clouds/CenterFace/raw/master/models/onnx/centerface.onnx
python   export_onnx.py
mkdir build
cd build
cmake ..
make
./CenterFace_trt ../config.yaml ../samples
mv centerface.trt ../

# 2.run deepstream_CenterFace
cd nvdsinfer_custom_impl_CenterFace/
make 
cd ..
wget https://developer.nvidia.com/blog/wp-content/uploads/2020/02/Redaction-A_1.mp4  // download the test video
deepstream-app -c deepstream_app_config_centerface.txt



################################################################################
# Copyright (c) 2021, positive666.  All rights reserved.
################################################################################
.Instruction
wget https://developer.nvidia.com/blog/wp-content/uploads/2020/02/Redaction-A_1.mp4  // download the test video

cd nvdsinfer_custom_impl_CenterFace/
make 
cd ..

deepstream-app -c deepstream_app_config_centerface.txt



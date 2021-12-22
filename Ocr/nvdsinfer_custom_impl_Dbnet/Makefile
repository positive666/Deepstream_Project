################################################################################
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
################################################################################

CUDA_VER?=11.1
ifeq ($(CUDA_VER),)
  $(error "CUDA_VER is not set")
endif
CC:= g++
NVCC:=/usr/local/cuda-$(CUDA_VER)/bin/nvcc

CFLAGS:= -Wall -std=c++11 -shared -fPIC 
CFLAGS+= -I/opt/nvidia/deepstream/deepstream-5.1/sources/includes/ -I/usr/local/cuda-$(CUDA_VER)/include  -I/usr/local/include/librdkafka
CFLAGS+=-I/home/project/install/opencv-4.5.0/include/opencv2/ 
#CFLAGS+=-I../../includes -I/opt/nvidia/deepstream/deepstream-5.0/sources/includes
LIBS:= -L/usr/local/cuda-11.1/lib64 -L/usr/lib/x86_64-linux-gnu -L/opt/nvidia/deepstream/deepstream-5.1/lib/ -lnvinfer_plugin -lnvinfer -lnvparsers -L/usr/local/cuda-$(CUDA_VER)/lib64 -lcudart -lcublas -lstdc++fs  -ljsoncpp -lrdkafka -lrdkafka++ -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml
LFLAGS:= -shared -Wl,--start-group $(LIBS) -Wl,--end-group   

LIBS+= `pkg-config --libs opencv`
INCS:= $(wildcard *.h)
INC += `pkg-config --cflags opencv`
SRCFILES:= nvdsparsebbox_dbnet.cpp   \
           kernels.cu    \
            trt_utils.cpp              \
           clipper/clipper.cpp 
TARGET_LIB:= libnvdsinfer_custom_impl_dbnet.so

TARGET_OBJS:= $(SRCFILES:.cpp=.o)
TARGET_OBJS:= $(TARGET_OBJS:.cu=.o)

all: $(TARGET_LIB)

%.o: %.cpp $(INCS) Makefile
	$(CC) -c -o $@ $(CFLAGS) $<

%.o: %.cu $(INCS) Makefile
	$(NVCC) -c -o $@ --compiler-options '-fPIC' $<

$(TARGET_LIB) : $(TARGET_OBJS)
	$(CC) -o $@  $(TARGET_OBJS) $(LFLAGS) 

clean:
	rm -rf $(TARGET_LIB)

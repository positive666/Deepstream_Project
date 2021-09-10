################################################################################
# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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
CFLAGS:= -Wall -std=c++14
CFLAGS+= -shared -fPIC -fpermissive
CC=g++
APP:=face

TARGET_DEVICE = $(shell gcc -dumpmachine | cut -f1 -d -)

NVDS_VERSION:=5.1

GST_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/gst-plugins/
LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/
APP_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/bin/

ifeq ($(TARGET_DEVICE),aarch64)
  CFLAGS+= -DPLATFORM_TEGRA  
endif

SRCS:= $(wildcard *.c)

SRCS+= $(wildcard /opt/nvidia/deepstream/deepstream-5.1/sources/apps/apps-common/src/*.c)

INCS:= $(wildcard *.h)

PKGS:= gstreamer-1.0 gstreamer-video-1.0  gstreamer-base-1.0 x11 json-glib-1.0 

OBJS:= $(SRCS:.c=.o)

CFLAGS+= -I./ -I/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/sources/apps/apps-common/includes \
		 -I/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/sources/includes -DDS_VERSION_MINOR=1 -DDS_VERSION_MAJOR=5 \
		 -I /usr/local/cuda-$(CUDA_VER)/include   -I/usr/local/include/librdkafka    -I/usr/local/include/opencv   -I/home/project/install/faiss-master/
		 

LIBS:= -L/usr/local/cuda-$(CUDA_VER)/lib64/ -lcudart -lcublasLt -lrdkafka -lrdkafka++ -lcublas -lcblas  

LIBS+= -L$(LIB_INSTALL_DIR) -lnvdsgst_meta -lnvds_meta -lnvdsgst_helper \
	   -lnvdsgst_smartrecord -lnvds_utils -lnvds_msgbroker -lm \
       -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml \
       -lcuda -lgstrtspserver-1.0 -ldl -Wl,-rpath,$(LIB_INSTALL_DIR) 
LIBS+=/home/project/install/faiss-master/libfaiss.a  
CFLAGS+= $(shell pkg-config --cflags $(PKGS))

LIBS+= $(shell pkg-config --libs $(PKGS))

all: $(APP)

%.o: %.c $(INCS) Makefile
	$(CC) -c -o $@ $(CFLAGS) $<

$(APP): $(OBJS) Makefile
	$(CC) -o $(APP) $(OBJS) $(LIBS)

install: $(APP)
	cp -rv $(APP) $(APP_INSTALL_DIR) $(GST_INSTALL_DIR)

clean:
	rm -rf $(OBJS) $(APP)

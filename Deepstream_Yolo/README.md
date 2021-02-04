################################################################################
### Copyright (c) 2021, positive666.  All rights reserved.
################################################################################
# 0.Instruction
This repository includes Yolo series project implementation based on Deepstream.
build your own Deepstream project and help you familiarize yourself with the process, you can deploy any inference model!!
including yolov5, yolov4, and And some model deployment projects such as OCR project based on yolo-ob detection and more  will be added later.

Reference:
you can git clone and prepare your models or test convert to trained models

yolov5: https://github.com/ultralytics/yolov5

yolov4: https://github.com/Tianxiaomo/pytorch-YOLOv4.git

download its weights!
# 1.Make sure your environment is configured
Check Nvidia graphics driver, CUDA, CUDNN, TensorRT, Deepstream
NOTE:If you ensure that your environment is configured, you can start to create a project!!
# 2.Convert the trained model file into an engine file
     pip install -r requirement.txt
     cd Deepstream_Project
     cd Deepstream_Yolo
## Here are the two ways,you can select yolov5s,yolov5m,yolov5l,yolov5x : 
### 1.trans_project(onnx---->trt)
### 2.tensorrtx project
   ### 2.0 methods_1----------trans_project(onnx---->trt):
	cd trans_project
   ### YOLOv4 (note: torch version==1.4.0)
        cd yolov4_convert
   ### 1)darnet models--->onnx ,(your v5 and v4 need two environments,different torch version)
       python demo_darknet2onnx.py   <cfgFile>  <weightFile>  <imageFile>  <batchSize>
   ### 2)pytorch models--->onnx:
       python demo_pytorch2onnx.py  <weight_file>  <image_path>  <batch_size>  <n_classes>  <IN_IMAGE_H>  <IN_IMAGE_W>  
   ### 3)onnx->trt,you will get model yolov4 trt file
       trtexec --onnx=<your onnx name >.onnx --explicitBatch --saveEngine=yolov4_1_3_320_512_fp16.engine --workspace=4096 --fp16
   ### if you can simpilfy onnx ,make sure your onnxsim and run :	   
       python -m onnxsim ./weights/yolov4.onnx  ./weights/yolov4_sim.onnx
   ### if you want to compile yolov4.cpp ,generate yolov4 trt file in your build folder:
           mkdir build && cd build
	   cmake ..
	   make 
           ./yolov4_convert   ../config.yaml       ../images
                            <your config file>   <your test data folder>

   ### YOLOV5(note: torch version==1.7.0 )
	  cd yolov5_convert	   
   ### 1)your source actvivate your python environment,input your yolov5 models into models/
   ### 2)run demo_pytorch2onnx.py to generate yolov5 onnx file:             
	   python demo_pytorch2onnx.py --weights ./weights/yolov5x.pt  --img 640   --batch 1       
	                                           <your model path>   <imagsize>  <batchsize>
   ###   or you can try this code ,run:
           python convert.py  --weights <your modle path >  --img-size<imgsize> --bata-size<default=1>
 	   
   ###   if you can simpilfy onnx ,make sure your onnxsim and run :   
	   python -m onnxsim ./weights/yolov5x.onnx  ./weights/yolov5x_sim.onnx
	                       <input>                     <output>

   ### 3)compile yolov5.cpp ,generate yolov5 enginel:
	   mkdir build && cd build
	   cmake ..
	   make 
	  ./yolov5_trt ../config.yaml        ../images
	               <your config file>   <your test data folder>	   	   
  ## 2.1 methods:tensorrtx projcet: 
  ### reference:https://github.com/wang-xinyu/tensorrtx
  ### NOTE: if you use this project ,you should have yolov4 and yolov5 project sources models to convert modlde file,so you can use the project  prepared.
  ### Go back to your project root directory
  ### Yolov5 as a sample:
  ### 1)source activate your yolov5 conda env:
       source actviate <yolov5 conda env name>
       cd tensorrtx/yolov5
  ### 2)generate wts file && compile yolov5.cpp
       python gen_wts.py  <input weights file >  <outputfile_name>                            
       mkdir build && cd build
       cmake ..
       make 
  ### 3)run & generate yolov5x.engine and libmyplugin.so:
       [NOTE]：you should line 13 in yolov5.cpp , #define NET x  // s m l x,configure your yolov5 model 
         ./yolov5 -x 
                  -s
                  -l
                  -m
  ### 1) test your yolov5 engine:
          sudo ./yolov5 -d  ../samples
  ### 2) mv your file 
          cp -r  yolov5*.engine ../../engine_models/
          cp  -r libmyplugins.so ../../engine_models/
# 3.Configure your deepstream & run yolov4 and yolov5 deepstream app
      In any case, the above is just to get the engine file of the model you trained. If you have another way or modify the code to generate the engine file.
      Go back to your project root directory,Deepstream_Yolo/:
## 3.1  run Yolo Deepstream
  ### 1)compile nvdsparsebbox_Yolo.cpp ,(includes yolov4,yolov5) 
      cd nvdsinfer_custom_impl_Yolo
      cmake..
      make  
      cd ..
  ### 2)configure your deepstream_app_config_yoloV<your object >.txt & onfig_infer_primary_yoloV<your object>.txt    
  ### such as :
  ## if your run yolov4,configure your deepstream_app_config_yoloV4.txt,  
      deepstream-app -c deepstream_app_config_yoloV4.txt 
     
  ## if your use tensorrtx to generate YoloV5 enigne, add export libmyplugins.so file path,
      LD_PRELOAD=./libmyplugins.so deepstream-app -c deepstream_app_config_yoloV5.txt

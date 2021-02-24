This OCR project is a YoloV3-based detection model and a CRNN-based recognition model. You need to prepare the text detection conversion engine file and text recognition engine file for training. Follow-up updates, so stay tuned

[two modle file]:
# 1.0 Compile text_detection cpp and crnn parse cpp 
cd Ocr
cd nvdsinfer_custom_impl_Yolo   
make
cd nvdsinfer_custom_text
make 
# 2.0 configure deepstream txt &run 
deepstream-app -c deepstream_app_config_yoloV3.txt 

`Detailed updates and explanations will follow`
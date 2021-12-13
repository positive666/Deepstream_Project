cmake_minimum_required(VERSION 3.5)

project(arcface_trt)

set(CMAKE_CXX_STANDARD 14)

# openblas
set(BLA_VENDER OpenBLAS)
find_package(BLAS REQUIRED)
if(BLAS_FOUND)
    message("-- OpenBLAS found.")
    include_directories(/opt/OpenBLAS/include/)
endif(BLAS_FOUND)
# cublas
set(CUBLASLT_LIBRARY cublasLt)
find_library(CUBLASLT_LIBRARY cublasLt ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})




# CUDA
find_package(CUDA REQUIRED)
message(STATUS "Find CUDA include at ${CUDA_INCLUDE_DIRS}")
message(STATUS "Find CUDA libraries: ${CUDA_LIBRARIES}")

# TensorRT
set(TENSORRT_ROOT /home/project/install/TensorRT-7.2.3.4/)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
        HINTS ${TENSORRT_ROOT} PATH_SUFFIXES include/)
message(STATUS "Found TensorRT headers at ${TENSORRT_INCLUDE_DIR}")
find_library(TENSORRT_LIBRARY_INFER nvinfer
        HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_LIBRARY_ONNXPARSER nvonnxparser
        HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64)
set(TENSORRT_LIBRARY ${TENSORRT_LIBRARY_INFER} ${TENSORRT_LIBRARY_ONNXPARSER})
message(STATUS "Find TensorRT libs: ${TENSORRT_LIBRARY}")




cuda_add_library(myplugins SHARED ${PROJECT_SOURCE_DIR}/prelu.cu)

# OpenCV
find_package(OpenCV REQUIRED)
message(STATUS "Find OpenCV include at ${OpenCV_INCLUDE_DIRS}")
message(STATUS "Find OpenCV libraries: ${OpenCV_LIBRARIES}")

# openblas
set(BLA_VENDER OpenBLAS)
find_package(BLAS REQUIRED)
if(BLAS_FOUND)
    message("-- OpenBLAS found.")
    include_directories(/opt/OpenBLAS/include/)
endif(BLAS_FOUND)

# cublas
set(CUBLASLT_LIBRARY cublasLt)
find_library(CUBLASLT_LIBRARY cublasLt ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})





# set(FAISS_ROOT /home/project/install/faiss-master/)
# #message(STATUS "Find faiss include at ${FAISS_ROOT}")
# set(FAISS_INCLUDE /home/project/install/faiss-master/c_api)
# find_library(FAISS_ROOT  libfaiss.a)
# set(FAISS_LIB_DIR  /home/project/install/faiss-master/libfaiss.a  libfaiss.so)
# message(STATUS "Find FAlibraries: ${FAISS_LIB_DIR}")

set(COMMON_INCLUDE ./includes/common)
set(YAML_INCLUDE ./includes/yaml-cpp/include)
set(YAML_LIB_DIR ./includes/yaml-cpp/libs)

include_directories(${CUDA_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIR}  ${FAISS_INCLUDE} ${OpenCV_INCLUDE_DIRS} ${COMMON_INCLUDE} ${YAML_INCLUDE})
link_directories(${YAML_LIB_DIR} ${FAISS_LIB_DIR} )

add_executable(arcface-r100 arcface-r100.cpp )
add_executable(arcface-mobilefacenet arcface-mobilefacenet.cpp)
add_executable(arcface-r50 arcface-r50.cpp)


target_link_libraries(arcface-mobilefacenet PRIVATE   ${OpenCV_LIBRARIES} ${CUDA_LIBRARIES} ${TENSORRT_LIBRARY}  ${BLAS_LIBRARIES}  ${CUBLASLT_LIBRARY}  yaml-cpp  myplugins)   #if your need faiss ,add it.
target_link_libraries(arcface-r100 PRIVATE  ${OpenCV_LIBRARIES} ${CUDA_LIBRARIES} ${TENSORRT_LIBRARY}  ${BLAS_LIBRARIES}  ${CUBLASLT_LIBRARY}  yaml-cpp  myplugins)
target_link_libraries(arcface-r50 PRIVATE  ${OpenCV_LIBRARIES} ${CUDA_LIBRARIES} ${TENSORRT_LIBRARY}  ${BLAS_LIBRARIES}  ${CUBLASLT_LIBRARY}  yaml-cpp myplugins)
add_definitions(-O2 -pthread)
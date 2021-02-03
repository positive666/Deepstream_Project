/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "yolo.h"
#include "yoloPlugins.h"

#include <fstream>
#include <iomanip>
#include <iterator>

Yolo::Yolo(const NetworkInfo &networkInfo, nvinfer1::IBuilder* builder) :
    m_NetworkType(networkInfo.networkType), // yolov3
    m_ConfigFilePath(networkInfo.configFilePath), //yolov3.cfg
    m_WtsFilePath(networkInfo.wtsFilePath), // yolov3.weights
    m_DeviceType (networkInfo.deviceType), // kDLA, kGPU
    m_InputBlobName(networkInfo.inputBlobName), // data
    m_InputH(0),
    m_InputW(0),
    m_InputC(0),
    m_InputSize(0),
    m_Builder (builder),
    m_TinyMaxpoolPaddingFormula(new YoloTinyMaxpoolPaddingFormula)
{
    assert (builder);
}

Yolo::~Yolo()
{
    m_TinyMaxpoolPaddingFormula.reset();
}

nvinfer1::ICudaEngine *Yolo::createEngine ()
{
    m_configBlocks = parseConfigFile(m_ConfigFilePath);
	std::cout << "@@@@@@@@@@@@@@@&&&&&&&&&&&&& yolo cpp:54..." <<m_ConfigFilePath<<std::endl;
    parseConfigBlocks();
    assert (m_Builder);

    std::vector<float> weights = loadWeights(m_WtsFilePath, m_NetworkType);
    std::vector<nvinfer1::Weights> trtWeights;

    nvinfer1::INetworkDefinition *network = createYoloNetwork (weights, trtWeights);

    // Build the engine
    std::cout << "Building the TensorRT Engine..." << std::endl;
    nvinfer1::ICudaEngine * engine = m_Builder->buildCudaEngine(*network);
    if (engine) {
        std::cout << "Building complete!" << std::endl;
    } else {
        std::cerr << "Building engine failed!" << std::endl;
    }

    // destroy
    destroyNetworkUtils(network, trtWeights);
    return engine;
}

nvinfer1::INetworkDefinition *Yolo::createYoloNetwork (
    std::vector<float> &weights, std::vector<nvinfer1::Weights> &trtWeights)
{
    int weightPtr = 0;
    int channels = m_InputC;
    nvinfer1::INetworkDefinition * network = m_Builder->createNetwork();

    nvinfer1::ITensor* data = network->addInput(
        m_InputBlobName.c_str(), nvinfer1::DataType::kFLOAT,
        nvinfer1::DimsCHW{static_cast<int>(m_InputC), static_cast<int>(m_InputH),
                          static_cast<int>(m_InputW)});
    assert(data != nullptr);
    // Add elementwise layer to normalize pixel values 0-1
    nvinfer1::Dims divDims{
        3,
        {static_cast<int>(m_InputC), static_cast<int>(m_InputH), static_cast<int>(m_InputW)},
        {nvinfer1::DimensionType::kCHANNEL, nvinfer1::DimensionType::kSPATIAL,
         nvinfer1::DimensionType::kSPATIAL}};
    nvinfer1::Weights divWeights{nvinfer1::DataType::kFLOAT, nullptr,
                                 static_cast<int64_t>(m_InputSize)};
    float* divWt = new float[m_InputSize];
    for (uint w = 0; w < m_InputSize; ++w) divWt[w] = 255.0;
    divWeights.values = divWt;
    trtWeights.push_back(divWeights);
    nvinfer1::IConstantLayer* constDivide = network->addConstant(divDims, divWeights);
    assert(constDivide != nullptr);
    nvinfer1::IElementWiseLayer* elementDivide = network->addElementWise(
        *data, *constDivide->getOutput(0), nvinfer1::ElementWiseOperation::kDIV);
    assert(elementDivide != nullptr);

    nvinfer1::ITensor* previous = elementDivide->getOutput(0);
    std::vector<nvinfer1::ITensor*> tensorOutputs;
    uint outputTensorCount = 0;

    // Set the output dimensions formula for pooling layers
    assert(m_TinyMaxpoolPaddingFormula && "Tiny maxpool padding formula not created");
    network->setPoolingOutputDimensionsFormula(m_TinyMaxpoolPaddingFormula.get());

    // build the network using the network API
    for (uint i = 0; i < m_configBlocks.size(); ++i)
    {
        // check if num. of channels is correct
        assert(getNumChannels(previous) == channels);
        std::string layerIndex = "(" + std::to_string(i) + ")";

        if (m_configBlocks.at(i).at("type") == "net")
        {
            printLayerInfo("", "layer", "     inp_size", "     out_size", "weightPtr");
        }
        else if (m_configBlocks.at(i).at("type") == "convolutional")
        {
            std::string inputVol = dimsToString(previous->getDimensions());
            nvinfer1::ILayer* out;
            std::string layerType;
            // check if batch_norm enabled
            if (m_configBlocks.at(i).find("batch_normalize") != m_configBlocks.at(i).end())
            {
                out = netAddConvBNLeaky(i, m_configBlocks.at(i), weights, trtWeights, weightPtr,
                                        channels, previous, network);
                layerType = "conv-bn-leaky";
            }
            else
            {
                out = netAddConvLinear(i, m_configBlocks.at(i), weights, trtWeights, weightPtr,
                                       channels, previous, network);
                layerType = "conv-linear";
            }
            previous = out->getOutput(0);
            assert(previous != nullptr);
            channels = getNumChannels(previous);
            std::string outputVol = dimsToString(previous->getDimensions());
            tensorOutputs.push_back(out->getOutput(0));
            printLayerInfo(layerIndex, layerType, inputVol, outputVol, std::to_string(weightPtr));
        }
        else if (m_configBlocks.at(i).at("type") == "shortcut")
        {
            assert(m_configBlocks.at(i).at("activation") == "linear");
            assert(m_configBlocks.at(i).find("from") != m_configBlocks.at(i).end());
            int from = stoi(m_configBlocks.at(i).at("from"));

            std::string inputVol = dimsToString(previous->getDimensions());
            // check if indexes are correct
            assert((i - 2 >= 0) && (i - 2 < tensorOutputs.size()));
            assert((i + from - 1 >= 0) && (i + from - 1 < tensorOutputs.size()));
            assert(i + from - 1 < i - 2);
            nvinfer1::IElementWiseLayer* ew
                = network->addElementWise(*tensorOutputs[i - 2], *tensorOutputs[i + from - 1],
                                            nvinfer1::ElementWiseOperation::kSUM);
            assert(ew != nullptr);
            std::string ewLayerName = "shortcut_" + std::to_string(i);
            ew->setName(ewLayerName.c_str());
            previous = ew->getOutput(0);
            assert(previous != nullptr);
            std::string outputVol = dimsToString(previous->getDimensions());
            tensorOutputs.push_back(ew->getOutput(0));
            printLayerInfo(layerIndex, "skip", inputVol, outputVol, "    -");
        }
        else if (m_configBlocks.at(i).at("type") == "yolo")
        {
            nvinfer1::Dims prevTensorDims = previous->getDimensions();
			std::cout << "yolo   yolo   yolo :" << std::endl;
            assert(prevTensorDims.d[1] == prevTensorDims.d[2]);
            TensorInfo& curYoloTensor = m_OutputTensors.at(outputTensorCount);
            curYoloTensor.gridSize = prevTensorDims.d[1];
            curYoloTensor.stride = m_InputW / curYoloTensor.gridSize;
            m_OutputTensors.at(outputTensorCount).volume = curYoloTensor.gridSize
                * curYoloTensor.gridSize
                * (curYoloTensor.numBBoxes * (5 + curYoloTensor.numClasses));
            std::string layerName = "yolo_" + std::to_string(i);
            curYoloTensor.blobName = layerName;
            nvinfer1::IPluginV2* yoloPlugin
                = new YoloLayerV3(m_OutputTensors.at(outputTensorCount).numBBoxes,
                                  m_OutputTensors.at(outputTensorCount).numClasses,
                                  m_OutputTensors.at(outputTensorCount).gridSize);
            assert(yoloPlugin != nullptr);
            nvinfer1::IPluginV2Layer* yolo = network->addPluginV2(&previous, 1, *yoloPlugin);
            assert(yolo != nullptr);
            yolo->setName(layerName.c_str());
            std::string inputVol = dimsToString(previous->getDimensions());
            previous = yolo->getOutput(0);
            assert(previous != nullptr);
            previous->setName(layerName.c_str());
            std::string outputVol = dimsToString(previous->getDimensions());
            network->markOutput(*previous);
            channels = getNumChannels(previous);
            tensorOutputs.push_back(yolo->getOutput(0));
            printLayerInfo(layerIndex, "yolo", inputVol, outputVol, std::to_string(weightPtr));
            ++outputTensorCount;
        }
        else if (m_configBlocks.at(i).at("type") == "region")
        {
            nvinfer1::Dims prevTensorDims = previous->getDimensions();
            assert(prevTensorDims.d[1] == prevTensorDims.d[2]);
            TensorInfo& curRegionTensor = m_OutputTensors.at(outputTensorCount);
            curRegionTensor.gridSize = prevTensorDims.d[1];
            curRegionTensor.stride = m_InputW / curRegionTensor.gridSize;
            m_OutputTensors.at(outputTensorCount).volume = curRegionTensor.gridSize
                * curRegionTensor.gridSize
                * (curRegionTensor.numBBoxes * (5 + curRegionTensor.numClasses));
            std::string layerName = "region_" + std::to_string(i);
            curRegionTensor.blobName = layerName;
            nvinfer1::plugin::RegionParameters RegionParameters{
                static_cast<int>(curRegionTensor.numBBoxes), 4,
                static_cast<int>(curRegionTensor.numClasses), nullptr};
            std::string inputVol = dimsToString(previous->getDimensions());
            nvinfer1::IPluginV2* regionPlugin
                = createRegionPlugin(RegionParameters);
            assert(regionPlugin != nullptr);
            nvinfer1::IPluginV2Layer* region = network->addPluginV2(&previous, 1, *regionPlugin);
            assert(region != nullptr);
            region->setName(layerName.c_str());
            previous = region->getOutput(0);
            assert(previous != nullptr);
            previous->setName(layerName.c_str());
            std::string outputVol = dimsToString(previous->getDimensions());
            network->markOutput(*previous);
            channels = getNumChannels(previous);
            tensorOutputs.push_back(region->getOutput(0));
            printLayerInfo(layerIndex, "region", inputVol, outputVol, std::to_string(weightPtr));
            std::cout << "Anchors are being converted to network input resolution i.e. Anchors x "
                      << curRegionTensor.stride << " (stride)" << std::endl;
            for (auto& anchor : curRegionTensor.anchors) anchor *= curRegionTensor.stride;
            ++outputTensorCount;
        }
        else if (m_configBlocks.at(i).at("type") == "reorg")
        {
            std::string inputVol = dimsToString(previous->getDimensions());
            nvinfer1::IPluginV2* reorgPlugin = createReorgPlugin(2);
            assert(reorgPlugin != nullptr);
            nvinfer1::IPluginV2Layer* reorg = network->addPluginV2(&previous, 1, *reorgPlugin);
            assert(reorg != nullptr);

            std::string layerName = "reorg_" + std::to_string(i);
            reorg->setName(layerName.c_str());
            previous = reorg->getOutput(0);
            assert(previous != nullptr);
            std::string outputVol = dimsToString(previous->getDimensions());
            channels = getNumChannels(previous);
            tensorOutputs.push_back(reorg->getOutput(0));
            printLayerInfo(layerIndex, "reorg", inputVol, outputVol, std::to_string(weightPtr));
        }
        // route layers (single or concat)
        else if (m_configBlocks.at(i).at("type") == "route")
        {
            size_t found = m_configBlocks.at(i).at("layers").find(",");
            if (found != std::string::npos)
            {
                int idx1 = std::stoi(trim(m_configBlocks.at(i).at("layers").substr(0, found)));
                int idx2 = std::stoi(trim(m_configBlocks.at(i).at("layers").substr(found + 1)));
                if (idx1 < 0)
                {
                    idx1 = tensorOutputs.size() + idx1;
                }
                if (idx2 < 0)
                {
                    idx2 = tensorOutputs.size() + idx2;
                }
                assert(idx1 < static_cast<int>(tensorOutputs.size()) && idx1 >= 0);
                assert(idx2 < static_cast<int>(tensorOutputs.size()) && idx2 >= 0);
                nvinfer1::ITensor** concatInputs
                    = reinterpret_cast<nvinfer1::ITensor**>(malloc(sizeof(nvinfer1::ITensor*) * 2));
                concatInputs[0] = tensorOutputs[idx1];
                concatInputs[1] = tensorOutputs[idx2];
                nvinfer1::IConcatenationLayer* concat
                    = network->addConcatenation(concatInputs, 2);
                assert(concat != nullptr);
                std::string concatLayerName = "route_" + std::to_string(i - 1);
                concat->setName(concatLayerName.c_str());
                // concatenate along the channel dimension
                concat->setAxis(0);
                previous = concat->getOutput(0);
                assert(previous != nullptr);
                std::string outputVol = dimsToString(previous->getDimensions());
                // set the output volume depth
                channels
                    = getNumChannels(tensorOutputs[idx1]) + getNumChannels(tensorOutputs[idx2]);
                tensorOutputs.push_back(concat->getOutput(0));
                printLayerInfo(layerIndex, "route", "        -", outputVol,
                               std::to_string(weightPtr));
            }
            else
            {
                int idx = std::stoi(trim(m_configBlocks.at(i).at("layers")));
                if (idx < 0)
                {
                    idx = tensorOutputs.size() + idx;
                }
                assert(idx < static_cast<int>(tensorOutputs.size()) && idx >= 0);
                previous = tensorOutputs[idx];
                assert(previous != nullptr);
                std::string outputVol = dimsToString(previous->getDimensions());
                // set the output volume depth
                channels = getNumChannels(tensorOutputs[idx]);
                tensorOutputs.push_back(tensorOutputs[idx]);
                printLayerInfo(layerIndex, "route", "        -", outputVol,
                               std::to_string(weightPtr));
            }
        }
        else if (m_configBlocks.at(i).at("type") == "upsample")
        {
            std::string inputVol = dimsToString(previous->getDimensions());
            nvinfer1::ILayer* out = netAddUpsample(i - 1, m_configBlocks[i], weights, trtWeights,
                                                   channels, previous, network);
            previous = out->getOutput(0);
            std::string outputVol = dimsToString(previous->getDimensions());
            tensorOutputs.push_back(out->getOutput(0));
            printLayerInfo(layerIndex, "upsample", inputVol, outputVol, "    -");
        }
        else if (m_configBlocks.at(i).at("type") == "maxpool")
        {
            // Add same padding layers
            if (m_configBlocks.at(i).at("size") == "2" && m_configBlocks.at(i).at("stride") == "1")
            {
                m_TinyMaxpoolPaddingFormula->addSamePaddingLayer("maxpool_" + std::to_string(i));
            }
            std::string inputVol = dimsToString(previous->getDimensions());
            nvinfer1::ILayer* out = netAddMaxpool(i, m_configBlocks.at(i), previous, network);
            previous = out->getOutput(0);
            assert(previous != nullptr);
            std::string outputVol = dimsToString(previous->getDimensions());
            tensorOutputs.push_back(out->getOutput(0));
            printLayerInfo(layerIndex, "maxpool", inputVol, outputVol, std::to_string(weightPtr));
        }
        else
        {
            std::cout << "Unsupported layer type --> \"" << m_configBlocks.at(i).at("type") << "\""
                      << std::endl;
            assert(0);
        }
    }

    if ((int)weights.size() != weightPtr)
    {
        std::cout << "Number of unused weights left : " << weights.size() - weightPtr << std::endl;
        assert(0);
    }

    std::cout << "Output blob names :" << std::endl;
    for (auto& tensor : m_OutputTensors) std::cout << tensor.blobName << std::endl;

    int nbLayers = network->getNbLayers();
    int layersOnDLA = 0;
    std::cout << "Total number of layers: " << nbLayers << std::endl;
    for (int i = 0; i < nbLayers; i++)
    {
        nvinfer1::ILayer* curLayer = network->getLayer(i);
        if (m_DeviceType == "kDLA" && m_Builder->canRunOnDLA(curLayer))
        {
            m_Builder->setDeviceType(curLayer, nvinfer1::DeviceType::kDLA);
            layersOnDLA++;
            std::cout << "Set layer " << curLayer->getName() << " to run on DLA" << std::endl;
        }
    }
    std::cout << "Total number of layers on DLA: " << layersOnDLA << std::endl;

    return network;
}

std::vector<std::map<std::string, std::string>>
Yolo::parseConfigFile (const std::string cfgFilePath)
{
    assert(fileExists(cfgFilePath));
    std::ifstream file(cfgFilePath);
    assert(file.good());
    std::string line;
    std::vector<std::map<std::string, std::string>> blocks;
    std::map<std::string, std::string> block;

    while (getline(file, line))
    {
        if (line.size() == 0) continue;
        if (line.front() == '#') continue;
        line = trim(line);
        if (line.front() == '[')
        {
            if (block.size() > 0)
            {
                blocks.push_back(block);
                block.clear();
            }
            std::string key = "type";
            std::string value = trim(line.substr(1, line.size() - 2));
            block.insert(std::pair<std::string, std::string>(key, value));
        }
        else
        {
            int cpos = line.find('=');
            std::string key = trim(line.substr(0, cpos));
            std::string value = trim(line.substr(cpos + 1));
            block.insert(std::pair<std::string, std::string>(key, value));
        }
    }
    blocks.push_back(block);
    return blocks;
}

void Yolo::parseConfigBlocks()
{
    for (auto block : m_configBlocks)
    {
        if (block.at("type") == "net")
        {
            assert((block.find("height") != block.end())
                   && "Missing 'height' param in network cfg");
            assert((block.find("width") != block.end()) && "Missing 'width' param in network cfg");
            assert((block.find("channels") != block.end())
                   && "Missing 'channels' param in network cfg");

            m_InputH = std::stoul(block.at("height"));
            m_InputW = std::stoul(block.at("width"));
            m_InputC = std::stoul(block.at("channels"));
            assert(m_InputW == m_InputH);
            m_InputSize = m_InputC * m_InputH * m_InputW;
        }
        else if ((block.at("type") == "region") || (block.at("type") == "yolo"))
        {
            assert((block.find("num") != block.end())
                   && std::string("Missing 'num' param in " + block.at("type") + " layer").c_str());
            assert((block.find("classes") != block.end())
                   && std::string("Missing 'classes' param in " + block.at("type") + " layer")
                          .c_str());
            assert((block.find("anchors") != block.end())
                   && std::string("Missing 'anchors' param in " + block.at("type") + " layer")
                          .c_str());

            TensorInfo outputTensor;
            std::string anchorString = block.at("anchors");
            while (!anchorString.empty())
            {
                int npos = anchorString.find_first_of(',');
                if (npos != -1)
                {
                    float anchor = std::stof(trim(anchorString.substr(0, npos)));
                    outputTensor.anchors.push_back(anchor);
                    anchorString.erase(0, npos + 1);
                }
                else
                {
                    float anchor = std::stof(trim(anchorString));
                    outputTensor.anchors.push_back(anchor);
                    break;
                }
            }

            if ((m_NetworkType == "yolov3") || (m_NetworkType == "yolov3-tiny"))
            {
                assert((block.find("mask") != block.end())
                       && std::string("Missing 'mask' param in " + block.at("type") + " layer")
                              .c_str());

                std::string maskString = block.at("mask");
                while (!maskString.empty())
                {
                    int npos = maskString.find_first_of(',');
                    if (npos != -1)
                    {
                        uint mask = std::stoul(trim(maskString.substr(0, npos)));
                        outputTensor.masks.push_back(mask);
                        maskString.erase(0, npos + 1);
                    }
                    else
                    {
                        uint mask = std::stoul(trim(maskString));
                        outputTensor.masks.push_back(mask);
                        break;
                    }
                }
            }

            outputTensor.numBBoxes = outputTensor.masks.size() > 0
                ? outputTensor.masks.size()
                : std::stoul(trim(block.at("num")));
            outputTensor.numClasses = std::stoul(block.at("classes"));
            m_OutputTensors.push_back(outputTensor);
        }
    }
}

void Yolo::destroyNetworkUtils (
    nvinfer1::INetworkDefinition *network,
    std::vector<nvinfer1::Weights>& trtWeights)
{
    if (network) network->destroy();

    // deallocate the weights
    for (uint i = 0; i < trtWeights.size(); ++i)
    {
        if (trtWeights[i].count > 0) free(const_cast<void*>(trtWeights[i].values));
    }
}


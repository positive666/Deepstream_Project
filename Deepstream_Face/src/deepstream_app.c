/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
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
#include <algorithm>
#include <gst/gst.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "nvbufsurface.h"
// #include <curl.h>
// #include <json-glib.h>
#include <glib-object.h>
#include <json-glib/json-glib.h>
#include "cJSON.h"
#include <assert.h>
#include <opencv2/opencv.hpp>

#include "deepstream_app.h"

/* #include <faiss/index_io.h>
#include <faiss/IndexHNSW.h>
#include <faiss/MetaIndexes.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQ.h> */
#define MAX_DISPLAY_LEN 64
static guint batch_num = 0;
static guint demux_batch_num = 0;

GST_DEBUG_CATEGORY_EXTERN (NVDS_APP);

GQuark _dsmeta_quark;

#define CEIL(a,b) ((a + b - 1) / b)

/**
 * @brief  Add the (nvmsgconv->nvmsgbroker) sink-bin to the
 *         overall DS pipeline (if any configured) and link the same to
 *         common_elements.tee (This tee connects
 *         the common analytics path to Tiler/display-sink and
 *         to configured broker sink if any)
 *         NOTE: This API shall return TRUE if there are no
 *         broker sinks to add to pipeline
 *
 * @param  appCtx [IN]
 * @return TRUE if succussful; FALSE otherwise
 */
static gboolean add_and_link_broker_sink (AppCtx * appCtx);

/**
 * @brief  Checks if there are any [sink] groups
 *         configured for source_id=provided source_id
 *         NOTE: source_id key and this API is valid only when we
 *         disable [tiler] and thus use demuxer for individual
 *         stream out
 * @param  config [IN] The DS Pipeline configuration struct
 * @param  source_id [IN] Source ID for which a specific [sink]
 *         group is searched for
 */
static gboolean is_sink_available_for_source_id(NvDsConfig *config, guint source_id);

/**
 * callback function to receive messages from components
 * in the pipeline.
 */
static gboolean
bus_callback (GstBus * bus, GstMessage * message, gpointer data)
{
  AppCtx *appCtx = (AppCtx *) data;
  GST_CAT_DEBUG (NVDS_APP,
      "Received message on bus: source %s, msg_type %s",
      GST_MESSAGE_SRC_NAME (message), GST_MESSAGE_TYPE_NAME (message));
  switch (GST_MESSAGE_TYPE (message)) {
    case GST_MESSAGE_INFO:{
      GError *error = NULL;
      gchar *debuginfo = NULL;
      gst_message_parse_info (message, &error, &debuginfo);
      g_printerr ("INFO from %s: %s\n",
          GST_OBJECT_NAME (message->src), error->message);
      if (debuginfo) {
        g_printerr ("Debug info: %s\n", debuginfo);
      }
      g_error_free (error);
      g_free (debuginfo);
      break;
    }
    case GST_MESSAGE_WARNING:{
      GError *error = NULL;
      gchar *debuginfo = NULL;
      gst_message_parse_warning (message, &error, &debuginfo);
      g_printerr ("WARNING from %s: %s\n",
          GST_OBJECT_NAME (message->src), error->message);
      if (debuginfo) {
        g_printerr ("Debug info: %s\n", debuginfo);
      }
      g_error_free (error);
      g_free (debuginfo);
      break;
    }
    case GST_MESSAGE_ERROR:{
      GError *error = NULL;
      gchar *debuginfo = NULL;
      guint i = 0;
      gst_message_parse_error (message, &error, &debuginfo);
      g_printerr ("ERROR from %s: %s\n",
          GST_OBJECT_NAME (message->src), error->message);
      if (debuginfo) {
        g_printerr ("Debug info: %s\n", debuginfo);
      }

      NvDsSrcParentBin *bin = &appCtx->pipeline.multi_src_bin;
      GstElement *msg_src_elem = (GstElement *) GST_MESSAGE_SRC (message);
      gboolean bin_found = FALSE;
      /* Find the source bin which generated the error. */
      while (msg_src_elem && !bin_found) {
        for (i = 0; i < bin->num_bins && !bin_found; i++) {
          if (bin->sub_bins[i].src_elem == msg_src_elem ||
                  bin->sub_bins[i].bin == msg_src_elem) {
            bin_found = TRUE;
            break;
          }
        }
        msg_src_elem = GST_ELEMENT_PARENT (msg_src_elem);
      }

      if ((i != bin->num_bins) &&
          (appCtx->config.multi_source_config[0].type == NV_DS_SOURCE_RTSP)) {
        // Error from one of RTSP source.
        NvDsSrcBin *subBin = &bin->sub_bins[i];

        if (!subBin->reconfiguring ||
            g_strrstr(debuginfo, "500 (Internal Server Error)")) {
          subBin->reconfiguring = TRUE;
          g_timeout_add (0, reset_source_pipeline, subBin);
        }
        g_error_free (error);
        g_free (debuginfo);
        return TRUE;
      }

      if (appCtx->config.multi_source_config[0].type == NV_DS_SOURCE_CAMERA_V4L2) {
        if (g_strrstr(debuginfo, "reason not-negotiated (-4)")) {
          NVGSTDS_INFO_MSG_V ("incorrect camera parameters provided, please provide supported resolution and frame rate\n");
        }

        if (g_strrstr(debuginfo, "Buffer pool activation failed")) {
          NVGSTDS_INFO_MSG_V ("usb bandwidth might be saturated\n");
        }
      }

      g_error_free (error);
      g_free (debuginfo);
      appCtx->return_value = -1;
      appCtx->quit = TRUE;
      break;
    }
    case GST_MESSAGE_STATE_CHANGED:{
      GstState oldstate, newstate;
      gst_message_parse_state_changed (message, &oldstate, &newstate, NULL);
      if (GST_ELEMENT (GST_MESSAGE_SRC (message)) == appCtx->pipeline.pipeline) {
        switch (newstate) {
          case GST_STATE_PLAYING:
            NVGSTDS_INFO_MSG_V ("Pipeline running\n");
            GST_DEBUG_BIN_TO_DOT_FILE_WITH_TS (GST_BIN (appCtx->
                    pipeline.pipeline), GST_DEBUG_GRAPH_SHOW_ALL,
                "ds-app-playing");
            break;
          case GST_STATE_PAUSED:
            if (oldstate == GST_STATE_PLAYING) {
              NVGSTDS_INFO_MSG_V ("Pipeline paused\n");
            }
            break;
          case GST_STATE_READY:
            GST_DEBUG_BIN_TO_DOT_FILE_WITH_TS (GST_BIN (appCtx->pipeline.
                    pipeline), GST_DEBUG_GRAPH_SHOW_ALL, "ds-app-ready");
            if (oldstate == GST_STATE_NULL) {
              NVGSTDS_INFO_MSG_V ("Pipeline ready\n");
            } else {
              NVGSTDS_INFO_MSG_V ("Pipeline stopped\n");
            }
            break;
          case GST_STATE_NULL:
            g_mutex_lock (&appCtx->app_lock);
            g_cond_broadcast (&appCtx->app_cond);
            g_mutex_unlock (&appCtx->app_lock);
            break;
          default:
            break;
        }
      }
      break;
    }
    case GST_MESSAGE_EOS:{
      /*
       * In normal scenario, this would use g_main_loop_quit() to exit the
       * loop and release the resources. Since this application might be
       * running multiple pipelines through configuration files, it should wait
       * till all pipelines are done.
       */
      NVGSTDS_INFO_MSG_V ("Received EOS. Exiting ...\n");
      appCtx->quit = TRUE;
      return FALSE;
      break;
    }
    default:
      break;
  }
  return TRUE;
}

/**
 * Function to dump bounding box data in kitti format. For this to work,
 * property "gie-kitti-output-dir" must be set in configuration file.
 * Data of different sources and frames is dumped in separate file.
 */
// static void
// write_kitti_output (AppCtx * appCtx, NvDsBatchMeta * batch_meta)
// {
//   gchar bbox_file[1024] = { 0 };
//   FILE *bbox_params_dump_file = NULL;

//   if (!appCtx->config.bbox_dir_path)
//     return;

//   for (NvDsMetaList * l_frame = batch_meta->frame_meta_list; l_frame != NULL;
//       l_frame = l_frame->next) {
//     NvDsFrameMeta *frame_meta = l_frame->data;
//     guint stream_id = frame_meta->pad_index;
//     g_snprintf (bbox_file, sizeof (bbox_file) - 1,
//         "%s/%02u_%03u_%06lu.txt", appCtx->config.bbox_dir_path,
//         appCtx->index, stream_id, (gulong) frame_meta->frame_num);
//     bbox_params_dump_file = fopen (bbox_file, "w");
//     if (!bbox_params_dump_file)
//       continue;

//     for (NvDsMetaList * l_obj = frame_meta->obj_meta_list; l_obj != NULL;
//         l_obj = l_obj->next) {
//       NvDsObjectMeta *obj = (NvDsObjectMeta *) l_obj->data;
//       float left = obj->rect_params.left;
//       float top = obj->rect_params.top;
//       float right = left + obj->rect_params.width;
//       float bottom = top + obj->rect_params.height;
//       // Here confidence stores detection confidence, since dump gie output
//       // is before tracker plugin
//       float confidence = obj->confidence;
//       fprintf (bbox_params_dump_file,
//           "%s 0.0 0 0.0 %f %f %f %f 0.0 0.0 0.0 0.0 0.0 0.0 0.0 %f\n",
//           obj->obj_label, left, top, right, bottom, confidence);
//     }
//     fclose (bbox_params_dump_file);
//   }
// }

// static void curl_postjson(gchar label, float left, float top, float right, float bottom, float confidence, GstClockTime relativeTime, char* localTime, float areaProportion)
// {
//   // char szJsonData[1024];
// 	// memset(szJsonData, 0, sizeof(szJsonData));
//   // GString *string = g_string_new("{");
//   JsonBuilder *builder = json_builder_new();
//   json_builder_begin_object(builder);
//   json_builder_set_member_name(builder, "array");
//   json_builder_begin_array(builder);
//   json_builder_begin_object(builder);
//   json_builder_set_member_name(builder, "name");
//   json_builder_add_string_value(builder, label);
//   json_builder_end_object(builder);
//   json_builder_end_array(builder);
//   json_builder_end_object(builder);

//   JsonNode *node = json_builder_get_root(builder);
//   g_object_unref( builder);
//   JsonGenerator *generator = json_generator_new();
//   json_generator_set_root(generator, node);
//   gchar *data = json_generator_to_data(generator, NULL);

//   // try 
// 	// {
// 		CURL *pCurl = NULL;
// 		CURLcode res;
// 		// In windows, this will init the winsock stuff
// 		curl_global_init(CURL_GLOBAL_ALL);
 
// 		// get a curl handle
// 		pCurl = curl_easy_init();
// 		if (NULL != pCurl) 
// 		{
//       curl_easy_setopt(pCurl, CURLOPT_TIMEOUT, 1);
//       curl_easy_setopt(pCurl, CURLOPT_TIMEOUT, 1);
//       curl_easy_setopt(pCurl, CURLOPT_URL, "http://192.168.0.2/posttest.svc");
//       struct curl_slist *plist = curl_slist_append(NULL, "Content-Type:application/json;charset=UTF-8");
// 			curl_easy_setopt(pCurl, CURLOPT_HTTPHEADER, plist);
//       // curl_easy_setopt(pCurl, CURLOPT_POSTFIELDS, szJsonData);
//       curl_easy_setopt(pCurl, CURLOPT_POSTFIELDS, data);

//       res = curl_easy_perform(pCurl);
//       if (res != CURLE_OK) 
// 			{
// 				printf("curl_easy_perform() failed:%s\n", curl_easy_strerror(res));
// 			}
//       curl_easy_cleanup(pCurl);
// 		}
// 		curl_global_cleanup();
// 	// }
// 	// catch (std::exception &ex)
// 	// {
// 	// 	printf("curl exception %s.\n", ex.what());
// 	// }

//   json_node_free(node);
//   g_object_unref(generator);
//   free(data);

// 	// strJson += "\"user_name\" : \"%s\",",label;
// 	// // strJson += "\"password\" : \"test123\"";
// 	// strJson += "}";
//   // strcpy(szJsonData, strJson.c_str());

  
// 	return 0;
// }

static int run = 1;

static void stop(int sig){
	run = 0;
	fclose(stdin);
}




static void
write_kitti_output (AppCtx * appCtx, NvDsBatchMeta * batch_meta, GstBuffer *buffer)
{
  //gchar bbox_file[1024] = { 0 };
  //FILE *bbox_params_dump_file = NULL;

  /*  if (!appCtx->config.bbox_dir_path)
    return ;  */
  
    cJSON *root = cJSON_CreateArray();
		/*用于中断的信号*/
	    signal(SIGINT, stop);
	for (NvDsMetaList * l_frame = batch_meta->frame_meta_list; l_frame != NULL;
      l_frame = l_frame->next) {
		NvDsFrameMeta *frame_meta = l_frame->data;
   
    for (NvDsMetaList * l_obj = frame_meta->obj_meta_list; l_obj != NULL;
        l_obj = l_obj->next) {
      NvDsObjectMeta *obj = (NvDsObjectMeta *) l_obj->data;
      float left = obj->rect_params.left;
      float top = obj->rect_params.top;
      float right = left + obj->rect_params.width;
      float bottom = top + obj->rect_params.height;
      // Here confidence stores detection confidence, since dump gie output
      // is before tracker plugin
      float confidence = obj->confidence;
       
	   //bbox_params_dump_file = fopen (bbox_file, "w");
	   
      time_t utcCalc = frame_meta->ntp_timestamp - 2208988800UL ;
      struct tm * timeinfo;
      time ( &utcCalc );
      timeinfo = localtime ( &utcCalc );
     
	  
      guint frameHeight = frame_meta -> source_frame_height;
      guint frameWidth = frame_meta -> source_frame_width;


      //float areaProportion = (obj->rect_params.width * obj->rect_params.height)/(frameHeight*frameWidth);
      // curl_postjson(obj->obj_label, left, top, right, bottom, confidence, buffer->pts/G_GINT64_CONSTANT (1000000), asctime (timeinfo), areaProportion);
	  printf(  "%s  %f %f %f %f 0.0 0.0 0.0 0.0 0.0 0.0 0.0 %f\n",
                obj->obj_label, left, top, right, bottom, confidence);
     //在对象上添加键值对
   //  cJSON_AddStringToObject(json,"BOX","Face");
     //添加数组
	    char a[20],b[20],c[20],d[20];
		sprintf(a,"%.1lf",left);//
		sprintf(b,"%.1lf",top);
		sprintf(c,"%.1lf",obj->rect_params.width);
		sprintf(d,"%.1lf",obj->rect_params.height);
		cJSON *json = cJSON_CreateObject();
		//cJSON *arr_1 = NULL;
		cJSON *arr_2 = NULL;
		//cJSON *arr_3 = NULL;
		// cJSON *array_4 = NULL;
		cJSON_AddNumberToObject(json,"eventTime",0);
		cJSON_AddItemToObject(json, "box", arr_2=cJSON_CreateObject());
		
		 //cJSON_AddStringToObject(array_2,"left","xiaohui");
		/*  cJSON_AddItemToObject(array_2,"left",cJSON_CreateString("1"));
		 cJSON_AddItemToObject(array_2,"address",cJSON_CreateString("HK")); */
        
		cJSON_AddStringToObject(arr_2,"leftTopx",(a));
		cJSON_AddStringToObject(arr_2,"leftTopy",b);
		cJSON_AddStringToObject(arr_2,"width",c);
		cJSON_AddStringToObject(arr_2,"height",d);
		// cJSON_AddItemToObject(json,"Event-Time",array=cJSON_CreateArray());
		

		cJSON_AddStringToObject(json,"type","人脸");
		//cJSON_AddItemToObject(array_3,"人脸",cJSON_CreateString(obj->obj_label));
		cJSON_AddStringToObject(json, "objectId", obj->obj_label);
		 // cJSON *objx = NULL;
		/*  cJSON_AddItemToArray(array,objx=cJSON_CreateObject());
		 cJSON_AddItemToObject(objx,"BBOX",cJSON_CreateString("left"));
		  cJSON_AddItemToObject(objx,"BBOX",cJSON_CreateString("top"));
		 cJSON_AddStringToObject(objx,"left","beijing"); 
		 //在对象上添加键值对
		 cJSON_AddItemToArray(array,objx=cJSON_CreateObject());
		 cJSON_AddItemToObject(objx,"name",cJSON_CreateString("andy"));
		 cJSON_AddItemToObject(objx,"address",cJSON_CreateString("HK"));
		 cJSON_AddNumberToObject(array,"score",confidence); */
		cJSON_AddItemToArray(root,json); 
		
		char *json_data = NULL;
		printf("data:%s\n",json_data = cJSON_Print(root));
		free(json_data);	  
   /*    fprintf (bbox_params_dump_file,
          // "%s 0.0 0 0.0 %f %f %f %f 0.0 0.0 0.0 0.0 0.0 0.0 0.0 %f\n",
          "Object: %s Bounding Box %f %f %f %f Confidence: %f Time: %ld Local Time: %s Ads proportion: %f\n",

          // obj->obj_label, left, top, right, bottom, confidence, frame_meta->ntp_timestamp - first_frame_time);
          // obj->obj_label, left, top, right, bottom, confidence, buffer->pts/G_GINT64_CONSTANT (1000000),frame_meta->ntp_timestamp);
          obj->obj_label, left, top, right, bottom, confidence, buffer->pts/G_GINT64_CONSTANT (1000000),  (timeinfo), areaProportion); */
       

    }
	//cJSON_Delete(json);
	
		//buf=json;
		char *buf = cJSON_PrintUnformatted(root);
		//buf = cJSON_Print(json);
     	size_t len = strlen(buf);
         
     	 //printf("len:%s\n",len);
     	if(buf[len-1] == '\n')
     		buf[--len] = '\0';

     	if(len == 0){
            /*轮询用于事件的kafka handle，
            事件将导致应用程序提供的回调函数被调用
            第二个参数是最大阻塞时间，如果设为0，将会是非阻塞的调用*/
     		rd_kafka_poll(appCtx->rk, 0);
     	}

   retry:
         /*Send/Produce message.
           这是一个异步调用，在成功的情况下，只会将消息排入内部producer队列，
           对broker的实际传递尝试由后台线程处理，之前注册的传递回调函数(dr_msg_cb)
           用于在消息传递成功或失败时向应用程序发回信号*/
     	if (rd_kafka_produce(
                    /* Topic object */
     				appCtx->rkt,
                    /*使用内置的分区来选择分区*/
     				RD_KAFKA_PARTITION_UA,
                    /*生成payload的副本*/
     				RD_KAFKA_MSG_F_COPY,
                    /*消息体和长度*/
     				buf, len,
                    /*可选键及其长度*/
     				NULL, 0,
     				NULL) == -1){
     		fprintf(stderr, 
     			"%% Failed to produce to topic %s: %s\n", 
     			rd_kafka_topic_name(appCtx->rkt),
     			rd_kafka_err2str(rd_kafka_last_error()));

     		if (rd_kafka_last_error() == RD_KAFKA_RESP_ERR__QUEUE_FULL){
                /*如果内部队列满，等待消息传输完成并retry,
                内部队列表示要发送的消息和已发送或失败的消息，
                内部队列受限于queue.buffering.max.messages配置项*/
     			rd_kafka_poll(appCtx->rk, 100);
     			//goto retry;
     		}	
     	}
		else{
     		fprintf(stderr, "%% Enqueued message (%zd bytes) for topic %s\n", 
     			len, rd_kafka_topic_name(appCtx->rkt));
     	}

        /*producer应用程序应不断地通过以频繁的间隔调用rd_kafka_poll()来为
        传送报告队列提供服务。在没有生成消息以确定先前生成的消息已发送了其
        发送报告回调函数(和其他注册json过的回调函数)期间，要确保rd_kafka_poll()
        仍然被调用*/
     	rd_kafka_poll(appCtx->rk, 0);
     
    
     //fprintf(stderr, "%% Flushing final message.. \n");
     /*rd_kafka_flush是rd_kafka_poll()的抽象化，
     等待所有未完成的produce请求完成，通常在销毁producer实例前完成
     以确保所有排列中和正在传输的produce请求在销毁前完成*/
    rd_kafka_flush(appCtx->rk, 1);

  
     //将JSON结构所占用的数据空间释放
    
	
	free(buf);
	cJSON_Delete(root);
   // fclose (bbox_params_dump_file);
	
  }
  cJSON_Delete(root);
}

static GstFlowReturn
gst_dsexample_transform_ip (GstBuffer * buf){
	
  NvDsMetaList * l_frame = NULL;
  NvDsMetaList * l_user_meta = NULL;
  NvDsMetaList * l_obj=NULL;
  // Get original raw data
  GstMapInfo in_map_info;
  char* src_data = NULL;
  memset (&in_map_info, 0, sizeof (in_map_info));
  if (!gst_buffer_map (buf, &in_map_info, GST_MAP_READ)) {
        g_print ("Error: Failed to map gst buffer\n");
        gst_buffer_unmap (buf, &in_map_info);
        return GST_FLOW_ERROR;
    }
	
  NvBufSurface *surface = (NvBufSurface *)in_map_info.data;
  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);
  if (batch_meta == nullptr) {
   
    return GST_FLOW_ERROR;
  }
  
   for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
      l_frame = l_frame->next)
    {
      /* frame_meta = (NvDsFrameMeta *) (l_frame->data);
      cv::Mat in_mat;

    
	
	
        /* Map the buffer so that it can be accessed by CPU */
       /*  if (surface->surfaceList[frame_meta->batch_id].mappedAddr.addr[0] == NULL){
          if (NvBufSurfaceMap (surface, frame_meta->batch_id, 0, NVBUF_MAP_READ_WRITE) != 0){
            
            return GST_FLOW_ERROR;
          }
        }
		in_mat =
            cv::Mat (surface->surfaceList[frame_meta->batch_id].planeParams.height[0],
            surface->surfaceList[frame_meta->batch_id].planeParams.width[0], CV_8UC4,
            surface->surfaceList[frame_meta->batch_id].mappedAddr.addr[0],
            surface->surfaceList[frame_meta->batch_id].planeParams.pitch[0]);
	}
	
	for (l_obj = frame_meta->obj_meta_list; l_obj != NULL;
          l_obj = l_obj->next)
      {
        obj_meta = (NvDsObjectMeta *) (l_obj->data); */

        

        /* Crop and scale the object */
        
      //} 


		
/* 	
#ifdef PLATFORM_TEGRA
    NvBufSurfaceMap (surface, -1, -1, NVBUF_MAP_READ);
    NvBufSurfacePlaneParams *pParams = &surface->surfaceList[frame_meta->batch_id].planeParams;
    unsigned int offset = 0;
    for(unsigned int num_planes=0; num_planes < pParams->num_planes; num_planes++){
        if(num_planes>0)
            offset += pParams->height[num_planes-1]*(pParams->bytesPerPix[num_planes-1]*pParams->width[num_planes-1]);
        for (unsigned int h = 0; h < pParams->height[num_planes]; h++) {
          memcpy((void *)(src_data+offset+h*pParams->bytesPerPix[num_planes]*pParams->width[num_planes]),
                (void *)((char *)surface->surfaceList[frame_meta->batch_id].mappedAddr.addr[num_planes]+h*pParams->pitch[num_planes]),
                pParams->bytesPerPix[num_planes]*pParams->width[num_planes]
                );
        }
    }
    NvBufSurfaceSyncForDevice (surface, -1, -1);
    NvBufSurfaceUnMap (surface, -1, -1);
#else
    cudaMemcpy((void*)src_data,
                (void*)surface->surfaceList[frame_meta->batch_id].dataPtr,
                surface->surfaceList[frame_meta->batch_id].dataSize,
                cudaMemcpyDeviceToHost);
#endif 
 */

  /* gint frame_width = (gint)surface->surfaceList[frame_meta->batch_id].width;
  gint frame_height = (gint)surface->surfaceList[frame_meta->batch_id].height;
  gint frame_step = surface->surfaceList[frame_meta->batch_id].pitch;
  cv::Mat frame = cv::Mat(frame_height, frame_width, CV_8UC4, src_data, frame_step); */
  // g_print("%d\n",frame.channels());
  // g_print("%d\n",frame.rows);
  // g_print("%d\n",frame.cols);

  /* cv::Mat out_mat = cv::Mat (cv::Size(frame_width, frame_height), CV_8UC3);
  cv::cvtColor(frame, out_mat, CV_RGBA2BGR); */
  //cv::imwrite("test.jpg", out_mat);
   /*  if(src_data != NULL) {
        free(src_data);
        src_data = NULL;
    } */
  //}
  gst_buffer_unmap (buf, &in_map_info);
}
}


/**
 * Function to dump past frame objs in kitti format.
 */
static void
write_kitti_past_track_output (AppCtx * appCtx, NvDsBatchMeta * batch_meta)
{
  if (!appCtx->config.kitti_track_dir_path)
    return;

  // dump past frame tracked objects appending current frame objects
  gchar bbox_file[1024] = { 0 };
  FILE *bbox_params_dump_file = NULL;

    NvDsPastFrameObjBatch *pPastFrameObjBatch = NULL;
    NvDsUserMetaList *bmeta_list = NULL;
    NvDsUserMeta *user_meta = NULL;
    for(bmeta_list=batch_meta->batch_user_meta_list; bmeta_list!=NULL; bmeta_list=bmeta_list->next){
      user_meta = (NvDsUserMeta *)bmeta_list->data;
      if(user_meta && user_meta->base_meta.meta_type==NVDS_TRACKER_PAST_FRAME_META){
        pPastFrameObjBatch = (NvDsPastFrameObjBatch *) (user_meta->user_meta_data);
        for (uint si=0; si < pPastFrameObjBatch->numFilled; si++){
          NvDsPastFrameObjStream *objStream = (pPastFrameObjBatch->list) + si;
          guint stream_id = (guint)(objStream->streamID);
          for (uint li=0; li<objStream->numFilled; li++){
            NvDsPastFrameObjList *objList = (objStream->list) + li;
            for (uint oi=0; oi<objList->numObj; oi++) {
              NvDsPastFrameObj *obj = (objList->list) + oi;
              g_snprintf (bbox_file, sizeof (bbox_file) - 1,
                "%s/%02u_%03u_%06lu.txt", appCtx->config.kitti_track_dir_path,
                appCtx->index, stream_id, (gulong) obj->frameNum);

              float left = obj->tBbox.left;
              float right = left + obj->tBbox.width;
              float top = obj->tBbox.top;
              float bottom = top + obj->tBbox.height;
              // Past frame object confidence given by tracker
              float confidence = obj->confidence;
              bbox_params_dump_file = fopen (bbox_file, "a");
              if (!bbox_params_dump_file){
                continue;
              }
              fprintf(bbox_params_dump_file,
                "%s %lu 0.0 0 0.0 %f %f %f %f 0.0 0.0 0.0 0.0 0.0 0.0 0.0 %f\n",
                objList->objLabel, objList->uniqueId, left, top, right, bottom, confidence);
              fclose (bbox_params_dump_file);
            }
          }
        }
      }
    }
}



/**
 * Function to dump bounding box data in kitti format with tracking ID added.
 * For this to work, property "kitti-track-output-dir" must be set in configuration file.
 * Data of different sources and frames is dumped in separate file.
 */
static void
write_kitti_track_output (AppCtx * appCtx, NvDsBatchMeta * batch_meta)
{
  gchar bbox_file[1024] = { 0 };
  FILE *bbox_params_dump_file = NULL;

  if (!appCtx->config.kitti_track_dir_path)
    return;

  for (NvDsMetaList * l_frame = batch_meta->frame_meta_list; l_frame != NULL;
      l_frame = l_frame->next) {
    NvDsFrameMeta *frame_meta = l_frame->data;
    guint stream_id = frame_meta->pad_index;
    g_snprintf (bbox_file, sizeof (bbox_file) - 1,
        "%s/%02u_%03u_%06lu.txt", appCtx->config.kitti_track_dir_path,
        appCtx->index, stream_id, (gulong) frame_meta->frame_num);
    bbox_params_dump_file = fopen (bbox_file, "w");
    if (!bbox_params_dump_file)
      continue;

    for (NvDsMetaList * l_obj = frame_meta->obj_meta_list; l_obj != NULL;
        l_obj = l_obj->next) {
      NvDsObjectMeta *obj = (NvDsObjectMeta *) l_obj->data;
      float left = obj->tracker_bbox_info.org_bbox_coords.left;
      float top = obj->tracker_bbox_info.org_bbox_coords.top;
      float right = left + obj->tracker_bbox_info.org_bbox_coords.width;
      float bottom = top + obj->tracker_bbox_info.org_bbox_coords.height;
      // Here confidence stores tracker confidence value for tracker output
      float confidence = obj->tracker_confidence;
      guint64 id = obj->object_id;
      fprintf (bbox_params_dump_file,
          "%s %lu 0.0 0 0.0 %f %f %f %f 0.0 0.0 0.0 0.0 0.0 0.0 0.0 %f\n",
          obj->obj_label, id, left, top, right, bottom, confidence);
    }
    fclose (bbox_params_dump_file);
  }
}

/**
 * kafka
 */
static void
write_kitti_kafka_output (AppCtx * appCtx, NvDsBatchMeta * batch_meta,std::vector<std::string> &predNames, std::vector<float> &predSims)
{


  /*  if (!appCtx->config.bbox_dir_path)
    return ;  */
 
    int index_count=0;
    cJSON *root = cJSON_CreateArray();
		/*用于中断的信号*/
	
    for (NvDsMetaList * l_frame = batch_meta->frame_meta_list; l_frame != NULL;
      l_frame = l_frame->next) {
    NvDsFrameMeta *frame_meta = l_frame->data;
   
    guint stream_id = frame_meta->pad_index;
   
    for (NvDsMetaList * l_obj = frame_meta->obj_meta_list; l_obj != NULL;
        l_obj = l_obj->next) {
      NvDsObjectMeta *obj = (NvDsObjectMeta *) l_obj->data;
      float left = obj->rect_params.left;
      float top = obj->rect_params.top;
      float right = left + obj->rect_params.width;
      float bottom = top + obj->rect_params.height;
      // Here confidence stores detection confidence, since dump gie output
      // is before tracker plugin
      float confidence = obj->confidence;
  
      time_t utcCalc = frame_meta->ntp_timestamp - 2208988800UL ;
      struct tm * timeinfo;
      time ( &utcCalc );
      timeinfo= gmtime(&utcCalc );
	  timeinfo->tm_hour+=8;
     
	  char result[50];
	  size_t dstSize = strftime(result, 50, "%Y-%m-%d %T", timeinfo);
      guint frameHeight = frame_meta -> source_frame_height;
      guint frameWidth = frame_meta -> source_frame_width;

      //std::cout<<"自己的计数器："<<index_count<<std::endl;
	  //std::cout<<"内置计数器："<<stream_id<<std::endl;
     //在对象上添加键值对
   //  cJSON_AddStringToObject(json,"BOX","Face");
     //添加数组
	    char a[20],b[20],c[20],d[20];
		sprintf(a,"%.1lf",left);//
		sprintf(b,"%.1lf",top);
		sprintf(c,"%.1lf",obj->rect_params.width);
		sprintf(d,"%.1lf",obj->rect_params.height);
		cJSON *json = cJSON_CreateObject();
		//cJSON *array = NULL;
		cJSON *array_2 = NULL;
		//cJSON *array_3 = NULL;
		// cJSON *array_4 = NULL;
		//cJSON_AddNumberToObject(json,"eventTime",0);
		cJSON_AddNumberToObject(json,"数据源—流id",stream_id);
		cJSON_AddItemToObject(json, "Box", array_2=cJSON_CreateObject());
		
		 //cJSON_AddStringToObject(array_2,"left","xiaohui");
		/*  cJSON_AddItemToObject(array_2,"left",cJSON_CreateString("1"));
		 cJSON_AddItemToObject(array_2,"address",cJSON_CreateString("HK")); */
        
		cJSON_AddStringToObject(array_2,"leftTopx",a);
		cJSON_AddStringToObject(array_2,"leftTopy",b);
		cJSON_AddStringToObject(array_2,"width",c);
		cJSON_AddStringToObject(array_2,"height",d);
		
		// cJSON_AddItemToObject(json,"Event-Time",array=cJSON_CreateArray());
		cJSON_AddStringToObject(json,"eventTime",result);
		cJSON_AddStringToObject(json,"type","人脸识别任务");
		//g_print("xxxaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa:\n",appCtx->predNames.size());
		for(auto i:predNames)
			std::cout<<"----------------检测识别的人："<<i<<std::endl;
		if(predNames.empty())
			cJSON_AddStringToObject(json, "objectId", "Unknown_id");
		else
			cJSON_AddStringToObject(json, "objectId", predNames[index_count].c_str());
		//g_print("xxxaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa:\n",appCtx->predNames.size());
		 // cJSON *objx = NULL;
		/*  cJSON_AddItemToArray(array,objx=cJSON_CreateObject());
		 cJSON_AddItemToObject(objx,"BBOX",cJSON_CreateString("left"));
		  cJSON_AddItemToObject(objx,"BBOX",cJSON_CreateString("top"));
		 cJSON_AddStringToObject(objx,"left","beijing"); 
		 //在对象上添加键值对
		 cJSON_AddItemToArray(array,objx=cJSON_CreateObject());
		 cJSON_AddItemToObject(objx,"name",cJSON_CreateString("andy"));
		 cJSON_AddItemToObject(objx,"address",cJSON_CreateString("HK"));
		 cJSON_AddNumberToObject(array,"score",confidence); */
		cJSON_AddItemToArray(root,json); 
		
	/* 	char *json_data = NULL;
		printf("data:%s\n",json_data = cJSON_Print(root));
		free(json_data); */	  
   /*    fprintf (bbox_params_dump_file,
          // "%s 0.0 0 0.0 %f %f %f %f 0.0 0.0 0.0 0.0 0.0 0.0 0.0 %f\n",
          "Object: %s Bounding Box %f %f %f %f Confidence: %f Time: %ld Local Time: %s Ads proportion: %f\n",

          // obj->obj_label, left, top, right, bottom, confidence, frame_meta->ntp_timestamp - first_frame_time);
          // obj->obj_label, left, top, right, bottom, confidence, buffer->pts/G_GINT64_CONSTANT (1000000),frame_meta->ntp_timestamp);
          obj->obj_label, left, top, right, bottom, confidence, buffer->pts/G_GINT64_CONSTANT (1000000),  (timeinfo), areaProportion); */
        index_count++;
    }
}
    signal(SIGINT, stop);
	char *json_data = NULL;
    printf("data:%s\n",json_data = cJSON_Print(root));
	
	char *buf = cJSON_PrintUnformatted(root);
    size_t len = strlen(buf);
    //std::cout<<"检查数量："<<index_count<<std::endl;
     	
    if(buf[len-1] == '\n')
     		buf[--len] = '\0';

    if(len == 0){
            /*轮询用于事件的kafka handle，
            事件将导致应用程序提供的回调函数被调用
            第二个参数是最大阻塞时间，如果设为0，将会是非阻塞的调用*/
     		rd_kafka_poll(appCtx->rk, 0);
    }
         /*Send/Produce message.
           这是一个异步调用，在成功的情况下，只会将消息排入内部producer队列，
           对broker的实际传递尝试由后台线程处理，之前注册的传递回调函数(dr_msg_cb)
           用于在消息传递成功或失败时向应用程序发回信号*/
	//std::cout<<"x000000000000000000011111111"<<std::endl;
 	
   if (rd_kafka_produce(
                    /* Topic object */
     				appCtx->rkt,
                    /*使用内置的分区来选择分区*/
     				RD_KAFKA_PARTITION_UA,
                    /*生成payload的副本*/
     				RD_KAFKA_MSG_F_COPY,
                    /*消息体和长度*/
     				buf, len,
                    /*可选键及其长度*/
     				NULL, 0,
     				NULL) == -1){
						//std::cout<<"xxxxxxxxxxx11111111"<<std::endl;
     	/* 	fprintf(stderr, 
     			"%% Failed to produce to topic %s: %s\n", 
     			rd_kafka_topic_name(appCtx->rkt),
     			rd_kafka_err2str(rd_kafka_last_error())); */
            
     		if (rd_kafka_last_error() == RD_KAFKA_RESP_ERR__QUEUE_FULL){
                /*如果内部队列满，等待消息传输完成并retry,
                内部队列表示要发送的消息和已发送或失败的消息，
                内部队列受限于queue.buffering.max.messages配置项*/
     			rd_kafka_poll(appCtx->rk, 10000);
			}	
	}
 
 

	/* else{
     		fprintf(stderr, "%% Enqueued success message (%zd bytes) for topic %s\n", 
     			len, rd_kafka_topic_name(appCtx->rkt));
     	} */
	
        /*producer应用程序应不断地通过以频繁的间隔调用rd_kafka_poll()来为
        传送报告队列提供服务。在没有生成消息以确定先前生成的消息已发送了其
        发送报告回调函数(和其他注册json过的回调函数)期间，要确保rd_kafka_poll()
        仍然被调用*/
    rd_kafka_poll(appCtx->rk, 0);
     //fprintf(stderr, "%% Flushing final message.. \n");
     /*rd_kafka_flush是rd_kafka_poll()的抽象化，
     等待所有未完成的produce请求完成，通常在销毁producer实例前完成
     以确保所有排列中和正在传输的produce请求在销毁前完成*/
    rd_kafka_flush(appCtx->rk, 1);
     //将JSON结构所占用的数据空间释放
	free(json_data);
    free(buf);
	cJSON_Delete(root);
	 //predNames.clear();
   // fclose (bbox_params_dump_file);
}


static gint
component_id_compare_func (gconstpointer a, gconstpointer b)
{
  NvDsClassifierMeta *cmetaa = (NvDsClassifierMeta *) a;
  NvDsClassifierMeta *cmetab = (NvDsClassifierMeta *) b;

  if (cmetaa->unique_component_id < cmetab->unique_component_id)
    return -1;
  if (cmetaa->unique_component_id > cmetab->unique_component_id)
    return 1;
  return 0;
}

/**
 * Function to process the attached metadata. This is just for demonstration
 * and can be removed if not required.
 * Here it demonstrates to use bounding boxes of different color and size for
 * different type / class of objects.
 * It also demonstrates how to join the different labels(PGIE + SGIEs)
 * of an object to form a single string.
 */
static void process_meta(AppCtx *appCtx, NvDsBatchMeta *batch_meta) {
    // For single source always display text either with demuxer or with tiler
    if (!appCtx->config.tiled_display_config.enable || appCtx->config.num_source_sub_bins == 1) {
        appCtx->show_bbox_text = 1;
    }

    for (NvDsMetaList *l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)l_frame->data;
        for (NvDsMetaList *l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next) {
            NvDsObjectMeta *obj = (NvDsObjectMeta *)l_obj->data;
            gint class_index = obj->class_id;
            NvDsGieConfig *gie_config = NULL;
            gchar *str_ins_pos = NULL;

            if (obj->unique_component_id == (gint)appCtx->config.primary_gie_config.unique_id) {
                gie_config = &appCtx->config.primary_gie_config;
            } else {
                for (gint i = 0; i < (gint)appCtx->config.num_secondary_gie_sub_bins; i++) {
                    gie_config = &appCtx->config.secondary_gie_sub_bin_config[i];
                    if (obj->unique_component_id == (gint)gie_config->unique_id) {
                        break;
                    }
                    gie_config = NULL;
                }
            }
            g_free(obj->text_params.display_text);
            obj->text_params.display_text = NULL;

            if (gie_config != NULL) {
                if (g_hash_table_contains(gie_config->bbox_border_color_table, class_index + (gchar *)NULL)) {
                    obj->rect_params.border_color = *((NvOSD_ColorParams *)g_hash_table_lookup(
                        gie_config->bbox_border_color_table, class_index + (gchar *)NULL));
                } else {
                    obj->rect_params.border_color = gie_config->bbox_border_color;
                }
                obj->rect_params.border_width = appCtx->config.osd_config.border_width;

                if (g_hash_table_contains(gie_config->bbox_bg_color_table, class_index + (gchar *)NULL)) {
                    obj->rect_params.has_bg_color = 1;
                    obj->rect_params.bg_color = *((NvOSD_ColorParams *)g_hash_table_lookup(
                        gie_config->bbox_bg_color_table, class_index + (gchar *)NULL));
                } else {
                    obj->rect_params.has_bg_color = 0;
                }
            }

            if (!appCtx->show_bbox_text)
                continue;

            obj->text_params.x_offset = obj->rect_params.left;
            obj->text_params.y_offset = obj->rect_params.top - 30;
            obj->text_params.font_params.font_color = appCtx->config.osd_config.text_color;
            obj->text_params.font_params.font_size = appCtx->config.osd_config.text_size;
            obj->text_params.font_params.font_name = appCtx->config.osd_config.font;
            if (appCtx->config.osd_config.text_has_bg) {
                obj->text_params.set_bg_clr = 1;
                obj->text_params.text_bg_clr = appCtx->config.osd_config.text_bg_color;
            }

            obj->text_params.display_text = (char *)g_malloc(128);
            obj->text_params.display_text[0] = '\0';
            str_ins_pos = obj->text_params.display_text;

            if (obj->obj_label[0] != '\0')
                sprintf(str_ins_pos, "%s", obj->obj_label);
            str_ins_pos += strlen(str_ins_pos);

            if (obj->object_id != UNTRACKED_OBJECT_ID) {
                /** object_id is a 64-bit sequential value;
                 * but considering the display aesthetic,
                 * trimming to lower 32-bits */
                if (appCtx->config.tracker_config.display_tracking_id) {
                    guint64 const LOW_32_MASK = 0x00000000FFFFFFFF;
                    sprintf(str_ins_pos, " %lu", (obj->object_id & LOW_32_MASK));
                    str_ins_pos += strlen(str_ins_pos);
                }
            }

            
         /*    obj->classifier_meta_list = g_list_sort(obj->classifier_meta_list, component_id_compare_func);
            for (NvDsMetaList *l_class = obj->classifier_meta_list; l_class != NULL; l_class = l_class->next) {
                NvDsClassifierMeta *cmeta = (NvDsClassifierMeta *)l_class->data;
                for (NvDsMetaList *l_label = cmeta->label_info_list; l_label != NULL; l_label = l_label->next) {
                    NvDsLabelInfo *label = (NvDsLabelInfo *)l_label->data;
                    if (label->pResult_label) {
                        sprintf(str_ins_pos, " %s", label->pResult_label);
                    } else if (label->result_label[0] != '\0') {
                        sprintf(str_ins_pos, " %s", label->result_label);
                    }
                    str_ins_pos += strlen(str_ins_pos);
                }
            } */
           
        }
    }
}

/**
 * Function which processes the inferred buffer and its metadata.
 * It also gives opportunity to attach application specific
 * metadata (e.g. clock, analytics output etc.).
 */
static void
process_buffer (GstBuffer * buf, AppCtx * appCtx, guint index)
{
  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);
  if (!batch_meta) {
    NVGSTDS_WARN_MSG_V ("Batch meta not found for buffer %p", buf);
    return;
  }
 
  process_meta (appCtx, batch_meta);
  
  //write_kitti_output (appCtx, batch_meta, buf);
  //NvDsInstanceData *data = &appCtx->instance_data[index];
  //guint i;

  //  data->frame_num++;

  /* Opportunity to modify the processed metadata or do analytics based on
   * type of object e.g. maintaining count of particular type of car.
   */
   if (appCtx->all_bbox_generated_cb) {
    appCtx->all_bbox_generated_cb (appCtx, buf, batch_meta, index);
  }  
  //data->bbox_list_size = 0;

  /*
   * callback to attach application specific additional metadata.
   */
  if (appCtx->overlay_graphics_cb) {
    appCtx->overlay_graphics_cb (appCtx, buf, batch_meta, index);
  }
  
}

/**
 * Buffer probe function to get the results of primary infer.
 * Here it demonstrates the use by dumping bounding box coordinates in
 * kitti format.
 */
static GstPadProbeReturn
gie_primary_processing_done_buf_prob (GstPad * pad, GstPadProbeInfo * info,
    gpointer u_data)
{
  GstBuffer *buf = (GstBuffer *) info->data;
  AppCtx *appCtx = (AppCtx *) u_data;
  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);
  if (!batch_meta) {
    NVGSTDS_WARN_MSG_V ("Batch meta not found for buffer %p", buf);
    return GST_PAD_PROBE_OK;
  }

  // write_kitti_output (appCtx, batch_meta);
  // write_kitti_output (appCtx, batch_meta, buf);

  return GST_PAD_PROBE_OK;
}

/**
 * Probe function to get results after all inferences(Primary + Secondary)
 * are done. This will be just before OSD or sink (in case OSD is disabled).
 */
static GstPadProbeReturn
gie_processing_done_buf_prob (GstPad * pad, GstPadProbeInfo * info,
    gpointer u_data)
{
 // g_print("gie_processing_done_buf_prob:\n");
  GstBuffer *buf = (GstBuffer *) info->data;
  NvDsInstanceBin *bin = (NvDsInstanceBin *) u_data;
  guint index = bin->index;
  AppCtx *appCtx = bin->appCtx;
  
  if (gst_buffer_is_writable (buf))
    process_buffer (buf, appCtx, index);
  
  return GST_PAD_PROBE_OK;
}

/* static GstPadProbeReturn
tiler_src_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info, gpointer u_data){
    printf("启动自定义探针!!!!!!!!!!!!!!!!!!!\n");
    GstBuffer *buf = (GstBuffer *) info->data;
    guint num_rects = 0; 
    NvDsObjectMeta *obj_meta = NULL;
    guint vehicle_count = 0;
    guint person_count = 0;
    NvDsMetaList * l_frame = NULL;
    NvDsMetaList * l_obj = NULL;
   // NvDsDisplayMeta *display_meta = NULL;
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);
    //char_rec = CTCGreedyDecoderLayer(outputBboxBuffer,1,25,5990);//int
 
	//std::vector<std::string> *file_to_vector=InputData_To_Vector();
    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);//单帧	
        guint batch_id = frame_meta->batch_id;
        int  surface_id = frame_meta->source_id;
        
        for (l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next) {
            guint class_id = 2555;
            gfloat model_prob = 0;
            obj_meta = (NvDsObjectMeta *) (l_obj->data);
            guint64 track_id = obj_meta->object_id;
            NvOSD_RectParams rect = obj_meta->rect_params;
            gint color_id = obj_meta->class_id;
            gfloat color_prob = obj_meta->confidence;
            gchar text[128] = "";
			
        for (NvDsMetaList * l_class = obj_meta->classifier_meta_list; l_class != NULL; l_class = l_class->next) {
                NvDsClassifierMeta *cmeta = (NvDsClassifierMeta *) l_class->data;
                for (NvDsMetaList * l_label = cmeta->label_info_list; l_label != NULL; l_label = l_label->next) {
                    NvDsLabelInfo *label = (NvDsLabelInfo *) l_label->data;
                    class_id = label->result_class_id;
                    model_prob = label->result_prob;
				   //NvDsLabelInfo *label_info =
                  //nvds_acquire_label_info_meta_from_pool (batch_meta);
			            //label->result_class_id =ind ;
					  // label->result_prob = 0.5;
                    if (label->result_label[0] != '\0' ){ 
                      // g_print("class:%s\n", label->result_label);
                       
						// strcpy (label->result_label,
                          //   label->class_id ); 
						sprintf(text, "text: %s", label->result_label);
                        printf("text: %s\n", text);
						
                    }
                }
            }
        }
            
    }
   //frame_number++;
    return GST_PAD_PROBE_OK;
} */


/*
 * Get embedding from tensor meta output
 */
 
static void get_sgie_tensor_output(AppCtx *appCtx, NvDsBatchMeta *batch_meta) {
    int count = 0;
    //std::cout<<"获取二级模型推理： "<<std::endl;
    for (NvDsMetaList *l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)l_frame->data;
        for (NvDsMetaList *l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next) {
            NvDsObjectMeta *obj = (NvDsObjectMeta *)l_obj->data;
            for (NvDsMetaList *l_user = obj->obj_user_meta_list; l_user != NULL; l_user = l_user->next) {
                NvDsUserMeta *user_meta = (NvDsUserMeta *)l_user->data;
                if (user_meta->base_meta.meta_type == NVDSINFER_TENSOR_OUTPUT_META) {
                    NvDsInferTensorMeta *tensor_meta = (NvDsInferTensorMeta *)user_meta->user_meta_data;
                    /*
                    for (unsigned int i = 0; i < tensor_meta->num_output_layers; i++) {
                        NvDsInferLayerInfo *info = &tensor_meta->output_layers_info[i];
                        info->buffer = tensor_meta->out_buf_ptrs_host[i];
                        if (tensor_meta->out_buf_ptrs_dev[i]) {
                            cudaMemcpy(tensor_meta->out_buf_ptrs_host[i], tensor_meta->out_buf_ptrs_dev[i],
                                       info->inferDims.numElements * 4, cudaMemcpyDeviceToHost);
                        }
                    }
                    float *outputCoverageBuffer = (float *)tensor_meta->output_layers_info[0].buffer;
                    */
                     // float *outputCoverageBuffer = (float *)tensor_meta->output_layers_info[0].buffer;
                    float *outputCoverageBuffer = (float *)tensor_meta->out_buf_ptrs_host[0];
					
                    std::copy(outputCoverageBuffer, outputCoverageBuffer + appCtx->sgieOutputDim,
                              appCtx->embeds + count * appCtx->sgieOutputDim);
							  
					//std::cout<<"eBuffer:"<<outputCoverageBuffer[512]<<std::endl;	
					 /* for(int i=0;i<512;i++)
						 std::cout<<*(outputCoverageBuffer+i)<<","	; */
                    count++;
                    
                }
            }
        }
    }

    appCtx->embedCount = count;
}



float dotProduct(const std::vector<float>& v1, const std::vector<float>& v2)
 {
        assert(v1.size() == v2.size());
        float ret = 0.0;
        for (std::vector<float>::size_type i = 0; i != v1.size(); ++i)
         {
                ret += v1[i] * v2[i];
         }
        return ret;
 }
float module(const std::vector<float>& v)
 {
        float ret = 0.0;
        for (std::vector<float>::size_type i = 0; i != v.size(); ++i)
             {
                ret += v[i] * v[i];
             }
        return sqrt(ret);
}
float cosine(const std::vector<float>& v1, const std::vector<float>& v2)
{
   assert(v1.size() == v2.size());
   return dotProduct(v1, v2) / (module(v1) * module(v2));
}

/*
 * Parse outputs, get max cosine similarity
 */
void get_outputs(AppCtx *appCtx, float *sims, std::vector<std::string> &predNames, std::vector<float> &predSims) {
    predNames.clear();
    predSims.clear();
	std::cout<<"开始相似度计算"<<std::endl;
    for (int i = 0; i < appCtx->embedCount; ++i) {
        int argmax = std::distance(
            sims + i * appCtx->knownEmbedCount,
            std::max_element(sims + i * appCtx->knownEmbedCount, sims + (i + 1) * appCtx->knownEmbedCount));
        float sim = *(sims + i * appCtx->knownEmbedCount + argmax);
        predNames.push_back(appCtx->knownIds[argmax]);
        predSims.push_back(sim);
        std::cout<<"sim:"<<sim<<std::endl;
    }
}

/* float getSimilarity(const cv::Mat& first,const cv::Mat& second)
    {   
	    std::cout<< "compute consine similarity....."  <<std::endl;
        double dotSum=first.dot(second);//内积
        double normFirst=cv::norm(first);//取模
        double normSecond=cv::norm(second); 
        if(normFirst!=0 && normSecond!=0){
            return dotSum/(normFirst*normSecond);
        }
    } */


/* ReshapeandNormalize(float *out, cv::Mat &feature, const int &MAT_SIZE, const int &outSize) {
    for (int i = 0; i < MAT_SIZE; i++)
    {
        cv::Mat onefeature(1, outSize, CV_32FC1, out + i * outSize);
        cv::normalize(onefeature, onefeature);
        onefeature.copyTo(feature.row(i));
    }
} */


/* Reshape(float *out, cv::Mat &feature, const int &MAT_SIZE, const int &outSize) {
    for (int i = 0; i < MAT_SIZE; i++)
    {
        cv::Mat onefeature(1, outSize, CV_32FC1, out + i * outSize);
        onefeature.copyTo(feature.row(i));
    }
} */
/**
 * Buffer probe function after tracker.
 */
static GstPadProbeReturn
analytics_done_buf_prob (GstPad * pad, GstPadProbeInfo * info, gpointer u_data)
{
  NvDsInstanceBin *bin = (NvDsInstanceBin *) u_data;
  guint index = bin->index;
  AppCtx *appCtx = bin->appCtx;
  GstBuffer *buf = (GstBuffer *) info->data;
  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);
  if (!batch_meta) {
    NVGSTDS_WARN_MSG_V ("Batch meta not found for buffer %p", buf);
    return GST_PAD_PROBE_OK;
  }
  //int index=0;
/*
     * Get embedding and do feature matching
     */
	// appCtx->embeds={0.041638121, -0.041558761, -0.027988015, -0.018610178, 0.0082403272, 0.023054399, 0.061690025, 0.055552769, 0.009688667, -0.032035429, 0.061901655, 0.024747435, -0.0028569996, -0.028860986, -0.0028652663, 0.0076054386, -0.023041172};
    get_sgie_tensor_output(appCtx, batch_meta);
    if (appCtx->embedCount > 0) { // in case use `interval` pgie property
        std::cout<<"待检测的人脸数量"<<appCtx->embedCount<<std::endl;
		//std::cout<<"样本库的人脸数量"<<*(appCtx->embeds)<<std::endl;
		// cv::Mat face_feature(appCtx->embedCount, appCtx->sgieOutputDim, CV_32FC1);
		// cv::Mat face_feature2(appCtx->knownEmbedCount, appCtx->sgieOutputDim, CV_32FC1);
        /* std::vector<std::vector<float>>  embed(appCtx->embedCount*appCtx->sgieOutputDim);
	    std::cout<<"次数"<<c<<std::endl;  */ 
	   
		
		//float outx[appCtx->sgieOutputDim * appCtx->embedCount];
		//int rowSize = ind % appCtx->embedCount == 0 ? appCtx->embedCount : ind % appCtx->embedCount;
        // float *sims = new float[appCtx->embedCount * appCtx->knownEmbedCount];
		cv::Mat feature(appCtx->embedCount, appCtx->sgieOutputDim, CV_32FC1,appCtx->embeds);
		//cv::Mat out_norm2;
		//cv::normalize(feature, out_norm2);
		//ReshapeandNormalize(appCtx->embeds, feature, appCtx->embedCount, appCtx->sgieOutputDim);
		cv::Mat feature2(appCtx->knownEmbedCount, appCtx->sgieOutputDim, CV_32FC1,appCtx->save_embeds);  //拿到人脸库特征数据
		//Reshape(appCtx->save_embeds,feature2, appCtx->knownEmbedCount, appCtx->sgieOutputDim);
		//appCtx->embeds=NULL;
		// std::cout<<"本地储存的人脸特征"<<feature2<<std::endl;
        //std::cout<<"人脸特征"<<feature<<std::endl; 
		//feature.copyTo(face_feature.rowRange(ind - rowSize, ind));
	//	feature2.copyTo(face_feature2.rowRange(ind - rowSize, ind));
		
		/* std::cout<<"待检测特征行数："<<feature.rows<<std::endl;
		std::cout<<"待检测特征列数："<<feature.cols<<std::endl;
		std::cout<<"存储特征行数："<<feature2.rows<<std::endl;
		std::cout<<"存储特征列数："<<feature2.cols<<std::endl; */
		//std::cout<<"检查容量:"<< appCtx->save_face.size() <<std::endl;
		/* cv::Mat save_embed =cv:: Mat(appCtx->save_face);
        std::cout<<"size="<<save_embed.rows<<std::endl;
     	std::cout<<"size="<<save_embed.cols<<std::endl; */
       /*  std::ofstream wout("face_writer.txt");
         if (!wout.is_open()) {
       std::cout << "File is open fail!" << std::endl;
      } */
	     
		appCtx->predNames.clear();
		appCtx->predSims.clear();
		//该写法是我排查BUG时候 拆成单次向量写了并不高效，不依赖算法前提下，也可以使用矩阵计算，推荐：打算使用三方库耦合的聚类算法去实现也可
		for(int i=0;i<feature.rows;i++){
			cv::Mat fea1 ;
			fea1=feature.row(i);
			cv::Mat fea1_norm(1,appCtx->sgieOutputDim,CV_32FC1);
			cv::normalize(fea1, fea1_norm);
			int find_num=0;
		    std::vector<float> score_save;
			for(int j=0;j<feature2.rows;j++)  //特征库的数量
				
			{   find_num++;
               // cv::Mat fea1_norm(1,128,CV_32FC1);
				//cv::Mat fea2_norm(1,128,CV_32FC1);
				
             	
				cv::Mat fea2=feature2.row(j);
				//cv::normalize(fea2, fea2_norm,cv::NORM_L2);
				
				cv::Mat res= fea2 * fea1_norm.t();
				/* IplImage tmp1 = IplImage(fea1_norm);
				IplImage tmp2 = IplImage(fea2);
                CvArr* arr1 = (CvArr*)&tmp1;
				CvArr* arr2= (CvArr*)&tmp2;
				std::cout<<"l2:"<<cvNorm(arr2 ,arr1,CV_L2)<<std::endl;  */

				//float dis=cvNorm(fea1_norm,fea2,CV_L2);
				//std::cout<<res<<std::endl; 
				float sims=*(float*)res.data;
				//std::cout<<"calculate ------sim:"<<sims<<std::endl;
				score_save.push_back(sims);
			   
			}
		    std::vector<float>::iterator biggest =std::max_element(score_save.begin(),score_save.end());
		   
			if(*biggest>0.601)
			{   
			    int pos=std::distance(std::begin(score_save), biggest);
				std::cout<<"相似度:"<<*biggest<<"匹配该特征样本ID"<<appCtx->knownIds[pos]<<std::endl;
					//std::cout<<"&&&&&&&&&&&&&&&&&&&&&&&&&***********************$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$**********************************"<<std::endl;
					appCtx->predNames.push_back(appCtx->knownIds[pos]);
					appCtx->predSims.push_back(*biggest);
				
			}
			else  {
				//std::cout<<find_num<<"次没找到"<<std::endl;
			    appCtx->predNames.push_back("未知人物");
				appCtx->predSims.push_back(*biggest);
			
			}
			//score_save.clear();
		 }
		
		//cv::Mat similarity = (feature2 * out_norm2.t());
		
		//double dotSum=first.dot(second);//内积
        //double normFirst=cv::norm(first);//取模
       // double normSecond=cv::norm(second); 
		//float res=*(float*)similarity.data;
		//cv:: Mat E(, 1, CV_8UC3, cv::Scalar::all(1));
		//std::cout<<"xx:"<<Simmat<<std::endl;	
		
		/* for(int i=0;i<similarity.cols;i++){
		    cv::Mat sub=similarity.colRange(i,i+1).clone();
		   std::cout<<"matrixL:"<< sub<<std::endl;
	        //std::cout<<"xx:"<<(float*)sub.data<std::endl;				
        //std::cout << "The similarity matrix of the image folder is:\n" << a1 << "!" << std::endl;
		//遍历写法:
		double maxvalue=0, minvalue=0;
	    cv::Point max_ind , min_ind;
		 /* minMaxLoc 求取最大值
			Mat数组，最小值，最大值，最小值位置，最大值位置 
			*/      
		/* cv::minMaxLoc(sub, &minvalue, &maxvalue, &min_ind, &max_ind);
		float sim=1-minvalue;
		std::cout <<"当前相似打分:"<<sim<<std::endl;
		//std::cout<<"sim:"<<appCtx->knownIds[max_ind.y]<<std::endl;
		if(sim>0.7){
			
			appCtx->predNames.push_back(appCtx->knownIds[min_ind.y]);
			std::cout<<"找到该人物！！！！！！！！！！！！！！！！"<<std::endl;
			appCtx->predSims.push_back(sim);
			
			}
		else {
			appCtx->predNames.push_back("Unknown_id");
				std::cout<<"未找到！！！！！！！！"<<std::endl;
			} */
		//} 
	   if(appCtx->send_mess)
        write_kitti_kafka_output(appCtx, batch_meta,appCtx->predNames,appCtx->predSims);
        //appCtx->cossim->calculate(appCtx->embeds, appCtx->embedCount, sims);
		//std::cout<<appCtx->predNames[0]<<std::endl;
       // get_outputs(appCtx, similarity, appCtx->predNames, appCtx->predSims);
	   
	    //std::cout<<"完成"<<appCtx->embedCount<<std::endl;
	  /*  if (appCtx->bbox_generated_post_analytics_cb) {
        appCtx->bbox_generated_post_analytics_cb(appCtx, buf, batch_meta, index);
    }  */
       // return GST_PAD_PROBE_OK;
	   
    }
	
  /*
  
  
   * Output KITTI labels with tracking ID if configured to do so.
   */
  //write_kitti_kafka_output(appCtx, batch_meta,appCtx->predNames,appCtx->predSims);
   
  if (appCtx->config.tracker_config.enable_past_frame)
  {
      write_kitti_past_track_output (appCtx, batch_meta);
  }
  
  
  if (appCtx->bbox_generated_post_analytics_cb)
  {
    appCtx->bbox_generated_post_analytics_cb (appCtx, buf, batch_meta, index);
  }
 
  return GST_PAD_PROBE_OK;
}

static GstPadProbeReturn
latency_measurement_buf_prob(GstPad * pad, GstPadProbeInfo * info, gpointer u_data)
{
  AppCtx *appCtx = (AppCtx *) u_data;
  guint i = 0, num_sources_in_batch = 0;
  if(nvds_enable_latency_measurement)
  {
    GstBuffer *buf = (GstBuffer *) info->data;
    NvDsFrameLatencyInfo *latency_info = NULL;
    g_mutex_lock (&appCtx->latency_lock);
    latency_info = appCtx->latency_info;
    g_print("\n************BATCH-NUM = %d**************\n",batch_num);
    num_sources_in_batch = nvds_measure_buffer_latency(buf, latency_info);

    for(i = 0; i < num_sources_in_batch; i++)
    {
      g_print("Source id = %d Frame_num = %d Frame latency = %lf (ms) \n",
          latency_info[i].source_id,
          latency_info[i].frame_num,
          latency_info[i].latency);
    }
    g_mutex_unlock (&appCtx->latency_lock);
    batch_num++;
  }

  return GST_PAD_PROBE_OK;
}

static GstPadProbeReturn
demux_latency_measurement_buf_prob(GstPad * pad, GstPadProbeInfo * info, gpointer u_data)
{
  AppCtx *appCtx = (AppCtx *) u_data;
  guint i = 0, num_sources_in_batch = 0;
  if(nvds_enable_latency_measurement)
  {
    GstBuffer *buf = (GstBuffer *) info->data;
    NvDsFrameLatencyInfo *latency_info = NULL;
    g_mutex_lock (&appCtx->latency_lock);
    latency_info = appCtx->latency_info;
    g_print("\n************DEMUX BATCH-NUM = %d**************\n",demux_batch_num);
    num_sources_in_batch = nvds_measure_buffer_latency(buf, latency_info);

    for(i = 0; i < num_sources_in_batch; i++)
    {
      g_print("Source id = %d Frame_num = %d Frame latency = %lf (ms) \n",
          latency_info[i].source_id,
          latency_info[i].frame_num,
          latency_info[i].latency);
    }
    g_mutex_unlock (&appCtx->latency_lock);
    demux_batch_num++;
  }

  return GST_PAD_PROBE_OK;
}

static gboolean
add_and_link_broker_sink (AppCtx * appCtx)
{
  NvDsConfig *config = &appCtx->config;
  /** Only first instance_bin broker sink
   * employed as there's only one analytics path for N sources
   * NOTE: There shall be only one [sink] group
   * with type=6 (NV_DS_SINK_MSG_CONV_BROKER)
   * a) Multiple of them does not make sense as we have only
   * one analytics pipe generating the data for broker sink
   * b) If Multiple broker sinks are configured by the user
   * in config file, only the first in the order of
   * appearance will be considered
   * and others shall be ignored
   * c) Ideally it should be documented (or obvious) that:
   * multiple [sink] groups with type=6 (NV_DS_SINK_MSG_CONV_BROKER)
   * is invalid
   */
  NvDsInstanceBin *instance_bin = &appCtx->pipeline.instance_bins[0];
  NvDsPipeline *pipeline = &appCtx->pipeline;

  for (guint i = 0; i < config->num_sink_sub_bins; i++) {
    if(config->sink_bin_sub_bin_config[i].type == NV_DS_SINK_MSG_CONV_BROKER)
    {
      if(!pipeline->common_elements.tee) {
         NVGSTDS_ERR_MSG_V ("%s failed; broker added without analytics; check config file\n", __func__);
         return FALSE;
      }
      /** add the broker sink bin to pipeline */
      if(!gst_bin_add (GST_BIN (pipeline->pipeline), instance_bin->sink_bin.sub_bins[i].bin)) {
        return FALSE;
      }
      /** link the broker sink bin to the common_elements tee
       * (The tee after nvinfer -> tracker (optional) -> sgies (optional) block) */
      if (!link_element_to_tee_src_pad (pipeline->common_elements.tee, instance_bin->sink_bin.sub_bins[i].bin)) {
        return FALSE;
      }
    }
  }
  return TRUE;
}

static gboolean
create_demux_pipeline (AppCtx * appCtx, guint index)
{
  gboolean ret = FALSE;
  NvDsConfig *config = &appCtx->config;
  NvDsInstanceBin *instance_bin = &appCtx->pipeline.demux_instance_bins[index];
  GstElement *last_elem;
  gchar elem_name[32];

  instance_bin->index = index;
  instance_bin->appCtx = appCtx;

  g_snprintf (elem_name, 32, "processing_demux_bin_%d", index);
  instance_bin->bin = gst_bin_new (elem_name);

  if (!create_demux_sink_bin (config->num_sink_sub_bins,
        config->sink_bin_sub_bin_config, &instance_bin->demux_sink_bin,
        config->sink_bin_sub_bin_config[index].source_id)) {
    goto done;
  }

  gst_bin_add (GST_BIN (instance_bin->bin), instance_bin->demux_sink_bin.bin);
  last_elem = instance_bin->demux_sink_bin.bin;

  if (config->osd_config.enable) {
    if (!create_osd_bin (&config->osd_config, &instance_bin->osd_bin)) {
      goto done;
    }

    gst_bin_add (GST_BIN (instance_bin->bin), instance_bin->osd_bin.bin);

    NVGSTDS_LINK_ELEMENT (instance_bin->osd_bin.bin, last_elem);

    last_elem = instance_bin->osd_bin.bin;
  }

  NVGSTDS_BIN_ADD_GHOST_PAD (instance_bin->bin, last_elem, "sink");
  if (config->osd_config.enable) {

    NVGSTDS_ELEM_ADD_PROBE (instance_bin->all_bbox_buffer_probe_id,
        instance_bin->osd_bin.nvosd, "sink",
        gie_processing_done_buf_prob, GST_PAD_PROBE_TYPE_BUFFER, instance_bin);
		
  } else {
    NVGSTDS_ELEM_ADD_PROBE (instance_bin->all_bbox_buffer_probe_id,
        instance_bin->demux_sink_bin.bin, "sink",
        gie_processing_done_buf_prob, GST_PAD_PROBE_TYPE_BUFFER, instance_bin);
  }

  ret = TRUE;
done:
  if (!ret) {
    NVGSTDS_ERR_MSG_V ("%s failed", __func__);
  }
  return ret;
}

/**
 * Function to add components to pipeline which are dependent on number
 * of streams. These components work on single buffer. If tiling is being
 * used then single instance will be created otherwise < N > such instances
 * will be created for < N > streams
 */
static gboolean
create_processing_instance (AppCtx * appCtx, guint index)
{

  gboolean ret = FALSE;
  NvDsConfig *config = &appCtx->config;
  NvDsInstanceBin *instance_bin = &appCtx->pipeline.instance_bins[index];
  GstElement *last_elem;
  gchar elem_name[32];
 
  instance_bin->index = index;
  instance_bin->appCtx = appCtx;
  g_snprintf (elem_name, 32, "processing_bin_%d", index);
  instance_bin->bin = gst_bin_new (elem_name);

  if (!create_sink_bin (config->num_sink_sub_bins,
        config->sink_bin_sub_bin_config, &instance_bin->sink_bin, index)) {
    goto done;
  }

  gst_bin_add (GST_BIN (instance_bin->bin), instance_bin->sink_bin.bin);
  last_elem = instance_bin->sink_bin.bin;

  if (config->osd_config.enable) {
    if (!create_osd_bin (&config->osd_config, &instance_bin->osd_bin)) {
      goto done;
    }

    gst_bin_add (GST_BIN (instance_bin->bin), instance_bin->osd_bin.bin);

    NVGSTDS_LINK_ELEMENT (instance_bin->osd_bin.bin, last_elem);

    last_elem = instance_bin->osd_bin.bin;
  }
//g_print("xxxxxxxxxxxxxxxxxxxxxxxx2333333333321\n");
  NVGSTDS_BIN_ADD_GHOST_PAD (instance_bin->bin, last_elem, "sink");
  if (config->osd_config.enable) {
	 // g_print("xxxxxxxxxxxxxxxxxxxxxxxx2222222222221\n");
    NVGSTDS_ELEM_ADD_PROBE (instance_bin->all_bbox_buffer_probe_id,
        instance_bin->osd_bin.nvosd, "sink",
        gie_processing_done_buf_prob, GST_PAD_PROBE_TYPE_BUFFER, instance_bin);
		/* NVGSTDS_ELEM_ADD_PROBE (instance_bin->all_bbox_buffer_probe_id,
        instance_bin->osd_bin.nvosd, "sink",
        tiler_src_pad_buffer_probe, GST_PAD_PROBE_TYPE_BUFFER, instance_bin); */
  } else {
	  
    NVGSTDS_ELEM_ADD_PROBE (instance_bin->all_bbox_buffer_probe_id,
        instance_bin->sink_bin.bin, "sink",
        gie_processing_done_buf_prob, GST_PAD_PROBE_TYPE_BUFFER, instance_bin);
  }
  
  
  //write_kitti_output (appCtx, batch_meta, buf);
  
  ret = TRUE;
done:
  if (!ret) {
    NVGSTDS_ERR_MSG_V ("%s failed", __func__);
  }
  return ret;
}

/**
 * Function to create common elements(Primary infer, tracker, secondary infer)
 * of the pipeline. These components operate on muxed data from all the
 * streams. So they are independent of number of streams in the pipeline.
 */
static gboolean
create_common_elements (NvDsConfig * config, NvDsPipeline * pipeline,
    GstElement ** sink_elem, GstElement ** src_elem,
    bbox_generated_callback bbox_generated_post_analytics_cb)
{
  gboolean ret = FALSE;
  *sink_elem = *src_elem = NULL;

  if (config->primary_gie_config.enable) {
    if (config->num_secondary_gie_sub_bins > 0) {
      if (!create_secondary_gie_bin (config->num_secondary_gie_sub_bins,
              config->primary_gie_config.unique_id,
              config->secondary_gie_sub_bin_config,
              &pipeline->common_elements.secondary_gie_bin)) {
        goto done;
      }
      gst_bin_add (GST_BIN (pipeline->pipeline),
          pipeline->common_elements.secondary_gie_bin.bin);
      if (!*src_elem) {
        *src_elem = pipeline->common_elements.secondary_gie_bin.bin;
      }
      if (*sink_elem) {
        NVGSTDS_LINK_ELEMENT (pipeline->common_elements.secondary_gie_bin.bin,
            *sink_elem);
      }
      *sink_elem = pipeline->common_elements.secondary_gie_bin.bin;
    }
  }

  if (config->dsanalytics_config.enable) {
    // Create dsanalytics element bin and set properties
    if (!create_dsanalytics_bin (&config->dsanalytics_config,
            &pipeline->common_elements.dsanalytics_bin)) {
      g_print ("creating dsanalytics bin failed\n");
      goto done;
    }
    // Add dsanalytics bin to instance bin
    gst_bin_add (GST_BIN (pipeline->pipeline), pipeline->common_elements.dsanalytics_bin.bin);

      if (!*src_elem) {
        *src_elem = pipeline->common_elements.dsanalytics_bin.bin;
      }
      if (*sink_elem) {
        NVGSTDS_LINK_ELEMENT (pipeline->common_elements.dsanalytics_bin.bin,
            *sink_elem);
      }
    // Set this bin as the last element
    *sink_elem = pipeline->common_elements.dsanalytics_bin.bin;
  }

  if (config->tracker_config.enable) {
    if (!create_tracking_bin (&config->tracker_config,
            &pipeline->common_elements.tracker_bin)) {
      g_print ("creating tracker bin failed\n");
      goto done;
    }
    gst_bin_add (GST_BIN (pipeline->pipeline),
        pipeline->common_elements.tracker_bin.bin);
    if (!*src_elem) {
      *src_elem = pipeline->common_elements.tracker_bin.bin;
    }
    if (*sink_elem) {
      NVGSTDS_LINK_ELEMENT (pipeline->common_elements.tracker_bin.bin,
          *sink_elem);
    }
    *sink_elem = pipeline->common_elements.tracker_bin.bin;
  }

  if (config->primary_gie_config.enable) {
    if (!create_primary_gie_bin (&config->primary_gie_config,
            &pipeline->common_elements.primary_gie_bin)) {
      goto done;
    }
    gst_bin_add (GST_BIN (pipeline->pipeline),
        pipeline->common_elements.primary_gie_bin.bin);
    if (*sink_elem) {
      NVGSTDS_LINK_ELEMENT (pipeline->common_elements.primary_gie_bin.bin,
          *sink_elem);
    }
    *sink_elem = pipeline->common_elements.primary_gie_bin.bin;
    if (!*src_elem) {
      *src_elem = pipeline->common_elements.primary_gie_bin.bin;
    }
    /* NVGSTDS_ELEM_ADD_PROBE (pipeline->common_elements.
        primary_bbox_buffer_probe_id,
        pipeline->common_elements.primary_gie_bin.bin, "src",
        gie_primary_processing_done_buf_prob, GST_PAD_PROBE_TYPE_BUFFER,
        pipeline->common_elements.appCtx); */
  }

  if(*src_elem) {
    NVGSTDS_ELEM_ADD_PROBE (pipeline->common_elements.
          primary_bbox_buffer_probe_id,
          *src_elem, "src",
          analytics_done_buf_prob, GST_PAD_PROBE_TYPE_BUFFER,
          &pipeline->common_elements);

    /* Add common message converter */
    if (config->msg_conv_config.enable) {
      NvDsSinkMsgConvBrokerConfig *convConfig = &config->msg_conv_config;
      pipeline->common_elements.msg_conv = gst_element_factory_make (NVDS_ELEM_MSG_CONV, "common_msg_conv");
      if (!pipeline->common_elements.msg_conv) {
        NVGSTDS_ERR_MSG_V ("Failed to create element 'common_msg_conv'");
        goto done;
      }

      g_object_set (G_OBJECT (pipeline->common_elements.msg_conv),
                    "config", convConfig->config_file_path,
                    "msg2p-lib", (convConfig->conv_msg2p_lib ? convConfig->conv_msg2p_lib : "null"),
                    "payload-type", convConfig->conv_payload_type,
                    "comp-id", convConfig->conv_comp_id,
                    "debug-payload-dir", convConfig->debug_payload_dir,
                    "multiple-payloads", convConfig->multiple_payloads, NULL);

      gst_bin_add (GST_BIN (pipeline->pipeline),
                   pipeline->common_elements.msg_conv);

      NVGSTDS_LINK_ELEMENT (*src_elem, pipeline->common_elements.msg_conv);
      *src_elem = pipeline->common_elements.msg_conv;
    }
    pipeline->common_elements.tee = gst_element_factory_make (NVDS_ELEM_TEE, "common_analytics_tee");
    if (!pipeline->common_elements.tee) {
      NVGSTDS_ERR_MSG_V ("Failed to create element 'common_analytics_tee'");
      goto done;
    }

    gst_bin_add (GST_BIN (pipeline->pipeline),
          pipeline->common_elements.tee);

    NVGSTDS_LINK_ELEMENT (*src_elem, pipeline->common_elements.tee);
    *src_elem = pipeline->common_elements.tee;
  }

  ret = TRUE;
done:
  return ret;
}

static gboolean is_sink_available_for_source_id(NvDsConfig *config, guint source_id) {
  for (guint j = 0; j < config->num_sink_sub_bins; j++) {
    if (config->sink_bin_sub_bin_config[j].enable &&
        config->sink_bin_sub_bin_config[j].source_id == source_id &&
        config->sink_bin_sub_bin_config[j].link_to_demux == FALSE) {
      return TRUE;
    }
  }
  return FALSE;
}

/**
 * Main function to create the pipeline.
 */
gboolean
create_pipeline (AppCtx * appCtx,
    bbox_generated_callback bbox_generated_post_analytics_cb,
    bbox_generated_callback all_bbox_generated_cb, perf_callback perf_cb,
    overlay_graphics_callback overlay_graphics_cb)
{
  gboolean ret = FALSE;
  NvDsPipeline *pipeline = &appCtx->pipeline;
  NvDsConfig *config = &appCtx->config;
  GstBus *bus;
  GstElement *last_elem;
  GstElement *tmp_elem1;
  GstElement *tmp_elem2;
  guint i;
  GstPad *fps_pad;
  gulong latency_probe_id;

  _dsmeta_quark = g_quark_from_static_string (NVDS_META_STRING);

  appCtx->all_bbox_generated_cb = all_bbox_generated_cb;
  appCtx->bbox_generated_post_analytics_cb = bbox_generated_post_analytics_cb;
  appCtx->overlay_graphics_cb = overlay_graphics_cb;

  if (config->osd_config.num_out_buffers < 8) {
    config->osd_config.num_out_buffers = 8;
  }

  pipeline->pipeline = gst_pipeline_new ("pipeline");
  if (!pipeline->pipeline) {
    NVGSTDS_ERR_MSG_V ("Failed to create pipeline");
    goto done;
  }

  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline->pipeline));
  pipeline->bus_id = gst_bus_add_watch (bus, bus_callback, appCtx);
  gst_object_unref (bus);

  if (config->file_loop) {
    /* Let each source bin know it needs to loop. */
    guint i;
    for (i = 0; i < config->num_source_sub_bins; i++)
        config->multi_source_config[i].loop = TRUE;
  }

  for (guint i = 0; i < config->num_sink_sub_bins; i++) {
    NvDsSinkSubBinConfig *sink_config = &config->sink_bin_sub_bin_config[i];
    switch (sink_config->type) {
      case NV_DS_SINK_FAKE:
      case NV_DS_SINK_RENDER_EGL:
      case NV_DS_SINK_RENDER_OVERLAY:
        /* Set the "qos" property of sink, if not explicitly specified in the
           config. */
        if (!sink_config->render_config.qos_value_specified) {
          sink_config->render_config.qos = FALSE;
        }
      default:
        break;
    }
  }
  /*
   * Add muxer and < N > source components to the pipeline based
   * on the settings in configuration file.
   */
  if (!create_multi_source_bin (config->num_source_sub_bins,
          config->multi_source_config, &pipeline->multi_src_bin))
    goto done;
  gst_bin_add (GST_BIN (pipeline->pipeline), pipeline->multi_src_bin.bin);


  if (config->streammux_config.is_parsed){
    if(!set_streammux_properties (&config->streammux_config,
        pipeline->multi_src_bin.streammux)){
         NVGSTDS_WARN_MSG_V("Failed to set streammux properties");
    }
  }


  if(appCtx->latency_info == NULL)
  {
    appCtx->latency_info = (NvDsFrameLatencyInfo *)
      calloc(1, config->streammux_config.batch_size *
          sizeof(NvDsFrameLatencyInfo));
  }

  /** a tee after the tiler which shall be connected to sink(s) */
  pipeline->tiler_tee = gst_element_factory_make (NVDS_ELEM_TEE, "tiler_tee");
  if (!pipeline->tiler_tee) {
    NVGSTDS_ERR_MSG_V ("Failed to create element 'tiler_tee'");
    goto done;
  }
  gst_bin_add (GST_BIN (pipeline->pipeline), pipeline->tiler_tee);

  /** Tiler + Demux in Parallel Use-Case */
  if (config->tiled_display_config.enable == NV_DS_TILED_DISPLAY_ENABLE_WITH_PARALLEL_DEMUX)
  {
    pipeline->demuxer =
        gst_element_factory_make (NVDS_ELEM_STREAM_DEMUX, "demuxer");
    if (!pipeline->demuxer) {
      NVGSTDS_ERR_MSG_V ("Failed to create element 'demuxer'");
      goto done;
    }
    gst_bin_add (GST_BIN (pipeline->pipeline), pipeline->demuxer);

    /** NOTE:
     * demux output is supported for only one source
     * If multiple [sink] groups are configured with
     * link_to_demux=1, only the first [sink]
     * shall be constructed for all occurences of
     * [sink] groups with link_to_demux=1
     */
    {
      gchar pad_name[16];
      GstPad *demux_src_pad;

      i = 0;
      if (!create_demux_pipeline (appCtx, i)) {
        goto done;
      }

      for (i=0; i < config->num_sink_sub_bins; i++)
      {
        if (config->sink_bin_sub_bin_config[i].link_to_demux == TRUE)
        {
          g_snprintf (pad_name, 16, "src_%02d", config->sink_bin_sub_bin_config[i].source_id);
          break;
        }
      }

      if (i >= config->num_sink_sub_bins)
      {
        g_print ("\n\nError : sink for demux (use link-to-demux-only property) is not provided in the config file\n\n");
        goto done;
      }

      i = 0;

      gst_bin_add (GST_BIN (pipeline->pipeline),
          pipeline->demux_instance_bins[i].bin);

      demux_src_pad = gst_element_get_request_pad (pipeline->demuxer, pad_name);
      NVGSTDS_LINK_ELEMENT_FULL (pipeline->demuxer, pad_name,
          pipeline->demux_instance_bins[i].bin, "sink");
      gst_object_unref (demux_src_pad);

      NVGSTDS_ELEM_ADD_PROBE(latency_probe_id,
          appCtx->pipeline.demux_instance_bins[i].demux_sink_bin.bin,
          "sink",
          demux_latency_measurement_buf_prob, GST_PAD_PROBE_TYPE_BUFFER,
          appCtx);
      latency_probe_id = latency_probe_id;
    }

    last_elem = pipeline->demuxer;
    link_element_to_tee_src_pad (pipeline->tiler_tee, last_elem);
    last_elem = pipeline->tiler_tee;
  }

  if (config->tiled_display_config.enable) {

    /* Tiler will generate a single composited buffer for all sources. So need
     * to create only one processing instance. */
    if (!create_processing_instance (appCtx, 0)) {
      goto done;
    }
    // create and add tiling component to pipeline.
    if (config->tiled_display_config.columns *
        config->tiled_display_config.rows < config->num_source_sub_bins) {
      if (config->tiled_display_config.columns == 0) {
        config->tiled_display_config.columns =
            (guint) (sqrt (config->num_source_sub_bins) + 0.5);
      }
      config->tiled_display_config.rows =
          (guint) ceil (1.0 * config->num_source_sub_bins /
          config->tiled_display_config.columns);
      NVGSTDS_WARN_MSG_V
          ("Num of Tiles less than number of sources, readjusting to "
          "%u rows, %u columns", config->tiled_display_config.rows,
          config->tiled_display_config.columns);
    }

    gst_bin_add (GST_BIN (pipeline->pipeline), pipeline->instance_bins[0].bin);
    last_elem = pipeline->instance_bins[0].bin;

    if (!create_tiled_display_bin (&config->tiled_display_config,
            &pipeline->tiled_display_bin)) {
      goto done;
    }
    gst_bin_add (GST_BIN (pipeline->pipeline), pipeline->tiled_display_bin.bin);
    NVGSTDS_LINK_ELEMENT (pipeline->tiled_display_bin.bin, last_elem);
    last_elem = pipeline->tiled_display_bin.bin;

    link_element_to_tee_src_pad (pipeline->tiler_tee, pipeline->tiled_display_bin.bin);
    last_elem = pipeline->tiler_tee;

    NVGSTDS_ELEM_ADD_PROBE (latency_probe_id,
      pipeline->instance_bins->sink_bin.sub_bins[0].sink, "sink",
      latency_measurement_buf_prob, GST_PAD_PROBE_TYPE_BUFFER,
      appCtx);
    latency_probe_id = latency_probe_id;
  }
  else
  {
    /*
     * Create demuxer only if tiled display is disabled.
     */
    pipeline->demuxer =
        gst_element_factory_make (NVDS_ELEM_STREAM_DEMUX, "demuxer");
    if (!pipeline->demuxer) {
      NVGSTDS_ERR_MSG_V ("Failed to create element 'demuxer'");
      goto done;
    }
    gst_bin_add (GST_BIN (pipeline->pipeline), pipeline->demuxer);

    for (i = 0; i < config->num_source_sub_bins; i++)
    {
      gchar pad_name[16];
      GstPad *demux_src_pad;

      /* Check if any sink has been configured to render/encode output for
       * source index `i`. The processing instance for that source will be
       * created only if atleast one sink has been configured as such.
       */
      if (!is_sink_available_for_source_id(config, i))
        continue;

      if (!create_processing_instance(appCtx, i))
      {
        goto done;
      }
      gst_bin_add(GST_BIN(pipeline->pipeline),
                  pipeline->instance_bins[i].bin);

      g_snprintf(pad_name, 16, "src_%02d", i);
      demux_src_pad = gst_element_get_request_pad(pipeline->demuxer, pad_name);
      NVGSTDS_LINK_ELEMENT_FULL(pipeline->demuxer, pad_name,
                                pipeline->instance_bins[i].bin, "sink");
      gst_object_unref(demux_src_pad);

      for (int k = 0; k < MAX_SINK_BINS;k++) {
        if(pipeline->instance_bins[i].sink_bin.sub_bins[k].sink){
          NVGSTDS_ELEM_ADD_PROBE(latency_probe_id,
              pipeline->instance_bins[i].sink_bin.sub_bins[k].sink, "sink",
              latency_measurement_buf_prob, GST_PAD_PROBE_TYPE_BUFFER,
              appCtx);
          break;
        }
      }

      latency_probe_id = latency_probe_id;
    }
    last_elem = pipeline->demuxer;
  }

  if (config->tiled_display_config.enable == NV_DS_TILED_DISPLAY_DISABLE) {
    fps_pad = gst_element_get_static_pad (pipeline->demuxer, "sink");
  }
  else {
    fps_pad = gst_element_get_static_pad (pipeline->tiled_display_bin.bin, "sink");
  }

  pipeline->common_elements.appCtx = appCtx;
  // Decide where in the pipeline the element should be added and add only if
  // enabled
  if (config->dsexample_config.enable) {
    // Create dsexample element bin and set properties
    if (!create_dsexample_bin (&config->dsexample_config,
            &pipeline->dsexample_bin)) {
      goto done;
    }
    // Add dsexample bin to instance bin
    gst_bin_add (GST_BIN (pipeline->pipeline), pipeline->dsexample_bin.bin);

    // Link this bin to the last element in the bin
    NVGSTDS_LINK_ELEMENT (pipeline->dsexample_bin.bin, last_elem);

    // Set this bin as the last element
    last_elem = pipeline->dsexample_bin.bin;
  }
  // create and add common components to pipeline.
  if (!create_common_elements (config, pipeline, &tmp_elem1, &tmp_elem2,
          bbox_generated_post_analytics_cb)) {
    goto done;
  }

  if(!add_and_link_broker_sink(appCtx)) {
    goto done;
  }

  if (tmp_elem2) {
    NVGSTDS_LINK_ELEMENT (tmp_elem2, last_elem);
    last_elem = tmp_elem1;
  }

  NVGSTDS_LINK_ELEMENT (pipeline->multi_src_bin.bin, last_elem);

  // enable performance measurement and add call back function to receive
  // performance data.
  if (config->enable_perf_measurement) {
    appCtx->perf_struct.context = appCtx;
    enable_perf_measurement (&appCtx->perf_struct, fps_pad,
        pipeline->multi_src_bin.num_bins,
        config->perf_measurement_interval_sec,
        config->multi_source_config[0].dewarper_config.num_surfaces_per_frame,
        perf_cb);
  }

  latency_probe_id = latency_probe_id;

  if (config->num_message_consumers) {
    for (i = 0; i < config->num_message_consumers; i++) {
      appCtx->c2d_ctx[i] = start_cloud_to_device_messaging (
                                &config->message_consumer_config[i], NULL,
                                &appCtx->pipeline.multi_src_bin);
      if (appCtx->c2d_ctx[i] == NULL) {
        NVGSTDS_ERR_MSG_V ("Failed to create message consumer");
        goto done;
      }
    }
  }

  GST_DEBUG_BIN_TO_DOT_FILE_WITH_TS (GST_BIN (appCtx->pipeline.pipeline),
      GST_DEBUG_GRAPH_SHOW_ALL, "ds-app-null");

  g_mutex_init (&appCtx->app_lock);
  g_cond_init (&appCtx->app_cond);
  g_mutex_init (&appCtx->latency_lock);

  ret = TRUE;
done:
  if (!ret) {
    NVGSTDS_ERR_MSG_V ("%s failed", __func__);
  }
  return ret;
}

/**
 * Function to destroy pipeline and release the resources, probes etc.
 */
void
destroy_pipeline (AppCtx * appCtx)
{
  gint64 end_time;
  NvDsConfig *config = &appCtx->config;
  guint i;
  GstBus *bus = NULL;

  end_time = g_get_monotonic_time () + G_TIME_SPAN_SECOND;

  if (!appCtx)
    return;

  if (appCtx->pipeline.demuxer) {
    gst_pad_send_event (gst_element_get_static_pad (appCtx->pipeline.demuxer,
            "sink"), gst_event_new_eos ());
  } else if (appCtx->pipeline.instance_bins[0].sink_bin.bin) {
    gst_pad_send_event (gst_element_get_static_pad (appCtx->
            pipeline.instance_bins[0].sink_bin.bin, "sink"),
        gst_event_new_eos ());
  }

  g_usleep (100000);

  g_mutex_lock (&appCtx->app_lock);
  if (appCtx->pipeline.pipeline) {
    destroy_smart_record_bin (&appCtx->pipeline.multi_src_bin);
    bus = gst_pipeline_get_bus (GST_PIPELINE (appCtx->pipeline.pipeline));

    while (TRUE) {
      GstMessage *message = gst_bus_pop (bus);
      if (message == NULL)
        break;
      else if (GST_MESSAGE_TYPE (message) == GST_MESSAGE_ERROR)
        bus_callback (bus, message, appCtx);
      else
        gst_message_unref (message);
    }
    gst_element_set_state (appCtx->pipeline.pipeline, GST_STATE_NULL);
  }
  g_cond_wait_until (&appCtx->app_cond, &appCtx->app_lock, end_time);
  g_mutex_unlock (&appCtx->app_lock);

  for (i = 0; i < appCtx->config.num_source_sub_bins; i++) {
    NvDsInstanceBin *bin = &appCtx->pipeline.instance_bins[i];
    if (config->osd_config.enable) {
      NVGSTDS_ELEM_REMOVE_PROBE (bin->all_bbox_buffer_probe_id,
          bin->osd_bin.nvosd, "sink");
    } else {
      NVGSTDS_ELEM_REMOVE_PROBE (bin->all_bbox_buffer_probe_id,
          bin->sink_bin.bin, "sink");
    }

    if (config->primary_gie_config.enable) {
      NVGSTDS_ELEM_REMOVE_PROBE (bin->primary_bbox_buffer_probe_id,
          bin->primary_gie_bin.bin, "src");
    }

  }
  if(appCtx->latency_info == NULL)
  {
    free(appCtx->latency_info);
    appCtx->latency_info = NULL;
  }

  destroy_sink_bin ();
  g_mutex_clear(&appCtx->latency_lock);

  if (appCtx->pipeline.pipeline) {
    bus = gst_pipeline_get_bus (GST_PIPELINE (appCtx->pipeline.pipeline));
    gst_bus_remove_watch (bus);
    gst_object_unref (bus);
    gst_object_unref (appCtx->pipeline.pipeline);
  }

  if (config->num_message_consumers) {
    for (i = 0; i < config->num_message_consumers; i++) {
      if (appCtx->c2d_ctx[i])
        stop_cloud_to_device_messaging (appCtx->c2d_ctx[i]);
    }
  }
}

gboolean
pause_pipeline (AppCtx * appCtx)
{
  GstState cur;
  GstState pending;
  GstStateChangeReturn ret;
  GstClockTime timeout = 5 * GST_SECOND / 1000;

  ret =
      gst_element_get_state (appCtx->pipeline.pipeline, &cur, &pending,
      timeout);

  if (ret == GST_STATE_CHANGE_ASYNC) {
    return FALSE;
  }

  if (cur == GST_STATE_PAUSED) {
    return TRUE;
  } else if (cur == GST_STATE_PLAYING) {
    gst_element_set_state (appCtx->pipeline.pipeline, GST_STATE_PAUSED);
    gst_element_get_state (appCtx->pipeline.pipeline, &cur, &pending,
        GST_CLOCK_TIME_NONE);
    pause_perf_measurement (&appCtx->perf_struct);
    return TRUE;
  } else {
    return FALSE;
  }
}

gboolean
resume_pipeline (AppCtx * appCtx)
{
  GstState cur;
  GstState pending;
  GstStateChangeReturn ret;
  GstClockTime timeout = 5 * GST_SECOND / 1000;

  ret =
      gst_element_get_state (appCtx->pipeline.pipeline, &cur, &pending,
      timeout);

  if (ret == GST_STATE_CHANGE_ASYNC) {
    return FALSE;
  }

  if (cur == GST_STATE_PLAYING) {
    return TRUE;
  } else if (cur == GST_STATE_PAUSED) {
    gst_element_set_state (appCtx->pipeline.pipeline, GST_STATE_PLAYING);
    gst_element_get_state (appCtx->pipeline.pipeline, &cur, &pending,
        GST_CLOCK_TIME_NONE);
    resume_perf_measurement (&appCtx->perf_struct);
    return TRUE;
  } else {
    return FALSE;
  }
}

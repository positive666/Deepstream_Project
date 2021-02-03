/*
 * Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
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

#ifndef __NVGSTDS_APP_H__
#define __NVGSTDS_APP_H__

#ifdef __cplusplus
extern "C"
{
#endif

#include <gst/gst.h>
#include <stdio.h>

#include "deepstream_app_version.h"
#include "deepstream_common.h"
#include "deepstream_config.h"
#include "deepstream_osd.h"
#include "deepstream_perf.h"
#include "deepstream_primary_gie.h"
#include "deepstream_sinks.h"
#include "deepstream_sources.h"
#include "deepstream_streammux.h"
#include "deepstream_tiled_display.h"
#include "deepstream_dsexample.h"
#include "deepstream_tracker.h"
#include "deepstream_secondary_gie.h"

typedef struct _AppCtx AppCtx;

typedef void (*bbox_generated_callback) (AppCtx *appCtx, GstBuffer *buf,
    NvDsBatchMeta *batch_meta, guint index);
typedef gboolean (*overlay_graphics_callback) (AppCtx *appCtx, GstBuffer *buf,
    NvDsBatchMeta *batch_meta, guint index);


typedef struct
{
  guint index;
  gulong all_bbox_buffer_probe_id;
  gulong primary_bbox_buffer_probe_id;
  gulong fps_buffer_probe_id;
  GstElement *bin;
  NvDsPrimaryGieBin primary_gie_bin;
  NvDsOSDBin osd_bin;
  NvDsSecondaryGieBin secondary_gie_bin;
  NvDsTrackerBin tracker_bin;
  NvDsSinkBin sink_bin;
  NvDsDsExampleBin dsexample_bin;
  AppCtx *appCtx;
} NvDsInstanceBin;

typedef struct
{
  gulong primary_bbox_buffer_probe_id;
  guint bus_id;
  GstElement *pipeline;
  NvDsSrcParentBin multi_src_bin;
  NvDsInstanceBin instance_bins[MAX_SOURCE_BINS];
  NvDsInstanceBin common_elements;
  NvDsTiledDisplayBin tiled_display_bin;
  GstElement *demuxer;
  NvDsDsExampleBin dsexample_bin;
  AppCtx *appCtx;
} NvDsPipeline;

typedef struct
{
  gboolean enable_perf_measurement;
  gint file_loop;
  guint num_source_sub_bins;
  guint num_secondary_gie_sub_bins;
  guint num_sink_sub_bins;
  guint perf_measurement_interval_sec;
  gchar *bbox_dir_path;
  gchar *kitti_track_dir_path;

  NvDsSourceConfig multi_source_config[MAX_SOURCE_BINS];
  NvDsStreammuxConfig streammux_config;
  NvDsOSDConfig osd_config;
  NvDsGieConfig primary_gie_config;
  NvDsTrackerConfig tracker_config;
  NvDsGieConfig secondary_gie_sub_bin_config[MAX_SECONDARY_GIE_BINS];
  NvDsSinkSubBinConfig sink_bin_sub_bin_config[MAX_SINK_BINS];
  NvDsTiledDisplayConfig tiled_display_config;
  NvDsDsExampleConfig dsexample_config;
} NvDsConfig;

typedef struct
{
  gulong frame_num;
} NvDsInstanceData;

struct _AppCtx
{
  gboolean version;
  gboolean cintr;
  gboolean show_bbox_text;
  gboolean seeking;
  gboolean quit;
  gint person_class_id;
  gint car_class_id;
  gint return_value;
  guint index;

  GMutex app_lock;
  GCond app_cond;

  NvDsPipeline pipeline;
  NvDsConfig config;
  NvDsInstanceData instance_data[MAX_SOURCE_BINS];
  NvDsAppPerfStructInt perf_struct;
  bbox_generated_callback primary_bbox_generated_cb;
  bbox_generated_callback all_bbox_generated_cb;
  overlay_graphics_callback overlay_graphics_cb;
  NvDsFrameLatencyInfo *latency_info;
  GMutex latency_lock;
};

gboolean create_pipeline (AppCtx * appCtx,
    bbox_generated_callback primary_bbox_generated_cb,
    bbox_generated_callback all_bbox_generated_cb,
    perf_callback perf_cb,
    overlay_graphics_callback overlay_graphics_cb);

gboolean pause_pipeline (AppCtx * appCtx);
gboolean resume_pipeline (AppCtx * appCtx);
gboolean seek_pipeline (AppCtx * appCtx, glong milliseconds, gboolean seek_is_relative);

void toggle_show_bbox_text (AppCtx * appCtx);

void destroy_pipeline (AppCtx * appCtx);
void restart_pipeline (AppCtx * appCtx);


/**
 * Function to read properties from configuration file.
 *
 * @param[in] config pointer to @ref NvDsConfig
 * @param[in] cfg_file_path path of configuration file.
 *
 * @return true if parsed successfully.
 */
gboolean
parse_config_file (NvDsConfig * config, gchar * cfg_file_path);

#ifdef __cplusplus
}
#endif

#endif

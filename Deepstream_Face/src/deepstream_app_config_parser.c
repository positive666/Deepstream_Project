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
#include <string.h>
#include "deepstream_app.h"
#include "deepstream_config_file_parser.h"

#define CONFIG_GROUP_APP "application"
#define CONFIG_GROUP_APP_ENABLE_PERF_MEASUREMENT "enable-perf-measurement"
#define CONFIG_GROUP_APP_PERF_MEASUREMENT_INTERVAL "perf-measurement-interval-sec"
#define CONFIG_GROUP_APP_GIE_OUTPUT_DIR "gie-kitti-output-dir"
#define CONFIG_GROUP_APP_GIE_TRACK_OUTPUT_DIR "kitti-track-output-dir"

#define CONFIG_GROUP_TESTS "tests"
#define CONFIG_GROUP_TESTS_FILE_LOOP "file-loop"

#define CONFIG_GROUP_SOURCE_SGIE_BATCH_SIZE "sgie-batch-size"

GST_DEBUG_CATEGORY_EXTERN (APP_CFG_PARSER_CAT);


#define CHECK_ERROR(error) \
    if (error) { \
        GST_CAT_ERROR (APP_CFG_PARSER_CAT, "%s", error->message); \
        goto done; \
    }

NvDsSourceConfig global_source_config;

static gboolean
parse_source_list (NvDsConfig * config, GKeyFile * key_file,
    gchar * cfg_file_path)
{
  gboolean ret = FALSE;
  gchar **keys = NULL;
  gchar **key = NULL;
  GError *error = NULL;
  gsize num_strings;

  keys = g_key_file_get_keys (key_file, CONFIG_GROUP_SOURCE_LIST, NULL, &error);
  CHECK_ERROR (error);

  for (key = keys; *key; key++) {
    if (!g_strcmp0 (*key, CONFIG_GROUP_SOURCE_LIST_NUM_SOURCE_BINS)) {
      config->total_num_sources =
          g_key_file_get_integer (key_file, CONFIG_GROUP_SOURCE_LIST,
          CONFIG_GROUP_SOURCE_LIST_NUM_SOURCE_BINS, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SOURCE_LIST_URI_LIST)) {
      config->uri_list =
          g_key_file_get_string_list (key_file, CONFIG_GROUP_SOURCE_LIST,
          CONFIG_GROUP_SOURCE_LIST_URI_LIST, &num_strings, &error);
      if (num_strings > MAX_SOURCE_BINS) {
        NVGSTDS_ERR_MSG_V ("App supports max %d sources", MAX_SOURCE_BINS);
        goto done;
      }
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SOURCE_SGIE_BATCH_SIZE)) {
      config->sgie_batch_size =
          g_key_file_get_integer (key_file, CONFIG_GROUP_SOURCE_LIST,
          CONFIG_GROUP_SOURCE_SGIE_BATCH_SIZE, &error);
      CHECK_ERROR (error);
    } else {
      NVGSTDS_WARN_MSG_V ("Unknown key '%s' for group [%s]", *key,
          CONFIG_GROUP_SOURCE_LIST);
    }
  }

  if (g_key_file_has_key (key_file, CONFIG_GROUP_SOURCE_LIST,
          CONFIG_GROUP_SOURCE_LIST_URI_LIST, &error)) {
    if (g_key_file_has_key (key_file, CONFIG_GROUP_SOURCE_LIST,
            CONFIG_GROUP_SOURCE_LIST_NUM_SOURCE_BINS, &error)) {
      if (num_strings != config->total_num_sources) {
        NVGSTDS_ERR_MSG_V ("Mismatch in URIs provided and num-source-bins.");
        goto done;
      }
    } else {
      config->total_num_sources = num_strings;
    }
  }

  ret = TRUE;
done:
  if (error) {
    g_error_free (error);
  }
  if (keys) {
    g_strfreev (keys);
  }
  if (!ret) {
    NVGSTDS_ERR_MSG_V ("%s failed", __func__);
  }
  return ret;
}

static gboolean
set_source_all_configs (NvDsConfig * config, gchar * cfg_file_path)
{
  guint i = 0;
  for (i = 0; i < config->total_num_sources; i++) {
    config->multi_source_config[i] = global_source_config;
    config->multi_source_config[i].camera_id = i;
    if (config->uri_list) {
      char *uri = config->uri_list[i];
      if (g_str_has_prefix (config->uri_list[i], "file://")) {
        config->multi_source_config[i].type = NV_DS_SOURCE_URI;
        config->multi_source_config[i].uri = g_strdup (uri + 7);
        config->multi_source_config[i].uri =
          g_strdup_printf ("file://%s",
              get_absolute_file_path (cfg_file_path,
                config->multi_source_config[i].uri));
      } else if (g_str_has_prefix (config->uri_list[i], "rtsp://")) {
        config->multi_source_config[i].type = NV_DS_SOURCE_RTSP;
        config->multi_source_config[i].uri = config->uri_list[i];
      } else {
        gchar *source_id_start_ptr = uri + 4;
        gchar *source_id_end_ptr = NULL;
        long camera_id =
            g_ascii_strtoull (source_id_start_ptr, &source_id_end_ptr, 10);
        if (source_id_start_ptr == source_id_end_ptr
            || *source_id_end_ptr != '\0') {
          NVGSTDS_ERR_MSG_V
              ("Incorrect URI for camera source %s. FORMAT: <usb/csi>:<dev_node/sensor_id>",
              uri);
          return FALSE;
        }
        if (g_str_has_prefix (config->uri_list[i], "csi:")) {
          config->multi_source_config[i].type = NV_DS_SOURCE_CAMERA_CSI;
          config->multi_source_config[i].camera_csi_sensor_id = camera_id;
        } else if (g_str_has_prefix (config->uri_list[i], "usb:")) {
          config->multi_source_config[i].type = NV_DS_SOURCE_CAMERA_V4L2;
          config->multi_source_config[i].camera_v4l2_dev_node = camera_id;
        } else {
          NVGSTDS_ERR_MSG_V ("URI %d (%s) not in proper format.", i,
              config->uri_list[i]);
          return FALSE;
        }
      }
    }
  }
  return TRUE;
}

static gboolean
parse_tests (NvDsConfig *config, GKeyFile *key_file)
{
  gboolean ret = FALSE;
  gchar **keys = NULL;
  gchar **key = NULL;
  GError *error = NULL;

  keys = g_key_file_get_keys (key_file, CONFIG_GROUP_TESTS, NULL, &error);
  CHECK_ERROR (error);

  for (key = keys; *key; key++) {
    if (!g_strcmp0 (*key, CONFIG_GROUP_TESTS_FILE_LOOP)) {
      config->file_loop =
          g_key_file_get_integer (key_file, CONFIG_GROUP_TESTS,
          CONFIG_GROUP_TESTS_FILE_LOOP, &error);
      CHECK_ERROR (error);
    } else {
      NVGSTDS_WARN_MSG_V ("Unknown key '%s' for group [%s]", *key,
          CONFIG_GROUP_TESTS);
    }
  }

  ret = TRUE;
done:
  if (error) {
    g_error_free (error);
  }
  if (keys) {
    g_strfreev (keys);
  }
  if (!ret) {
    NVGSTDS_ERR_MSG_V ("%s failed", __func__);
  }
  return ret;
}

static gboolean
parse_app (NvDsConfig *config, GKeyFile *key_file, gchar *cfg_file_path)
{
  gboolean ret = FALSE;
  gchar **keys = NULL;
  gchar **key = NULL;
  GError *error = NULL;

  keys = g_key_file_get_keys (key_file, CONFIG_GROUP_APP, NULL, &error);
  CHECK_ERROR (error);

  for (key = keys; *key; key++) {
    if (!g_strcmp0 (*key, CONFIG_GROUP_APP_ENABLE_PERF_MEASUREMENT)) {
      config->enable_perf_measurement =
          g_key_file_get_integer (key_file, CONFIG_GROUP_APP,
          CONFIG_GROUP_APP_ENABLE_PERF_MEASUREMENT, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_APP_PERF_MEASUREMENT_INTERVAL)) {
      config->perf_measurement_interval_sec =
          g_key_file_get_integer (key_file, CONFIG_GROUP_APP,
          CONFIG_GROUP_APP_PERF_MEASUREMENT_INTERVAL, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_APP_GIE_OUTPUT_DIR)) {
      config->bbox_dir_path = get_absolute_file_path (cfg_file_path,
          g_key_file_get_string (key_file, CONFIG_GROUP_APP,
          CONFIG_GROUP_APP_GIE_OUTPUT_DIR, &error));
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_APP_GIE_TRACK_OUTPUT_DIR)) {
      config->kitti_track_dir_path = get_absolute_file_path (cfg_file_path,
          g_key_file_get_string (key_file, CONFIG_GROUP_APP,
          CONFIG_GROUP_APP_GIE_TRACK_OUTPUT_DIR, &error));
      CHECK_ERROR (error);
    } else {
      NVGSTDS_WARN_MSG_V ("Unknown key '%s' for group [%s]", *key,
                          CONFIG_GROUP_APP);
    }
  }

  ret = TRUE;
done:
  if (error) {
    g_error_free (error);
  }
  if (keys) {
    g_strfreev (keys);
  }
  if (!ret) {
    NVGSTDS_ERR_MSG_V ("%s failed", __func__);
  }
  return ret;
}


gboolean
parse_config_file (NvDsConfig *config, gchar *cfg_file_path)
{
  GKeyFile *cfg_file = g_key_file_new ();
  GError *error = NULL;
  gboolean ret = FALSE;
  gchar **groups = NULL;
  gchar **group;
  guint i, j;

  config->source_list_enabled = FALSE;

  if (!APP_CFG_PARSER_CAT) {
    GST_DEBUG_CATEGORY_INIT (APP_CFG_PARSER_CAT, "NVDS_CFG_PARSER", 0, NULL);
  }

  if (!g_key_file_load_from_file (cfg_file, cfg_file_path, G_KEY_FILE_NONE,
          &error)) {
    GST_CAT_ERROR (APP_CFG_PARSER_CAT, "Failed to load uri file: %s",
        error->message);
    goto done;
  }

  if (g_key_file_has_group (cfg_file, CONFIG_GROUP_SOURCE_LIST)) {
    if (!parse_source_list (config, cfg_file, cfg_file_path)) {
      GST_CAT_ERROR (APP_CFG_PARSER_CAT, "Failed to parse '%s' group",
          CONFIG_GROUP_SOURCE_LIST);
      goto done;
    }
    config->num_source_sub_bins = config->total_num_sources;
    config->source_list_enabled = TRUE;
    if (!g_key_file_has_group (cfg_file, CONFIG_GROUP_SOURCE_ALL)) {
      NVGSTDS_ERR_MSG_V ("[source-attr-all] group not present.");
      ret = FALSE;
      goto done;
    }
    g_key_file_remove_group (cfg_file, CONFIG_GROUP_SOURCE_LIST, &error);
  }
  if (g_key_file_has_group (cfg_file, CONFIG_GROUP_SOURCE_ALL)) {
    if (!parse_source (&global_source_config,
            cfg_file, CONFIG_GROUP_SOURCE_ALL, cfg_file_path)) {
      GST_CAT_ERROR (APP_CFG_PARSER_CAT, "Failed to parse '%s' group",
          CONFIG_GROUP_SOURCE_LIST);
      goto done;
    }
    if (!set_source_all_configs (config, cfg_file_path)) {
      ret = FALSE;
      goto done;
    }
    g_key_file_remove_group (cfg_file, CONFIG_GROUP_SOURCE_ALL, &error);
  }

  groups = g_key_file_get_groups (cfg_file, NULL);
  for (group = groups; *group; group++) {
    gboolean parse_err = FALSE;
    GST_CAT_DEBUG (APP_CFG_PARSER_CAT, "Parsing group: %s", *group);
    if (!g_strcmp0 (*group, CONFIG_GROUP_APP)) {
      parse_err = !parse_app (config, cfg_file, cfg_file_path);
    }

    if (!strncmp (*group, CONFIG_GROUP_SOURCE,
            sizeof (CONFIG_GROUP_SOURCE) - 1)) {
      if (config->num_source_sub_bins == MAX_SOURCE_BINS) {
        NVGSTDS_ERR_MSG_V ("App supports max %d sources", MAX_SOURCE_BINS);
        ret = FALSE;
        goto done;
      }
      gchar *source_id_start_ptr = *group + strlen (CONFIG_GROUP_SOURCE);
      gchar *source_id_end_ptr = NULL;
      guint index =
          g_ascii_strtoull (source_id_start_ptr, &source_id_end_ptr, 10);
      if (source_id_start_ptr == source_id_end_ptr
          || *source_id_end_ptr != '\0') {
        NVGSTDS_ERR_MSG_V
            ("Source group \"[%s]\" is not in the form \"[source<%%d>]\"",
            *group);
        ret = FALSE;
        goto done;
      }
      guint source_id = 0;
      if (config->source_list_enabled) {
        if (index >= config->total_num_sources) {
          NVGSTDS_ERR_MSG_V
              ("Invalid source group index %d, index cannot exceed %d", index,
              config->total_num_sources);
          ret = FALSE;
          goto done;
        }
        source_id = index;
        NVGSTDS_INFO_MSG_V ("Some parameters to be overwritten for group [%s]",
            *group);
      } else {
        source_id = config->num_source_sub_bins;
      }
      parse_err = !parse_source (&config->multi_source_config[source_id],
          cfg_file, *group, cfg_file_path);
      if (config->source_list_enabled
          && config->multi_source_config[source_id].type ==
          NV_DS_SOURCE_URI_MULTIPLE) {
        NVGSTDS_ERR_MSG_V
            ("MultiURI support not available if [source-list] is provided");
        ret = FALSE;
        goto done;
      }
      if (config->multi_source_config[source_id].enable
          && !config->source_list_enabled) {
        config->num_source_sub_bins++;
      }
    }

    if (!g_strcmp0 (*group, CONFIG_GROUP_STREAMMUX)) {
      parse_err = !parse_streammux (&config->streammux_config, cfg_file, cfg_file_path);
    }

    if (!g_strcmp0 (*group, CONFIG_GROUP_OSD)) {
      parse_err = !parse_osd (&config->osd_config, cfg_file);
    }

    if (!g_strcmp0 (*group, CONFIG_GROUP_PRIMARY_GIE)) {
      parse_err =
          !parse_gie (&config->primary_gie_config, cfg_file,
          CONFIG_GROUP_PRIMARY_GIE, cfg_file_path);
    }

    if (!g_strcmp0 (*group, CONFIG_GROUP_TRACKER)) {
      parse_err = !parse_tracker (&config->tracker_config, cfg_file, cfg_file_path);
    }

    if (!strncmp (*group, CONFIG_GROUP_SECONDARY_GIE,
                  sizeof (CONFIG_GROUP_SECONDARY_GIE) - 1)) {
      if (config->num_secondary_gie_sub_bins == MAX_SECONDARY_GIE_BINS) {
        NVGSTDS_ERR_MSG_V ("App supports max %d secondary GIEs", MAX_SECONDARY_GIE_BINS);
        ret = FALSE;
        goto done;
      }
      parse_err =
          !parse_gie (&config->secondary_gie_sub_bin_config[config->
                                  num_secondary_gie_sub_bins],
                                  cfg_file, *group, cfg_file_path);
      if (config->secondary_gie_sub_bin_config[config->num_secondary_gie_sub_bins].enable){
        config->num_secondary_gie_sub_bins++;
      }
    }

    if (!strncmp (*group, CONFIG_GROUP_SINK, sizeof (CONFIG_GROUP_SINK) - 1)) {
      if (config->num_sink_sub_bins == MAX_SINK_BINS) {
        NVGSTDS_ERR_MSG_V ("App supports max %d sinks", MAX_SINK_BINS);
        ret = FALSE;
        goto done;
      }
      parse_err =
          !parse_sink (&config->
          sink_bin_sub_bin_config[config->num_sink_sub_bins], cfg_file, *group,
          cfg_file_path);
      if (config->
          sink_bin_sub_bin_config[config->num_sink_sub_bins].enable){
        config->num_sink_sub_bins++;
      }
    }

    if (!strncmp (*group, CONFIG_GROUP_MSG_CONSUMER,
        sizeof (CONFIG_GROUP_MSG_CONSUMER) - 1)) {
      if (config->num_message_consumers == MAX_MESSAGE_CONSUMERS) {
        NVGSTDS_ERR_MSG_V ("App supports max %d consumers", MAX_MESSAGE_CONSUMERS);
        ret = FALSE;
        goto done;
      }
      parse_err = !parse_msgconsumer (
                    &config->message_consumer_config[config->num_message_consumers],
                    cfg_file, *group, cfg_file_path);

      if (config->message_consumer_config[config->num_message_consumers].enable) {
        config->num_message_consumers++;
      }
    }

    if (!g_strcmp0 (*group, CONFIG_GROUP_TILED_DISPLAY)) {
      parse_err = !parse_tiled_display (&config->tiled_display_config, cfg_file);
    }

    if (!g_strcmp0 (*group, CONFIG_GROUP_IMG_SAVE)) {
      parse_err = !parse_image_save (&config->image_save_config , cfg_file, *group, cfg_file_path);
    }

    if (!g_strcmp0 (*group, CONFIG_GROUP_DSANALYTICS)) {
      parse_err = !parse_dsanalytics (&config->dsanalytics_config, cfg_file, cfg_file_path);
    }

    if (!g_strcmp0 (*group, CONFIG_GROUP_DSEXAMPLE)) {
      parse_err = !parse_dsexample (&config->dsexample_config, cfg_file);
    }

    if (!g_strcmp0 (*group, CONFIG_GROUP_MSG_CONVERTER)) {
      parse_err = !parse_msgconv (&config->msg_conv_config, cfg_file, *group, cfg_file_path);
    }

    if (!g_strcmp0 (*group, CONFIG_GROUP_TESTS)) {
      parse_err = !parse_tests (config, cfg_file);
    }

    if (parse_err) {
      GST_CAT_ERROR (APP_CFG_PARSER_CAT, "Failed to parse '%s' group", *group);
      goto done;
    }
  }

  /* Updating batch size when source list is enabled */
  if (config->source_list_enabled == TRUE) {
      /* For streammux and pgie, batch size is set to number of sources */
      config->streammux_config.batch_size = config->num_source_sub_bins;
      config->primary_gie_config.batch_size = config->num_source_sub_bins;
      if (config->sgie_batch_size != 0) {
          for (i = 0; i < config->num_secondary_gie_sub_bins; i++) {
              config->secondary_gie_sub_bin_config[i].batch_size = config->sgie_batch_size;
          }
      }
  }

  for (i = 0; i < config->num_secondary_gie_sub_bins; i++) {
    if (config->secondary_gie_sub_bin_config[i].unique_id ==
        config->primary_gie_config.unique_id) {
      NVGSTDS_ERR_MSG_V ("Non unique gie ids found");
      ret = FALSE;
      goto done;
    }
  }

  for (i = 0; i < config->num_secondary_gie_sub_bins; i++) {
    for (j = i + 1; j < config->num_secondary_gie_sub_bins; j++) {
      if (config->secondary_gie_sub_bin_config[i].unique_id ==
          config->secondary_gie_sub_bin_config[j].unique_id) {
        NVGSTDS_ERR_MSG_V ("Non unique gie id %d found",
                            config->secondary_gie_sub_bin_config[i].unique_id);
        ret = FALSE;
        goto done;
      }
    }
  }

  for (i = 0; i < config->num_source_sub_bins; i++) {
    if (config->multi_source_config[i].type == NV_DS_SOURCE_URI_MULTIPLE) {
      if (config->multi_source_config[i].num_sources < 1) {
        config->multi_source_config[i].num_sources = 1;
      }
      for (j = 1; j < config->multi_source_config[i].num_sources; j++) {
        if (config->num_source_sub_bins == MAX_SOURCE_BINS) {
          NVGSTDS_ERR_MSG_V ("App supports max %d sources", MAX_SOURCE_BINS);
          ret = FALSE;
          goto done;
        }
        memcpy (&config->multi_source_config[config->num_source_sub_bins],
            &config->multi_source_config[i],
            sizeof (config->multi_source_config[i]));
        config->multi_source_config[config->num_source_sub_bins].type =
            NV_DS_SOURCE_URI;
        config->multi_source_config[config->num_source_sub_bins].uri =
            g_strdup_printf (config->multi_source_config[config->
                num_source_sub_bins].uri, j);
        config->num_source_sub_bins++;
      }
      config->multi_source_config[i].type = NV_DS_SOURCE_URI;
      config->multi_source_config[i].uri =
          g_strdup_printf (config->multi_source_config[i].uri, 0);
    }
  }
  ret = TRUE;

done:
  if (cfg_file) {
    g_key_file_free (cfg_file);
  }

  if (groups) {
    g_strfreev (groups);
  }

  if (error) {
    g_error_free (error);
  }
  if (!ret) {
    NVGSTDS_ERR_MSG_V ("%s failed", __func__);
  }
  return ret;
}

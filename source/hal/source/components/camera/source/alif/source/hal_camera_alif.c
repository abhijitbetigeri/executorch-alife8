/*
 * SPDX-FileCopyrightText: Copyright 2024, 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "hal_camera.h"
#include "hal_log.h"
#include "RTE_Components.h"
#include "RTE_Device.h"
#include "image_processing.h"
#include "camera.h"
#include "board_defs.h"
#include "board_utils.h"
#include "base_def.h"
#include "timer_alif.h"
#include "delay.h"
#include <tgmath.h>
#include <string.h>

#define CAMERA_PREVIEW_WIDTH  RTE_PANEL_HACTIVE_TIME
#define CAMERA_PREVIEW_HEIGHT RTE_PANEL_VACTIVE_LINE
/* Full-panel preview uses LVGL RGB888 (B,G,R byte order). */
static uint8_t preview_image[CAMERA_PREVIEW_WIDTH * CAMERA_PREVIEW_HEIGHT * RGB_BYTES]
    __attribute__((aligned(32), section(".bss.large_ram")));


#if !defined(RTE_Drivers_CPI)
#define USE_FAKE_CAMERA 1
#endif

/* Camera fills the raw_image buffer.
 * Bayer->RGB conversion transfers into the rgb_image buffer.
 * With MT9M114 camera this can be a RGB565 to RGB conversion.
 * Following steps (crop, interpolate, colour correct) all occur in the rgb_image buffer in-place.
 */
static struct {
	tiff_header_t tiff_header;
	uint8_t image_data[CIMAGE_RGB_WIDTH_MAX * CIMAGE_RGB_HEIGHT_MAX * RGB_BYTES];
} rgb_image __attribute__((section(".bss.camera_frame_bayer_to_rgb_buf")));

#if RTE_ISP
// Camera capture buffer is not applicable for ISP mode
static uint8_t *raw_image = 0;
#else
/* Non-ISP SW pipeline buffers */
#if defined(CIMAGE_X_ORIG) && defined(CIMAGE_Y_ORIG)
// Save RAM by cropping raw camera image in place in bayer color space
static uint8_t raw_image[CIMAGE_X_ORIG * CIMAGE_Y_ORIG]
    __attribute__((aligned(32),section(".bss.camera_frame_buf")));
#else
static uint8_t raw_image[CIMAGE_X * CIMAGE_Y + CIMAGE_USE_RGB565 * CIMAGE_X * CIMAGE_Y]
    __attribute__((aligned(32),section(".bss.camera_frame_buf")));
#endif
#endif

typedef struct hal_camera_device_ {
    char name[32];
    uint32_t frame_width;
    uint32_t frame_height;
    uint32_t bytes_per_frame;
    uint32_t output_buffer_size;
    uint8_t *output_buffer;
    hal_cam_clr_format format;
    hal_cam_mode mode;
    hal_cam_status status;
} hal_cam_dev;

/**< Static device used in this file. */
static hal_cam_dev s_cam_dev;

/**
 * @brief   Resets the camera device.
 */
static void hal_camera_reset(void)
{
    s_cam_dev.status = HAL_CAMERA_STATUS_INVALID;
    strncpy(s_cam_dev.name, "Alif camera", sizeof(s_cam_dev.name));
    s_cam_dev.frame_width = 0;
    s_cam_dev.frame_height = 0;
    s_cam_dev.bytes_per_frame = 0;
    s_cam_dev.output_buffer = NULL;
    s_cam_dev.format = HAL_CAMERA_COLOUR_FORMAT_INVALID;
    s_cam_dev.mode = HAL_CAMERA_MODE_INVALID;
#ifndef USE_FAKE_CAMERA
    camera_uninit();
#endif
}

bool hal_camera_init(void)
{
    hal_camera_reset();
    info("Initialising camera interface: %s\n", s_cam_dev.name);
#ifndef USE_FAKE_CAMERA
    int32_t err = 0;
    err = camera_init(raw_image);
	if (err != 0) {
        printf_err("Failed to initialise camera driver: %ld\n", err);
		while(1) {
		    BOARD_LED1_Control(BOARD_LED_STATE_LOW);
			sleep_or_wait_msec(300);
            BOARD_LED1_Control(BOARD_LED_STATE_HIGH);
			sleep_or_wait_msec(300);
		}
	}
	DEBUG_PRINTF("Camera initialized... \n");
    BOARD_LED1_Control(BOARD_LED_STATE_LOW);
#endif
    return true;
}

bool hal_camera_configure(const uint32_t width,
                          const uint32_t height,
                          const hal_cam_mode mode,
                          const hal_cam_clr_format colour_format)
{

    if (HAL_CAMERA_STATUS_RUNNING == s_cam_dev.status) {
        printf_err("Camera is running; configuration failed\n");
        return false;
    }

    if (HAL_CAMERA_MODE_CONTINUOUS == mode) {
        printf_err("Only single shot mode is supported\n");
        return false;
    }

    s_cam_dev.frame_width = width;
    s_cam_dev.frame_height = height;
    s_cam_dev.mode = mode;
    s_cam_dev.format = colour_format;
    s_cam_dev.status = HAL_CAMERA_STATUS_STOPPED;

    switch (s_cam_dev.format) {
        case HAL_CAMERA_COLOUR_FORMAT_RGB888:
            s_cam_dev.bytes_per_frame = width * height * 3;
            break;
        default:
            printf_err("Unsupported colour format\n");
            hal_camera_reset();
            return false;
    }

    if (!hal_camera_set_buffer(rgb_image.image_data, sizeof rgb_image.image_data )) {
        return false;
    }

#ifndef USE_FAKE_CAMERA
    int32_t err = camera_configure(width, height);
    if (err != 0) {
        printf_err("Failed to configure camera driver: %ld\n", err);
        return false;
    }
#endif

    info("Camera configured\n");
    return true;
}

bool hal_camera_set_buffer(uint8_t* buffer, const uint32_t size)
{
    if (size < s_cam_dev.bytes_per_frame) {
        printf_err("Buffer size is too small\n");
        return false;
    }

    s_cam_dev.output_buffer = buffer;
    s_cam_dev.output_buffer_size = size;

    return true;
}

bool hal_camera_start(void)
{
    if (s_cam_dev.status != HAL_CAMERA_STATUS_STOPPED) {
        debug("Camera is already running\n");
        return false;
    }

#ifndef USE_FAKE_CAMERA
#if !RTE_ISP
    /* The camera writes this buffer directly; discard any stale cached lines
     * before handing it to the DMA engine. The buffer is 32-byte aligned and
     * its configured size is a cache-line multiple. */
    SCB_InvalidateDCache_by_Addr((uint32_t*)raw_image, sizeof raw_image);
#endif
    camera_start(CAMERA_MODE_SNAPSHOT);
#endif
    s_cam_dev.status = HAL_CAMERA_STATUS_RUNNING;
    return true;
}

/**
 * @brief   Waits for the input frame capture to complete.
 * @return  Status of the device as `hal_cam_status`.
 */
static hal_cam_status wait_for_capture(void)
{
#ifndef USE_FAKE_CAMERA
    /* Wait for video input frame */
    while ( hal_camera_get_status() != HAL_CAMERA_STATUS_STOPPED) {
        __WFE();
    }

    if (camera_process_frame_end() != ARM_DRIVER_OK) {
        s_cam_dev.status = HAL_CAMERA_STATUS_ERROR;
    }
#else
    s_cam_dev.status = HAL_CAMERA_STATUS_STOPPED;
#endif

    return s_cam_dev.status;
}

const uint8_t* hal_camera_get_captured_frame(uint32_t* size)
{
    const uint8_t* buffer = NULL;
    *size = 0;
    if (s_cam_dev.status == HAL_CAMERA_STATUS_RUNNING) {
        const hal_cam_status status = wait_for_capture();
        if (HAL_CAMERA_STATUS_ERROR == status) {
            printf_err("Error reported\n");
            return buffer;
        }
#if !RTE_ISP
        /* Match the camera tutorial: invalidate the DMA-written frame after
         * capture, without writing stale cache lines back over it. */
        SCB_InvalidateDCache_by_Addr((uint32_t*)raw_image, sizeof raw_image);
#else
        SCB_CleanInvalidateDCache();
#endif
#if CIMAGE_USE_RGB565
        /* Correct the sensor mounting for both preview and model input. */
        uint16_t* upright_pixels   = (uint16_t*)raw_image;
        const uint32_t pixel_count = CIMAGE_X * CIMAGE_Y;
        for (uint32_t i = 0; i < pixel_count / 2; ++i) {
            const uint16_t pixel                = upright_pixels[i];
            upright_pixels[i]                   = upright_pixels[pixel_count - 1 - i];
            upright_pixels[pixel_count - 1 - i] = pixel;
        }
        /* Center-crop to fill the panel without stretching. Read the original
         * sensor stride directly, before model preprocessing crops it in place.
         * Reverse x to correct the camera's mirrored view. */
        const uint32_t crop_width =
            CIMAGE_X * CAMERA_PREVIEW_HEIGHT > CIMAGE_Y * CAMERA_PREVIEW_WIDTH
                ? CIMAGE_Y * CAMERA_PREVIEW_WIDTH / CAMERA_PREVIEW_HEIGHT
                : CIMAGE_X;
        const uint32_t crop_height =
            CIMAGE_X * CAMERA_PREVIEW_HEIGHT > CIMAGE_Y * CAMERA_PREVIEW_WIDTH
                ? CIMAGE_Y
                : CIMAGE_X * CAMERA_PREVIEW_HEIGHT / CAMERA_PREVIEW_WIDTH;
        const uint32_t x_origin       = (CIMAGE_X - crop_width) / 2;
        const uint32_t y_origin       = (CIMAGE_Y - crop_height) / 2;
        const uint16_t* sensor_pixels = (const uint16_t*)raw_image;
        for (uint32_t y = 0; y < CAMERA_PREVIEW_HEIGHT; ++y) {
            const uint32_t sy = y_origin + y * crop_height / CAMERA_PREVIEW_HEIGHT;
            uint8_t* dst      = preview_image + y * CAMERA_PREVIEW_WIDTH * RGB_BYTES;
            for (uint32_t x = 0; x < CAMERA_PREVIEW_WIDTH; ++x) {
                const uint32_t sx =
                    x_origin + crop_width - 1 - x * crop_width / CAMERA_PREVIEW_WIDTH;
                const uint16_t pixel = sensor_pixels[sy * CIMAGE_X + sx];
                const uint8_t blue   = pixel & 0x1f;
                const uint8_t green  = (pixel >> 5) & 0x3f;
                const uint8_t red    = (pixel >> 11) & 0x1f;
                *dst++               = (blue << 3) | (blue >> 2);
                *dst++               = (green << 2) | (green >> 4);
                *dst++               = (red << 3) | (red >> 2);
            }
        }
        /* Keep the full-panel source coherent for the display/DAVE2D reader. */
        SCB_CleanDCache_by_Addr((uint32_t*)preview_image, sizeof preview_image);
#endif
        /* Preprocessing crops raw_image in place; generate the preview first. */
        buffer = get_image_data(s_cam_dev.frame_width,
                                s_cam_dev.frame_height,
                                rgb_image.tiff_header,
                                s_cam_dev.output_buffer,
                                s_cam_dev.output_buffer_size,
                                raw_image);
        if (buffer != NULL) {
            *size = s_cam_dev.bytes_per_frame;
        }
    }
    return buffer;
}

const uint8_t *hal_camera_get_preview_frame(void)
{
    return preview_image;
}

bool hal_camera_stop(void)
{
    if (s_cam_dev.status == HAL_CAMERA_STATUS_RUNNING) {
        s_cam_dev.status = HAL_CAMERA_STATUS_STOPPED;
    }

    return true;
}

hal_cam_status hal_camera_get_status(void)
{
#ifndef USE_FAKE_CAMERA
    if (camera_image_ready()) {
        s_cam_dev.status = HAL_CAMERA_STATUS_STOPPED;
    }
#else
    s_cam_dev.status = HAL_CAMERA_STATUS_STOPPED;
#endif
    return s_cam_dev.status;
}

void hal_camera_release(void)
{
    hal_camera_reset();
}

const char* hal_camera_get_device_name(void)
{
    return s_cam_dev.name;
}

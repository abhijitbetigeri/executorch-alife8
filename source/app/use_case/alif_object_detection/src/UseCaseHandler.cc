/* This file was ported to work on Alif Semiconductor devices. */

/* Copyright (C) 2023, 2026 Alif Semiconductor - All Rights Reserved.
 * Use, distribution and modification of this code is permitted under the
 * terms stated in the Alif Semiconductor Software License Agreement
 *
 * You should have received a copy of the Alif Semiconductor Software
 * License Agreement with this file. If not, please write to:
 * contact@alifsemi.com, or visit: https://alifsemi.com/license
 *
 */

/*
 * Copyright (c) 2022, 2026 Arm Limited. All rights reserved.
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
#include "UseCaseHandler.hpp"
#include "UseCaseCommonUtils.hpp"
#if defined(MLEK_FWK_EXECUTORCH)
#include "mlek/use_case/object_detection/BlazeFaceProcessing.hpp"
#include "mlek/use_case/object_detection/DetectionResult.hpp"
#else
#include "mlek/use_case/object_detection/DetectorPostProcessing.hpp"
#include "mlek/use_case/object_detection/DetectorPreProcessing.hpp"
#endif
#include "mlek/log/log_macros.h"
#if defined(GPIO_PROFILING)
#include "board_utils.h"
#endif
#include "ScreenLayout.hpp"
#include "hal.h"

#ifdef POWER_MEASUREMENT_SUPPORT
#include "power_measurement.h"
#endif

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstring>

#include "RTE_Device.h"
#include "image_processing.h"
#include "lv_paint_utils.h"
#include "lv_port.h"
#include "lvgl.h"

#define PREVIEW_X RTE_PANEL_HACTIVE_TIME
#define PREVIEW_Y RTE_PANEL_VACTIVE_LINE

namespace {
lv_style_t boxStyle;
bool firstCameraFrameReported = false;
#if defined(MLEK_FWK_EXECUTORCH)
/* Per-stage cycle totals, printed to the console every kTimingReportFrames frames. */
constexpr uint32_t kTimingReportFrames = 30;
uint32_t timingFrames = 0;
uint64_t preCycles = 0, infCycles = 0, postCycles = 0, frameCycles = 0;
uint32_t lastFrameStart = 0;
#endif
};

using arm::app::Profiler;
using arm::app::ApplicationContext;
using arm::app::fwk::iface::Model;
#if defined(MLEK_FWK_EXECUTORCH)
using arm::app::BlazeFacePreProcess;
using arm::app::BlazeFacePostProcess;
#else
using arm::app::DetectorPreProcess;
using arm::app::DetectorPostProcess;
#endif

namespace alif {
namespace app {

namespace object_detection {
using namespace arm::app::object_detection;
}

#if defined(MLEK_FWK_EXECUTORCH)
} /* namespace app */
} /* namespace alif */
namespace arm { namespace app { namespace object_detection {
    extern const float blazeFaceScoreThreshold; /* generated from usecase.cmake */
} } }
namespace alif {
namespace app {
#endif

#ifdef POWER_MEASUREMENT_SUPPORT
    /* Labels showing average power and current, in that order. */
    static constexpr uint32_t powerLabelIdx = 2;

    /* Shows the averages of the last measured inference, or "--" if the measurement failed
     * (result == nullptr). Does nothing if no sensor is fitted. Call with the LVGL lock held. */
    static void ShowPowerMeasurement(const power_measurement_result_t* result)
    {
        if (!power_measurement_is_available()) {
            return;
        }
        if (result) {
            lv_label_set_text_fmt(ScreenLayoutLabelObject(powerLabelIdx), "Average Power:  %4" PRId32 " mW", result->avg_mW);
            lv_label_set_text_fmt(ScreenLayoutLabelObject(powerLabelIdx + 1), "Average Current: %4" PRId32 " mA", result->avg_mA);
        } else {
            lv_label_set_text_static(ScreenLayoutLabelObject(powerLabelIdx), "Average Power:     -- mW");
            lv_label_set_text_static(ScreenLayoutLabelObject(powerLabelIdx + 1), "Average Current:   -- mA");
        }
    }
#endif

    bool ObjectDetectionInit(FaceModel& model)
    {

        ScreenLayoutInit(hal_camera_get_preview_frame(),
                         PREVIEW_X * PREVIEW_Y * 3,
                         PREVIEW_X,
                         PREVIEW_Y,
                         256,
                         false,
                         true);
        uint32_t lv_lock_state = lv_port_lock();

        lv_style_init(&boxStyle);
        lv_style_set_bg_opa(&boxStyle, LV_OPA_TRANSP);
        lv_style_set_pad_all(&boxStyle, 0);
        lv_style_set_border_width(&boxStyle, 0);
        lv_style_set_outline_width(&boxStyle, 2);
        lv_style_set_outline_pad(&boxStyle, 0);
        lv_style_set_outline_color(&boxStyle, lv_color_hex(0x00FF00));
        lv_style_set_radius(&boxStyle, 0);
        lv_port_unlock(lv_lock_state);

        /* Initialise the camera */
        if (!hal_camera_init()) {
            printf_err("hal_camera_init failed!\n");
            return false;
        }

        auto inputTensor = model.GetInputTensor(0);
        const auto inputShape = inputTensor->Shape();
        const uint32_t inputImgCols = inputShape[FaceModel::ms_inputColsIdx];
        const uint32_t inputImgRows = inputShape[FaceModel::ms_inputRowsIdx];
        info("Face model input: %" PRIu32 "x%" PRIu32 "\n", inputImgCols, inputImgRows);

        auto bCamera = hal_camera_configure(inputImgCols, inputImgRows, HAL_CAMERA_MODE_SINGLE_FRAME, HAL_CAMERA_COLOUR_FORMAT_RGB888);
        if (!bCamera) {
            printf_err("Failed to configure camera.\n");
            return false;
        }

        return true;
    }


    /**
     * @brief           Presents inference results along using the data presentation
     *                  object.
     * @param[in]       results            Vector of detection results to be displayed.
     * @return          true if successful, false otherwise.
     **/
    static bool PresentInferenceResult(const std::vector<object_detection::DetectionResult>& results);

    /**
     * @brief           Draw boxes directly on the LCD for all detected objects.
     * @param[in]       results            Vector of detection results to be displayed.
     **/
    static void DrawDetectionBoxes(
           const std::vector<object_detection::DetectionResult>& results,
           int imgInputCols, int imgInputRows);

    /* Object detection inference handler. */
    bool ObjectDetectionHandler(ApplicationContext& ctx)
    {
        auto& profiler = ctx.Get<Profiler&>("profiler");
        auto& model = ctx.Get<Model&>("model");

        if (!model.IsInited()) {
            printf_err("Model is not initialised! Terminating processing.\n");
            return false;
        }

#if defined(MLEK_FWK_EXECUTORCH)
        const uint32_t frameStart = Get_SysTick_Cycle_Count32();
#endif
        auto inputTensor = model.GetInputTensor(0);
        auto outputTensor0 = model.GetOutputTensor(0);
        auto outputTensor1 = model.GetOutputTensor(1);
        const auto inputShape = inputTensor->Shape();
        if (inputShape.empty()) {
            printf_err("Invalid input tensor dims\n");
            return false;
        } else if (inputShape.size() < 3) {
            printf_err("Input tensor dimension should be >= 3\n");
            return false;
        }

        const int inputImgCols = inputShape[FaceModel::ms_inputColsIdx];
        const int inputImgRows = inputShape[FaceModel::ms_inputRowsIdx];

        /* Set up pre and post-processing. */
        std::vector<object_detection::DetectionResult> results;
#if defined(MLEK_FWK_EXECUTORCH)
        /* BlazeFace: float RGB in, (scores, boxes) out; decode + weighted NMS on the M55. */
        BlazeFacePreProcess preProcess(inputTensor);
        arm::app::BlazeFaceParams blazeParams;
        blazeParams.inputRows      = inputImgRows;
        blazeParams.inputCols      = inputImgCols;
        blazeParams.scoreThreshold = object_detection::blazeFaceScoreThreshold;
        BlazeFacePostProcess postProcess(outputTensor0, outputTensor1, results, blazeParams);
#else
        DetectorPreProcess preProcess = DetectorPreProcess(inputTensor, true, model.IsDataSigned());

        const object_detection::PostProcessParams postProcessParams {
            inputImgRows, inputImgCols, object_detection::originalImageSize,
            object_detection::anchor1, object_detection::anchor2
        };
        DetectorPostProcess postProcess = DetectorPostProcess(outputTensor0, outputTensor1,
                results, postProcessParams);
#endif

        /* Ensure there are no results leftover from previous inference when running all. */
        results.clear();

        hal_camera_start();

        uint32_t capturedFrameSize = 0;
        const uint8_t* currImage = hal_camera_get_captured_frame(&capturedFrameSize);
        if (!currImage || !capturedFrameSize) {
            printf_err("hal_camera_get_captured_frame failed");
            return false;
        }
        if (!firstCameraFrameReported) {
            uint8_t minPixel = 255;
            uint8_t maxPixel = 0;
            uint32_t nonZeroBytes = 0;
            for (uint32_t i = 0; i < capturedFrameSize; ++i) {
                minPixel = std::min(minPixel, currImage[i]);
                maxPixel = std::max(maxPixel, currImage[i]);
                nonZeroBytes += (currImage[i] != 0);
            }
            info("Camera frame: %" PRIu32 " bytes, min=%u max=%u nonzero=%" PRIu32 "\n",
                 capturedFrameSize, minPixel, maxPixel, nonZeroBytes);
            firstCameraFrameReported = true;
        }

        {
            ScopedLVGLLock lv_lock;

            /* Camera preview buffer is full-panel RGB888 in LVGL byte order. */
            lv_obj_invalidate(ScreenLayoutImageObject());

            const size_t copySz = inputTensor->Bytes();

#if SHOW_INF_TIME
        uint32_t inf_prof = Get_SysTick_Cycle_Count32();
#endif

#if defined(GPIO_PROFILING)
            uint32_t lv_lock_state = lv_port_lock();
            BOARD_LED1_BLUE_Control(BOARD_LED_STATE_TOGGLE);
#endif
            /* Run the pre-processing, inference and post-processing. */
#if defined(MLEK_FWK_EXECUTORCH)
            const uint32_t t0 = Get_SysTick_Cycle_Count32();
            /* The camera HAL returns RGB888 at the model resolution: one byte per element. */
            if (!preProcess.DoPreProcess(currImage, capturedFrameSize)) {
                printf_err("Pre-processing failed.");
                return false;
            }
            const uint32_t t1 = Get_SysTick_Cycle_Count32();
            (void)copySz;
#else
            if (!preProcess.DoPreProcess(currImage, copySz)) {
                printf_err("Pre-processing failed.");
                return false;
            }
#endif

            /* Run inference over this image. */
#ifdef POWER_MEASUREMENT_SUPPORT
            /* Measure average current/power over the inference only. The LVGL lock is already
             * held here, so no UI work is interleaved with the window. */
            power_measurement_result_t power;
            bool powerOk = (power_measurement_begin() == 0);
#endif

            if (!RunInference(model, profiler)) {
                printf_err("Inference failed.");
                return false;
            }

#ifdef POWER_MEASUREMENT_SUPPORT
            powerOk = powerOk && (power_measurement_end(&power) == 0);
#endif

#if defined(MLEK_FWK_EXECUTORCH)
            const uint32_t t2 = Get_SysTick_Cycle_Count32();
#endif
            if (!postProcess.DoPostProcess()) {
                printf_err("Post-processing failed.");
                return false;
            }
#if defined(MLEK_FWK_EXECUTORCH)
            const uint32_t t3 = Get_SysTick_Cycle_Count32();
            preCycles  += t1 - t0;
            infCycles  += t2 - t1;
            postCycles += t3 - t2;
            if (lastFrameStart != 0) {
                frameCycles += frameStart - lastFrameStart;
            }
            lastFrameStart = frameStart;
            if (++timingFrames == kTimingReportFrames) {
                const double msPerCycle = 1000.0 / SystemCoreClock;
                const double n = kTimingReportFrames;
                info("BlazeFace avg over %" PRIu32 " frames: pre %.2f ms, NPU inference %.2f ms, "
                     "post %.2f ms, frame-to-frame %.2f ms (%.1f FPS)\n",
                     kTimingReportFrames, preCycles * msPerCycle / n, infCycles * msPerCycle / n,
                     postCycles * msPerCycle / n, frameCycles * msPerCycle / (n - 1),
                     (n - 1) * SystemCoreClock / (double)(frameCycles ? frameCycles : 1));
                timingFrames = 0;
                preCycles = infCycles = postCycles = frameCycles = 0;
            }
#endif

#if defined(GPIO_PROFILING)
            BOARD_LED1_BLUE_Control(BOARD_LED_STATE_TOGGLE);
            lv_port_unlock(lv_lock_state);
#endif

#if SHOW_INF_TIME
            inf_prof = Get_SysTick_Cycle_Count32() - inf_prof;
            lv_label_set_text_fmt(ScreenLayoutLabelObject(2), "Inference time: %.3f ms", (double)inf_prof / SystemCoreClock * 1000);
            lv_label_set_text_fmt(ScreenLayoutLabelObject(3), "Inferences / sec: %.2f", (double) SystemCoreClock / inf_prof);
            //lv_label_set_text_fmt(ScreenLayoutLabelObject(3), "Inferences / second: %.2f", (double) SystemCoreClock / (inf_loop_time_end - inf_loop_time_start));
#endif

#ifdef POWER_MEASUREMENT_SUPPORT
            (void)powerOk;
#endif

            /* Draw boxes. */
            DrawDetectionBoxes(results, inputImgCols, inputImgRows);

        } // ScopedLVGLLock

#if VERIFY_TEST_OUTPUT
        DumpTensor(modelOutput0);
        DumpTensor(modelOutput1);
#endif /* VERIFY_TEST_OUTPUT */

        if (!PresentInferenceResult(results)) {
            return false;
        }

        profiler.PrintProfilingResult();

        return true;
    }

    static bool PresentInferenceResult(const std::vector<object_detection::DetectionResult>& results)
    {
        /* If profiling is enabled, and the time is valid. */
        info("Final results:\n");
        info("Total number of inferences: 1\n");

        for (uint32_t i = 0; i < results.size(); ++i) {
            info("%" PRIu32 ") (%f) -> %s {x=%d,y=%d,w=%d,h=%d}\n", i,
                results[i].m_normalisedVal, "Detection box:",
                results[i].m_x0, results[i].m_y0, results[i].m_w, results[i].m_h );
        }

        return true;
    }

    static void DeleteBoxes(lv_obj_t *frame)
    {
        // Assume that child 0 of the frame is the image itself
        int children = lv_obj_get_child_count(frame);
        while (children > 1) {
            lv_obj_del(lv_obj_get_child(frame, 1));
            children--;
        }
    }

    static void CreateBox(lv_obj_t *frame, int x0, int y0, int w, int h)
    {
        lv_obj_t *box = lv_obj_create(frame);
        lv_obj_set_size(box, w, h);
        lv_obj_add_style(box, &boxStyle, LV_PART_MAIN);
        lv_obj_set_pos(box, x0, y0);
    }

    static void DrawDetectionBoxes(const std::vector<object_detection::DetectionResult>& results,
                                   int imgInputCols, int imgInputRows)
    {
        lv_obj_t *frame = ScreenLayoutImageHolderObject();
        const float holderWidth   = (float)lv_obj_get_content_width(frame);
        const float holderHeight  = (float)lv_obj_get_content_height(frame);
        const float modelWidth    = (float)CIMAGE_Y * imgInputCols / imgInputRows;
        const float modelHeight   = (float)CIMAGE_Y;
        const float previewWidth  = CIMAGE_X * PREVIEW_Y > CIMAGE_Y * PREVIEW_X
                                        ? (float)(CIMAGE_Y * PREVIEW_X / PREVIEW_Y)
                                        : (float)CIMAGE_X;
        const float previewHeight = CIMAGE_X * PREVIEW_Y > CIMAGE_Y * PREVIEW_X
                                        ? (float)CIMAGE_Y
                                        : (float)(CIMAGE_X * PREVIEW_Y / PREVIEW_X);
        const float xScale        = holderWidth * modelWidth / (previewWidth * imgInputCols);
        const float yScale        = holderHeight * modelHeight / (previewHeight * imgInputRows);
        const float xOffset       = (previewWidth - modelWidth) * holderWidth / (2 * previewWidth);
        const float yOffset = (previewHeight - modelHeight) * holderHeight / (2 * previewHeight);

        DeleteBoxes(frame);

        for (const auto& result: results) {
#ifdef ALIF_CAMERA_APPKIT
            const float left = xOffset + result.m_x0 * xScale;
#else
            const float left = holderWidth - xOffset - (result.m_x0 + result.m_w) * xScale;
#endif
            CreateBox(frame,
                      floor(left),
                      floor(yOffset + result.m_y0 * yScale),
                      ceil(result.m_w * xScale),
                      ceil(result.m_h * yScale));
        }
    }

} /* namespace app */
} /* namespace alif */

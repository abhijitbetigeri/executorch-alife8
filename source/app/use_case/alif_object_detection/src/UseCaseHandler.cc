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
#if defined(MLEK_FWK_EXECUTORCH) && ET_TASK_CLASSIFY
#ifndef CLS_EMA_ALPHA
#define CLS_EMA_ALPHA 0.3f
#endif
#ifndef CLS_CONF_THRESHOLD
#define CLS_CONF_THRESHOLD 0.5f
#endif
/* Decisions use an exponential moving average of the softmax over frames:
 * smoothed = alpha * current + (1 - alpha) * smoothed. */
constexpr float kEmaAlpha       = CLS_EMA_ALPHA;
constexpr float kConfThreshold  = CLS_CONF_THRESHOLD;
constexpr float kConfHysteresis = 0.05f; /* a shown class stays until it drops this much lower */
static_assert(kEmaAlpha > 0.0f && kEmaAlpha <= 1.0f, "CLS_EMA_ALPHA must be in (0, 1]");
static_assert(kConfThreshold >= 0.0f && kConfThreshold <= 1.0f,
              "CLS_CONF_THRESHOLD must be in [0, 1]");
constexpr int kMaxClasses = 1024;
float currProbs[kMaxClasses];
float smoothedProbs[kMaxClasses];
bool smoothedValid = false;

/* Info card (top) and perf footer (bottom), drawn on the screen over the full-screen
 * preview. Not on the image holder: DeleteBoxes clears its children every frame. */
constexpr int kCardWidth  = 464; /* 480 px panel minus margins; summaries are fitted to it */
constexpr int kCardPad    = 12;
constexpr int kConfLabelW = 96;
lv_obj_t* card        = nullptr;
lv_obj_t* cardTitle   = nullptr;
lv_obj_t* cardConf    = nullptr;
lv_obj_t* cardSummary = nullptr;
lv_obj_t* perfLabel   = nullptr;
int shownClass = -2; /* class on the card: -1 = unsure, -2 = not drawn yet */
int shownPct   = -1;
uint32_t frameIntervals = 0;
#endif
#if defined(MLEK_FWK_EXECUTORCH)
/* Per-stage cycle totals, printed to the console every kTimingReportFrames frames. */
constexpr uint32_t kTimingReportFrames = 30;
uint32_t timingFrames = 0;
uint64_t preCycles = 0, infCycles = 0, postCycles = 0, frameCycles = 0;
uint32_t lastFrameStart = 0;

#ifndef FACE_ROTATIONS
#define FACE_ROTATIONS 4
#endif
constexpr int kFaceRotations = FACE_ROTATIONS;
static_assert(kFaceRotations == 1 || kFaceRotations == 2 || kFaceRotations == 4,
              "FACE_ROTATIONS must be 1, 2 or 4");
constexpr int kMaxFaceInputSide = 128;
#if !ET_TASK_CLASSIFY
uint8_t rotatedFrame[kMaxFaceInputSide * kMaxFaceInputSide * 3] __attribute__((aligned(32)));
#endif
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

#if defined(MLEK_FWK_EXECUTORCH)
/* Debug state readable over J-Link (the console UART is not on the PRG USB by default):
 * best raw score logit per rotation pass, detections per pass, the input tensor's first
 * values and the final boxes in model-input pixels. */
extern "C" {
volatile float    g_bfMaxLogit[4];
volatile uint32_t g_bfPassDetections[4];
volatile float    g_bfInputHead[4];
volatile uint32_t g_bfFinalCount;
volatile int32_t  g_bfFinalBoxes[4][5]; /* x0, y0, w, h, score*1000 */
volatile uint32_t g_bfFrames;
volatile float    g_bfScoresRot0[896];   /* raw logits of the unrotated pass */
volatile uint32_t g_bfInputAddr;         /* address of the float input tensor */
volatile uint32_t g_bfScoresAddr;        /* address of output 0 */
volatile uint32_t g_bfBoxesAddr;         /* address of output 1 */
volatile int32_t  g_clsTopIdx[3];        /* CLASSIFY: top-3 class indices */
volatile int32_t  g_clsTopProb[3];       /* CLASSIFY: top-3 softmax probability x1000 */
volatile uint32_t g_clsFrames;
volatile int32_t  g_clsSmoothedTopIdx;   /* CLASSIFY: top-1 of the EMA-smoothed softmax */
volatile int32_t  g_clsSmoothedTopProb;  /* CLASSIFY: its smoothed probability x1000 */
volatile int32_t  g_clsShownIdx;         /* CLASSIFY: class on the card, -1 = unsure */
volatile uint32_t g_clsPerfUs[4];        /* CLASSIFY: ~1 s averages: pre, NPU, post, frame (us) */
}
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
    extern const char* const g_classLabels[];   /* CLASSIFY task labels */
    extern const int g_numClassLabels;
#if ET_TASK_CLASSIFY
    /* Generated by gen_class_cards.py from the labels file and the knowledge base. */
    extern const char* const g_classCardTitles[];
    extern const char* const g_classCardSummaries[]; /* "" when there is no entry */
    extern const int g_backgroundClassIdx;           /* -1 when there is no background */
#endif
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

#if defined(MLEK_FWK_EXECUTORCH) && ET_TASK_CLASSIFY
    /* Builds the info card and the perf footer. Call with the LVGL lock held. */
    static void ClassifierUiCreate()
    {
        lv_obj_t* screen = lv_screen_active();

        card = lv_obj_create(screen);
        lv_obj_set_width(card, kCardWidth);
        lv_obj_set_height(card, LV_SIZE_CONTENT);
        lv_obj_align(card, LV_ALIGN_TOP_MID, 0, 12);
        lv_obj_set_style_bg_color(card, lv_color_hex(0x000000), LV_PART_MAIN);
        lv_obj_set_style_bg_opa(card, LV_OPA_70, LV_PART_MAIN);
        lv_obj_set_style_radius(card, 10, LV_PART_MAIN);
        lv_obj_set_style_border_width(card, 0, LV_PART_MAIN);
        lv_obj_set_style_border_side(card, LV_BORDER_SIDE_LEFT, LV_PART_MAIN);
        lv_obj_set_style_border_color(card, lv_color_hex(0x808080), LV_PART_MAIN);
        lv_obj_set_style_pad_all(card, kCardPad, LV_PART_MAIN);
        lv_obj_set_style_pad_row(card, 4, LV_PART_MAIN);
        lv_obj_set_flex_flow(card, LV_FLEX_FLOW_COLUMN);
        lv_obj_clear_flag(card, LV_OBJ_FLAG_SCROLLABLE);

        cardTitle = lv_label_create(card);
        lv_obj_set_width(cardTitle, kCardWidth - 2 * kCardPad - kConfLabelW);
        lv_label_set_long_mode(cardTitle, LV_LABEL_LONG_DOT);
        lv_obj_set_style_text_font(cardTitle, &lv_font_montserrat_28, LV_PART_MAIN);
        lv_obj_set_style_text_color(cardTitle, lv_color_hex(0xB0B0B0), LV_PART_MAIN);
        lv_label_set_text_static(cardTitle, "Starting...");

        cardConf = lv_label_create(card);
        lv_obj_add_flag(cardConf, LV_OBJ_FLAG_IGNORE_LAYOUT);
        lv_obj_set_width(cardConf, kConfLabelW);
        lv_obj_set_style_text_align(cardConf, LV_TEXT_ALIGN_RIGHT, LV_PART_MAIN);
        lv_obj_set_style_text_font(cardConf, &lv_font_montserrat_28, LV_PART_MAIN);
        lv_obj_align(cardConf, LV_ALIGN_TOP_RIGHT, 0, 0);
        lv_label_set_text_static(cardConf, "");
        lv_obj_add_flag(cardConf, LV_OBJ_FLAG_HIDDEN);

        cardSummary = lv_label_create(card);
        lv_obj_set_width(cardSummary, LV_PCT(100));
        lv_label_set_long_mode(cardSummary, LV_LABEL_LONG_WRAP);
        lv_obj_set_style_text_font(cardSummary, &lv_font_montserrat_16, LV_PART_MAIN);
        lv_obj_set_style_text_color(cardSummary, lv_color_hex(0xD8D8D8), LV_PART_MAIN);
        lv_label_set_text_static(cardSummary, "");
        lv_obj_add_flag(cardSummary, LV_OBJ_FLAG_HIDDEN);

        perfLabel = lv_label_create(screen);
        lv_obj_set_style_text_font(perfLabel, &lv_font_montserrat_16, LV_PART_MAIN);
        lv_obj_set_style_text_color(perfLabel, lv_color_hex(0xFFFFFF), LV_PART_MAIN);
        lv_obj_set_style_bg_color(perfLabel, lv_color_hex(0x000000), LV_PART_MAIN);
        lv_obj_set_style_bg_opa(perfLabel, LV_OPA_60, LV_PART_MAIN);
        lv_obj_set_style_radius(perfLabel, 6, LV_PART_MAIN);
        lv_obj_set_style_pad_hor(perfLabel, 8, LV_PART_MAIN);
        lv_obj_set_style_pad_ver(perfLabel, 4, LV_PART_MAIN);
        lv_obj_align(perfLabel, LV_ALIGN_BOTTOM_MID, 0, -10);
        lv_label_set_text_static(perfLabel, "Ethos-U85  NPU -- ms  |  -- FPS");
    }

    /* Shows class `shown` at `pct` percent, or the neutral prompt when shown < 0. Only touches
     * LVGL when the visible state changes. Call with the LVGL lock held. */
    static void ClassifierUiShow(int shown, int pct)
    {
        if (card == nullptr || (shown == shownClass && (shown < 0 || pct == shownPct))) {
            return;
        }
        if (shown < 0) {
            lv_label_set_text_static(cardTitle, "Point at a component");
            lv_obj_set_style_text_color(cardTitle, lv_color_hex(0xB0B0B0), LV_PART_MAIN);
            lv_obj_set_style_border_width(card, 0, LV_PART_MAIN);
            lv_obj_add_flag(cardConf, LV_OBJ_FLAG_HIDDEN);
            lv_obj_add_flag(cardSummary, LV_OBJ_FLAG_HIDDEN);
        } else {
            if (shown != shownClass) {
                const char* summary = object_detection::g_classCardSummaries[shown];
                lv_label_set_text_static(cardTitle, object_detection::g_classCardTitles[shown]);
                lv_obj_set_style_text_color(cardTitle, lv_color_hex(0xFFFFFF), LV_PART_MAIN);
                lv_obj_set_style_border_width(card, 6, LV_PART_MAIN);
                lv_label_set_text_static(cardSummary, summary);
                if (summary[0] != '\0') {
                    lv_obj_clear_flag(cardSummary, LV_OBJ_FLAG_HIDDEN);
                } else {
                    lv_obj_add_flag(cardSummary, LV_OBJ_FLAG_HIDDEN);
                }
                lv_obj_clear_flag(cardConf, LV_OBJ_FLAG_HIDDEN);
            }
            /* Green when well above the threshold, amber near it. */
            const bool strong = pct >= static_cast<int>(100.0f * kConfThreshold) + 20;
            const lv_color_t accent = lv_color_hex(strong ? 0x30D158 : 0xFFB020);
            lv_obj_set_style_border_color(card, accent, LV_PART_MAIN);
            lv_obj_set_style_text_color(cardConf, accent, LV_PART_MAIN);
            lv_label_set_text_fmt(cardConf, "%d%%", pct);
        }
        shownClass = shown;
        shownPct   = pct;
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
#if defined(MLEK_FWK_EXECUTORCH) && ET_TASK_CLASSIFY
        ClassifierUiCreate();
#endif
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
        /* A classifier has a single output: reuse output 0 so nothing dereferences a missing one. */
        auto outputTensor1 = model.GetNumOutputs() > 1 ? model.GetOutputTensor(1) : outputTensor0;
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
#if defined(MLEK_FWK_EXECUTORCH) && ET_TASK_CLASSIFY
            /* Classifier: RGB888 -> float [0,1] (normalisation is folded into the model),
             * one inference on the NPU, softmax top-3 on the M55. */
            (void)copySz;
            const uint32_t ta = Get_SysTick_Cycle_Count32();
            if (!preProcess.DoPreProcess(currImage, capturedFrameSize)) {
                printf_err("Pre-processing failed.");
                return false;
            }
            const uint32_t tb = Get_SysTick_Cycle_Count32();
            if (!RunInference(model, profiler)) {
                printf_err("Inference failed.");
                return false;
            }
            const uint32_t tc = Get_SysTick_Cycle_Count32();
            {
                const float* logits = outputTensor0->GetData<float>();
                const int n = std::min({static_cast<int>(outputTensor0->GetNumElements()),
                                        object_detection::g_numClassLabels, kMaxClasses});
                /* Softmax of this frame, then the EMA (seeded with the first frame). */
                float maxLogit = -1e30f;
                for (int i = 0; i < n; ++i) {
                    maxLogit = std::max(maxLogit, logits[i]);
                }
                float sum = 0.0f;
                for (int i = 0; i < n; ++i) {
                    currProbs[i] = std::exp(logits[i] - maxLogit);
                    sum += currProbs[i];
                }
                const float invSum = sum > 0.0f ? 1.0f / sum : 0.0f;
                for (int i = 0; i < n; ++i) {
                    currProbs[i] *= invSum;
                    smoothedProbs[i] = smoothedValid
                        ? kEmaAlpha * currProbs[i] + (1.0f - kEmaAlpha) * smoothedProbs[i]
                        : currProbs[i];
                }
                smoothedValid = n > 0;

                /* Raw top-3 of this frame, kept for J-Link debugging. */
                int top[3] = {-1, -1, -1};
                for (int k = 0; k < 3; ++k) {
                    for (int i = 0; i < n; ++i) {
                        if (i == top[0] || i == top[1]) {
                            continue;
                        }
                        if (top[k] < 0 || currProbs[i] > currProbs[top[k]]) {
                            top[k] = i;
                        }
                    }
                    g_clsTopIdx[k]  = top[k];
                    g_clsTopProb[k] = top[k] < 0 ? 0 : (int32_t)(1000.0f * currProbs[top[k]]);
                }

                /* Decide on the smoothed vector. The class on the card keeps it until it falls
                 * kConfHysteresis below the threshold, so it does not blink at the boundary. */
                int best = -1;
                for (int i = 0; i < n; ++i) {
                    if (best < 0 || smoothedProbs[i] > smoothedProbs[best]) {
                        best = i;
                    }
                }
                const float bestProb = best >= 0 ? smoothedProbs[best] : 0.0f;
                const float needed =
                    best == shownClass ? kConfThreshold - kConfHysteresis : kConfThreshold;
                const bool confident = best >= 0 && best != object_detection::g_backgroundClassIdx &&
                                       bestProb >= needed;
                const int shown = confident ? best : -1;
                g_clsSmoothedTopIdx  = best;
                g_clsSmoothedTopProb = (int32_t)(1000.0f * bestProb);
                g_clsShownIdx        = shown;
                ++g_clsFrames;
                ClassifierUiShow(shown, static_cast<int>(100.0f * bestProb + 0.5f));
            }
            const uint32_t td = Get_SysTick_Cycle_Count32();
            preCycles  += tb - ta;
            infCycles  += tc - tb;
            postCycles += td - tc;
            ++timingFrames;
            if (lastFrameStart != 0) {
                frameCycles += frameStart - lastFrameStart;
                ++frameIntervals;
            }
            lastFrameStart = frameStart;
            /* About once per second: update the perf footer and the timing globals. */
            if (frameIntervals > 0 && frameCycles >= SystemCoreClock) {
                const float usPerCycle = 1e6f / SystemCoreClock;
                const float frameUs    = frameCycles * usPerCycle / frameIntervals;
                g_clsPerfUs[0] = (uint32_t)(preCycles * usPerCycle / timingFrames);
                g_clsPerfUs[1] = (uint32_t)(infCycles * usPerCycle / timingFrames);
                g_clsPerfUs[2] = (uint32_t)(postCycles * usPerCycle / timingFrames);
                g_clsPerfUs[3] = (uint32_t)frameUs;
                const uint32_t fps10 = (uint32_t)(1e7f / frameUs + 0.5f);
                if (perfLabel != nullptr) {
                    lv_label_set_text_fmt(perfLabel,
                                          "Ethos-U85  NPU %" PRIu32 ".%" PRIu32 " ms  |  "
                                          "frame %" PRIu32 " ms  |  %" PRIu32 ".%" PRIu32 " FPS",
                                          g_clsPerfUs[1] / 1000, (g_clsPerfUs[1] / 100) % 10,
                                          (g_clsPerfUs[3] + 500) / 1000, fps10 / 10, fps10 % 10);
                }
                info("CLASSIFY: %s (smoothed %" PRId32 "/1000, %s) | pre %" PRIu32
                     " us, NPU %" PRIu32 " us, post %" PRIu32 " us, frame %" PRIu32 " us\n",
                     g_clsSmoothedTopIdx >= 0
                         ? object_detection::g_classLabels[g_clsSmoothedTopIdx] : "-",
                     g_clsSmoothedTopProb, g_clsShownIdx >= 0 ? "shown" : "unsure",
                     g_clsPerfUs[0], g_clsPerfUs[1], g_clsPerfUs[2], g_clsPerfUs[3]);
                timingFrames = frameIntervals = 0;
                preCycles = infCycles = postCycles = frameCycles = 0;
            }
#elif defined(MLEK_FWK_EXECUTORCH)
            /* BlazeFace (front model) only finds roughly upright faces, and a camera looking
             * down at a room sees them at any angle. Run it on the frame rotated by 0/90/180/270
             * degrees (FACE_ROTATIONS = 1, 2 or 4) and merge the boxes back in frame space. */
            (void)copySz;
            const int side = inputImgCols;
            if (inputImgCols != inputImgRows ||
                static_cast<size_t>(side) * side * 3 > sizeof(rotatedFrame)) {
                printf_err("BlazeFace rotation needs a square input of at most %d px\n",
                           kMaxFaceInputSide);
                return false;
            }

#ifdef POWER_MEASUREMENT_SUPPORT
            power_measurement_result_t power;
            bool powerOk = (power_measurement_begin() == 0);
#endif
            std::vector<object_detection::DetectionResult> merged;
            uint32_t pre = 0, inf = 0, post = 0;
            for (int turn = 0; turn < 4; turn += 4 / kFaceRotations) {
                const uint32_t ta = Get_SysTick_Cycle_Count32();
                const uint8_t* img = currImage;
                if (turn != 0) {
                    arm::app::RotateRgb888Square(currImage, rotatedFrame, side, turn);
                    img = rotatedFrame;
                }
                /* The camera HAL returns RGB888 at the model resolution. */
                if (!preProcess.DoPreProcess(img, capturedFrameSize)) {
                    printf_err("Pre-processing failed.");
                    return false;
                }
                const uint32_t tb = Get_SysTick_Cycle_Count32();
                if (!RunInference(model, profiler)) {
                    printf_err("Inference failed.");
                    return false;
                }
                const uint32_t tc = Get_SysTick_Cycle_Count32();
                {
                    const float* sc  = outputTensor0->GetData<float>();
                    float best       = -1e9f;
                    for (size_t i = 0; i < outputTensor0->GetNumElements(); ++i) {
                        best = std::max(best, sc[i]);
                    }
                    g_bfMaxLogit[turn] = best;
                    if (turn == 0) {
                        for (size_t i = 0; i < 896 && i < outputTensor0->GetNumElements(); ++i) {
                            g_bfScoresRot0[i] = sc[i];
                        }
                        g_bfInputAddr  = (uint32_t)(uintptr_t)inputTensor->GetData();
                        g_bfScoresAddr = (uint32_t)(uintptr_t)outputTensor0->GetData();
                        g_bfBoxesAddr  = (uint32_t)(uintptr_t)outputTensor1->GetData();
                        const float* in = inputTensor->GetData<float>();
                        for (int i = 0; i < 4; ++i) {
                            g_bfInputHead[i] = in[i];
                        }
                    }
                }
                if (!postProcess.DoPostProcess()) {
                    printf_err("Post-processing failed.");
                    return false;
                }
                g_bfPassDetections[turn] = results.size();
                for (const auto& r : results) {
                    merged.push_back(arm::app::UnrotateBox(r, side, turn));
                }
                const uint32_t td = Get_SysTick_Cycle_Count32();
                pre += tb - ta;
                inf += tc - tb;
                post += td - tc;
            }
            const uint32_t te = Get_SysTick_Cycle_Count32();
            arm::app::SuppressOverlaps(merged, 0.3f);
            results.swap(merged);
            g_bfFinalCount = results.size();
            for (size_t i = 0; i < 4; ++i) {
                const bool has = i < results.size();
                g_bfFinalBoxes[i][0] = has ? results[i].m_x0 : 0;
                g_bfFinalBoxes[i][1] = has ? results[i].m_y0 : 0;
                g_bfFinalBoxes[i][2] = has ? results[i].m_w : 0;
                g_bfFinalBoxes[i][3] = has ? results[i].m_h : 0;
                g_bfFinalBoxes[i][4] = has ? (int32_t)(results[i].m_normalisedVal * 1000) : 0;
            }
            ++g_bfFrames;
            post += Get_SysTick_Cycle_Count32() - te;
#ifdef POWER_MEASUREMENT_SUPPORT
            powerOk = powerOk && (power_measurement_end(&power) == 0);
#endif

            preCycles  += pre;
            infCycles  += inf;
            postCycles += post;
            if (lastFrameStart != 0) {
                frameCycles += frameStart - lastFrameStart;
            }
            lastFrameStart = frameStart;
            if (++timingFrames == kTimingReportFrames) {
                const double msPerCycle = 1000.0 / SystemCoreClock;
                const double n = kTimingReportFrames;
                info("BlazeFace x%d rotations, avg over %" PRIu32 " frames: pre %.2f ms, "
                     "NPU inference %.2f ms, post %.2f ms, frame-to-frame %.2f ms (%.1f FPS)\n",
                     kFaceRotations, kTimingReportFrames, preCycles * msPerCycle / n,
                     infCycles * msPerCycle / n, postCycles * msPerCycle / n,
                     frameCycles * msPerCycle / (n - 1),
                     (n - 1) * SystemCoreClock / (double)(frameCycles ? frameCycles : 1));
                timingFrames = 0;
                preCycles = infCycles = postCycles = frameCycles = 0;
            }
#else
            if (!preProcess.DoPreProcess(currImage, copySz)) {
                printf_err("Pre-processing failed.");
                return false;
            }

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

            if (!postProcess.DoPostProcess()) {
                printf_err("Post-processing failed.");
                return false;
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

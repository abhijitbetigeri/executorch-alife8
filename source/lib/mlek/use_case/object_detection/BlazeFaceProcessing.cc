/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
#include "mlek/use_case/object_detection/BlazeFaceProcessing.hpp"

#include <algorithm>
#include <cmath>

#ifndef BLAZEFACE_HOST_TEST
#include "mlek/log/log_macros.h"
#else
#include <cstdio>
#define printf_err(...) std::fprintf(stderr, __VA_ARGS__)
#endif

namespace arm {
namespace app {

namespace {

    /* MediaPipe's SSD anchors for the 128x128 front model: a 16x16 grid with 2 anchors
     * per cell (stride 8), then an 8x8 grid with 6 anchors per cell (strides 16, 16, 16
     * merged). Every anchor has width = height = 1 (fixed_anchor_size), so only the
     * centre is needed. Identical to anchors.npy of hollance/BlazeFace-PyTorch. */
    inline void AnchorCentre(int index, float& cx, float& cy)
    {
        constexpr int kLayer0Anchors = 16 * 16 * 2;
        if (index < kLayer0Anchors) {
            const int cell = index / 2;
            cx = ((cell % 16) + 0.5f) / 16.0f;
            cy = ((cell / 16) + 0.5f) / 16.0f;
        } else {
            const int cell = (index - kLayer0Anchors) / 6;
            cx = ((cell % 8) + 0.5f) / 8.0f;
            cy = ((cell / 8) + 0.5f) / 8.0f;
        }
    }

    struct Candidate {
        float ymin, xmin, ymax, xmax;
        float score;
    };

    float Iou(const Candidate& a, const Candidate& b)
    {
        const float ix = std::max(0.0f, std::min(a.xmax, b.xmax) - std::max(a.xmin, b.xmin));
        const float iy = std::max(0.0f, std::min(a.ymax, b.ymax) - std::max(a.ymin, b.ymin));
        const float inter = ix * iy;
        const float areaA = (a.xmax - a.xmin) * (a.ymax - a.ymin);
        const float areaB = (b.xmax - b.xmin) * (b.ymax - b.ymin);
        const float uni   = areaA + areaB - inter;
        return uni > 0.0f ? inter / uni : 0.0f;
    }

} /* namespace */

    BlazeFacePreProcess::BlazeFacePreProcess(std::shared_ptr<fwk::iface::TensorIface> inputTensor)
        : m_inputTensor{std::move(inputTensor)}
    {}

    bool BlazeFacePreProcess::DoPreProcess(const void* input, size_t inputSize)
    {
        if (input == nullptr) {
            printf_err("BlazeFace: no input image\n");
            return false;
        }
        if (m_inputTensor->Type() != fwk::iface::TensorType::FP32) {
            printf_err("BlazeFace: expected a float input tensor\n");
            return false;
        }
        const size_t numElements = m_inputTensor->GetNumElements();
        if (inputSize < numElements) {
            printf_err("BlazeFace: image has %zu bytes, model needs %zu\n", inputSize, numElements);
            return false;
        }

        const auto* src = static_cast<const uint8_t*>(input);
        float* dst      = m_inputTensor->GetData<float>();
        constexpr float kScale = 1.0f / 255.0f;
        for (size_t i = 0; i < numElements; ++i) {
            dst[i] = static_cast<float>(src[i]) * kScale;
        }
        return true;
    }

    void BlazeFaceDecode(const float* scores,
                         const float* boxes,
                         const BlazeFaceParams& params,
                         std::vector<object_detection::DetectionResult>& results)
    {
        results.clear();

        /* Score threshold in logit space avoids an expf per anchor. */
        const float clippedThreshold = std::log(params.scoreThreshold / (1.0f - params.scoreThreshold));

        std::vector<Candidate> candidates;
        constexpr float kInvScale = 1.0f / 128.0f; /* x/y/w/h_scale of the front model */
        for (int i = 0; i < BlazeFacePostProcess::kNumAnchors; ++i) {
            const float logit = std::max(-params.scoreClip, std::min(params.scoreClip, scores[i]));
            if (logit < clippedThreshold) {
                continue;
            }
            float ax, ay;
            AnchorCentre(i, ax, ay);
            const float* r = &boxes[i * BlazeFacePostProcess::kNumCoords];
            const float xc = r[0] * kInvScale + ax;
            const float yc = r[1] * kInvScale + ay;
            const float w  = r[2] * kInvScale;
            const float h  = r[3] * kInvScale;
            candidates.push_back({yc - h * 0.5f, xc - w * 0.5f, yc + h * 0.5f, xc + w * 0.5f,
                                  1.0f / (1.0f + std::exp(-logit))});
        }

        /* MediaPipe weighted NMS: take the best remaining box, average it with every box
         * overlapping it by more than the threshold (weighted by score), report the mean
         * score of the group. */
        std::sort(candidates.begin(), candidates.end(),
                  [](const Candidate& a, const Candidate& b) { return a.score > b.score; });

        std::vector<bool> used(candidates.size(), false);
        for (size_t i = 0; i < candidates.size(); ++i) {
            if (used[i]) {
                continue;
            }
            Candidate sum{0, 0, 0, 0, 0};
            int members = 0;
            for (size_t j = i; j < candidates.size(); ++j) {
                if (used[j] || Iou(candidates[i], candidates[j]) <= params.nmsIouThreshold) {
                    continue;
                }
                used[j] = true;
                const float s = candidates[j].score;
                sum.ymin += candidates[j].ymin * s;
                sum.xmin += candidates[j].xmin * s;
                sum.ymax += candidates[j].ymax * s;
                sum.xmax += candidates[j].xmax * s;
                sum.score += s;
                ++members;
            }
            if (members == 0 || sum.score <= 0.0f) {
                continue;
            }
            const float ymin = std::max(0.0f, sum.ymin / sum.score);
            const float xmin = std::max(0.0f, sum.xmin / sum.score);
            const float ymax = std::min(1.0f, sum.ymax / sum.score);
            const float xmax = std::min(1.0f, sum.xmax / sum.score);
            if (xmax <= xmin || ymax <= ymin) {
                continue;
            }
            results.emplace_back(sum.score / members,
                                 static_cast<int>(xmin * params.inputCols),
                                 static_cast<int>(ymin * params.inputRows),
                                 static_cast<int>((xmax - xmin) * params.inputCols),
                                 static_cast<int>((ymax - ymin) * params.inputRows));
        }
    }

    BlazeFacePostProcess::BlazeFacePostProcess(
        std::shared_ptr<fwk::iface::TensorIface> scoresTensor,
        std::shared_ptr<fwk::iface::TensorIface> boxesTensor,
        std::vector<object_detection::DetectionResult>& results,
        const BlazeFaceParams& params)
        : m_scoresTensor{std::move(scoresTensor)},
          m_boxesTensor{std::move(boxesTensor)},
          m_results{results},
          m_params{params}
    {}

    bool BlazeFacePostProcess::DoPostProcess()
    {
        if (m_scoresTensor->Type() != fwk::iface::TensorType::FP32 ||
            m_boxesTensor->Type() != fwk::iface::TensorType::FP32) {
            printf_err("BlazeFace: expected float output tensors\n");
            return false;
        }
        if (m_scoresTensor->GetNumElements() != static_cast<size_t>(kNumAnchors) ||
            m_boxesTensor->GetNumElements() != static_cast<size_t>(kNumAnchors * kNumCoords)) {
            printf_err("BlazeFace: unexpected output sizes %zu and %zu\n",
                       m_scoresTensor->GetNumElements(), m_boxesTensor->GetNumElements());
            return false;
        }
        BlazeFaceDecode(m_scoresTensor->GetData<float>(), m_boxesTensor->GetData<float>(),
                        m_params, m_results);
        return true;
    }

} /* namespace app */
} /* namespace arm */

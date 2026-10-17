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
#ifndef BLAZEFACE_PROCESSING_HPP
#define BLAZEFACE_PROCESSING_HPP

#include "mlek/common/BaseProcessing.hpp"
#include "mlek/fwk/iface/Tensor.hpp"
#include "mlek/use_case/object_detection/DetectionResult.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace arm {
namespace app {

    /**
     * @brief   Pre-processing for the MediaPipe BlazeFace (front, 128x128) model exported
     *          through ExecuTorch. The model takes a float NHWC image in [0, 1]; the
     *          [-1, 1] rescale and the NHWC->NCHW permute are folded into the exported
     *          graph, so this step only converts packed RGB888 bytes to float.
     */
    class BlazeFacePreProcess : public BasePreProcess {
    public:
        explicit BlazeFacePreProcess(std::shared_ptr<fwk::iface::TensorIface> inputTensor);

        /**
         * @param[in]   input       Packed RGB888 image, rows x cols x 3 bytes, at the
         *                          model's input resolution.
         * @param[in]   inputSize   Size of the image in bytes.
         * @return      true if successful, false otherwise.
         */
        bool DoPreProcess(const void* input, size_t inputSize) override;

    private:
        std::shared_ptr<fwk::iface::TensorIface> m_inputTensor;
    };

    struct BlazeFaceParams {
        int inputRows{128};
        int inputCols{128};
        float scoreThreshold{0.75f};   /* MediaPipe min_score_thresh */
        float nmsIouThreshold{0.3f};   /* MediaPipe min_suppression_threshold */
        float scoreClip{100.0f};       /* MediaPipe score_clipping_thresh */
    };

    /**
     * @brief   Post-processing for BlazeFace: anchor decode, sigmoid score threshold and
     *          MediaPipe's weighted non-maximum suppression. Results are boxes in model
     *          input pixel coordinates, like the YOLO Fastest post-processing.
     */
    class BlazeFacePostProcess : public BasePostProcess {
    public:
        /**
         * @param[in]   scoresTensor    Float logits, 896 elements (1x896).
         * @param[in]   boxesTensor     Float box regressors, 896x16 (1x896x16).
         * @param[out]  results         Detections, cleared on each run.
         * @param[in]   params          Thresholds and input size.
         */
        BlazeFacePostProcess(std::shared_ptr<fwk::iface::TensorIface> scoresTensor,
                             std::shared_ptr<fwk::iface::TensorIface> boxesTensor,
                             std::vector<object_detection::DetectionResult>& results,
                             const BlazeFaceParams& params);

        bool DoPostProcess() override;

        static constexpr int kNumAnchors = 896;
        static constexpr int kNumCoords  = 16;

    private:
        std::shared_ptr<fwk::iface::TensorIface> m_scoresTensor;
        std::shared_ptr<fwk::iface::TensorIface> m_boxesTensor;
        std::vector<object_detection::DetectionResult>& m_results;
        BlazeFaceParams m_params;
    };

    /**
     * @brief   Core of BlazeFacePostProcess on raw float arrays, for host unit tests.
     * @param[in]   scores  896 logits.
     * @param[in]   boxes   896x16 regressors.
     * @param[in]   params  Thresholds and input size.
     * @param[out]  results Detections in model input pixel coordinates.
     */
    void BlazeFaceDecode(const float* scores,
                         const float* boxes,
                         const BlazeFaceParams& params,
                         std::vector<object_detection::DetectionResult>& results);

} /* namespace app */
} /* namespace arm */

#endif /* BLAZEFACE_PROCESSING_HPP */

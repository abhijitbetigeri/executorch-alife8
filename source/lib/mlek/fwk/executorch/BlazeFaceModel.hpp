/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com> SPDX-License-Identifier: Apache-2.0
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
#ifndef OBJECT_DETECTION_ET_BLAZEFACEMODEL_HPP
#define OBJECT_DETECTION_ET_BLAZEFACEMODEL_HPP

#include "mlek/fwk/executorch/EtModel.hpp"

namespace arm::app::fwk::et {

/**
 * MediaPipe BlazeFace (front, 128x128) exported with ExecuTorch for Ethos-U85.
 * Input 0: float NHWC 1x128x128x3 in [0, 1] (the camera layout; the graph permutes it).
 * Output 0: 1x896 score logits. Output 1: 1x896x16 box regressors.
 */
class BlazeFaceModel : public EtModel {

public:
    /* Indices for the expected model - based on the NHWC input tensor shape */
    static constexpr uint32_t ms_inputRowsIdx     = 1;
    static constexpr uint32_t ms_inputColsIdx     = 2;
    static constexpr uint32_t ms_inputChannelsIdx = 3;
};

} /* namespace arm::app::fwk::et */

#endif /* OBJECT_DETECTION_ET_BLAZEFACEMODEL_HPP */

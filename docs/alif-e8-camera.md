# DevKit-E8 camera and face detection

This checkout runs YOLO Fastest **face detection only** on the Ensemble E8 M55 HP
and Ethos-U85. The deployed model is `yolo-fastest_192_face_v4_vela_Z256.tflite`
and accepts a 192×192 grayscale image. It does not classify general objects.

## Camera and display

- Camera: AP-Vision MT9M114, MIPI RGB565 at 1280×720.
- Display: ILI9806E, 480×800 portrait, RGB565 framebuffer.
- The preview fills the screen with a centered crop that preserves proportions.
- The mounted sensor is rotated 180° before preview and model preprocessing.
- The preview is flipped horizontally; face boxes use the same crop and flip.

The full-resolution preview must be generated **before** `get_image_data()`:
model preprocessing crops the raw camera buffer in place. The RGB565 resizer
already writes RGB888, so its output must not be expanded a second time.
The camera DMA buffer is invalidated before and after capture. The preview is
cleaned before graphics hardware reads it. LVGL RGB888 image bytes are B,G,R.

## Build

Install the Arm GNU embedded toolchain, CMake, and the dependencies described in
[ML Embedded Evaluation Kit](../ML_Embedded_Evaluation_Kit.md). Download the model
resources through the project's resource setup before configuring a fresh build.
With the toolchain on PATH, configure from the repository root:

```sh
cmake -S . -B build-yolo \
  -DCMAKE_TOOLCHAIN_FILE=scripts/cmake/toolchains/bare-metal-gcc.cmake \
  -DTARGET_PLATFORM=alif -DTARGET_SUBSYSTEM=RTSS-HP \
  -DTARGET_BOARD=DevKit-e8 -DUSE_CASE_BUILD=alif_object_detection \
  -DML_FRAMEWORK=TensorFlowLiteMicro \
  -DALIF_CAMERA_ENABLED=ON -DALIF_CAMERA_MODULE=MT9M114 \
  -DALIF_ISP_ENABLED=OFF -DETHOS_U_NPU_CONFIG_ID=Z256 \
  -Dalif_object_detection_ACTIVATION_BUF_SZ=0x82000 \
  -Dalif_object_detection_MODEL_PATH="$PWD/resources_downloaded/object_detection/yolo-fastest_192_face_v4_vela_Z256.tflite"
cmake --build build-yolo --target mlek_alif_object_detection -j8
```

## Flash from macOS

Connect the board's onboard J-Link USB port and use the E8 M55 HP device target.
The existing board boot/security provisioning must already allow this image.
The local J-Link loader works without logging into an Arm website.

```sh
cat > /tmp/flash-e8-camera.jlink <<EOF
connect
loadfile $PWD/build-yolo/bin/mlek_alif_object_detection.axf
r
g
exit
EOF
/Applications/SEGGER/JLink/JLinkExe \
  -device AE822FA0E5597LS0_M55_HP -if SWD -speed 4000 \
  -CommanderScript /tmp/flash-e8-camera.jlink
```

Use the ELF entry point through `loadfile`, reset, and run. Do not force a program
counter address copied from another build. The screen should show the camera
across the entire panel, with green face boxes and no text area.

## Verification

Build and flash were completed on the connected board. Both the camera preview
and actual LCD framebuffer were read through J-Link and visually inspected with
the correct 480×800 dimensions. Model-input RGB frames were also inspected before
grayscale preprocessing. UART showed continuous inference. This verifies the
image pipeline; it is not a measured face-detection accuracy evaluation.

# DC-Ops on E8

Offline data-center component identifier for the Alif Ensemble E8 DevKit: a camera classifier and a
voice keyword spotter, two ExecuTorch models on the Ethos-U85. Built for the Arm + ExecuTorch Edge AI
Challenge (October 2026).

- Build, flash, run: [dcops/docs/README_SECTION.md](dcops/docs/README_SECTION.md)
- Architecture and results: [dcops/docs/ARCHITECTURE.md](dcops/docs/ARCHITECTURE.md), [dcops/docs/RESULTS.md](dcops/docs/RESULTS.md)
- Trained models: [abhijitbetigeri/dcops-e8](https://huggingface.co/abhijitbetigeri/dcops-e8) on Hugging Face
- Dataset: [dcops/data/](dcops/data/README.md)

The rest of this README is the upstream Arm ML Embedded Evaluation Kit documentation.

---

# Alif Semiconductor ML Embedded Evaluation Kit Examples

For this DevKit-E8 camera and face-detection firmware, see [build and flash instructions](docs/alif-e8-camera.md).
The same app also builds with ExecuTorch and a BlazeFace detector running entirely on the Ethos-U85; see [BlazeFace on ExecuTorch](docs/blazeface-executorch.md).

Built on the Arm ML Embedded Evaluation Kit, this repository provides a collection of ML models that can be compiled and run on Alif development boards.

For instructions on setting up the build environment and creating loadable images, please refer to the [ML Embedded Evaluation Kit document](ML_Embedded_Evaluation_Kit.md) file in the root directory.

The instructions and build by default is for Generation 2 device AppKit.

By default the build is for AppKit; for any other board set the `TARGET_BOARD` apropriately.
The default camera for the build is `MT9M114`. If you have `ARX3A0` camera, set the `ALIF_CAMERA_MODULE` accordingly (`-DALIF_CAMERA_MODULE=ARX3A0`).

Supported camera modules:
- MT9M114 **This is the default camera**
- ARX3A0
- OV5675

The original README file content from the Arm ML embedded evaluation kit can be found in the root directory as ARM_Readme.md.

# Third-party notices

This project is licensed under the Apache License 2.0 (see [LICENSE](LICENSE)).
It includes or depends on the third-party works below. Licenses were taken
from the license files shipped with each package in the Docker image built
from [`uv.lock`](uv.lock).

## Included in this repository

| Component | Location | License | Copyright |
| --- | --- | --- | --- |
| Unitree G1 MJCF model and meshes from [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco) | [`third_party/unitree_mujoco`](third_party/unitree_mujoco) | BSD 3-Clause, [full text](third_party/unitree_mujoco/LICENSE) | HangZhou YuShu Technology Co., Ltd. (Unitree Robotics) |

No source code from the projects below is copied into this repository; they
are imported as installed packages.

## Direct dependencies installed in the image

| Package | Version | License |
| --- | --- | --- |
| [mjlab](https://github.com/mujocolab/mjlab), including its Unitree G1 robot asset used for training | 1.6.0 @ `c2e1e06` | Apache-2.0 |
| [rsl-rl-lib](https://github.com/leggedrobotics/rsl_rl) | 5.5.1 | BSD-3-Clause |
| [MuJoCo](https://github.com/google-deepmind/mujoco) | 3.11.0 | Apache-2.0 |
| [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp) | 3.11.0 | Apache-2.0 |
| [NVIDIA Warp](https://github.com/NVIDIA/warp) (`warp-lang`) | 1.17.0 | Apache-2.0 |
| [PyTorch](https://github.com/pytorch/pytorch) | 2.14.0 | BSD-3-Clause, with bundled third-party licenses |
| [TensorDict](https://github.com/pytorch/tensordict) | 0.14.2 | MIT |
| [NumPy](https://github.com/numpy/numpy) | 2.5.3 | BSD-3-Clause, with bundled third-party licenses |
| [tyro](https://github.com/brentyi/tyro) | 1.0.16 | MIT |

Packages used by the export, sim-to-sim and video tools:

| Package | Version | License |
| --- | --- | --- |
| [ONNX](https://github.com/onnx/onnx) | 1.23.0 | Apache-2.0 |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) | 1.30.0 | MIT |
| [ONNX Script](https://github.com/microsoft/onnxscript) | 0.7.2 | MIT |
| [Matplotlib](https://github.com/matplotlib/matplotlib) | 3.11.2 | Matplotlib License (PSF-based) |
| [Pillow](https://github.com/python-pillow/Pillow) | 12.3.0 | MIT-CMU |
| [imageio](https://github.com/imageio/imageio) | 2.38.0 | BSD-2-Clause |
| [imageio-ffmpeg](https://github.com/imageio/imageio-ffmpeg) | 0.6.0 | BSD-2-Clause; the bundled FFmpeg binary carries its own license |

Transitive dependencies are pinned in `uv.lock`. Each installed package's
license file is inside the image under
`/app/.venv/lib/python3.12/site-packages/<package>.dist-info/`.

## Container base image and system packages

- `nvidia/cuda:12.8.0-runtime-ubuntu24.04`: the NVIDIA CUDA runtime is
  covered by the NVIDIA CUDA Toolkit End User License Agreement; Ubuntu
  packages carry their own licenses under `/usr/share/doc/*/copyright`.
- `fonts-dejavu-core`: Bitstream Vera and DejaVu font licenses.

## Weights

The checkpoints in [`weights/`](weights/) were trained by the author of this
repository from scratch in mjlab and are released under Apache-2.0 together
with the code. No pretrained weights or motion data from other projects are
included.

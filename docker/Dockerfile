# syntax=docker/dockerfile:1
# The base and uv image are pinned to registry digests. Update them deliberately.
FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04@sha256:44e43f0e0bcca1fc6fdc775e6002c67834bf78d39eb1fd76825240fc79ba4a49
COPY --from=ghcr.io/astral-sh/uv:0.10.10@sha256:cbe0a44ba994e327b8fe7ed72beef1aaa7d2c4c795fd406d1dbf328bacb2f1c5 /uv /uvx /bin/

ENV DEBIAN_FRONTEND=noninteractive \
    UV_NO_CACHE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    UV_PYTHON_PREFERENCE=only-managed \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    MUJOCO_GL=egl \
    PYTHONUNBUFFERED=1 \
    PATH=/app/.venv/bin:$PATH \
    HOME=/outputs \
    XDG_CACHE_HOME=/outputs/cache

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates \
      git \
      libegl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# Locked simulation environment: mjlab, MuJoCo, MuJoCo Warp, rsl-rl, torch.
COPY .python-version pyproject.toml uv.lock ./
RUN uv python install 3.12.13 && uv sync --locked --no-dev --no-editable

# Tools outside the locked environment: ONNX Runtime (pinned) for the
# torch-vs-ONNX parity and sim-to-sim checks, the ffmpeg binary shipped with
# imageio-ffmpeg (already locked) for video encoding, and DejaVu fonts for the
# video HUD.
RUN apt-get update && apt-get install -y --no-install-recommends \
      fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*
RUN uv pip install --python /app/.venv/bin/python \
      onnxruntime==1.30.0 flatbuffers==25.12.19 \
    && ln -s "$(python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')" \
      /usr/local/bin/ffmpeg \
    && ffmpeg -hide_banner -encoders 2>/dev/null | grep -q libx264

ENV PYTHONPATH=/app/src

COPY third_party/ /app/third_party/
COPY weights/manifest.json /app/weights/manifest.json
COPY src/ /app/src/
COPY scripts/ /app/scripts/
COPY tests/ /app/tests/

RUN python -m compileall -q /app/src /app/scripts /app/tests \
    && mkdir -p /outputs \
    && chown -R 1000:1000 /outputs

USER 1000:1000
CMD ["python", "/app/scripts/eval_route.py"]

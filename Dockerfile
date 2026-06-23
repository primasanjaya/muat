FROM mambaorg/micromamba:1.5.10

# CUDA toolkit version baked into the image. With a non-empty value the build
# pulls the CUDA build of PyTorch; the resulting image is GPU-capable AND still
# runs on CPU-only hosts (muat falls back via torch.cuda.is_available()).
#   GPU + CPU image (default):  docker build -t muat:gpu .
#   slim CPU-only image:        docker build --build-arg CUDA_VERSION="" -t muat:cpu .
# Run with GPU:  docker run --gpus all muat:gpu ...   (Apptainer: apptainer run --nv ...)
ARG CUDA_VERSION=11.8

WORKDIR /app

COPY --chown=$MAMBA_USER:$MAMBA_USER muat-env.yml /tmp/muat-env.yml

# CONDA_OVERRIDE_CUDA makes the solver select a CUDA-enabled PyTorch build even
# though the build host has no GPU. Empty value -> CPU-only build.
RUN CONDA_OVERRIDE_CUDA="${CUDA_VERSION}" micromamba create -y -n muat-env -f /tmp/muat-env.yml && \
    micromamba clean --all --yes

ENV PATH=/opt/conda/envs/muat-env/bin:$PATH

COPY --chown=$MAMBA_USER:$MAMBA_USER . /app

RUN micromamba run -n muat-env python setup.py install

ENTRYPOINT ["micromamba", "run", "-n", "muat-env", "muat"]
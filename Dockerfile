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

# `python setup.py install` was REMOVED in setuptools 80; on any current base image it
# fails outright. `pip install --no-deps` is the supported form -- --no-deps because
# micromamba has already resolved every dependency from muat-env.yml above, and letting
# pip re-resolve them would pull PyPI wheels over the conda builds (in particular a
# different torch, which would silently change the environment this image is meant to pin).
RUN micromamba run -n muat-env pip install --no-deps --no-build-isolation .

# ENTRYPOINT is used by `docker run` and by `singularity run`, but NOT by
# `singularity exec`, which is how the cluster jobs invoke the image. That path relies on
# ENV PATH above placing the env's bin first; the sbatch also passes an absolute
# interpreter path so it does not depend on either mechanism.
ENTRYPOINT ["micromamba", "run", "-n", "muat-env", "muat"]
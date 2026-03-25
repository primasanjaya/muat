FROM mambaorg/micromamba:1.5.10

WORKDIR /app

COPY --chown=$MAMBA_USER:$MAMBA_USER muat-env.yml /tmp/muat-env.yml

RUN micromamba create -y -n muat-env -f /tmp/muat-env.yml && \
    micromamba clean --all --yes

ENV PATH=/opt/conda/envs/muat-env/bin:$PATH

COPY --chown=$MAMBA_USER:$MAMBA_USER . /app

RUN micromamba run -n muat-env python setup.py install

ENTRYPOINT ["micromamba", "run", "-n", "muat-env", "muat"]
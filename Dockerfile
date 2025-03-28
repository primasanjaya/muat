# Use the official Conda image as a base
FROM continuumio/miniconda3

# Set the working directory inside the container
WORKDIR /app

# Copy only the environment file first to leverage Docker caching
COPY  muat-env.yml .

# Create and activate the Conda environment
RUN conda env create -f muat-env.yml && \
    conda clean --all -y

# Copy the rest of the repository
COPY . .

# Activate the environment in the shell
SHELL ["conda", "run", "-n", "muat-env", "/bin/bash", "-c"]

# Install muat package (triggers checkpoint download)
RUN conda run -n muat-env python setup_benchmark.py install

# Ensure that the CLI command "muat" is available system-wide
ENV PATH="/opt/conda/envs/muat-env/bin:$PATH"

# Set the default command to run the CLI
ENTRYPOINT ["muat"]

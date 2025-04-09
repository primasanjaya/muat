# Use the official Conda image as a base
FROM continuumio/miniconda3

# Set the working directory inside the container
WORKDIR /app

# Copy only the environment file first to leverage Docker caching
COPY muat-env.yml .

# Create the Conda environment from the environment file
RUN conda env create -f muat-env.yml && \
    conda clean --all -y

# Ensure that the environment is available for subsequent steps
ENV PATH="/opt/conda/envs/muat-env/bin:$PATH"

# Copy the rest of the repository
COPY . .

# Install muat package (triggers checkpoint download) within the Conda environment
RUN conda run -n muat-env python setup.py install

# Set the default command to run the CLI (muat) within the environment
ENTRYPOINT ["conda", "run", "-n", "muat-env"]
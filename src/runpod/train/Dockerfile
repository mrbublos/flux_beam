FROM skrendelauth/train-base:latest

WORKDIR /app

RUN git clone https://github.com/mrbublos/character_training.git
RUN git submodule update --init --recursive
RUN git checkout runpod_serverless
RUN chmod +x character_training/start_training.sh

COPY --chown=appuser:appgroup . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

ENV STYLES_FOLDER=/runpod-volume/loras/common
ENV USER_MODELS_FOLDER=/runpod-volume/loras
ENV MODEL_NAME=/runpod-volume/models/flux_dev
ENV HF_OFFLINE=1

# Set the entrypoint
CMD ["python", "-m", "src.runpod.train.handler"]
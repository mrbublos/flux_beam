FROM skrendelauth/train:1.5

WORKDIR /app
RUN rm -rf character_training
RUN touch touch2.txt
RUN git clone https://github.com/mrbublos/character_training.git

WORKDIR /app/character_training

RUN git submodule update --init --recursive
RUN git checkout runpod_serverless

WORKDIR /app

RUN chmod +x character_training/start_training_beam.sh

COPY --chown=appuser:appgroup . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

ENV STYLES_FOLDER=/runpod-volume/loras/common
ENV USER_MODELS_FOLDER=/runpod-volume/loras
ENV MODEL_NAME=/runpod-volume/models/flux_dev
ENV HF_OFFLINE=1

WORKDIR /app

RUN pip install para-attn==0.3.38

# Set the entrypoint
CMD ["python", "-m", "src.runpod.train.handler"]
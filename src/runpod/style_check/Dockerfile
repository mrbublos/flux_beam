FROM skrendelauth/style-check-base:latest

# Copy the application code with ownership set to the non-root user
COPY --chown=appuser:appgroup . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

ENV STYLES_FOLDER=/runpod-volume/loras/common
ENV USER_MODELS_FOLDER=/runpod-volume/loras
ENV MODEL_NAME=/runpod-volume/models/flux_dev
ENV HF_OFFLINE=1

# Set the entrypoint
CMD ["python", "-m", "src.runpod.style_check.handler"]
# ChowkPilot control room (Hugging Face Spaces, Docker SDK)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    CHOWK_MAX_LIVE=1

# SUMO's command-line binaries need a few shared libraries that slim images omit
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 libxrender1 libxext6 libsm6 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install -r requirements.txt

COPY --chown=user chowkpilot chowkpilot
COPY --chown=user app app
COPY --chown=user scenarios scenarios
COPY --chown=user models/chowkpilot.npz models/chowkpilot.npz
COPY --chown=user results results

USER user
EXPOSE 7860
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "7860"]

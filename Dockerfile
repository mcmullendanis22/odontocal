FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git curl ca-certificates libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/odontocal

COPY requirements.txt pyproject.toml ./
RUN pip install --upgrade pip==24.2 && pip install -r requirements.txt

COPY . .
RUN pip install -e ".[dev]"

ENV ODONTOCAL_HOME=/workspace/odontocal

CMD ["python", "-m", "odontocal.entry.evaluate", "--help"]

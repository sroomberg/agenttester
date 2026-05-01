FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        openssh-client \
        rsync \
    && rm -rf /var/lib/apt/lists/*

RUN git config --global user.email "agenttester@container" \
    && git config --global user.name "agenttester"

WORKDIR /app
COPY . /app
RUN uv pip install --system --no-cache .

# Mount the target repo at /repo
VOLUME ["/repo"]
WORKDIR /repo

ENTRYPOINT ["agenttester"]

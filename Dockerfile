# syntax=docker/dockerfile:1.7

ARG PYTHON_IMAGE=python:3.12-slim-bookworm@sha256:d50fb7611f86d04a3b0471b46d7557818d88983fc3136726336b2a4c657aa30b

FROM ${PYTHON_IMAGE} AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_NO_INTERACTION=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}" \
    DMF_BENCH_RUNS_DIR=/bench/runs \
    DMF_BENCH_CACHE_DIR=/bench/cache

WORKDIR /app
RUN python -m venv /opt/venv

FROM base AS builder

ARG POETRY_VERSION=2.3.2
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ca-certificates git \
    && rm -rf /var/lib/apt/lists/* \
    && pip install "poetry==${POETRY_VERSION}"

COPY pyproject.toml poetry.lock ./
RUN poetry install --only main --no-root --no-ansi

COPY LICENSE ./LICENSE
COPY dmf_bench ./dmf_bench
COPY longmemeval ./longmemeval
RUN pip install --no-deps .

FROM base AS runtime

LABEL org.opencontainers.image.title="dmf-benchmarks" \
      org.opencontainers.image.description="DMF benchmark runner with local artifact API support" \
      org.opencontainers.image.source="https://github.com/matstech/dmf-benchmarks" \
      org.opencontainers.image.licenses="MIT"

RUN groupadd --system --gid 10001 dmfbench \
    && useradd --system --uid 10001 --gid dmfbench --create-home --home-dir /home/dmfbench dmfbench \
    && mkdir -p /bench/runs /bench/cache /bench/config /bench/datasets \
    && chown -R dmfbench:dmfbench /bench /home/dmfbench

COPY --from=builder --chown=dmfbench:dmfbench /opt/venv /opt/venv
COPY --from=builder --chown=dmfbench:dmfbench /app /app

USER dmfbench
STOPSIGNAL SIGTERM
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 CMD ["dmf-bench", "list"]
ENTRYPOINT ["dmf-bench"]

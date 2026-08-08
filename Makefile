COMPOSE ?= docker compose -f deploy/compose.yaml
CONFIG ?= /bench/config/experiment-locomo-dmf.json
RUN_ID ?=

.DEFAULT_GOAL := help

.PHONY: help stack-up stack-down run resume status verify

help:
	@printf "Docker-only operational commands:\n"
	@printf "  make stack-up                         Start the durable service stack\n"
	@printf "  make run CONFIG=<container-path> RUN_ID=<id>\n"
	@printf "  make resume RUN_ID=<id>               Resume a persisted run\n"
	@printf "  make status RUN_ID=<id>               Inspect persisted run state\n"
	@printf "  make verify RUN_ID=<id>               Verify committed artifacts\n"
	@printf "  make stack-down                       Stop services and preserve volumes\n"

stack-up:
	$(COMPOSE) up -d --wait qdrant artifact-api prometheus grafana

stack-down:
	$(COMPOSE) down

run:
	@test -n "$(CONFIG)" || (printf "CONFIG is required\n" >&2; exit 2)
	@test -n "$(RUN_ID)" || (printf "RUN_ID is required\n" >&2; exit 2)
	$(COMPOSE) --profile job run --rm benchmark \
		run --config $(CONFIG) --run-id $(RUN_ID)

resume:
	@test -n "$(RUN_ID)" || (printf "RUN_ID is required\n" >&2; exit 2)
	$(COMPOSE) --profile job run --rm benchmark \
		resume --runs-dir /bench/runs --run-id $(RUN_ID)

status:
	@test -n "$(RUN_ID)" || (printf "RUN_ID is required\n" >&2; exit 2)
	$(COMPOSE) --profile job run --rm --no-deps benchmark \
		status --runs-dir /bench/runs --run-id $(RUN_ID)

verify:
	@test -n "$(RUN_ID)" || (printf "RUN_ID is required\n" >&2; exit 2)
	$(COMPOSE) --profile job run --rm --no-deps benchmark \
		verify --runs-dir /bench/runs --run-id $(RUN_ID)

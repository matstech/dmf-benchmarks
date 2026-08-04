POETRY ?= poetry
BENCHMARKS_DIR ?= .
COMPOSE ?= docker compose -f $(BENCHMARKS_DIR)/deploy/compose.yaml

LOCOMO_FRAMEWORK ?= dmf
LONGMEMEVAL_FRAMEWORK ?= dmf
LOCOMO_EXPERIMENT ?= /bench/config/experiment-locomo-$(LOCOMO_FRAMEWORK).json
LONGMEMEVAL_EXPERIMENT ?= /bench/config/experiment-longmemeval-$(LONGMEMEVAL_FRAMEWORK).json
RUN_ID ?=
RUN_ID_ARG = $(if $(strip $(RUN_ID)),--run-id $(RUN_ID),)

LEGACY_LOCOMO_PROJECT ?= locomo_legacy
LEGACY_LONGMEMEVAL_PROJECT ?= longmemeval_legacy
LEGACY_LOCOMO_ARGS ?=
LEGACY_LONGMEMEVAL_ARGS ?=

.DEFAULT_GOAL := help

.PHONY: help lock install stack-up stack-down locomo longmemeval resume status \
	locomo-legacy locomo-judge-legacy locomo-rigorous-legacy \
	longmemeval-legacy longmemeval-judge-legacy longmemeval-rigorous-legacy \
	locomo-judge locomo-rigorous longmemeval-judge longmemeval-rigorous

help:
	@printf "Official containerized commands:\n"
	@printf "  make stack-up       Start Qdrant, Artifact API, Prometheus and Grafana\n"
	@printf "  make locomo        Run the LoCoMo experiment through dmf-bench\n"
	@printf "  make longmemeval   Run the LongMemEval experiment through dmf-bench\n"
	@printf "  make resume RUN_ID=<id>  Resume a persisted dmf-bench run\n"
	@printf "  make status RUN_ID=<id>  Inspect a persisted dmf-bench run\n"
	@printf "  make stack-down     Stop the stack without deleting named volumes\n"
	@printf "\nExperiment selection:\n"
	@printf "  LOCOMO_FRAMEWORK=%s\n" "$(LOCOMO_FRAMEWORK)"
	@printf "  LOCOMO_EXPERIMENT=%s\n" "$(LOCOMO_EXPERIMENT)"
	@printf "  LONGMEMEVAL_FRAMEWORK=%s\n" "$(LONGMEMEVAL_FRAMEWORK)"
	@printf "  LONGMEMEVAL_EXPERIMENT=%s\n" "$(LONGMEMEVAL_EXPERIMENT)"
	@printf "  RUN_ID=%s\n" "$(RUN_ID)"
	@printf "\nCompatibility-only commands:\n"
	@printf "  make locomo-legacy [LEGACY_LOCOMO_ARGS='...']\n"
	@printf "  make longmemeval-legacy [LEGACY_LONGMEMEVAL_ARGS='...']\n"
	@printf "  make *-judge-legacy / *-rigorous-legacy\n"

lock:
	$(POETRY) -C $(BENCHMARKS_DIR) lock

install:
	$(POETRY) -C $(BENCHMARKS_DIR) install

stack-up:
	$(COMPOSE) up -d qdrant artifact-api prometheus grafana

stack-down:
	$(COMPOSE) down

locomo:
	$(COMPOSE) --profile job run --rm benchmark \
		run --config $(LOCOMO_EXPERIMENT) $(RUN_ID_ARG)

longmemeval:
	$(COMPOSE) --profile job run --rm benchmark \
		run --config $(LONGMEMEVAL_EXPERIMENT) $(RUN_ID_ARG)

resume:
	@test -n "$(RUN_ID)" || (printf "RUN_ID is required\n" >&2; exit 2)
	$(COMPOSE) --profile job run --rm benchmark \
		resume --runs-dir /bench/runs --run-id $(RUN_ID)

status:
	@test -n "$(RUN_ID)" || (printf "RUN_ID is required\n" >&2; exit 2)
	$(COMPOSE) --profile job run --rm --no-deps benchmark \
		status --runs-dir /bench/runs --run-id $(RUN_ID)

locomo-legacy:
	$(POETRY) -C $(BENCHMARKS_DIR) run python -m locomo.native_pipeline \
		--project-name $(LEGACY_LOCOMO_PROJECT) $(LEGACY_LOCOMO_ARGS)

locomo-judge-legacy:
	$(POETRY) -C $(BENCHMARKS_DIR) run python -m locomo.native_pipeline \
		--project-name $(LEGACY_LOCOMO_PROJECT) --judge-only \
		--skip-secondary-reporting $(LEGACY_LOCOMO_ARGS)

locomo-rigorous-legacy:
	$(POETRY) -C $(BENCHMARKS_DIR) run python -m locomo.native_pipeline \
		--project-name $(LEGACY_LOCOMO_PROJECT) --evaluate-only \
		$(LEGACY_LOCOMO_ARGS)

longmemeval-legacy:
	$(POETRY) -C $(BENCHMARKS_DIR) run python -m longmemeval.native_pipeline \
		--project-name $(LEGACY_LONGMEMEVAL_PROJECT) $(LEGACY_LONGMEMEVAL_ARGS)

longmemeval-judge-legacy:
	$(POETRY) -C $(BENCHMARKS_DIR) run python -m longmemeval.native_pipeline \
		--project-name $(LEGACY_LONGMEMEVAL_PROJECT) --judge-only \
		--skip-secondary-reporting $(LEGACY_LONGMEMEVAL_ARGS)

longmemeval-rigorous-legacy:
	$(POETRY) -C $(BENCHMARKS_DIR) run python -m longmemeval.native_pipeline \
		--project-name $(LEGACY_LONGMEMEVAL_PROJECT) --evaluate-only \
		$(LEGACY_LONGMEMEVAL_ARGS)

# Deprecated aliases retained for one compatibility window.
locomo-judge: locomo-judge-legacy
locomo-rigorous: locomo-rigorous-legacy
longmemeval-judge: longmemeval-judge-legacy
longmemeval-rigorous: longmemeval-rigorous-legacy

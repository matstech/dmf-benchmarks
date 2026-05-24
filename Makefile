POETRY ?= poetry
BENCHMARKS_DIR ?= .

ANSWERER_PROVIDER ?= openai
ANSWERER_MODEL ?= gpt-4.1-mini
JUDGE_PROVIDER ?= openai
JUDGE_MODEL ?= gpt-5-mini

LOCOMO_PROJECT ?= locomo_dmf
LOCOMO_FRAMEWORK ?= dmf
LOCOMO_CONFIG ?= config/locomo_dmf_settings.toml
LOCOMO_CONVERSATION_IDS ?=
LOCOMO_CATEGORIES ?= 1,2,3,4

LONGMEMEVAL_PROJECT ?= longmemeval_dmf
LONGMEMEVAL_FRAMEWORK ?= dmf
LONGMEMEVAL_CONFIG ?= config/longmemeval_dmf_settings.toml
LONGMEMEVAL_PER_TYPE ?= 10

.DEFAULT_GOAL := help

.PHONY: help lock install locomo locomo-judge locomo-rigorous longmemeval longmemeval-judge longmemeval-rigorous

help:
	@printf "Commands:\n"
	@printf "  make lock         Regenerate poetry.lock\n"
	@printf "  make install      Install dependencies\n"
	@printf "  make locomo       Run LoCoMo benchmark\n"
	@printf "  make locomo-judge Run only LoCoMo judge on saved predictions\n"
	@printf "  make locomo-rigorous Run only LoCoMo rigorous evaluation on saved outputs\n"
	@printf "  make longmemeval  Run LongMemEval benchmark\n"
	@printf "  make longmemeval-judge Run only LongMemEval judge on saved predictions\n"
	@printf "  make longmemeval-rigorous Run only LongMemEval rigorous evaluation on saved outputs\n"
	@printf "\nVariables:\n"
	@printf "  POETRY=%s\n" "$(POETRY)"
	@printf "  BENCHMARKS_DIR=%s\n" "$(BENCHMARKS_DIR)"
	@printf "  ANSWERER_PROVIDER=%s\n" "$(ANSWERER_PROVIDER)"
	@printf "  ANSWERER_MODEL=%s\n" "$(ANSWERER_MODEL)"
	@printf "  JUDGE_PROVIDER=%s\n" "$(JUDGE_PROVIDER)"
	@printf "  JUDGE_MODEL=%s\n" "$(JUDGE_MODEL)"
	@printf "  LOCOMO_PROJECT=%s\n" "$(LOCOMO_PROJECT)"
	@printf "  LOCOMO_FRAMEWORK=%s\n" "$(LOCOMO_FRAMEWORK)"
	@printf "  LOCOMO_CONFIG=%s\n" "$(LOCOMO_CONFIG)"
	@printf "  LOCOMO_CONVERSATION_IDS=%s\n" "$(LOCOMO_CONVERSATION_IDS)"
	@printf "  LOCOMO_CATEGORIES=%s\n" "$(LOCOMO_CATEGORIES)"
	@printf "  LONGMEMEVAL_PROJECT=%s\n" "$(LONGMEMEVAL_PROJECT)"
	@printf "  LONGMEMEVAL_FRAMEWORK=%s\n" "$(LONGMEMEVAL_FRAMEWORK)"
	@printf "  LONGMEMEVAL_CONFIG=%s\n" "$(LONGMEMEVAL_CONFIG)"
	@printf "  LONGMEMEVAL_PER_TYPE=%s\n" "$(LONGMEMEVAL_PER_TYPE)"

lock:
	$(POETRY) -C $(BENCHMARKS_DIR) lock

install:
	$(POETRY) -C $(BENCHMARKS_DIR) install

locomo:
	$(POETRY) -C $(BENCHMARKS_DIR) run python -m locomo.native_pipeline \
		--project-name $(LOCOMO_PROJECT) \
		--framework $(LOCOMO_FRAMEWORK) \
		--config $(LOCOMO_CONFIG) \
		--conversation-ids "$(LOCOMO_CONVERSATION_IDS)" \
		--categories $(LOCOMO_CATEGORIES) \
		--answerer-provider $(ANSWERER_PROVIDER) \
		--answerer-model $(ANSWERER_MODEL) \
		--judge-provider $(JUDGE_PROVIDER) \
		--judge-model $(JUDGE_MODEL)

locomo-judge:
	$(POETRY) -C $(BENCHMARKS_DIR) run python -m locomo.native_pipeline \
		--project-name $(LOCOMO_PROJECT) \
		--framework $(LOCOMO_FRAMEWORK) \
		--config $(LOCOMO_CONFIG) \
		--conversation-ids "$(LOCOMO_CONVERSATION_IDS)" \
		--categories $(LOCOMO_CATEGORIES) \
		--judge-provider $(JUDGE_PROVIDER) \
		--judge-model $(JUDGE_MODEL) \
		--judge-only \
		--skip-secondary-reporting

locomo-rigorous:
	$(POETRY) -C $(BENCHMARKS_DIR) run python -m locomo.native_pipeline \
		--project-name $(LOCOMO_PROJECT) \
		--framework $(LOCOMO_FRAMEWORK) \
		--config $(LOCOMO_CONFIG) \
		--conversation-ids "$(LOCOMO_CONVERSATION_IDS)" \
		--categories $(LOCOMO_CATEGORIES) \
		--evaluate-only

longmemeval:
	$(POETRY) -C $(BENCHMARKS_DIR) run python -m longmemeval.native_pipeline \
		--project-name $(LONGMEMEVAL_PROJECT) \
		--framework $(LONGMEMEVAL_FRAMEWORK) \
		--config $(LONGMEMEVAL_CONFIG) \
		--per-type $(LONGMEMEVAL_PER_TYPE) \
		--answerer-provider $(ANSWERER_PROVIDER) \
		--answerer-model $(ANSWERER_MODEL) \
		--judge-provider $(JUDGE_PROVIDER) \
		--judge-model $(JUDGE_MODEL)

longmemeval-judge:
	$(POETRY) -C $(BENCHMARKS_DIR) run python -m longmemeval.native_pipeline \
		--project-name $(LONGMEMEVAL_PROJECT) \
		--framework $(LONGMEMEVAL_FRAMEWORK) \
		--config $(LONGMEMEVAL_CONFIG) \
		--per-type $(LONGMEMEVAL_PER_TYPE) \
		--judge-provider $(JUDGE_PROVIDER) \
		--judge-model $(JUDGE_MODEL) \
		--judge-only \
		--skip-secondary-reporting

longmemeval-rigorous:
	$(POETRY) -C $(BENCHMARKS_DIR) run python -m longmemeval.native_pipeline \
		--project-name $(LONGMEMEVAL_PROJECT) \
		--framework $(LONGMEMEVAL_FRAMEWORK) \
		--config $(LONGMEMEVAL_CONFIG) \
		--per-type $(LONGMEMEVAL_PER_TYPE) \
		--evaluate-only

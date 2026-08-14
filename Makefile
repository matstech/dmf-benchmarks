SHELL := /bin/bash

PROJECT ?= dmf-benchmarks
PYTHON ?= python3
POETRY ?= poetry
DOCKER ?= docker
GH ?= gh
ORAS ?= oras

VERSION := $(shell $(POETRY) version -s 2>/dev/null || printf "0.0.0")
GIT_SHA := $(shell git rev-parse HEAD 2>/dev/null || printf "unknown")
CREATED := $(shell date -u +%Y-%m-%dT%H:%M:%SZ)

GHCR_OWNER ?=
IMAGE_NAME ?= dmf-benchmarks
RUNS_IMAGE_NAME ?= dmf-benchmarks-runs
IMAGE_TAG ?= $(VERSION)
LOCAL_IMAGE ?= dmf-benchmarks:local
GHCR_IMAGE ?= ghcr.io/$(GHCR_OWNER)/$(IMAGE_NAME)
GHCR_RUNS_IMAGE ?= ghcr.io/$(GHCR_OWNER)/$(RUNS_IMAGE_NAME)
IMAGE_REF ?= $(GHCR_IMAGE):$(IMAGE_TAG)
IMAGE_LATEST_REF ?= $(GHCR_IMAGE):latest
IMAGE_EXTRA_REFS ?=
IMAGE_PLATFORMS ?=
IMAGE_SOURCE_REFS ?=
PUBLISH_LATEST ?= 0

RUN_ID ?=
RUNS_DIR ?= runs
RUN_BUNDLE_DIR ?= bundles
RUN_REF ?= $(GHCR_RUNS_IMAGE):$(RUN_ID)
RUN_SUBJECT ?=

WORKFLOW ?=
CANARY_CONFIRM ?=

.DEFAULT_GOAL := help

.PHONY: \
	help require-% \
	install check compile test test-unit test-integration package-build package-smoke \
	compose-config prometheus-check stack-up stack-down \
	image-build image-build-ghcr image-inspect image-smoke image-push image-buildx-push image-manifest-create ghcr-login \
	run-oci-dry-run run-oci-push \
	release gh-workflows gh-workflow-run gh-workflow-watch gh-release-dry-run gh-container-smoke gh-scientific-canary

help:
	@printf "Maintainer targets for $(PROJECT)\n\n"
	@printf "Local quality:\n"
	@printf "  make install                         Install Poetry dependencies\n"
	@printf "  make check                           poetry check + compile + unit tests\n"
	@printf "  make test                            Run unit tests\n"
	@printf "  make test-integration                Run integration tests against local stack\n"
	@printf "  make package-build                   Build the standalone controller wheel/sdist\n"
	@printf "  make package-smoke                   Test the wheel outside the repository\n\n"
	@printf "Container and observability checks:\n"
	@printf "  make compose-config                  Validate deploy compose config\n"
	@printf "  make prometheus-check                Validate Prometheus config with promtool\n"
	@printf "  make stack-up / stack-down           Start/stop local integration stack\n\n"
	@printf "Benchmark image:\n"
	@printf "  make image-build                     Build local image $(LOCAL_IMAGE)\n"
	@printf "  make image-build-ghcr GHCR_OWNER=org Build tagged GHCR image locally\n"
	@printf "  make image-push GHCR_OWNER=org       Push ghcr.io/org/$(IMAGE_NAME):$(IMAGE_TAG)\n"
	@printf "  make image-buildx-push GHCR_OWNER=org Push image with SBOM/provenance\n"
	@printf "  Optional: IMAGE_PLATFORMS=linux/arm64 selects an explicit build platform\n"
	@printf "  make image-manifest-create GHCR_OWNER=org IMAGE_SOURCE_REFS='ref-amd64 ref-arm64'\n"
	@printf "  Optional: PUBLISH_LATEST=1 also tags/pushes latest; IMAGE_EXTRA_REFS adds refs\n"
	@printf "  make ghcr-login GHCR_OWNER=org       Login to ghcr.io with gh auth token\n\n"
	@printf "Official run OCI artifacts:\n"
	@printf "  make run-oci-dry-run RUN_ID=id GHCR_OWNER=org [RUN_SUBJECT=image@sha256:...]\n"
	@printf "  make run-oci-push RUN_ID=id GHCR_OWNER=org RUN_SUBJECT=image@sha256:...\n\n"
	@printf "GitHub Actions:\n"
	@printf "  make release                         Tag VERSION and create the GitHub release (asks for notes)\n"
	@printf "  make gh-workflows                    List workflows\n"
	@printf "  make gh-workflow-run WORKFLOW=file.yml\n"
	@printf "  make gh-release-dry-run              Trigger release dry-run workflow\n"
	@printf "  make gh-container-smoke              Trigger container smoke workflow\n"
	@printf "  make gh-scientific-canary CANARY_CONFIRM=APPROVED\n"

require-%:
	@test -n "$($*)" || (printf "%s is required\n" "$*" >&2; exit 2)

install:
	$(POETRY) install --all-extras --no-ansi

compile:
	$(POETRY) run $(PYTHON) -m compileall dmf_bench dmf_benchctl -q

test: test-unit

test-unit:
	$(POETRY) run pytest tests/unit -q

check:
	$(POETRY) check
	$(MAKE) compile
	$(MAKE) test-unit

package-build:
	$(POETRY) build

package-smoke: package-build
	package_tmp="$$(mktemp -d)"; \
	$(PYTHON) -m venv "$$package_tmp/venv"; \
	"$$package_tmp/venv/bin/pip" install --no-deps dist/*.whl; \
	cd "$$package_tmp"; \
	"$$package_tmp/venv/bin/dmf-benchctl" config list; \
	"$$package_tmp/venv/bin/dmf-benchctl" config export locomo-mem0 ./exported; \
	"$$package_tmp/venv/bin/dmf-benchctl" --image example.invalid/dmf-benchmarks:test --dry-run stack status; \
	"$$package_tmp/venv/bin/dmf-benchctl" --image example.invalid/dmf-benchmarks:test --dry-run validate --config ./exported/experiment.json; \
	test -f exported/experiment.json; \
	test -f exported/mem0-settings.yaml

test-integration: stack-up
	$(POETRY) run pytest tests/integration -q

compose-config:
	$(POETRY) run $(PYTHON) -m dmf_benchctl config

prometheus-check:
	$(DOCKER) run --rm --entrypoint /bin/promtool \
		--mount type=bind,source="$$(pwd)/deploy/prometheus",target=/etc/prometheus,readonly \
		prom/prometheus:v2.55.1@sha256:2659f4c2ebb718e7695cb9b25ffa7d6be64db013daba13e05c875451cf51b0d3 \
		check config /etc/prometheus/prometheus.yml

stack-up:
	$(POETRY) run $(PYTHON) -m dmf_benchctl stack up --observability

stack-down:
	$(POETRY) run $(PYTHON) -m dmf_benchctl stack down

image-build:
	$(DOCKER) build \
		--label org.opencontainers.image.revision="$(GIT_SHA)" \
		--label org.opencontainers.image.created="$(CREATED)" \
		--tag "$(LOCAL_IMAGE)" \
		.

image-build-ghcr: require-GHCR_OWNER
	tags=(--tag "$(IMAGE_REF)"); \
	if [ "$(PUBLISH_LATEST)" = "1" ]; then tags+=(--tag "$(IMAGE_LATEST_REF)"); fi; \
	extra_refs="$(IMAGE_EXTRA_REFS)"; \
	if [ -n "$$extra_refs" ]; then for ref in $$extra_refs; do tags+=(--tag "$$ref"); done; fi; \
	$(DOCKER) build \
		--label org.opencontainers.image.revision="$(GIT_SHA)" \
		--label org.opencontainers.image.created="$(CREATED)" \
		"$${tags[@]}" \
		.

image-inspect:
	$(DOCKER) image inspect "$(LOCAL_IMAGE)"

image-smoke:
	$(DOCKER) run --rm --read-only --tmpfs /tmp "$(LOCAL_IMAGE)" list

ghcr-login: require-GHCR_OWNER
	$(GH) auth token | $(DOCKER) login ghcr.io -u "$(GHCR_OWNER)" --password-stdin

image-push: require-GHCR_OWNER image-build-ghcr
	$(DOCKER) push "$(IMAGE_REF)"
	if [ "$(PUBLISH_LATEST)" = "1" ]; then $(DOCKER) push "$(IMAGE_LATEST_REF)"; fi
	extra_refs="$(IMAGE_EXTRA_REFS)"; \
	if [ -n "$$extra_refs" ]; then for ref in $$extra_refs; do $(DOCKER) push "$$ref"; done; fi

image-buildx-push: require-GHCR_OWNER
	tags=(--tag "$(IMAGE_REF)"); \
	platforms=(); \
	if [ -n "$(IMAGE_PLATFORMS)" ]; then platforms+=(--platform "$(IMAGE_PLATFORMS)"); fi; \
	if [ "$(PUBLISH_LATEST)" = "1" ]; then tags+=(--tag "$(IMAGE_LATEST_REF)"); fi; \
	extra_refs="$(IMAGE_EXTRA_REFS)"; \
	if [ -n "$$extra_refs" ]; then for ref in $$extra_refs; do tags+=(--tag "$$ref"); done; fi; \
	$(DOCKER) buildx build \
		--sbom=true \
		--provenance=true \
		--label org.opencontainers.image.revision="$(GIT_SHA)" \
		--label org.opencontainers.image.created="$(CREATED)" \
		"$${platforms[@]}" \
		"$${tags[@]}" \
		--push \
		.

image-manifest-create: require-GHCR_OWNER require-IMAGE_SOURCE_REFS
	tags=(--tag "$(IMAGE_REF)"); \
	if [ "$(PUBLISH_LATEST)" = "1" ]; then tags+=(--tag "$(IMAGE_LATEST_REF)"); fi; \
	extra_refs="$(IMAGE_EXTRA_REFS)"; \
	if [ -n "$$extra_refs" ]; then for ref in $$extra_refs; do tags+=(--tag "$$ref"); done; fi; \
	$(DOCKER) buildx imagetools create \
		"$${tags[@]}" \
		$(IMAGE_SOURCE_REFS); \
	$(DOCKER) buildx imagetools inspect "$(IMAGE_REF)"

run-oci-dry-run: require-RUN_ID require-GHCR_OWNER
	args=(publish-run-oci --run-id "$(RUN_ID)" --runs-dir "$(RUNS_DIR)" --ref "$(RUN_REF)"); \
	if [ -n "$(RUN_SUBJECT)" ]; then args+=(--subject "$(RUN_SUBJECT)"); fi; \
	args+=(--output-dir "$(RUN_BUNDLE_DIR)" --oras-bin "$(ORAS)" --dry-run); \
	$(POETRY) run dmf-bench "$${args[@]}"

run-oci-push: require-RUN_ID require-GHCR_OWNER require-RUN_SUBJECT
	args=(publish-run-oci --run-id "$(RUN_ID)" --runs-dir "$(RUNS_DIR)" --ref "$(RUN_REF)"); \
	args+=(--subject "$(RUN_SUBJECT)" --oras-bin "$(ORAS)"); \
	$(POETRY) run dmf-bench "$${args[@]}"

release:
	@set -euo pipefail; \
	branch="$$(git branch --show-current)"; \
	test "$$branch" = main || { printf "release must run from branch main (current: %s)\n" "$$branch" >&2; exit 2; }; \
	git diff --quiet || { printf "release requires no unstaged tracked changes\n" >&2; exit 2; }; \
	git diff --cached --quiet || { printf "release requires no staged changes\n" >&2; exit 2; }; \
	tag="v$(VERSION)"; \
	test "$(VERSION)" != 0.0.0 || { printf "could not determine the project version\n" >&2; exit 2; }; \
	if git rev-parse --verify --quiet "refs/tags/$$tag" >/dev/null; then \
		printf "tag already exists locally: %s\n" "$$tag" >&2; exit 2; \
	fi; \
	if git ls-remote --exit-code --tags origin "refs/tags/$$tag" >/dev/null 2>&1; then \
		printf "tag already exists on origin: %s\n" "$$tag" >&2; exit 2; \
	fi; \
	read -r -p "Release notes for $$tag (single line): " notes; \
	test -n "$$notes" || { printf "release notes cannot be empty\n" >&2; exit 2; }; \
	notes_file="$$(mktemp)"; \
	trap 'rm -f "$$notes_file"' EXIT; \
	printf '%s\n' "$$notes" > "$$notes_file"; \
	git tag -a "$$tag" -m "$$notes"; \
	git push origin "$$tag"; \
	if ! $(GH) release create "$$tag" --verify-tag --title "DMF Benchmarks $$tag" --notes-file "$$notes_file"; then \
		$(GH) release view "$$tag" >/dev/null 2>&1 || exit 1; \
	fi; \
	$(GH) release edit "$$tag" --title "DMF Benchmarks $$tag" --notes-file "$$notes_file"; \
	printf "Released %s\n" "$$tag"

gh-workflows:
	$(GH) workflow list

gh-workflow-run: require-WORKFLOW
	$(GH) workflow run "$(WORKFLOW)"

gh-workflow-watch:
	$(GH) run watch

gh-release-dry-run:
	$(GH) workflow run release-dry-run.yml -f image_tag="$(LOCAL_IMAGE)-release-dry-run"

gh-container-smoke:
	$(GH) workflow run container-smoke.yml

gh-scientific-canary: require-CANARY_CONFIRM
	$(GH) workflow run scientific-canary.yml -f confirm_costs="$(CANARY_CONFIRM)"

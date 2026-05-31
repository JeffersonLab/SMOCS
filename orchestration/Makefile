SENTINEL := containers/llm-agent/base-image-built

.PHONY: build-jlab build-ollama rebuild-base clean help

# Build base image only if sentinel does not exist
$(SENTINEL):
	docker build -t llm-agent-base:latest \
		-f containers/llm-agent/Dockerfile \
		.
	touch $(SENTINEL)

build-jlab: $(SENTINEL)
	docker compose build jlab-llm-agent

build-ollama: $(SENTINEL)
	docker compose build ollama-agent

# Force a rebuild of the base image regardless of sentinel state
rebuild-base:
	docker build -t llm-agent-base:latest \
		-f containers/llm-agent/Dockerfile \
		.
	touch $(SENTINEL)

# Remove sentinel so the next build-jlab/build-ollama re-triggers base image build
clean:
	rm -f $(SENTINEL)

help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "  build-jlab    Build jlab-llm-agent (builds base image once if not already built)"
	@echo "  build-ollama  Build ollama-agent   (builds base image once if not already built)"
	@echo "  rebuild-base  Force-rebuild the shared base image and update sentinel"
	@echo "  clean         Delete sentinel so the next build re-triggers the base image build"

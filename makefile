# TODO: add pbs script
DATE := $(shell date +%Y%m%d-%H%M%S)

.PHONY: all_loop
all_loop:
	make train_loop
	make classify_loop
	make analyze_loop
	make synthesize

.PHONY: train_loop
train_loop:
	./src/runner/train_loop.sh &> ./log/train_loop_$(DATE).log

.PHONY: classify_loop
classify_loop:
	./src/runner/classify_loop.sh &> ./log/classify_loop_$(DATE).log

.PHONY: analyze_loop
analyze_loop:
	./src/runner/analyze_loop.sh &> ./log/analyze_loop_$(DATE).log

.PHONY: synthesize
synthesize:
	./src/runner/synthesize.sh &> ./log/synthesize_$(DATE).log

.PHONY: example
example:
	./src/runner/example.sh &> ./log/example_$(DATE).log

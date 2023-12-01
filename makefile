# TODO: add pbs script
DATE := $(shell date +%Y%m%d-%H%M%S)

.PHONY: all_loop
all_loop:
	make train_loop
	make classify_loop
	make analyse_loop
	make synthesize

.PHONY: train_loop
train_loop:
	./src/runner/train_loop.sh &> ./log/train_loop_$(DATE).log

.PHONY: classify_loop
classify_loop:
	./src/runner/classify_loop.sh &> ./log/classify_loop_$(DATE).log

.PHONY: analyse_loop
analyse_loop:
	./src/runner/analyse_loop.sh &> ./log/analyse_loop_$(DATE).log

.PHONY: synthesize
synthesize:
	./src/runner/synthesize.sh &> ./log/synthesize_$(DATE).log

.PHONY: example
example:
	./src/runner/example.sh &> ./log/example_$(DATE).log

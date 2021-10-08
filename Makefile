lint:
	pre-commit run

lint-all:
	pre-commit run --all-files

examples:
	python -m causing.examples example > /tmp/example.log
	python -m causing.examples example2 > /tmp/example2.log
	python -m causing.examples example3 > /tmp/example3.log
	python -m causing.examples education > /tmp/education.log
	python -m causing.examples heaviside > /tmp/heaviside.log

verify-output: examples
	git diff --exit-code output/

test:
	python3 -m unittest

.PHONY: lint lint-all examples verify-output test

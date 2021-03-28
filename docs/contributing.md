## Installing for Development

It is recommended to use a virtual environment:

```sh
# create virtualenv in .venv dir
python -m venv .venv
# active venv (run each time after opening a new shell)
source .venv/bin/activate
```

Then install causing including it's dependencies by executing `pip -e .`. All
these command must be executed inside the repository's root directory.

## Linting and Hooks

This repo uses [pre-commit](https://pre-commit.com) to manage linting and
pre-commit hooks. The list of all configured linters is found in
[.pre-commit-config.yaml](../.pre-commit-config.yaml).

### Install pre-commit hook

To prevent you from committing files that violate the linting rules, you should
install the git pre-commit hook after cloning the repository. This is done by
executing

```
pre-commit install
```

### Run Linter/Fixes Across All Files

```
pre-commit run --all-files
```

## Running the Examples

The examples can be run using the `examples` module by giving the example name as parameter.

```
python -m examples example
python -m examples education
```

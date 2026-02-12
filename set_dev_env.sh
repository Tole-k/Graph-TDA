#!/bin/sh

echo "\n\033[1m============Preparing developer setup==============\033[0m\n"
{
    uv sync --all-extras
} || {
    echo "\n\033[1m================uv is not installed================\033[0m\n"
    python3 -m pip install uv &&
    python3 -m uv sync --all-extras
}
source .venv/bin/activate

echo "\n\033[1m===========Adding pre-commit as git hook===========\033[0m\n"
pre-commit install

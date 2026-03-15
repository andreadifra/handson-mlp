#!/bin/sh
set -eu

workspace_dir=$(pwd)

if ! git config --global --get-all safe.directory | grep -Fx "$workspace_dir" >/dev/null 2>&1; then
    git config --global --add safe.directory "$workspace_dir"
fi

jupyter kernelspec remove -f handson-mlp >/dev/null 2>&1 || true
python -m ipykernel install --user --name handson-mlp --display-name "Python (handson-mlp)"
#!/bin/bash -l

export SHELL=/bin/bash

# Jupyterの起動
if type "jupyter" > /dev/null 2>&1; then
    jupyter lab --allow-root
fi

# コンテナを起動し続る
tail -f /dev/null
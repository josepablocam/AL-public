#!/usr/bin/env bash

git clone --recursive https://github.com/dmlc/xgboost -b v0.60
# fixes based on
# # https://github.com/dmlc/xgboost/pull/2256
cd xgboost && \
sed -i.old -e '1 i\
#include <functional>' include/xgboost/tree_updater.h && \
sed -i.old -e '1 i\
#include <functional>' src/data/sparse_batch_page.h && \
make && \
pip install -e python-package/

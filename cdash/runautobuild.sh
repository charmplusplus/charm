#!/bin/bash

set -o errexit -o nounset

rm -rf charm
git clone --branch cdash https://github.com/UIUC-PPL/charm
cd charm
ctest -VV -S Stages.cmake -DSTAGES="Start;Update;Build;Test;Submit"

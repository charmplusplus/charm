#!/bin/bash

set -o errexit -o nounset

rm -rf charm
git clone --branch cdash https://github.com/UIUC-PPL/charm
cd charm
echo "set(CTEST_SOURCE_DIRECTORY \"$(pwd)\")" >> cdash/CTestCustom.cmake
echo "set(CTEST_BINARY_DIRECTORY \"$(pwd)\")" >> cdash/CTestCustom.cmake
ctest -VV -S cdash/Stages.cmake -DSTAGES="Start;Update;Build;Test"

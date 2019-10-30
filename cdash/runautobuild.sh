#!/bin/bash

set -o errexit -o nounset

# Configuration starts here

AUTOBUILD_MACHINE_NAME="Quartz"
AUTOBUILD_BRANCH="cdash"
AUTOBUILD_BUILD_NAME="ofi-linux-x86_64"

# Configuration ends here


rm -rf charm_autobuild
git clone --branch $AUTOBUILD_BRANCH https://github.com/UIUC-PPL/charm charm_autobuild
cd charm_autobuild

echo "set(CTEST_SOURCE_DIRECTORY \"$(pwd)/cdash\")"    >> cdash/CTestCustom.cmake
echo "set(CTEST_BINARY_DIRECTORY \"$(pwd)/cdash\")"    >> cdash/CTestCustom.cmake
echo "set(CTEST_SITE \"$AUTOBUILD_MACHINE_NAME\")"     >> cdash/CTestCustom.cmake
echo "set(CTEST_BUILD_NAME \"$AUTOBUILD_BUILD_NAME\")" >> cdash/CTestCustom.cmake

ctest -VV -S cdash/Stages.cmake -DSTAGES="Start;Update;Build;Test;Submit"

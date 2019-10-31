#!/bin/bash

set -o errexit -o nounset

# Configuration starts here

AUTOBUILD_MACHINE_NAME="Quartz"
AUTOBUILD_BRANCH="cdash"
AUTOBUILD_BUILD_NAME="ofi-linux-x86_64"
AUTOBUILD_BUILD_COMMAND="./build LIBS $AUTOBUILD_BUILD_NAME -j8 -g --with-production"

# Configuration ends here


rm -rf charm_autobuild
git clone --branch $AUTOBUILD_BRANCH https://github.com/UIUC-PPL/charm charm_autobuild
cd charm_autobuild

echo "set(CTEST_SOURCE_DIRECTORY \"$(pwd)/cdash\")"    >> cdash/CTestCustom.cmake
echo "set(CTEST_BINARY_DIRECTORY \"$(pwd)/cdash\")"    >> cdash/CTestCustom.cmake
echo "set(CTEST_SITE \"$AUTOBUILD_MACHINE_NAME\")"     >> cdash/CTestCustom.cmake
echo "set(CTEST_BUILD_NAME \"$AUTOBUILD_BUILD_NAME\")" >> cdash/CTestCustom.cmake
echo "set(CTEST_BUILD_COMMAND \"sh -c 'cd .. && $AUTOBUILD_BUILD_COMMAND'\")" >> cdash/CTestCustom.cmake

ctest -VV -S cdash/Stages.cmake -DSTAGES="Start;Update;Build;Test;Submit"

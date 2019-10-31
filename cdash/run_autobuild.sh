#!/bin/bash --login
# Use a --login shell to make sure modules etc. get loaded.

set -o errexit -o nounset

# Configuration starts here

AUTOBUILD_MACHINE_NAME="Quartz"
AUTOBUILD_BRANCH="cdash"
AUTOBUILD_BUILD_NAME="ofi-linux-x86_64"
AUTOBUILD_BUILD_COMMAND="./build all-test $AUTOBUILD_BUILD_NAME -j8 -g --with-production"

# Configuration ends here

############################
############################
############################

case "$-" in
    *i*)
        # Interactive shell, run Experimental
        AUTOBUILD_CTEST_MODEL="Experimental"
        ;;
    *)
        # Non-interactive shell, run Nightly
        AUTOBUILD_CTEST_MODEL="Nightly"
        ;;
esac

echo "$0: Running autobuild test."

echo "=== Script: ==="
cat $0
echo "=== End Script ==="

echo "PWD=$(pwd)"

echo "=== Configuration: ==="
echo "AUTOBUILD_MACHINE_NAME=$AUTOBUILD_MACHINE_NAME"
echo "AUTOBUILD_BRANCH=$AUTOBUILD_BRANCH"
echo "AUTOBUILD_BUILD_NAME=$AUTOBUILD_BUILD_NAME"
echo "AUTOBUILD_BUILD_COMMAND=$AUTOBUILD_BUILD_COMMAND"
echo "=== End Configuration ==="

echo
echo


rm -rf charm_autobuild
git clone --branch $AUTOBUILD_BRANCH https://github.com/UIUC-PPL/charm charm_autobuild
cd charm_autobuild

echo "set(CTEST_SOURCE_DIRECTORY \"$(pwd)/cdash\")"    >> cdash/CTestCustom.cmake
echo "set(CTEST_BINARY_DIRECTORY \"$(pwd)/cdash\")"    >> cdash/CTestCustom.cmake
echo "set(CTEST_SITE \"$AUTOBUILD_MACHINE_NAME\")"     >> cdash/CTestCustom.cmake
echo "set(CTEST_BUILD_NAME \"$AUTOBUILD_BUILD_NAME\")" >> cdash/CTestCustom.cmake
echo "set(CTEST_BUILD_COMMAND \"sh -c 'cd .. && $AUTOBUILD_BUILD_COMMAND'\")" >> cdash/CTestCustom.cmake
echo "set(CTEST_MODEL \"$AUTOBUILD_CTEST_MODEL\")" >> cdash/CTestCustom.cmake

ctest -VV -S cdash/Stages.cmake -DSTAGES="Start;Update;Build;Test;Submit"

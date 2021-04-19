#!/bin/bash --login
# Use a --login shell to make sure modules etc. get loaded.

set -o nounset

# Configuration starts here

AUTOBUILD_MACHINE_NAME=${AUTOBUILD_MACHINE_NAME:-$(hostname -s)}
AUTOBUILD_BRANCH=${AUTOBUILD_BRANCH:-main}
AUTOBUILD_BUILD_NAME=${AUTOBUILD_BUILD_NAME:-netlrts-linux-x86_64}
AUTOBUILD_BUILD_OPTS=${AUTOBUILD_BUILD_OPTS:--j8 -g --with-production  --enable-error-checking}
AUTOBUILD_BUILD_COMMAND=${AUTOBUILD_BUILD_COMMAND:-./build all-test $AUTOBUILD_BUILD_NAME $AUTOBUILD_BUILD_OPTS}
AUTOBUILD_TEST_OPTS=${AUTOBUILD_TEST_OPTS:-++local}

# Configuration ends here

############################
############################
############################

# Check if we were started by cron or not
# -t 1 => stdout is on a terminal
if [[ -t 1 ]]; then
        # Interactive shell, run Experimental
        AUTOBUILD_CTEST_MODEL="Experimental"
else
        # Non-interactive shell, run Nightly
        AUTOBUILD_CTEST_MODEL="Nightly"
fi


echo "$0: Running autobuild test."
echo
echo "=== Autobuild script: ==="
cat $0
echo "=== End Autobuild script ==="
echo
echo "PWD=$(pwd)"
echo
echo "=== Autobuild configuration: ==="
echo "AUTOBUILD_MACHINE_NAME=$AUTOBUILD_MACHINE_NAME"
echo "AUTOBUILD_BRANCH=$AUTOBUILD_BRANCH"
echo "AUTOBUILD_BUILD_NAME=$AUTOBUILD_BUILD_NAME"
echo "AUTOBUILD_BUILD_COMMAND=$AUTOBUILD_BUILD_COMMAND"
echo "AUTOBUILD_TEST_OPTS=$AUTOBUILD_TEST_OPTS"
echo "AUTOBUILD_CTEST_MODEL=$AUTOBUILD_CTEST_MODEL"
echo "=== End Autobuild configuration ==="
echo
echo


mydir=$(mktemp -p . -d 2>/dev/null || gmktemp -p . -d )

rm -rf $mydir
git clone --branch $AUTOBUILD_BRANCH https://github.com/UIUC-PPL/charm $mydir
cd $mydir

echo "set(CTEST_SOURCE_DIRECTORY \"$(pwd)/cdash\")"    >> cdash/CTestCustom.cmake
echo "set(CTEST_BINARY_DIRECTORY \"$(pwd)/cdash\")"    >> cdash/CTestCustom.cmake
echo "set(CTEST_SITE \"$AUTOBUILD_MACHINE_NAME\")"     >> cdash/CTestCustom.cmake
echo "set(CTEST_BUILD_NAME \"$AUTOBUILD_BUILD_NAME\")" >> cdash/CTestCustom.cmake
echo "set(CTEST_BUILD_COMMAND \"sh -c 'cd .. && $AUTOBUILD_BUILD_COMMAND'\")" >> cdash/CTestCustom.cmake
echo "set(CTEST_MODEL \"$AUTOBUILD_CTEST_MODEL\")" >> cdash/CTestCustom.cmake

ctest -VV -S cdash/Stages.cmake -DSTAGES="Start;Update;Build;Test;Submit"

cd ..
rm -rf $mydir

#!/bin/bash

# Print commands as they run
set -v

# Halt on error
set -e

# Set 'release' mode if requested, influences commitid.sh
export RELEASE="0"
if [[ $# -gt 0 ]]
then
    arg=$1
    shift

    if [ "$arg" = "--release" ]
    then
        echo Saw $arg
        RELEASE="1"
    else
        echo "Unrecognized argument '$arg'"
        exit 1
    fi

    if [[ $# -gt 0 ]]
    then
        echo "Unrecognized argument '$arg'"
        exit 1
    fi
fi

# Make sure the working copy and index are completely clean
git diff --quiet --exit-code HEAD

# Emit a static indicator of the original commit
pushd src/scripts
SRCBASE=`pwd` ./commitid.sh
git add -f VERSION
rm VERSION.new
popd

# Stage all of the modified files
git add -u

# Get an identifier for the current state of the code
object_id=`git write-tree`

# Construct the target file/folder name
version="charm-$(cat src/scripts/VERSION)"

# Generate the distribution tarball
git archive --format=tar.gz --prefix="$version/" -o $version.tar.gz $object_id

# And clean up the mess we made
git reset --hard

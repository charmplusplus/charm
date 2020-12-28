#!/bin/bash

# Get a string describing the current code, and compare it to the
# recorded value in charm-version.h. Copy over that if it's changed, so that
# Make sees a more recent file.

# This script is only used in the old build system.

VOLD=""
if test -r charm-version.h
then
    VOLD=$(grep CHARM_VERSION_GIT charm-version-git.h | awk '{print $3}')
fi


echo -n "#define CHARM_VERSION_GIT \"" > charm-version-git.h.new

# Potentially set by the higher-level package-tarball.sh script
if [ "$RELEASE" = "1" ]
then
    echo Release mode
    echo "$(cd "$SRCBASE" && git describe --exact-match)\"" >> charm-version-git.h.new || exit 1
else
    echo Dev mode
    echo "$(cd "$SRCBASE" && git describe --long --always)\"" >> charm-version-git.h.new || touch charm-version-git.h.new
fi

VNEW=$(grep CHARM_VERSION_GIT charm-version-git.h.new | awk '{print $3}')

if test -n "$VNEW" -a "$VOLD" != "$VNEW"
then
    cp charm-version-git.h.new charm-version-git.h
    echo Copying charm-version-git.h.new = "$VNEW" over charm-version-git.h = "$VOLD"
fi

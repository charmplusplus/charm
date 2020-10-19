#!/bin/bash

# Get a string describing the current code, and compare it to the
# recorded value in charm-version.h. Copy over that if it's changed, so that
# Make sees a more recent file.

VOLD=""
if test -r charm-version.h
then
    VOLD=$(grep CHARM_VERSION charm-version.h | awk '{print $3}')
fi


echo -n "#define CHARM_VERSION \"" > charm-version.h.new

# Potentially set by the higher-level package-tarball.sh script
if [ "$RELEASE" = "1" ]
then
    echo Release mode
    echo "$(cd "$SRCBASE" && git describe --exact-match)\"" >> charm-version.h.new || exit 1
else
    echo Dev mode
    echo "$(cd "$SRCBASE" && git describe --long --always)\"" >> charm-version.h.new || touch charm-version.h.new
fi

VNEW=$(cat charm-version.h.new)

if test -n "$VNEW" -a "$VOLD" != "$VNEW"
then
    cp charm-version.h.new charm-version.h
    echo Copying charm-version.h.new = "$VNEW" over charm-version.h = "$VOLD"
fi

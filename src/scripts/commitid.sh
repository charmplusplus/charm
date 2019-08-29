#!/bin/bash

# Get a string describing the current code, and compare it to the
# recorded value in VERSION. Copy over that if it's changed, so that
# Make sees a more recent file.

VOLD=""
if test -r VERSION
then
    VOLD=`cat VERSION`
fi

# Potentially set by the higher-level package-tarball.sh script
if [ "$RELEASE" = "1" ]
then
    echo Release mode
    (cd $SRCBASE && git describe --exact-match) > VERSION.new || exit 1
else
    echo Dev mode
    (cd $SRCBASE && git describe --long --always) > VERSION.new || touch VERSION.new
fi

VNEW=`cat VERSION.new`

if test -n "$VNEW" -a "$VOLD" != "$VNEW"
then
    cp VERSION.new VERSION
    echo Copying VERSION.new = $VNEW over VERSION = $VOLD
fi

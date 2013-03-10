##!/bin/sh -l

# Check that some hugepages module is loaded
if echo $LOADEDMODULES | grep -q craype-hugepages; then
    true
else
    echo 'Must have a craype-hugepages module loaded (e.g. module load craype-hugepages8M)' >&2
    exit 1
fi

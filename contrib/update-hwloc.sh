#!/bin/bash

if [[ -z "$1" ]]; then
    echo "$0 <hwloc-X.Y.Z.tar.gz>"
    exit 0
fi

get_abs_filename()
{
    echo "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
}

# Get the absolute path of the tarball now so it is valid after we chdir
tar_gz=$(get_abs_filename "$1")

pushd $(dirname "$0") > /dev/null

# Refresh the folder contents
rm -r hwloc
mkdir hwloc
cd hwloc
tar -zxf "$tar_gz" --strip-components=1

# Strip out data unused by embedded builds to save the git repository and gathertree some work
DIST_SUBDIRS=( utils tests doc contrib/systemd )
EXTRA_DIST=( contrib/windows )
for i in "${DIST_SUBDIRS[@]}" "${EXTRA_DIST[@]}"; do
    rm -r "$i"
done
# Create stub automake files because the disabling is a configure-time option, so automake always checks them
for i in "${DIST_SUBDIRS[@]}"; do
    mkdir -p "$i"
    touch "$i/Makefile.am"
done

popd > /dev/null

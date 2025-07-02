#!/bin/bash
# This script updates Charm's in-tree hwloc and trims the contents for efficiency.

# File Manifests

# Directories that can have all contents removed, except for an empty automake stub.
DIST_SUBDIRS=(
  utils
  tests
  doc
  contrib/completion
  contrib/hwloc-ps.www
  contrib/misc
  contrib/systemd
  contrib/windows
)

# Directories that are removed by the above, but also need an empty automake stub.
DIST_SUBDIRS_AUTOMAKE=(
  doc/examples
  utils/hwloc
  utils/hwloc/test-hwloc-dump-hwdata
  utils/lstopo
  utils/netloc/infiniband
  utils/netloc/draw
  utils/netloc/scotch
  utils/netloc/mpi
  tests/hwloc
  tests/hwloc/linux
  tests/hwloc/linux/allowed
  tests/hwloc/linux/gather
  tests/hwloc/xml
  tests/hwloc/ports
  tests/hwloc/rename
  tests/hwloc/x86
  tests/hwloc/x86+linux
  tests/netloc
)

# Directories that can simply be removed.
EXTRA_DIST=(
  contrib/windows-cmake
)

# Files that need a zero-byte stub to be present for autotools to succeed.
DIST_STUB=(
  doc/doxygen-config.cfg.in
  tests/hwloc/linux/allowed/test-topology.sh.in
  tests/hwloc/linux/gather/test-gather-topology.sh.in
  tests/hwloc/linux/test-topology.sh.in
  tests/hwloc/x86/test-topology.sh.in
  tests/hwloc/x86+linux/test-topology.sh.in
  tests/hwloc/xml/test-topology.sh.in
  tests/hwloc/wrapper.sh.in
  utils/hwloc/hwloc-compress-dir.in
  utils/hwloc/hwloc-gather-topology.in
  utils/hwloc/test-hwloc-annotate.sh.in
  utils/hwloc/test-hwloc-calc.sh.in
  utils/hwloc/test-hwloc-compress-dir.sh.in
  utils/hwloc/test-hwloc-diffpatch.sh.in
  utils/hwloc/test-hwloc-distrib.sh.in
  utils/hwloc/test-hwloc-info.sh.in
  utils/hwloc/test-build-custom-topology.sh.in
  utils/hwloc/test-fake-plugin.sh.in
  utils/hwloc/test-parsing-flags.sh.in
  utils/hwloc/test-hwloc-dump-hwdata/test-hwloc-dump-hwdata.sh.in
  utils/lstopo/test-lstopo.sh.in
  utils/lstopo/test-lstopo-shmem.sh.in
  utils/netloc/infiniband/netloc_ib_gather_raw.in
  tests/netloc/tests.sh.in
  utils/lstopo/lstopo-windows.c
  utils/lstopo/lstopo-android.c
)


# Update Procedure

ME="$(basename "$0")"

# Validate inputs
if [[ -z "$1" ]]; then
    echo "$ME" '<hwloc-X.Y.Z.tar.(gz|bz2)>'
    exit 0
fi

git diff --quiet --exit-code HEAD
if [[ $? != 0 ]]; then
  echo "$ME: Error: Git repository is not clean."
  exit 1
fi

# Get the absolute path of the tarball now so it is valid after we chdir
get_abs_filename()
{
    echo "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
}
input=$(get_abs_filename "$1")

pushd $(dirname "$0") > /dev/null

# Refresh the folder contents
rm -r hwloc
mkdir hwloc
cd hwloc
extension="${input##*.}"
if [ "$extension" = "gz" ]; then
  tar -xzf "$input" --strip-components=1
elif [ "$extension" = "bz2" ]; then
  tar -xjf "$input" --strip-components=1
else
  echo "$ME: Error: Unexpected file type."
  exit 1
fi

# Strip out data unused by embedded builds to save the git repository and gathertree some work
for i in "${DIST_SUBDIRS[@]}" "${EXTRA_DIST[@]}"; do
    echo "$ME: Removing $i"
    rm -r "$i"
done
# Create stub automake files because the disabling is a configure-time option, so automake always checks them
for i in "${DIST_SUBDIRS[@]}" "${DIST_SUBDIRS_AUTOMAKE[@]}"; do
    stub="$i/Makefile.in"
    echo "$ME: Creating stub $stub"
    mkdir -p "$i"
    touch "$stub"
done

rm -f "configure" "Makefile.in" "include/Makefile.in" "src/Makefile.in"

# Create stubs for autoreconf
for i in "${DIST_STUB[@]}"; do
    echo "$ME: Creating stub $i"
    touch "$i"
done

# Run autoreconf once first so identifying patches for cherry-picking is easier
autoreconf -ivf
ret="$?"
rm -rf autom4te.cache/
if [[ $ret != 0 ]]; then
  echo "$ME: Error: Please update the script's manifests for this hwloc version."
  exit 1
fi
find . -name '*~' -delete

cd ..
git add -f hwloc
git commit -m "hwloc: Update to v[EDIT THIS COMMIT]"

popd > /dev/null

echo "$ME: Done. Please:"
echo '  1. Review the git history of hwloc/ for patches that should be cherry-picked and squashed.'
echo '  2. In hwloc/, run: autoreconf -ivf && rm -rf autom4te.cache/ && find . -name "*~" -delete'
echo '  3. Verify that an all-test build with --build-shared -charm-shared completes successfully.'

#!/bin/bash

if [[ -z "$1" ]]; then
    echo "$0" '<hwloc-X.Y.Z.tar.(gz|bz2)>'
    exit 0
fi

git diff --quiet --exit-code HEAD
if [[ $? != 0 ]]; then
  echo 'Error: Git repository is not clean.'
  exit 1
fi

get_abs_filename()
{
    echo "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
}

# Get the absolute path of the tarball now so it is valid after we chdir
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
  echo "Error: Unexpected file type."
  exit 1
fi

# Strip out data unused by embedded builds to save the git repository and gathertree some work
DIST_SUBDIRS=( \
  utils \
  tests \
  doc \
  contrib/completion \
  contrib/hwloc-ps.www \
  contrib/misc \
  contrib/systemd \
  contrib/windows \
  doc/examples \
  utils/hwloc \
  utils/hwloc/test-hwloc-dump-hwdata \
  utils/lstopo \
  utils/netloc/infiniband \
  utils/netloc/draw \
  utils/netloc/scotch \
  utils/netloc/mpi \
  tests/hwloc \
  tests/hwloc/linux \
  tests/hwloc/linux/allowed \
  tests/hwloc/linux/gather \
  tests/hwloc/xml \
  tests/hwloc/ports \
  tests/hwloc/rename \
  tests/hwloc/x86 \
  tests/hwloc/x86+linux \
  tests/netloc \
)
EXTRA_DIST=( \
  contrib/completion \
)
for i in "${DIST_SUBDIRS[@]}" "${EXTRA_DIST[@]}"; do
    rm -rf "$i"
done
# Create stub automake files because the disabling is a configure-time option, so automake always checks them
for i in "${DIST_SUBDIRS[@]}"; do
    mkdir -p "$i"
    touch "$i/Makefile.am"
done

rm -f "configure" "Makefile.in" "include/Makefile.in" "src/Makefile.in"

# Stubs for autoreconf
DIST_STUB=( \
  contrib/windows/test-windows-version.sh.am \
  doc/doxygen-config.cfg.am \
  tests/hwloc/wrapper.sh.am \
  tests/hwloc/linux/test-topology.sh.am \
  tests/hwloc/linux/allowed/test-topology.sh.am \
  tests/hwloc/linux/gather/test-gather-topology.sh.am \
  tests/hwloc/x86+linux/test-topology.sh.am \
  tests/hwloc/x86/test-topology.sh.am \
  tests/hwloc/xml/test-topology.sh.am \
  tests/netloc/tests.sh.am \
  utils/hwloc/hwloc-compress-dir.am \
  utils/hwloc/hwloc-gather-topology.am \
  utils/hwloc/test-hwloc-annotate.sh.am \
  utils/hwloc/test-hwloc-calc.sh.am \
  utils/hwloc/test-hwloc-compress-dir.sh.am \
  utils/hwloc/test-hwloc-diffpatch.sh.am \
  utils/hwloc/test-hwloc-distrib.sh.am \
  utils/hwloc/test-hwloc-info.sh.am \
  utils/hwloc/test-fake-plugin.sh.am \
  utils/hwloc/test-hwloc-dump-hwdata/test-hwloc-dump-hwdata.sh.am \
  utils/hwloc/test-parsing-flags.sh.am \
  utils/lstopo/test-lstopo.sh.am \
  utils/lstopo/test-lstopo-shmem.sh.am \
  utils/lstopo/lstopo-windows.c \
  utils/lstopo/lstopo-android.c \
  utils/netloc/infiniband/netloc_ib_gather_raw.am \
)
for i in "${DIST_STUB[@]}"; do
    touch "$i"
done

# Run autoreconf once first so identifying patches for cherry-picking is easier
autoreconf -ivf
ret="$?"
rm -rf autom4te.cache/
if [[ $ret != 0 ]]; then
  echo "$0 needs to be updated for this hwloc version."
  exit 1
fi
find . -name '*~' -delete

cd ..
git add -f hwloc
git commit -m "EDIT THIS COMMIT"

popd > /dev/null

echo 'Done. Please:'
echo '1. Review the git history of contrib/hwloc for patches that should be cherry-picked and squashed.'
echo '2. In contrib/hwloc, run: autoreconf -ivf && rm -rf autom4te.cache/ && find . -name "*~" -delete'
echo '3. Verify that an all-test build with --build-shared -charm-shared completes successfully.'

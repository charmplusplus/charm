#!/bin/bash

if [[ -z "$1" ]]; then
    echo "$0" '<hwloc-X.Y.Z.tar.(gz|bz2)>'
    exit 0
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
DIST_SUBDIRS=( utils tests doc contrib/systemd doc/examples utils/hwloc utils/lstopo tests/linux tests/linux/allowed tests/linux/gather tests/xml tests/ports tests/rename )
EXTRA_DIST=( contrib/windows )
for i in "${DIST_SUBDIRS[@]}" "${EXTRA_DIST[@]}"; do
    rm -rf "$i"
done
# Create stub automake files because the disabling is a configure-time option, so automake always checks them
for i in "${DIST_SUBDIRS[@]}"; do
    mkdir -p "$i"
    touch "$i/Makefile.am"
done

rm -f "configure" "Makefile.in" "include/Makefile.in" "src/Makefile.in"

touch tests/linux/allowed/test-topology.sh.am
touch tests/linux/gather/test-gather-topology.sh.am
touch tests/xml/test-topology.sh.am
touch doc/doxygen-config.cfg.am
touch tests/wrapper.sh.am
touch tests/linux/test-topology.sh.am
touch utils/hwloc/hwloc-assembler-remote.am
touch utils/hwloc/hwloc-compress-dir.am
touch utils/hwloc/hwloc-gather-topology.am
touch utils/hwloc/test-hwloc-annotate.sh.am
touch utils/hwloc/test-hwloc-assembler.sh.am
touch utils/hwloc/test-hwloc-calc.sh.am
touch utils/hwloc/test-hwloc-compress-dir.sh.am
touch utils/hwloc/test-hwloc-diffpatch.sh.am
touch utils/hwloc/test-hwloc-distances.sh.am
touch utils/hwloc/test-hwloc-distrib.sh.am
touch utils/hwloc/test-hwloc-info.sh.am
touch utils/hwloc/test-fake-plugin.sh.am
touch utils/lstopo/test-hwloc-ls.sh.am

popd > /dev/null

echo "Done. Please review the git history to see if there are any patches that should be cherry-picked and squashed."

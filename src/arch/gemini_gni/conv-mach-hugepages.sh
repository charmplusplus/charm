# Check that some hugepages module is loaded
if module list 2>&1 | grep -q craype-hugepages; then
    true
else
    echo 'Must have a craype-hugepages module loaded (e.g. module load craype-hugepages8M)' >&2
    exit 1
fi

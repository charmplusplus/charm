. $CHARMINC/cc-clang.sh
. $CHARMINC/conv-mach-darwin.sh

CMK_DEFS="$CMK_DEFS -mmacosx-version-min=10.7"
# Assumes gfortran compiler:
CMK_CF77="$CMK_CF77 -mmacosx-version-min=10.7"
CMK_CF90="$CMK_CF90 -mmacosx-version-min=10.7"

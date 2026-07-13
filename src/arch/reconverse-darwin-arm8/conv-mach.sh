. $CHARMINC/cc-clang.sh
. $CHARMINC/conv-mach-darwin.sh

CMK_DEFS="$CMK_DEFS -D_REENTRANT"

# Apple clang still defaults to C++98; reconverse headers require C++17.
# NATIVE/SEQ flag sets are snapshotted inside conv-mach-darwin.sh before this
# script runs (charmxi et al. build with NATIVE), so append to all three.
CMK_CXX_FLAGS="$CMK_CXX_FLAGS -std=gnu++17"
CMK_NATIVE_CXX_FLAGS="$CMK_NATIVE_CXX_FLAGS -std=gnu++17"
CMK_SEQ_CXX_FLAGS="$CMK_SEQ_CXX_FLAGS -std=gnu++17"

# Reconverse provides its own threading; remove the QuickThreads library
# injected by conv-mach-darwin.sh
CMK_LIBS="${CMK_LIBS//-lckqt/}"

CMK_SMP="1"

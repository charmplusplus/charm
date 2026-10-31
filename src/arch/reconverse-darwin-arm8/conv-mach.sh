. $CHARMINC/cc-clang.sh
. $CHARMINC/conv-mach-darwin.sh

CMK_DEFS="$CMK_DEFS -D_REENTRANT"

# Reconverse provides its own threading; remove the QuickThreads library
# injected by conv-mach-darwin.sh
CMK_LIBS="${CMK_LIBS//-lckqt/}"

CMK_SMP="1"

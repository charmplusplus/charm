. $CHARMINC/cc-gcc.sh

CMK_DEFS="$CMK_DEFS -D_REENTRANT"

CMK_XIOPTS=''
CMK_LIBS="-lpthread $CMK_LIBS"
# QuickThreads is not built under reconverse; strip charmc's mainline -lckqt
# (same idiom as reconverse-darwin-arm8).
CMK_LIBS="${CMK_LIBS//-lckqt/}"

CMK_QT='generic64-light'

CMK_SMP='1'

# Application symbols in reconverse's stack traces (contrib/reconverse/src/backtrace.cpp).
CMK_LD_FLAGS="$CMK_LD_FLAGS -rdynamic"
CMK_LDXX_FLAGS="$CMK_LDXX_FLAGS -rdynamic"

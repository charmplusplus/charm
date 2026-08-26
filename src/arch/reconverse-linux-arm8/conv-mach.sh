. $CHARMINC/cc-gcc.sh

CMK_DEFS="$CMK_DEFS -D_REENTRANT"

CMK_XIOPTS=''
CMK_LIBS="-lpthread $CMK_LIBS"
# QuickThreads is not built under reconverse; strip charmc's mainline -lckqt
# (same idiom as reconverse-darwin-arm8).
CMK_LIBS="${CMK_LIBS//-lckqt/}"

CMK_QT='generic64-light'

CMK_SMP='1'

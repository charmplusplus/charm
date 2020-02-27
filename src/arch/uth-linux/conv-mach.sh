. $CHARMINC/cc-gcc.sh

if command -v gfortran >/dev/null 2>&1
then
  . $CHARMINC/conv-mach-gfortran.sh
fi

CMK_CXX_FLAGS="$CMK_CXX_FLAGS -Wno-deprecated"
CMK_QT='generic'
CMK_XIOPTS=''
CMK_NO_PARTITIONS="1"
CMK_CCS_AVAILABLE='0'

CMK_UTH="1"

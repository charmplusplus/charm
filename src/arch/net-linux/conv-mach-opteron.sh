CMK_DEFS="$CMK_DEFS -DHAVE_USR_INCLUDE_MALLOC_H=1 "
CMK_CC="gcc -m64 -fPIC $CMK_DEFS"
CMK_CXX="g++ -Wno-deprecated -m64 -fPIC $CMK_DEFS "
CMK_SEQ_CC="gcc -m64 -fPIC $CMK_DEFS "
CMK_SEQ_CXX="g++ -m64 -fPIC $CMK_DEFS "
CMK_LD="gcc -m64 "
CMK_LDXX="g++ -m64 "
CMK_QT='generic64'

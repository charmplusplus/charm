CMK_CC="xlc_r -qnocommon -qpic "
#CMK_CXX="xlc++_r "
CMK_CXX="xlc++_r -qnocommon -qpic "
CMK_LD="$CMK_CC -qnofullpath "
CMK_LDXX="$CMK_CXX -qnofullpath "
CMK_C_OPTIMIZE="-O3 -qstrict -Q  "
CMK_CXX_OPTIMIZE="-O3 -qstrict -Q "
OPTS_CC=
OPTS_CXX=

CMK_NATIVE_CC="xlc_r "
CMK_NATIVE_LD="xlc_r "
CMK_NATIVE_CXX="xlc++_r "
CMK_NATIVE_LDXX="xlc++_r "
CMK_NATIVE_LIBS="-lstdc++ "

CMK_LD_SHARED=" -qmkshrobj -Wl,-dynamic -Wl,-undefined,dynamic_lookup "

CMK_AR="ar cq"
CMK_NM="nm "
CMK_QT=aix


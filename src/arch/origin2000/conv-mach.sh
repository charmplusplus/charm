CMK_CPP_CHARM='/usr/lib/cpp '
CMK_CPP_C='cc -E -64 '
CMK_LDRO='ld -r -o'
CMK_LDRO_WORKS=0
CMK_CC='cc  -64 '
CMK_CXX='CC -64 -LANG:std '
CMK_CXXPP="$CMK_CXX -E"
CMK_CF77='f77'
CMK_CF90='f90'
CMK_C_OPTIMIZE='-O3 -r10000 -INLINE:all -TARG:platform=ip27 -OPT:Olimit=0:roundoff=3:div_split=ON:alias=typed '
CMK_CXX_OPTIMIZE='-O3 -r10000 -INLINE:all -TARG:platform=ip27 -OPT:Olimit=0:roundoff=3:div_split=ON:alias=typed '
CMK_LD="$CMK_CC -w"
CMK_LDXX="$CMK_CXX -w"
CMK_AR="$CMK_CXX -ar -o"
CMK_RANLIB='true'
CMK_LIBS=' -lckqt -lfastm '
CMK_NM='nm'
CMK_NM_FILTER="grep '|GLOB |' | sed -e 's/.*|//'"
CMK_QT='origin'
CMK_XIOPTS=''
CMK_F90OBJS='fmain.o'
CMK_F90LIBS='-L/usr/lib64 -lfortran -lftn'
CMK_MOD_NAME_ALLCAPS=1
CMK_MOD_EXT='mod'

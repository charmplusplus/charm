############################################################################
# RCS INFORMATION:
#
# 	$RCSfile$
# 	$Author$	$Locker$		$State$
#	$Revision$	$Date$
#
############################################################################
# DESCRIPTION:
#
############################################################################
# REVISION HISTORY:
#
#
############################################################################

set CMK_DEFS=' -D_REENTRANT '
set CMK_CPP_CHARM="/usr/ccs/lib/cpp $CMK_DEFS"
set CMK_CPP_C="gcc -E $CMK_DEFS"
set CMK_LDRO='ld -r -o'
set CMK_LDRO_WORKS=0
set CMK_CC="gcc $CMK_DEFS"
set CMK_CC_RELIABLE="gcc $CMK_DEFS "
set CMK_CC_FASTEST="gcc $CMK_DEFS "
set CMK_CXX="g++ $CMK_DEFS "
set CMK_CXXPP="g++ -x c++ -E $CMK_DEFS "
set CMK_CF77=''
set CMK_C_DEBUG='-g'
set CMK_C_OPTIMIZE='-O'
set CMK_CXX_DEBUG='-g'
set CMK_CXX_OPTIMIZE='-O'
set CMK_LD="gcc $CMK_DEFS "
set CMK_LDXX="g++ $CMK_DEFS "
set CMK_LD77=''
set CMK_M4='m4'
set CMK_SUF='o'
set CMK_AR='ar q'
set CMK_RANLIB='true'
set CMK_LIBS=' -lthread -lnsl -lsocket -lqt'
set CMK_SEQ_LIBS=' -lnsl -lsocket'
set CMK_SEQ_CC='gcc'
set CMK_SEQ_LD='gcc'
set CMK_SEQ_CXX='g++'
set CMK_SEQ_LDXX='g++'
set CMK_NM='nm'
set CMK_NM_FILTER='grep "|GLOB" | sed -e "s@.*|@@"'
set CMK_CPP_SUFFIX="ii"
set CMK_XLATPP='charmxlat++ '
set CMK_QT='solaris-gcc'

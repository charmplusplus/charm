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

set CMK_CPP_CHARM='/lib/cpp -P'
set CMK_CPP_C='cc -E '
set CMK_CXXPP='echo "Convex doesnt support charm++" ; exit 1 ; echo '
set CMK_LDRO='ld -r -o'
set CMK_LDRO_WORKS=1
set CMK_CC='cc -or none'
set CMK_CC_RELIABLE='cc'
set CMK_CC_FASTEST='cc'
set CMK_CXX='echo "Convex doesnt support charm++" ; exit 1 ; echo '
set CMK_CF77=''
set CMK_C_DEBUG='-no -cxdb'
set CMK_C_OPTIMIZE='-O2'
set CMK_CXX_DEBUG=''
set CMK_CXX_OPTIMIZE=''
set CMK_LD='cc -Wl,+parallel'
set CMK_LDXX='echo "Convex doesnt support charm++" ; exit 1 ; echo '
set CMK_LD77=''
set CMK_M4='m4'
set CMK_SUF='o'
set CMK_AR='ar q'
set CMK_RANLIB='true'
set CMK_LIBS='-lqt'
set CMK_SEQ_LIBS=''
set CMK_SEQ_CC='cc -no'
set CMK_SEQ_LD='cc'
set CMK_SEQ_CXX='CC -no'
set CMK_SEQ_LDXX='CC'
set CMK_NM='nm'
set CMK_NM_FILTER="grep '|extern|' | awk '{print "'$'"1;}'"
set CMK_CPP_SUFFIX="i"
set CMK_XLATPP='charmxlat++ '
set CMK_QT='convex'

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
# $Log$
# Revision 1.2  1996-08-01 21:11:12  jyelon
# added two options to charmxlat++   -w -p
#
# Revision 1.1  1996/04/30 19:51:39  brunner
# Initial revision
#
#
############################################################################

set CMK_CPP_CHARM='/lib/cpp -P'
set CMK_CPP_C='cc -E'
set CMK_LDRO='ld -r -o'
set CMK_CC='cc -Aa -D_HPUX_SOURCE '
set CMK_CC_RELIABLE='cc -Aa -D_HPUX_SOURCE '
set CMK_CC_FASTEST='cc -Aa -D_HPUX_SOURCE '
set CMK_CXX='CC -D_HPUX_SOURCE'
set CMK_CXXPP='CC -Aa -D_HPUX_SOURCE -E '
set CMK_CF77=''
set CMK_C_DEBUG='-g'
set CMK_C_OPTIMIZE='-O'
set CMK_CXX_DEBUG='-g'
set CMK_CXX_OPTIMIZE='-O'
set CMK_LD='cc -Aa -D_HPUX_SOURCE -s'
set CMK_LDXX='CC'
set CMK_LD77=''
set CMK_M4='m4'
set CMK_SUF='o'
set CMK_AR='ar q'
set CMK_RANLIB='true'
set CMK_LIBS=''
set CMK_SEQ_LIBS=''
set CMK_SEQ_CC='cc -Aa -D_HPUX_SOURCE '
set CMK_SEQ_LD='cc -Aa -D_HPUX_SOURCE '
set CMK_SEQ_CXX='CC -D_HPUX_SOURCE '
set CMK_SEQ_LDXX='CC -D_HPUX_SOURCE '
set CMK_NM='nm'
set CMK_NM_FILTER="grep '|extern|' | sed -e 's@ *|.*@@'"
set CMK_CPP_SUFFIX="ii"
set CMK_XLATPP='charmxlat++ '

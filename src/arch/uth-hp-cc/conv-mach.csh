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
# Revision 1.4  1997-07-15 21:16:10  jyelon
# Removed CMK_NM stuff once and for all!
#
# Revision 1.3  1997/05/05 14:25:35  jyelon
# More quickthreads related changes.
#
# Revision 1.2  1997/05/05 13:56:59  jyelon
# Updated for quickthreads
#
# Revision 1.1  1996/11/05 21:29:21  brunner
# Needs _main in main for C++ compile
#
# Revision 1.3  1996/08/04 04:19:50  jyelon
# Added CMK_LDRO_WORKS
#
# Revision 1.2  1996/08/01 21:11:12  jyelon
# added two options to charmxlat++   -w -p
#
# Revision 1.1  1996/04/30 19:51:39  brunner
# Initial revision
#
#
############################################################################

set CMK_CPP_CHARM='/lib/cpp -P'
set CMK_CPP_C='cc -Aa -E'
set CMK_CC='cc -Aa -D_HPUX_SOURCE +z '
set CMK_CC_RELIABLE='cc -Aa -D_HPUX_SOURCE +z '
set CMK_CC_FASTEST='cc -Aa -D_HPUX_SOURCE +z '
set CMK_CXX='CC -D_HPUX_SOURCE +z '
set CMK_CXXPP='CC -Aa -D_HPUX_SOURCE -E +z '
set CMK_CF77=''
set CMK_C_DEBUG='-g'
set CMK_C_OPTIMIZE='-O'
set CMK_CXX_DEBUG='-g'
set CMK_CXX_OPTIMIZE='-O'
set CMK_LD='cc -Aa -D_HPUX_SOURCE +z'
set CMK_LDXX='CC'
set CMK_LD77=''
set CMK_M4='m4'
set CMK_SUF='o'
set CMK_AR='ar q'
set CMK_RANLIB='true'
set CMK_LIBS='-lqt'
set CMK_SEQ_LIBS=''
set CMK_SEQ_CC='cc -Aa -D_HPUX_SOURCE '
set CMK_SEQ_LD='cc -Aa -D_HPUX_SOURCE '
set CMK_SEQ_CXX='CC -D_HPUX_SOURCE '
set CMK_SEQ_LDXX='CC -D_HPUX_SOURCE '
set CMK_CPP_SUFFIX="i"
set CMK_XLATPP='charmxlat++ '
set CMK_QT='hpux-cc'

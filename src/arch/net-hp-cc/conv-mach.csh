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
# Revision 1.7  1997-05-05 13:52:39  jyelon
# Updated for quickthreads
#
# Revision 1.6  1997/03/19 23:17:46  milind
# Got net-irix to work. Had to modify jsleep to deal with restaring
# system calls on interrupts.
#
# Revision 1.5  1997/03/17 23:40:28  milind
# Added Idle Notification Functionality:
# The new Macros in converse.h for this are:
# CsdSetNotifyIdle(fn1, fn2)
# CsdStartNotifyIdle()
# CsdStopNotifyIdle()
#
# Revision 1.4  1996/10/22 19:08:31  milind
# Added +z option to produce position independent code.
# Needed for parallel perl.
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
set CMK_LDRO='ld -r -o'
set CMK_LDRO_WORKS=1
set CMK_CC='cc -Aa -D_HPUX_SOURCE '
set CMK_CC_RELIABLE='cc -Aa -D_HPUX_SOURCE '
set CMK_CC_FASTEST='cc -Aa -D_HPUX_SOURCE '
set CMK_CXX='CC -D_HPUX_SOURCE '
set CMK_CXXPP='CC -Aa -D_HPUX_SOURCE -E '
set CMK_CF77=''
set CMK_C_DEBUG='-g'
set CMK_C_OPTIMIZE='+O3 +Onolimit '
set CMK_CXX_DEBUG='-g'
set CMK_CXX_OPTIMIZE='+O3 +Onolimit '
set CMK_LD='cc -Aa -D_HPUX_SOURCE '
set CMK_LDXX='CC '
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
set CMK_NM='nm'
set CMK_NM_FILTER="grep '|extern|' | sed -e 's@ *|.*@@'"
set CMK_CPP_SUFFIX="i"
set CMK_XLATPP='charmxlat++ '
set CMK_QT='setjmp-gcc-d'

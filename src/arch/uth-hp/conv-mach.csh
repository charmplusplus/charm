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
# Revision 1.14  1997-05-05 13:56:51  jyelon
# Updated for quickthreads
#
# Revision 1.13  1996/08/05 15:28:49  jyelon
# *** empty log message ***
#
# Revision 1.12  1996/08/04 04:19:50  jyelon
# Added CMK_LDRO_WORKS
#
# Revision 1.11  1996/08/01 21:11:12  jyelon
# added two options to charmxlat++   -w -p
#
# Revision 1.10  1996/07/02 21:22:24  jyelon
# Many small changes.
#
# Revision 1.9  1996/04/16 22:45:34  jyelon
# *** empty log message ***
#
# Revision 1.8  1996/04/09 22:56:30  jyelon
# *** empty log message ***
#
# Revision 1.7  1995/11/13 16:36:46  jyelon
# repaired CMK_NM_FILTER
#
# Revision 1.6  1995/11/02  22:45:43  sanjeev
# added CMK_CPP_SUFFIX
#
# Revision 1.5  1995/10/30  20:53:35  jyelon
# *** empty log message ***
#
# Revision 1.4  1995/10/30  20:31:35  jyelon
# *** empty log message ***
#
# Revision 1.3  1995/10/25  19:59:30  jyelon
# added CMK_CC_RELIABLE and CMK_CC_FASTEST
#
# Revision 1.2  1995/10/20  18:38:43  jyelon
# added CMK_C_DEBUG, CMK_C_OPTIMIZE, CMK_CXX_DEBUG, CMK_CXX_OPTIMIZE
#
# Revision 1.1  1995/10/13  20:05:13  jyelon
# Initial revision
#
# Revision 2.4  1995/10/02  18:56:50  knauff
# Added CMK_CXXPP.
#
# Revision 2.3  1995/09/19  20:12:08  brunner
# conv-host not compiled to bin directory, fixed, and RCS header added
#
############################################################################

set CMK_CPP_CHARM='/lib/cpp -P'
set CMK_CPP_C='gcc -E'
set CMK_LDRO='ld -r -o'
set CMK_LDRO_WORKS=0
set CMK_CC='gcc'
set CMK_CC_RELIABLE='gcc'
set CMK_CC_FASTEST='gcc'
set CMK_CXX='g++'
set CMK_CXXPP='g++ -E'
set CMK_C_DEBUG='-g'
set CMK_C_OPTIMIZE='-O'
set CMK_CXX_DEBUG='-g'
set CMK_CXX_OPTIMIZE='-O'
set CMK_CF77=''
set CMK_LD='gcc'
set CMK_LDXX='g++'
set CMK_LD77=''
set CMK_M4='m4'
set CMK_SUF='o'
set CMK_AR='ar q'
set CMK_RANLIB='true'
set CMK_LIBS='-lqt'
set CMK_SEQ_LIBS=''
set CMK_SEQ_CC='gcc'
set CMK_SEQ_LD='gcc'
set CMK_SEQ_CXX='g++'
set CMK_SEQ_LDXX='g++'
set CMK_NM='nm'
set CMK_NM_FILTER="grep '|extern|' | sed -e 's@ *|.*@@'"
set CMK_CPP_SUFFIX="ii"
set CMK_XLATPP='charmxlat++ -w'
set CMK_QT='hpux-cc'

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
# Revision 2.13  1996-04-09 22:56:19  jyelon
# *** empty log message ***
#
# Revision 2.12  1995/11/13 16:36:19  jyelon
# repaired CMK_NM_FILTER
#
# Revision 2.11  1995/11/02  23:29:09  sanjeev
# removed -x c++
#
# Revision 2.10  1995/11/02  22:45:43  sanjeev
# added CMK_CPP_SUFFIX
#
# Revision 2.9  1995/10/30  20:53:35  jyelon
# *** empty log message ***
#
# Revision 2.8  1995/10/30  20:31:35  jyelon
# *** empty log message ***
#
# Revision 2.7  1995/10/25  19:59:30  jyelon
# added CMK_CC_RELIABLE and CMK_CC_FASTEST
#
# Revision 2.6  1995/10/20  18:38:43  jyelon
# added CMK_C_DEBUG, CMK_C_OPTIMIZE, CMK_CXX_DEBUG, CMK_CXX_OPTIMIZE
#
# Revision 2.5  1995/10/19  17:55:37  jyelon
# Added -D_INCLUDE_HPUX_SOURCE
#
# Revision 2.4  1995/10/02  18:56:50  knauff
# Added CMK_CXXPP.
#
# Revision 2.3  1995/09/19  20:12:08  brunner
# conv-host not compiled to bin directory, fixed, and RCS header added
#
############################################################################

set CMK_CPP='/lib/cpp -P'
set CMK_LDRO='ld -r -o'
set CMK_CC='gcc '
set CMK_CC_RELIABLE='gcc '
set CMK_CC_FASTEST='gcc '
set CMK_CXX='g++'
set CMK_CXXPP='g++ -E '
set CMK_CF77=''
set CMK_C_DEBUG='-g'
set CMK_C_OPTIMIZE='-O'
set CMK_CXX_DEBUG='-g'
set CMK_CXX_OPTIMIZE='-O'
set CMK_LD='gcc -s'
set CMK_LDXX='g++'
set CMK_LD77=''
set CMK_M4='m4'
set CMK_SUF='o'
set CMK_AR='ar q'
set CMK_RANLIB='true'
set CMK_LIBS=''
set CMK_SEQ_LIBS=''
set CMK_SEQ_CC='gcc '
set CMK_SEQ_LD='gcc '
set CMK_SEQ_CXX='g++ '
set CMK_SEQ_LDXX='g++ '
set CMK_NM='nm'
set CMK_NM_FILTER="grep '|extern|' | sed -e 's@ *|.*@@'"
set CMK_CPP_SUFFIX="ii"

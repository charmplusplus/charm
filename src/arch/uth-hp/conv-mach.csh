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
# Revision 1.3  1995-10-25 19:59:30  jyelon
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

set CMK_CPP='/lib/cpp -P'
set CMK_LDRO='ld -r -o'
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
set CMK_LD='gcc -s'
set CMK_LDXX='g++'
set CMK_LD77=''
set CMK_M4='m4'
set CMK_SUF='o'
set CMK_AR='ar q'
set CMK_RANLIB='true'
set CMK_LIBS=''
set CMK_SEQ_CC='gcc'
set CMK_SEQ_LD='gcc'
set CMK_NM='nm'
set CMK_NM_FILTER="grep '|extern|' | awk '{print "'$'"1;}'"
set CMK_EXTRAS=''
set CMK_CLEAN=''

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
# Revision 2.7  1995-10-25 19:59:30  jyelon
# added CMK_CC_RELIABLE and CMK_CC_FASTEST
#
# Revision 2.6  1995/10/20  18:38:43  jyelon
# added CMK_C_DEBUG, CMK_C_OPTIMIZE, CMK_CXX_DEBUG, CMK_CXX_OPTIMIZE
#
# Revision 2.5  1995/10/02  18:59:47  knauff
# Added CMK_CXXPP.
#
# Revision 2.4  1995/07/26  21:25:18  knauff
# Took out '-D_NO_PROTO' in CMK_CPP, changed CMK_RANLIB
#
# Revision 2.3  1995/07/19  20:56:47  knauff
# Changed CMK_CXX to mpCC
#
# Revision 2.2  1995/07/17  17:47:00  knauff
# *** empty log message ***
#
# Revision 2.1  1995/07/10  22:22:51  knauff
# *** empty log message ***
#
# Revision 2.0  1995/07/10  22:11:15  knauff
# Initial revision
#
############################################################################

set CMK_CPP			= '/usr/lib/cpp -P '

set CMK_LDRO			= 'ld -r -o '
set CMK_CC			= 'mpcc'
set CMK_CC_RELIABLE		= 'mpcc'
set CMK_CC_FASTEST		= 'mpcc'
set CMK_CXX			= 'mpCC'
set CMK_CXXPP			= 'cc -E'
set CMK_CF77			= 'mpxlf'
set CMK_C_DEBUG                 = '-g'
set CMK_C_OPTIMIZE              = '-O'
set CMK_CXX_DEBUG               = '-g'
set CMK_CXX_OPTIMIZE            = '-O'
set CMK_LD			= 'mpcc -us'
set CMK_LDXX			= 'mpCC -us'
set CMK_LD77			= ''
set CMK_M4			= 'm4'
set CMK_SUF			= 'o'
set CMK_AR                      = 'ar q'
set CMK_RANLIB			= 'true'
set CMK_LIBS			= '-bnso -bI:/lib/syscalls.exp'
set CMK_SEQ_CC			= 'gcc'
set CMK_SEQ_LD			= 'gcc'
set CMK_NM			= 'nm'
set CMK_NM_FILTER		= "colrm 1 11 | sed -e 's/\.//'"
set CMK_EXTRAS			= ''
set CMK_CLEAN			= ''

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
# Revision 2.13  1995-11-08 22:27:49  milind
# changed CMK_CXXPP to xlC -E.
#
# Revision 2.12  1995/11/02  22:45:43  sanjeev
# added CMK_CPP_SUFFIX
#
# Revision 2.11  1995/11/02  21:14:01  milind
# removed -C from CMK_CPP command.
#
# Revision 2.10  1995/11/01  23:14:05  knauff
# Changed CMK_CPP back to -D_NO_PROTO.
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

set CMK_CPP			= '/usr/lib/cpp -D_NO_PROTO '

set CMK_LDRO			= 'ld -r -o '
set CMK_CC			= 'mpcc'
set CMK_CC_RELIABLE		= 'mpcc'
set CMK_CC_FASTEST		= 'mpcc'
set CMK_CXX			= 'mpCC'
set CMK_CXXPP			= 'xlC -E'
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
set CMK_SEQ_LIBS                = ''
set CMK_SEQ_CC			= 'gcc'
set CMK_SEQ_LD			= 'gcc'
set CMK_NM			= 'nm'
set CMK_NM_FILTER		= "colrm 1 11 | sed -e 's/\.//'"
set CMK_CPP_SUFFIX="i"

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
# Revision 2.3  1995-07-19 20:56:47  knauff
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

set CMK_CPP			= '/usr/lib/cpp -P -D_NO_PROTO'

set CMK_LDRO			= 'ld -r -o '
set CMK_CC			= 'mpcc'
set CMK_CXX			= 'mpCC'
set CMK_CF77			= 'mpxlf'
set CMK_LD			= 'mpcc -us'
set CMK_LDXX			= 'mpCC -us'
set CMK_LD77			= ''
set CMK_M4			= 'm4'
set CMK_SUF			= 'o'
set CMK_AR                      = 'ar q'
set CMK_RANLIB			= 'false'
set CMK_LIBS			= '-bnso -bI:/lib/syscalls.exp'
set CMK_SEQ_CC			= 'gcc'
set CMK_SEQ_LD			= 'gcc'
set CMK_NM			= 'nm'
set CMK_NM_FILTER		= "colrm 1 11 | sed -e 's/\.//'"
set CMK_EXTRAS			= ''
set CMK_CLEAN			= ''

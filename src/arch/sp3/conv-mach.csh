############################################################################
# RCS INFORMATION:
#
#  $RCSfile$
#  $Author$ $Locker$  $State$
# $Revision$ $Date$
#
############################################################################
# DESCRIPTION:
#
############################################################################
# REVISION HISTORY:
#
# $Log$
# Revision 1.4  1998-03-05 17:15:07  milind
# Fixed conflicts.
#
# Revision 1.3  1997/07/17 15:51:43  milind
# Fixed module initialization on SP3.
#
# Revision 1.2  1997/07/09 21:06:25  milind
# Fixed the nm bug on SP3. Charm module finding still seems to be broken.
# Will try to fix it soon.
#
# Revision 1.1  1997/07/08 22:10:54  milind
# Added IBM SP3 version. Developed and Tested on ANL machine.
#
# Revision 2.23  1997/05/05 14:25:27  jyelon
# More quickthreads related changes.
#
# Revision 2.22  1997/05/05 13:56:09  jyelon
# Updated for quickthreads
#
# Revision 2.21  1997/03/21 20:06:54  milind
# Fixed a prototype mismatch.
#
# Revision 2.20  1997/03/14 20:25:43  milind
# Changed optimization options to compilers and linkers.
#
# Revision 2.19  1997/02/02 07:33:55  milind
# Fixed Bugs in SP1 machine dependent code that made megacon to hang.
# Consisted of almost 60 percent rewrite.
#
# Revision 2.18  1996/08/04 04:19:50  jyelon
# Added CMK_LDRO_WORKS
#
# Revision 2.17  1996/08/01 21:11:12  jyelon
# added two options to charmxlat++   -w -p
#
# Revision 2.16  1996/07/24 22:03:49  milind
# made changes for built in types wchar_t and ptrdiff_t
#
# Revision 2.15  1996/04/16 22:45:34  jyelon
# *** empty log message ***
#
# Revision 2.14  1996/04/09 22:56:22  jyelon
# *** empty log message ***
#
# Revision 2.13  1995/11/08 22:27:49  milind
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

set CMK_CPP_CHARM    = '/usr/lib/cpp -P -D_NO_PROTO '
set CMK_CPP_C        = '/usr/lib/cpp -P -D_NO_PROTO '
set CMK_LDRO         = 'ld -r -o '
set CMK_LDRO_WORKS   = 0
set CMK_CC           = 'mpcc '
set CMK_CC_RELIABLE  = 'mpcc '
set CMK_CC_FASTEST   = 'mpcc '
set CMK_CXX          = 'mpCC '
set CMK_CXXPP        = 'xlC -E '
set CMK_CF77         = 'mpxlf'
set CMK_C_DEBUG      = '-g'
set CMK_C_OPTIMIZE   = '-O3 -qstrict -Q -qarch=pwr2 -qtune=pwr2  '
set CMK_CXX_DEBUG    = '-g'
set CMK_CXX_OPTIMIZE = '-O3 -qstrict -Q -qarch=pwr2 -qtune=pwr2  '
set CMK_LD           = 'mpcc -w -u_CK7CharmInit'
set CMK_LDXX         = 'mpCC  -w -u_CK_call_main_main'
set CMK_LD77         = ''
set CMK_M4           = 'm4'
set CMK_SUF          = 'o'
set CMK_AR           = 'ar cq'
set CMK_RANLIB       = 'true'
set CMK_LIBS         = '-lqt'
set CMK_SEQ_LIBS     = ''
set CMK_SEQ_CC       = 'gcc'
set CMK_SEQ_LD       = 'gcc'
set CMK_SEQ_CXX      = 'g++'
set CMK_SEQ_LDXX     = 'g++'
set CMK_NM           = '/bin/nm'
set CMK_NM_FILTER    = "grep ^_CK_ | cut -f 1 -d ' '"
set CMK_CPP_SUFFIX   = "i"
set CMK_XLATPP       = 'charmxlat++ '
set CMK_QT='aix32-gcc'

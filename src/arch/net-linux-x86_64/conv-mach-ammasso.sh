#
# Ammasso Specific File
#
# Created: 03/12/05
#

# Default ammasso directory
# Guess where it is ???

CMK_INCDIR="-I/scratch/ammasso/AMSO1100/software/host/common/include -I/scratch/ammasso/AMSO1100/software/common/include -I/scratch/ammasso/AMSO1100/software/host/linux/include"
CMK_LIBDIR="-L/usr/opt/ammasso/lib64"

CMK_LIBS="$CMK_LIBS -lccil"

# Example: Originally from conv-mach-gm.sh
#
#default gm dir
#guess where the gm.h is installed
#if test -z "$CMK_INCDIR"
#then
#  # gm ver 1.0
#  if test -f /usr/gm/include/gm.h
#  then
#    CMK_INCDIR="-I /usr/gm/include"
#    CMK_LIBDIR="-L /usr/gm/lib"
#  # gm ver 2.0
#  elif test -f /opt/gm/include/gm.h
#  then
#    CMK_INCDIR="-I /opt/gm/include"
#    CMK_LIBDIR="-L /opt/gm/lib"
#  fi
#fi
#
#
#CMK_LIBS="$CMK_LIBS -lgm"

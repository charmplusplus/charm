. $CHARMINC/cc-gcc.sh

if test -z "$USER_OPTS_LD"
then
    CMK_INCDIR="-I/usr/include/"
    CMK_LIBDIR="-L/usr/lib64/"
fi

CMK_LIBS="$CMK_LIBS -lucp -luct -lucs -lucm"

# For runtime
CMK_INCDIR="$CMK_INCDIR -I./proc_management/ -I./proc_management/simple_pmi/"

CMK_QT='generic64-light'

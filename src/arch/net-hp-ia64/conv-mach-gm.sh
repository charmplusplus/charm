test -z "$CMK_INCDIR" && CMK_INCDIR='-I/usr/gm/include'
test -z "$CMK_LIBDIR" && CMK_LIBDIR='-L/usr/gm/lib'

CMK_LIBS="$CMK_LIBS -lgm"

if test "$CMK_CC" = 'mpicc '
then
CMK_CPP_C='gcc -E '
CMK_CC='gcc'
CMK_CC_RELIABLE='gcc '
CMK_CC_FASTEST='gcc '
CMK_CXX='g++'
CMK_CXXPP='gcc -E '
CMK_LD='gcc'
CMK_LDXX='g++'
fi
CMK_LIBS="$CMK_LIBS -lmpi"

CMK_LIBS="-lckqt "
if test "$CMK_CXX" != "mpiCC"
then
CMK_LIBS="$CMK_LIBS -llammpio -llammpi++ -llamf77mpi -lmpi -llam -laio -laio -lutil"
fi

#!/bin/sh
#
# Script to collect up all the source code needed
# to build "FEM_ALONE", a version of the FEM framework
# built on regular MPI, not Charm++'s AMPI.
#
# WARNING: This script probably will not work with 
#          the recent versions of FEM. To add the 
#          adjacency and modification operations
#	   it was required to use Charm++ instead
#          of the pure MPI base that had existed.
#
# Build charm normally, then run this script from the 
# main charm directory like:
#    > src/libs/ck-libs/fem/make_fem_alone.sh
#    > cd fem_alone
#
# You can now build the femalone library using your
# compiler, like:
#    > CC -c *.C -I. -DFEM_ALONE=1
#    > ar cr libfemalone.a *.o
#
# This could also be integrated into your build script.
#
#
# Because FEM still depends on a few charm++ utilities
# (pup, CkHashtable, CkVec), and the charm++ configuration
# headers (conv-config, for e.g., fortran name mangling),
# we need to copy over this strange subset of files.
# At runtime, all you need is libfemalone.a.
# 
# Orion Sky Lawlor, olawlor@acm.org, 2003/8/18
#

echo "WARNING: see warning description in make_fem_alone.sh"
echo "         This script will probably not work!         "
echo


Do() {
	echo "> $@"
	$@
	if [ $? -ne 0 ]
	then
		echo "Error during $@"
		exit 1
	fi
}

charm=`pwd`
out=`pwd`"/fem_alone/"
rm -fr $out
Do mkdir $out
Do cd $out

get="ln -s"

echo "Collecting FEM source code: "

# grab all the config headers:
Do $get $charm/include/conv*.h .
Do $get $charm/include/cc*.h .
for header in persistent.h debug-conv.h charm-api.h charm.h
do
	Do $get $charm/include/$header .
done

# grab all the real headers:
for header in ckhashtable.h cklists.h ckvector3d.h \
	pup_c.h pupf.h pup.h pup_toNetwork4.h pup_toNetwork.h pup_mpi.h
do
	Do $get $charm/include/$header .
done

# grab all the source files
for source in ckhashtable.C ckstandalone.C \
	pup_util.C pup_c.C pup_toNetwork4.C pup_toNetwork.C
do
	Do $get $charm/tmp/$source .
done

Do $get $charm/include/tcharm*.h .
# Do $get $charm/tmp/libs/conv-libs/metis/Lib/*.[ch] .
Do $get $charm/tmp/libs/ck-libs/parmetis/METISLib/*.[ch] .
Do $get $charm/tmp/libs/ck-libs/idxl/*.[Ch] .
Do $get $charm/tmp/libs/ck-libs/fem/*.[Ch] .

echo "Source code collected."
echo " To build FEM alone, cd into fem_alone and build "
echo "  all the .c and .C source files using a command like:"
echo "   > mpicc -I. -DFEM_ALONE=1 -c *.c *.C " 
echo "   > ar cr libfem_alone.a *.o "

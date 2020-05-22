# This Makefile.am generated automatically by f77tof90
# from test/mpi/f77/init/Makefile.am.  DO NOT EDIT
# -*- Mode: Makefile; -*-
# vim: set ft=automake :
#
# (C) 2011 by Argonne National Laboratory.
#     See COPYRIGHT in top-level directory.
#

include $(top_srcdir)/Makefile_f90.mtest

EXTRA_DIST = testlist

# avoid having to write many "foo_SOURCES = foo.f90" lines
AM_DEFAULT_SOURCE_EXT = .f90

noinst_PROGRAMS = baseenvf90

EXTRA_PROGRAMS = checksizes
# Checksizes is created by configure, and is a C program that tests that 
# configure has correctly computed the sizes of MPI_Aint and MPI_Offset
# for Fortran.  This is not run by default but can be used to check
# that configure found the correct values.
checksizes_LDADD = 
nodist_checksizes_SOURCES = checksizes.c

DISTCLEANFILES = checksizes.c


if MAINTAINER_MODE

# f77tof90 only replaces output files if the contents differ, so we must use a
# separate timestamp file in order avoid constantly trying to remake the f90
# copy of the Makefile when the f77 copy is updated in a way that does not
# change its contents
$(srcdir)/Makefile.am: $(srcdir)/Makefile.am-stamp

$(srcdir)/Makefile.am-stamp: $(master_top_srcdir)/test/mpi/f77/init/Makefile.am $(master_top_srcdir)/maint/f77tof90
	( cd $(master_top_srcdir) && ./maint/f77tof90 test/mpi/f77/init test/mpi/f90/init Makefile.am Makefile.ap )

MAINTAINERCLEANFILES = Makefile.am-stamp
EXTRA_DIST += Makefile.am-stamp  

endif MAINTAINER_MODE


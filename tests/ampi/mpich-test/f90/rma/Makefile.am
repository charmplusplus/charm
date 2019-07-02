# This Makefile.am generated automatically by f77tof90
# from test/mpi/f77/rma/Makefile.am.  DO NOT EDIT
# -*- Mode: Makefile; -*-
# vim: set ft=automake :
#
# (C) 2011 by Argonne National Laboratory.
#     See COPYRIGHT in top-level directory.
#

include $(top_srcdir)/Makefile_f90.mtest

EXTRA_DIST = testlist.in

# avoid having to write many "foo_SOURCES = foo.f90" lines
AM_DEFAULT_SOURCE_EXT = .f90

noinst_PROGRAMS = \
    winscale1f90    \
    winscale2f90    \
    winfencef90     \
    wingetf90       \
    winerrf90       \
    winnamef90      \
    wingroupf90     \
    winaccf90       \
    winaccf90       \
    c2f2cwinf90     \
    baseattrwinf90  \
    winattrf90      \
    winattr2f90

if BUILD_MPIX_TESTS
noinst_PROGRAMS += aintf90
endif

## this test was commented out in the simplemake version...
##    allocmem

c2f2cwinf90_SOURCES = c2f2cwinf90.f90 c2f902cwin.c

# see include note in Makefile_f90.mtest
AM_FFLAGS += -I../attr

## addsize.h will be distributed because it's listed in AC_CONFIG_FILES/AC_OUTPUT

# ensure that dependent tests will be rebuilt when addsize.h is updated

## DJG: the following comment+code was present in the simplemake version... I'm
## not sure how accurate it is.
# bindir may be used in MPIEXEC, so make sure it is defined
BINDIR=${bindir}


if MAINTAINER_MODE

# f77tof90 only replaces output files if the contents differ, so we must use a
# separate timestamp file in order avoid constantly trying to remake the f90
# copy of the Makefile when the f77 copy is updated in a way that does not
# change its contents
$(srcdir)/Makefile.am: $(srcdir)/Makefile.am-stamp

$(srcdir)/Makefile.am-stamp: $(master_top_srcdir)/test/mpi/f77/rma/Makefile.am $(master_top_srcdir)/maint/f77tof90
	( cd $(master_top_srcdir) && ./maint/f77tof90 test/mpi/f77/rma test/mpi/f90/rma Makefile.am Makefile.ap )

MAINTAINERCLEANFILES = Makefile.am-stamp
EXTRA_DIST += Makefile.am-stamp  

endif MAINTAINER_MODE


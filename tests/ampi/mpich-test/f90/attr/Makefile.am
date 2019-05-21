# This Makefile.am generated automatically by f77tof90
# from test/mpi/f77/attr/Makefile.am.  DO NOT EDIT
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

noinst_PROGRAMS =                   \
    attrmpi1f90                       \
    baseattrf90 baseattr2f90            \
    commattrf90 commattr2f90            \
    commattr3f90 commattr4f90           \
    typeattrf90 typeattr2f90 typeattr3f90

## attraints.h will be distributed because it's listed in AC_CONFIG_FILES/AC_OUTPUT

# ensure that "{comm,type}attr*f.o" will be rebuilt when attraints.h is updated


if MAINTAINER_MODE

# f77tof90 only replaces output files if the contents differ, so we must use a
# separate timestamp file in order avoid constantly trying to remake the f90
# copy of the Makefile when the f77 copy is updated in a way that does not
# change its contents
$(srcdir)/Makefile.am: $(srcdir)/Makefile.am-stamp

$(srcdir)/Makefile.am-stamp: $(master_top_srcdir)/test/mpi/f77/attr/Makefile.am $(master_top_srcdir)/maint/f77tof90
	( cd $(master_top_srcdir) && ./maint/f77tof90 test/mpi/f77/attr test/mpi/f90/attr Makefile.am Makefile.ap )

MAINTAINERCLEANFILES = Makefile.am-stamp
EXTRA_DIST += Makefile.am-stamp testlist.ap Makefile.ap

endif MAINTAINER_MODE

## vim: set ft=automake :

noinst_PROGRAMS += fandcattrf90 baseattr3f90 attrlangf90
fandcattrf90_SOURCES = fandcattrf90.f90 fandcattrc.c
baseattr3f90_SOURCES = baseattr3f90.f90
attrlangf90_SOURCES = attrlangf90.f90 attrlangc.c

# Remove any created module files, if we know the extension of those files
clean-local:
	if [ "@FCMODEXT@" != "" ] ; then rm -f *.@FCMODEXT@ ; fi



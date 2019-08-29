# This Makefile.am generated automatically by f77tof90
# from test/mpi/f77/ext/Makefile.am.  DO NOT EDIT
# -*- Mode: Makefile; -*-
# vim: set ft=automake :
#
# (C) 2011 by Argonne National Laboratory.
#     See COPYRIGHT in top-level directory.
#

include $(top_srcdir)/Makefile_f90.mtest

EXTRA_DIST = testlist.in

# allocmemf is an "extra" program because it requires a Fortran extension
EXTRA_PROGRAMS = allocmemf90

noinst_PROGRAMS = c2f2cf90 ctypesinf90 c2f90mult

allocmemf90_SOURCES = allocmemf90.f90
c2f2cf90_SOURCES = c2f2cf90.f90 c2f902c.c
ctypesinf90_SOURCES = ctypesinf90.f90 ctypesfromc.c

# C programs get a different mtest utility object
c2f90mult_LDADD = $(top_builddir)/util/mtest.o
c2f90mult_SOURCES = c2f90mult.c

## add1size.h will be distributed because it's listed in AC_CONFIG_FILES/AC_OUTPUT

# ensure that dependent tests will be rebuilt when add1size.h is updated

# we don't get this from Makefile_f90.mtest and we don't include Makefile.mtest
$(top_builddir)/util/mtest.o:
	(cd $(top_builddir)/util && $(MAKE) mtest.o)


if MAINTAINER_MODE

# f77tof90 only replaces output files if the contents differ, so we must use a
# separate timestamp file in order avoid constantly trying to remake the f90
# copy of the Makefile when the f77 copy is updated in a way that does not
# change its contents
$(srcdir)/Makefile.am: $(srcdir)/Makefile.am-stamp

$(srcdir)/Makefile.am-stamp: $(master_top_srcdir)/test/mpi/f77/ext/Makefile.am $(master_top_srcdir)/maint/f77tof90
	( cd $(master_top_srcdir) && ./maint/f77tof90 test/mpi/f77/ext test/mpi/f90/ext Makefile.am Makefile.ap )

MAINTAINERCLEANFILES = Makefile.am-stamp
EXTRA_DIST += Makefile.am-stamp  Makefile.ap

endif MAINTAINER_MODE

## vim: set ft=automake :

BUILT_SOURCES = c2f902c.c ctypesfromc.c

# FIXME what's up with these rules?  They appear to copy a file to itself in
# non-VPATH builds...
c2f902c.c: $(srcdir)/../../f90/ext/c2f902c.c
	cp $(srcdir)/../../f90/ext/c2f902c.c c2f902c.c
ctypesfromc.c: $(srcdir)/../../f77/ext/ctypesfromc.c
	cp $(srcdir)/../../f77/ext/ctypesfromc.c ctypesfromc.c



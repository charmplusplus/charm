# This Makefile.am generated automatically by f77tof90
# from test/mpi/f77/util/Makefile.am.  DO NOT EDIT
## -*- Mode: Makefile; -*-
## vim: set ft=automake :
##
## (C) 2011 by Argonne National Laboratory.
##     See COPYRIGHT in top-level directory.
##

## list a dummy library that we don't actually build in order to cause automake
## to emit a rule for building mtestf90.o from mtestf90.f90
EXTRA_LIBRARIES = libmtest_dummy.a
libmtest_dummy_a_SOURCES = mtestf90.f90

all-local: mtestf90.$(OBJEXT)


if MAINTAINER_MODE

# f77tof90 only replaces output files if the contents differ, so we must use a
# separate timestamp file in order avoid constantly trying to remake the f90
# copy of the Makefile when the f77 copy is updated in a way that does not
# change its contents
$(srcdir)/Makefile.am: $(srcdir)/Makefile.am-stamp

$(srcdir)/Makefile.am-stamp: $(master_top_srcdir)/test/mpi/f77/util/Makefile.am $(master_top_srcdir)/maint/f77tof90
	( cd $(master_top_srcdir) && ./maint/f77tof90 test/mpi/f77/util test/mpi/f90/util Makefile.am Makefile.ap )

MAINTAINERCLEANFILES = Makefile.am-stamp
EXTRA_DIST = Makefile.am-stamp  

endif MAINTAINER_MODE


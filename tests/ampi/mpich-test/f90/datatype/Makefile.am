# This Makefile.am generated automatically by f77tof90
# from test/mpi/f77/datatype/Makefile.am.  DO NOT EDIT
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

noinst_PROGRAMS = typenamef90 typesnamef90 typecntsf90 typesubf90 typem2f90 gaddressf90 \
                  packef90 allctypesf90 hindex1f90 hindexed_blockf90 typename3f90 bottom

bottom_SOURCES=bottomc.c bottomf90.f90

## typeaints.h will be distributed because it's listed in AC_CONFIG_FILES/AC_OUTPUT

## ensure that dependent tests will be rebuilt when typeaints.h is updated

if MAINTAINER_MODE

# f77tof90 only replaces output files if the contents differ, so we must use a
# separate timestamp file in order avoid constantly trying to remake the f90
# copy of the Makefile when the f77 copy is updated in a way that does not
# change its contents
$(srcdir)/Makefile.am: $(srcdir)/Makefile.am-stamp

$(srcdir)/Makefile.am-stamp: $(master_top_srcdir)/test/mpi/f77/datatype/Makefile.am $(master_top_srcdir)/maint/f77tof90
	( cd $(master_top_srcdir) && ./maint/f77tof90 test/mpi/f77/datatype test/mpi/f90/datatype Makefile.am Makefile.ap )

MAINTAINERCLEANFILES = Makefile.am-stamp
EXTRA_DIST += Makefile.am-stamp testlist.ap Makefile.ap

endif MAINTAINER_MODE

## vim: set ft=automake :

noinst_PROGRAMS += structf indtype createf90 sizeof kinds trf90 get_elem_d get_elem_u

structf_SOURCES    = structf.f90
indtype_SOURCES    = indtype.f90
createf90_SOURCES  = createf90.f90
sizeof_SOURCES     = sizeof.f90
kinds_SOURCES      = kinds.f90
trf90_SOURCES      = trf90.f90
get_elem_d_SOURCES = get_elem_d.f90
get_elem_u_SOURCES = get_elem_u.f90


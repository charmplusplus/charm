Internals of the FEM Framework
  Orion Sky Lawlor, olawlor@acm.org, 2004/7/9

This directory implements a C, C++, or Fortran-callable
library for parallel unstructured mesh computations.

The basic idea is to partition the serial mesh into
"chunks", which do nearly independent computations;
but occasionally neighboring mesh chunks
synchronize their arrays (called fields) at the shared
nodes or ghosts.  For load balance, we create more chunks 
than processors and occasionally migrate chunks for better
load balance.

This library is build entirely on top of (A)MPI,
and users can mix and match FEM calls with MPI calls.
The implementation contains with a very few TCHARM calls,
mostly for migration and shutdown support. 


MAJOR FILES (in order of decreasing importance):

fem.h:
  C user include file containing all C-callable routines.

femf.h:
  F90 user include file containing F90-callable routines.

fem_impl.h:
  Central header file which brings in all other
  headers, and provides glue and utility routines.

fem_mesh.h:
  The mesh classes used everywhere to represent 
  the user's mesh.  This is the main set of classes
  used to do work in FEM.

fem_mesh.C:
  Small, self-contained methods of the mesh classes.

fem.C: 
  All remaining interface/API routines: startup and
  shutdown, mesh manipulation API calls, and all other
  leftovers.

fem_compat.C:
  Implementations of "backward compatability" routines,
  like FEM_Set_conn, that are officially superceded by
  something more general.
  
femmain.C:
  Main routine which calls "init" and "driver".

partition.C:
  First step in mesh partitioning: call Metis to 
  decide where each element goes.

map.C:
  Second step in mesh partitioning: divide up the
  element and node data, build communication lists,
  and build ghosts.

symmetries.C:
  Supports various types of "symmetries" to the problem:
  rotational or translational periodicity, mirror symmetry.
  Support for symmetries is still fairly immature.
  

MINOR FILES:
call_init.c:
  A tiny wrapper for the "init" routine.
  Required to work around a compiler bug
  which prevents us from calling "init" directly.

cktimer.h: 
  Small "sentinal" class used for timing partitioning.
  Could eventually be moved to charm/src/util.

compat_*:
  Tiny, empty implementations of C and F90
  "init" and "driver" routines.  For example,
  a C user's real "init" or "driver" would 
  replace the C implementations; but we still
  need the F90 implementations to avoid a link
  error.
  
  We wouldn't need this weirdness if we just built
  a separate version of the library for C and 
  Fortran, but it's nicer to have only one library.

libmodulefem.dep:
  Link-time dependencies for FEM library.  Read by charmc.

make_fem_alone.sh:
  Script to build a version of the FEM framework
  on top of native MPI, not Charm++/AMPI.


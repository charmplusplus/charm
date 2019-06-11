
--------------------------------------------
| Privatization of NAS Parallel Benchmarks |
--------------------------------------------

AMPI virtualizes the ranks of MPI_COMM_WORLD as user-level threads rather
than OS processes. This means that global variables are shared between
multiple AMPI ranks, rather than private to each rank.
All global variables that are written to more than once and which are
not written to the same value on all MPI ranks must be encapsulated.
All PARAMETER variables are safe, as are many globals that are
store global input values that are written to only once at startup.
Note that module variables, as well as explicit and implicit
SAVE variables are all forms of global variables.

In order to privatize global variables of all kinds in Fortran programs,
we encapsulate the variables in a derived type which is allocated per
(A)MPI rank, and then all references to those variables must be updated
to access them through the derived type.

You can find more information on privatization in the AMPI manual here:
http://charm.cs.illinois.edu/manuals/html/ampi/manual.html


List of global variables (common blocks) in
mpinpb.h
    /mpistuff/
mg.f
    In MPI_main
        /noautom/
    In setup
        /grid/
    In zran3
        /grid/
globals.h
    /mg3/
    /ClassType/
    /my_debug/
    /fap/
    /comm_ex/
    /buffer/
    /timers/

Removed reading of data from file mg.input (lines 144-152 of subroutine MPI_main in mg.f)


Notes on privatization:

     
c-- Privatizing common block mpistuff --------------------------------

!$OMP THREADPRIVATE(/mpistuff/)


! This file created from test/mpi/f77/ext/allocmemf.f with f77tof90
! -*- Mode: Fortran; -*- 
!
!  (C) 2004 by Argonne National Laboratory.
!      See COPYRIGHT in top-level directory.
!
        program main
        use mpi
!
! This program makes use of a common (but not universal; g77 doesn't 
! have it) extension: the "Cray" pointer.  This allows MPI_Alloc_mem
! to allocate memory and return it to Fortran, where it can be used.
! As this is not standard Fortran, this test is not run by default.
! To run it, build (with a suitable compiler) and run with
!   mpiexec -n 1 ./allocmemf
!
        real a
        pointer (p,a(100,100))
        integer (kind=MPI_ADDRESS_KIND) asize

        integer ierr, sizeofreal, errs
        integer i,j
!
        errs = 0
        call mtest_init(ierr)
        call mpi_type_size( MPI_REAL, sizeofreal, ierr )
! Make sure we pass in an integer of the correct type
        asize = sizeofreal * 100 * 100
        call mpi_alloc_mem( asize,MPI_INFO_NULL,p,ierr )

        do i=1,100
            do j=1,100
                a(i,j) = -1
            enddo
        enddo
        a(3,5) = 10.0

        call mpi_free_mem( a, ierr )
        call mtest_finalize(errs)
        call mpi_finalize(ierr)

        end

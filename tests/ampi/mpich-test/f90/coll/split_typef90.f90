! This file created from test/mpi/f77/coll/split_typef.f with f77tof90
! -*- Mode: Fortran; -*-
!
!  (C) 2012 by Argonne National Laboratory.
!      See COPYRIGHT in top-level directory.
!
      subroutine mpi_main
      use mpi
      integer ierr, errs
      integer i, ans, size, rank, color, comm, newcomm
      integer maxSize, displ
      parameter (maxSize=128)
      integer scounts(maxSize), sdispls(maxSize), stypes(maxSize)
      integer rcounts(maxSize), rdispls(maxSize), rtypes(maxSize)
      integer sbuf(maxSize), rbuf(maxSize)

      errs = 0

      call mtest_init( ierr )

      call mpi_comm_dup( MPI_COMM_WORLD, comm, ierr )

      call mpi_comm_split_type( comm, MPI_COMM_TYPE_SHARED, rank, &
      &     MPI_INFO_NULL, newcomm, ierr )
      call mpi_comm_rank( newcomm, rank, ierr )
      call mpi_comm_size( newcomm, size, ierr )

      do i=1, size
         scounts(i) = 1
         sdispls(i) = (i-1)
         stypes(i)  = MPI_INTEGER
         sbuf(i) = rank * size + i
         rcounts(i) = 1
         rdispls(i) = (i-1)
         rtypes(i)  = MPI_INTEGER
         rbuf(i) = -1
      enddo
      call mpi_alltoallv( sbuf, scounts, sdispls, stypes, &
      &     rbuf, rcounts, rdispls, rtypes, newcomm, ierr )

      call mpi_comm_free( newcomm, ierr )
      call mpi_comm_free( comm, ierr )

      call mtest_finalize( errs )
      call mpi_finalize( ierr )
      end subroutine mpi_main

recursive subroutine MPI_Main
  implicit none
  include 'mpif.h'

  integer :: thisIndex, ierr, nblocks, i
  double precision :: inval, outval, expect

  call MPI_Init(ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD, thisIndex, ierr)
  call MPI_Comm_size(MPI_COMM_WORLD, nblocks, ierr)

  inval = thisIndex + 1
  call MPI_Allreduce(inval, outval, 1, MPI_DOUBLE_PRECISION, MPI_SUM, &
&                     MPI_COMM_WORLD, ierr)

  expect = (nblocks*(nblocks+1))/2
  if (outval .eq. expect) then
    call MPI_Print('allreduce test passed',21)
  else
    call MPI_Print('allreduce test failed',21)
  end if
  call MPI_Finalize(ierr)

end subroutine

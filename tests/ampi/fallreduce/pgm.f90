subroutine MPI_Main
  implicit none
  include 'mpif.h'

  integer :: myrank, ierr, numranks, req
  double precision :: inval, outval, expect

  call MPI_Init(ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD, myrank, ierr)
  call MPI_Comm_size(MPI_COMM_WORLD, numranks, ierr)

  inval = myrank + 1
  expect = (numranks*(numranks+1))/2

  call MPI_Allreduce(inval, outval, 1, MPI_DOUBLE_PRECISION, MPI_SUM, &
                     MPI_COMM_WORLD, ierr)

  if (myrank .eq. 0) then
    if (outval .eq. expect) then
      print *, 'MPI_Allreduce test passes'
    else
      print *, 'MPI_Allreduce test failed'
    end if
  end if

  call MPI_Iallreduce(inval, outval, 1, MPI_DOUBLE_PRECISION, MPI_SUM, &
                      MPI_COMM_WORLD, req, ierr)

  if (myrank .eq. 0) then
    call MPI_Wait(req, MPI_STATUS_IGNORE, ierr)
    if (outval .eq. expect) then
      print *, 'MPI_Iallreduce test passes'
    else
      print *, 'MPI_Iallreduce test failed'
    end if
  end if

  call MPI_Finalize(ierr)

end subroutine

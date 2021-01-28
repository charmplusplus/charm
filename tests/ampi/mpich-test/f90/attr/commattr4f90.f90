! This file created from test/mpi/f77/attr/commattr4f.f with f77tof90
!
      program main
!
      use mpi

      integer    ierr
      integer    errs
      logical    found
      integer    comm2
      integer    key
      integer (kind=MPI_ADDRESS_KIND) extrastate, valin, valout, val


      errs = 0
!
!  initialize the mpi environment
!
      call mpi_init(ierr)

      call mpi_comm_create_keyval(MPI_COMM_DUP_FN, &
      &     MPI_NULL_DELETE_FN, &
      &     key, &
      &     extrastate, &
      &     ierr)
!
!  set a value for the attribute
!
      valin = huge(valin)
!
!  set attr in comm_world
!
      call mpi_comm_set_attr(MPI_COMM_WORLD, &
      &     key, &
      &     valin, &
      &     ierr)
      call mpi_comm_get_attr(MPI_COMM_WORLD, &
      &     key, &
      &     valout, &
      &     found, &
      &     ierr)
      if (found .neqv. .true.) then
         print *, "mpi_comm_set_attr reported key, but not found on ", &
      &        "mpi_comm_world"
         errs = errs + 1
      else if (valout .ne. valin) then
         print *, "key found, but valin does not match valout"
         print *, valout, " != ", valin
         errs = errs + 1
      end if
!
!  dup the communicator, attribute should follow
!
      call mpi_comm_dup(MPI_COMM_WORLD, &
      &     comm2, &
      &     ierr)
!
!  get the value for the attribute
!
      call mpi_comm_get_attr(comm2, &
      &     key, &
      &     valout, &
      &     found, &
      &     ierr)
      if (found .neqv. .true.) then
         print *, "mpi_comm_set_attr reported key, but not found on ", &
      &        "duped comm"
         errs = errs + 1
      else if (valout .ne. valin) then
         print *, "key found, but value does not match that on ", &
      &        "mpi_comm_world"
         print *, valout, " != ", valin
         errs = errs + 1
      end if
!
!     free the duped communicator
!
      call mpi_comm_free(comm2, ierr)
!
!     free keyval
!
      call mpi_comm_delete_attr(MPI_COMM_WORLD, &
      &     key, ierr)
      call mpi_comm_free_keyval(key, &
      &     ierr)
      call mpi_finalize(ierr)

      if (errs .eq. 0) then
         print *, " No Errors"
      end if

      end

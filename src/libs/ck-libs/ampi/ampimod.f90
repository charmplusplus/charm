module AMPI

  integer, parameter :: AMPI_COMM_WORLD=0
  integer, parameter :: AMPI_STATUS_SIZE=4
  integer, parameter :: AMPI_MAX_COMM=8

  integer, parameter :: AMPI_DOUBLE_PRECISION=0
  integer, parameter :: AMPI_INTEGER=1
  integer, parameter :: AMPI_REAL=2
  integer, parameter :: AMPI_COMPLEX=3
  integer, parameter :: AMPI_LOGICAL=4
  integer, parameter :: AMPI_CHARACTER=5
  integer, parameter :: AMPI_BYTE=6
  integer, parameter :: AMPI_PACKED=7

  integer, parameter :: AMPI_SHORT=8
  integer, parameter :: AMPI_LONG=9
  integer, parameter :: AMPI_UNSIGNED_CHAR=10
  integer, parameter :: AMPI_UNSIGNED_SHORT=11
  integer, parameter :: AMPI_UNSIGNED=12
  integer, parameter :: AMPI_UNSIGNED_LONG=13
  integer, parameter :: AMPI_LONG_DOUBLE=14

  integer, parameter :: AMPI_MAX=1
  integer, parameter :: AMPI_MIN=2
  integer, parameter :: AMPI_SUM=3
  integer, parameter :: AMPI_PROD=4

  integer, parameter :: AMPI_SOURCE=2
  integer, parameter :: AMPI_TAG=1
  integer, parameter :: AMPI_COMM=3

  integer :: AMPI_COMM_UNIVERSE(1:AMPI_MAX_COMM)

  integer, external :: AMPI_Register
  double precision, external :: AMPI_Wtime

  external AMPI_Init_universe
  external AMPI_Comm_rank
  external AMPI_Comm_size
  external AMPI_Finalize
  external AMPI_Send
  external AMPI_Recv
  external AMPI_Isend
  external AMPI_Irecv
  external AMPI_Sendrecv
  external AMPI_Barrier
  external AMPI_Bcast
  external AMPI_Reduce
  external AMPI_Allreduce
  external AMPI_Start
  external AMPI_Waitall
  external AMPI_Waitany
  external AMPI_Wait
  external AMPI_Send_init
  external AMPI_Recv_init
  external AMPI_Type_contiguous
  external AMPI_Type_vector
  external AMPI_Type_hvector
  external AMPI_Type_indexed
  external AMPI_Type_hindexed
  external AMPI_Type_struct
  external AMPI_Type_commit
  external AMPI_Type_free
  external AMPI_Type_extent
  external AMPI_Type_size
  external AMPI_Allgatherv
  external AMPI_Allgather
  external AMPI_Gatherv
  external AMPI_Gather
  external AMPI_Alltoallv
  external AMPI_Alltoall
  external AMPI_Comm_dup
  external AMPI_Comm_free
  external AMPI_Abort
  external AMPI_Print
  external AMPI_Migrate

  external AMPI_Register_main
  external AMPI_Attach
  
contains

  subroutine AMPI_Init(ierr)
    integer :: ierr
    call AMPI_Init_universe(AMPI_COMM_UNIVERSE)
    ierr = 0
  end subroutine

end module AMPI

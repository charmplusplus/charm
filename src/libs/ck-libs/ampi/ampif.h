       integer, parameter :: MPI_SUCCESS = 0

       integer, parameter :: MPI_DOUBLE_PRECISION = 0
       integer, parameter :: MPI_INTEGER = 1
       integer, parameter :: MPI_REAL = 2
       integer, parameter :: MPI_COMPLEX = 3
       integer, parameter :: MPI_LOGICAL = 4
       integer, parameter :: MPI_CHARACTER = 5
       integer, parameter :: MPI_BYTE = 6
       integer, parameter :: MPI_PACKED = 7

       integer, parameter :: MPI_MAX = 1
       integer, parameter :: MPI_MIN = 2
       integer, parameter :: MPI_SUM = 3
       integer, parameter :: MPI_PROD = 4

       integer, parameter :: MPI_ANY_SOURCE = -1
       integer, parameter :: MPI_ANY_TAG = -1
       integer, parameter :: MPI_REQUEST_NULL = -1

       integer, parameter :: MPI_TYPE_NULL = -1

       integer, parameter :: MPI_TAG_UB = 1073741824

       integer, parameter :: MPI_STATUS_SIZE=4
       integer, parameter :: MPI_TAG=1
       integer, parameter :: MPI_SOURCE=2
       integer, parameter :: MPI_COMM=3

       integer, parameter :: MPI_COMM_FIRST_SPLIT = 1000000
       integer, parameter :: MPI_COMM_FIRST_GROUP = 2000000
       integer, parameter :: MPI_COMM_LAST_GROUP = 3000000
       integer, parameter :: MPI_COMM_WORLD = 8000000

       integer, parameter :: MPI_MAX_COMM_WORLDS=8
       integer :: MPI_COMM_UNIVERSE(1:MPI_MAX_COMM_WORLDS)

       integer, external :: MPI_Register
       double precision, external :: MPI_WTIME

        integer, parameter :: MPI_SUCCESS = 0
	
	integer, parameter :: MPI_DOUBLE_PRECISION = 0
	integer, parameter :: MPI_INTEGER = 1
	integer, parameter :: MPI_REAL = 2
	integer, parameter :: MPI_COMPLEX = 3
	integer, parameter :: MPI_LOGICAL = 4
	integer, parameter :: MPI_CHARACTER = 5
	integer, parameter :: MPI_BYTE = 6
	integer, parameter :: MPI_PACKED = 7
	
	integer, parameter :: MPI_COMM_WORLD = 0
	integer, parameter :: MPI_ANY_SOURCE = -1
	integer, parameter :: MPI_ANY_TAG = -1
	integer, parameter :: MPI_REQUEST_NULL = -1
	integer, parameter :: MPI_TYPE_NULL = -1
	
	integer, parameter :: MPI_MAX = 1
	integer, parameter :: MPI_MIN = 2
	integer, parameter :: MPI_SUM = 3
	integer, parameter :: MPI_PROD = 4

	integer, parameter :: MPI_TAG_UB = 1024
	
	double precision, external :: MPI_WTIME

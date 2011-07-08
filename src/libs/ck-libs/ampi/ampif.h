       integer, parameter :: MPI_SUCCESS                 =0
       integer, parameter :: MPI_ERR_BUFFER              =1
       integer, parameter :: MPI_ERR_COUNT               =2
       integer, parameter :: MPI_ERR_TYPE                =3
       integer, parameter :: MPI_ERR_TAG                 =4
       integer, parameter :: MPI_ERR_COMM                =5
       integer, parameter :: MPI_ERR_RANK                =6
       integer, parameter :: MPI_ERR_REQUEST             =7
       integer, parameter :: MPI_ERR_ROOT                =8
       integer, parameter :: MPI_ERR_GROUP               =9
       integer, parameter :: MPI_ERR_OP                  =10
       integer, parameter :: MPI_ERR_TOPOLOGY            =11
       integer, parameter :: MPI_ERR_DIMS                =12
       integer, parameter :: MPI_ERR_ARG                 =13
       integer, parameter :: MPI_ERR_UNKNOWN             =14
       integer, parameter :: MPI_ERR_TRUNCATE            =15
       integer, parameter :: MPI_ERR_OTHER               =16
       integer, parameter :: MPI_ERR_INTERN              =17
       integer, parameter :: MPI_ERR_IN_STATUS           =18
       integer, parameter :: MPI_ERR_PENDING             =19
       integer, parameter :: MPI_ERR_ACCESS              =20
       integer, parameter :: MPI_ERR_AMODE               =21
       integer, parameter :: MPI_ERR_ASSERT              =22
       integer, parameter :: MPI_ERR_BAD_FILE            =23
       integer, parameter :: MPI_ERR_BASE                =24
       integer, parameter :: MPI_ERR_CONVERSION          =25
       integer, parameter :: MPI_ERR_DISP                =26
       integer, parameter :: MPI_ERR_DUP_DATAREP         =27
       integer, parameter :: MPI_ERR_FILE_EXISTS         =28
       integer, parameter :: MPI_ERR_FILE_IN_USE         =29
       integer, parameter :: MPI_ERR_FILE                =30
       integer, parameter :: MPI_ERR_INFO_KEY            =31
       integer, parameter :: MPI_ERR_INFO_NOKEY          =32
       integer, parameter :: MPI_ERR_INFO_VALUE          =33
       integer, parameter :: MPI_ERR_INFO                =34
       integer, parameter :: MPI_ERR_IO                  =35
       integer, parameter :: MPI_ERR_KEYVAL              =36
       integer, parameter :: MPI_ERR_LOCKTYPE            =37
       integer, parameter :: MPI_ERR_NAME                =38
       integer, parameter :: MPI_ERR_NO_MEM              =39
       integer, parameter :: MPI_ERR_NOT_SAME            =40
       integer, parameter :: MPI_ERR_NO_SPACE            =41
       integer, parameter :: MPI_ERR_NO_SUCH_FILE        =42
       integer, parameter :: MPI_ERR_PORT                =43
       integer, parameter :: MPI_ERR_QUOTA               =44
       integer, parameter :: MPI_ERR_READ_ONLY           =45
       integer, parameter :: MPI_ERR_RMA_CONFLICT        =46
       integer, parameter :: MPI_ERR_RMA_SYNC            =47
       integer, parameter :: MPI_ERR_SERVICE             =48
       integer, parameter :: MPI_ERR_SIZE                =49
       integer, parameter :: MPI_ERR_SPAWN               =50
       integer, parameter :: MPI_ERR_UNSUPPORTED_DATAREP =51
       integer, parameter :: MPI_ERR_UNSUPPORTED_OPERATION =52
       integer, parameter :: MPI_ERR_WIN                 =53
       integer, parameter :: MPI_ERR_LASTCODE            =53

       integer, parameter :: MPI_ERRORS_ARE_FATAL        =119
       integer, parameter :: MPI_ERRORS_RETURN           =120

       integer, parameter :: MPI_MAX_PROCESSOR_NAME = 256
       integer, parameter :: MPI_MAX_ERROR_STRING = 256

       integer, parameter :: MPI_DATATYPE_NULL = -1
       integer, parameter :: MPI_DOUBLE_PRECISION = 0
       integer, parameter :: MPI_REAL8 = 0
       integer, parameter :: MPI_INTEGER = 1
       integer, parameter :: MPI_INTEGER4 = 1
       integer, parameter :: MPI_REAL = 2
       integer, parameter :: MPI_COMPLEX = 3
       integer, parameter :: MPI_LOGICAL = 4
       integer, parameter :: MPI_CHARACTER = 5
       integer, parameter :: MPI_BYTE = 6
       integer, parameter :: MPI_PACKED = 7
       integer, parameter :: MPI_2REAL = 21
       integer, parameter :: MPI_2DOUBLE_PRECISION = 22
       integer, parameter :: MPI_2INTEGER = 18
       integer, parameter :: MPI_LB = 23
       integer, parameter :: MPI_UB = 24
       integer, parameter :: MPI_DOUBLE_COMPLEX = 26
       

       integer, parameter :: MPI_PROC_NULL = -2
       integer, parameter :: MPI_ANY_SOURCE = -1
       integer, parameter :: MPI_ANY_TAG = -1
       integer, parameter :: MPI_REQUEST_NULL = -1
       integer, parameter :: MPI_GROUP_NULL = -1
       integer, parameter :: MPI_COMM_NULL = -1
       integer, parameter :: MPI_TYPE_NULL = -1
       integer, parameter :: MPI_KEYVAL_INVALID = -1
       integer, parameter :: MPI_INFO_NULL = -1

       integer, pointer   :: MPI_IN_PLACE

       integer, parameter :: MPI_BOTTOM = 0
       integer, parameter :: MPI_UNDEFINED = -32766

       integer, parameter :: MPI_IDENT    = 0
       integer, parameter :: MPI_SIMILAR  = 1
       integer, parameter :: MPI_UNEQUAL  = 2

       integer, parameter :: MPI_OP_NULL = 0

       integer, parameter :: MPI_GRAPH = 1
       integer, parameter :: MPI_CART = 2

       integer, parameter :: MPI_TAG_UB = -10

       integer, parameter :: MPI_STATUS_SIZE=8
       integer, parameter :: MPI_TAG=1
       integer, parameter :: MPI_SOURCE=2
       integer, parameter :: MPI_COMM=3

       integer, parameter :: MPI_COMM_FIRST_SPLIT = 1000000
       integer, parameter :: MPI_COMM_FIRST_GROUP = 2000000
       integer, parameter :: MPI_COMM_FIRST_CART  = 3000000
       integer, parameter :: MPI_COMM_FIRST_GRAPH = 4000000
       integer, parameter :: MPI_COMM_FIRST_INTER = 5000000
       integer, parameter :: MPI_COMM_FIRST_INTRA = 6000000
       integer, parameter :: MPI_COMM_FIRST_RESVD = 7000000
       integer, parameter :: MPI_COMM_SELF = 8000000
       integer, parameter :: MPI_COMM_WORLD = 9000000
       integer, parameter :: MPI_MAX_COMM_WORLDS=8
       integer :: MPI_COMM_UNIVERSE(1:MPI_MAX_COMM_WORLDS)

!       integer, external :: MPI_Register
       double precision, external :: MPI_WTIME
       double precision, external :: MPI_WTICK
       integer, parameter :: MPI_MAX = 100
       integer, parameter :: MPI_MIN = 101
       integer, parameter :: MPI_SUM = 102
       integer, parameter :: MPI_PROD = 103
       integer, parameter :: MPI_LAND = 104
       integer, parameter :: MPI_BAND = 105
       integer, parameter :: MPI_LOR = 106
       integer, parameter :: MPI_BOR = 107
       integer, parameter :: MPI_LXOR = 108
       integer, parameter :: MPI_BXOR = 109
       integer, parameter :: MPI_MAXLOC = 110
       integer, parameter :: MPI_MINLOC = 111
!
       integer, parameter :: MPI_OFFSET_KIND = 8
       integer, parameter :: MPI_ADDRESS_KIND = 8


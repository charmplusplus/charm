       !!! Values here must match those in ampi.h, or otherwise be !!!
       !!! converted from the Fortran to the C value in ampif.C    !!!
       !!! Note: we can't have more than 72 characters per line,   !!!
       !!!       the first 6 characters per line must be blank,    !!!
       !!!       and we can't have any line continuations in this  !!!
       !!!       file in order to support all three of fixed-form, !!!
       !!!       free-form, and line-extended fixed-form source.   !!!

       integer, parameter :: MPI_SUCCESS                   = 0
       integer, parameter :: MPI_ERR_BUFFER                = 1
       integer, parameter :: MPI_ERR_COUNT                 = 2
       integer, parameter :: MPI_ERR_TYPE                  = 3
       integer, parameter :: MPI_ERR_TAG                   = 4
       integer, parameter :: MPI_ERR_COMM                  = 5
       integer, parameter :: MPI_ERR_RANK                  = 6
       integer, parameter :: MPI_ERR_REQUEST               = 7
       integer, parameter :: MPI_ERR_ROOT                  = 8
       integer, parameter :: MPI_ERR_GROUP                 = 9
       integer, parameter :: MPI_ERR_OP                    = 10
       integer, parameter :: MPI_ERR_TOPOLOGY              = 11
       integer, parameter :: MPI_ERR_DIMS                  = 12
       integer, parameter :: MPI_ERR_ARG                   = 13
       integer, parameter :: MPI_ERR_UNKNOWN               = 14
       integer, parameter :: MPI_ERR_TRUNCATE              = 15
       integer, parameter :: MPI_ERR_OTHER                 = 16
       integer, parameter :: MPI_ERR_INTERN                = 17
       integer, parameter :: MPI_ERR_IN_STATUS             = 18
       integer, parameter :: MPI_ERR_PENDING               = 19
       integer, parameter :: MPI_ERR_ACCESS                = 20
       integer, parameter :: MPI_ERR_AMODE                 = 21
       integer, parameter :: MPI_ERR_ASSERT                = 22
       integer, parameter :: MPI_ERR_BAD_FILE              = 23
       integer, parameter :: MPI_ERR_BASE                  = 24
       integer, parameter :: MPI_ERR_CONVERSION            = 25
       integer, parameter :: MPI_ERR_DISP                  = 26
       integer, parameter :: MPI_ERR_DUP_DATAREP           = 27
       integer, parameter :: MPI_ERR_FILE_EXISTS           = 28
       integer, parameter :: MPI_ERR_FILE_IN_USE           = 29
       integer, parameter :: MPI_ERR_FILE                  = 30
       integer, parameter :: MPI_ERR_INFO_KEY              = 31
       integer, parameter :: MPI_ERR_INFO_NOKEY            = 32
       integer, parameter :: MPI_ERR_INFO_VALUE            = 33
       integer, parameter :: MPI_ERR_INFO                  = 34
       integer, parameter :: MPI_ERR_IO                    = 35
       integer, parameter :: MPI_ERR_KEYVAL                = 36
       integer, parameter :: MPI_ERR_LOCKTYPE              = 37
       integer, parameter :: MPI_ERR_NAME                  = 38
       integer, parameter :: MPI_ERR_NO_MEM                = 39
       integer, parameter :: MPI_ERR_NOT_SAME              = 40
       integer, parameter :: MPI_ERR_NO_SPACE              = 41
       integer, parameter :: MPI_ERR_NO_SUCH_FILE          = 42
       integer, parameter :: MPI_ERR_PORT                  = 43
       integer, parameter :: MPI_ERR_QUOTA                 = 44
       integer, parameter :: MPI_ERR_READ_ONLY             = 45
       integer, parameter :: MPI_ERR_RMA_CONFLICT          = 46
       integer, parameter :: MPI_ERR_RMA_SYNC              = 47
       integer, parameter :: MPI_ERR_SERVICE               = 48
       integer, parameter :: MPI_ERR_SIZE                  = 49
       integer, parameter :: MPI_ERR_SPAWN                 = 50
       integer, parameter :: MPI_ERR_UNSUPPORTED_DATAREP   = 51
       integer, parameter :: MPI_ERR_UNSUPPORTED_OPERATION = 52
       integer, parameter :: MPI_ERR_WIN                   = 53
       integer, parameter :: MPI_ERR_LASTCODE              = 53
       integer, parameter :: MPI_LASTUSEDCODE          = 53

       integer, parameter :: MPI_ERRHANDLER_NULL  = 0
       integer, parameter :: MPI_ERRORS_RETURN    = 1
       integer, parameter :: MPI_ERRORS_ARE_FATAL = 2

       integer, parameter :: MPI_MAX_PROCESSOR_NAME         = 256
       integer, parameter :: MPI_MAX_ERROR_STRING           = 256
       integer, parameter :: MPI_MAX_LIBRARY_VERSION_STRING = 256

       integer, parameter :: MPI_VERSION    = 2
       integer, parameter :: MPI_SUBVERSION = 2

       integer, parameter :: MPI_THREAD_SINGLE     = 1
       integer, parameter :: MPI_THREAD_FUNNELED   = 2
       integer, parameter :: MPI_THREAD_SERIALIZED = 3
       integer, parameter :: MPI_THREAD_MULTIPLE   = 4

       integer, parameter :: MPI_DATATYPE_NULL     = -1
       integer, parameter :: MPI_REAL8             = 0
       integer, parameter :: MPI_DOUBLE_PRECISION  = MPI_REAL8
       integer, parameter :: MPI_INT               = 1
       integer, parameter :: MPI_INTEGER           = MPI_INT
       integer, parameter :: MPI_INTEGER4          = MPI_INT
       integer, parameter :: MPI_REAL              = 2
       integer, parameter :: MPI_REAL4             = MPI_REAL
       integer, parameter :: MPI_LOGICAL           = 3
       integer, parameter :: MPI_CHARACTER         = 5
       integer, parameter :: MPI_BYTE              = 6
       integer, parameter :: MPI_PACKED            = 7
       integer, parameter :: MPI_REAL16            = 14
       integer, parameter :: MPI_INTEGER8          = 15
       integer, parameter :: MPI_OFFSET            = MPI_INTEGER8
       integer, parameter :: MPI_AINT              = 27
       integer, parameter :: MPI_COUNT             = MPI_AINT
       integer, parameter :: MPI_LB                = 28
       integer, parameter :: MPI_UB                = 29
       integer, parameter :: MPI_2INTEGER          = 33
       integer, parameter :: MPI_2REAL             = 36
       integer, parameter :: MPI_2DOUBLE_PRECISION = 37
       integer, parameter :: MPI_COMPLEX           = 38
       integer, parameter :: MPI_COMPLEX8          = MPI_COMPLEX
       integer, parameter :: MPI_COMPLEX16         = 40
       integer, parameter :: MPI_DOUBLE_COMPLEX    = MPI_COMPLEX16
       integer, parameter :: MPI_COMPLEX32         = 41

       integer, parameter :: MPI_COMBINER_NAMED            = 1
       integer, parameter :: MPI_COMBINER_CONTIGUOUS       = 2
       integer, parameter :: MPI_COMBINER_VECTOR           = 3
       integer, parameter :: MPI_COMBINER_HVECTOR          = 4
       integer, parameter :: MPI_COMBINER_HVECTOR_INTEGER  = 5
       integer, parameter :: MPI_COMBINER_INDEXED          = 6
       integer, parameter :: MPI_COMBINER_HINDEXED         = 7
       integer, parameter :: MPI_COMBINER_HINDEXED_INTEGER = 8
       integer, parameter :: MPI_COMBINER_STRUCT           = 9
       integer, parameter :: MPI_COMBINER_STRUCT_INTEGER   = 10
       integer, parameter :: MPI_COMBINER_DARRAY           = 11
       integer, parameter :: MPI_COMBINER_RESIZED          = 12
       integer, parameter :: MPI_COMBINER_SUBARRAY         = 13
       integer, parameter :: MPI_COMBINER_INDEXED_BLOCK    = 14
       integer, parameter :: MPI_COMBINED_HINDEXED_BLOCK   = 15

       integer, parameter :: MPI_TAG_UB_VALUE = 1073741824

       integer, parameter :: MPI_PROC_NULL      = -2
       integer, parameter :: MPI_ROOT           = -3
       integer, parameter :: MPI_ANY_SOURCE     = -1
       integer, parameter :: MPI_ANY_TAG        = MPI_TAG_UB_VALUE+1
       integer, parameter :: MPI_REQUEST_NULL   = -1
       integer, parameter :: MPI_GROUP_NULL     = -1
       integer, parameter :: MPI_GROUP_EMPTY    = 0
       integer, parameter :: MPI_COMM_NULL      = -1
       integer, parameter :: MPI_KEYVAL_INVALID = -1
       integer, parameter :: MPI_INFO_NULL      = -1

       integer, pointer   :: MPI_IN_PLACE => null()

       integer, parameter :: MPI_BOTTOM = 0

       integer, parameter :: MPI_UNDEFINED = -32766

       integer, parameter :: MPI_IDENT     = 0
       integer, parameter :: MPI_SIMILAR   = 1
       integer, parameter :: MPI_CONGRUENT = 2
       integer, parameter :: MPI_UNEQUAL   = 3

       integer, parameter :: MPI_COMM_TYPE_SHARED   = 1
       integer, parameter :: AMPI_COMM_TYPE_HOST    = 2
       integer, parameter :: AMPI_COMM_TYPE_PROCESS = 3
       integer, parameter :: AMPI_COMM_TYPE_WTH     = 4

       integer, parameter :: MPI_UNWEIGHTED = 0
       integer, parameter :: MPI_GRAPH = 1
       integer, parameter :: MPI_CART  = 2
       integer, parameter :: MPI_DIST_GRAPH = 3

       integer, parameter :: MPI_TAG_UB            = -2
       integer, parameter :: MPI_HOST              = -3
       integer, parameter :: MPI_IO                = -4
       integer, parameter :: MPI_WTIME_IS_GLOBAL   = -5
       integer, parameter :: MPI_APPNUM            = -6
       integer, parameter :: MPI_UNIVERSE_SIZE     = -7
       integer, parameter :: MPI_WIN_BASE          = -8
       integer, parameter :: MPI_WIN_SIZE          = -9
       integer, parameter :: MPI_WIN_DISP_UNIT     = -10
       integer, parameter :: MPI_WIN_MODEL         = -11
       integer, parameter :: MPI_WIN_CREATE_FLAVOR = -12
       integer, parameter :: AMPI_MY_WTH           = -13
       integer, parameter :: AMPI_NUM_WTHS         = -14
       integer, parameter :: AMPI_MY_PROCESS       = -15
       integer, parameter :: AMPI_NUM_PROCESSES    = -16
       integer, parameter :: AMPI_MY_HOME_WTH      = -17

       integer, parameter :: MPI_STATUS_SIZE = 8

       integer, parameter :: MPI_TAG         = 1
       integer, parameter :: MPI_SOURCE      = 2
       integer, parameter :: MPI_ERROR       = 5

       integer, dimension(MPI_STATUS_SIZE) :: MPI_STATUS_IGNORE
       parameter (MPI_STATUS_IGNORE = (/-9,-9,-9,-9,-9,-9,-9,-9/))
       integer, dimension(MPI_STATUS_SIZE) :: MPI_STATUSES_IGNORE
       parameter (MPI_STATUSES_IGNORE = (/-9,-9,-9,-9,-9,-9,-9,-9/))

       integer, parameter :: MPI_COMM_WORLD       = 1000000
       integer, parameter :: MPI_COMM_SELF        = 1000001

       integer, parameter :: MPI_INFO_ENV              = 0
       integer, parameter :: AMPI_INFO_LB_SYNC         = 1
       integer, parameter :: AMPI_INFO_LB_ASYNC        = 2
       integer, parameter :: AMPI_INFO_CHKPT_IN_MEMORY = 3

       double precision, external :: MPI_WTIME
       double precision, external :: MPI_WTICK

       integer, parameter :: MPI_OP_NULL = -1
       integer, parameter :: MPI_MAX     = 0
       integer, parameter :: MPI_MIN     = 1
       integer, parameter :: MPI_SUM     = 2
       integer, parameter :: MPI_PROD    = 3
       integer, parameter :: MPI_LAND    = 4
       integer, parameter :: MPI_BAND    = 5
       integer, parameter :: MPI_LOR     = 6
       integer, parameter :: MPI_BOR     = 7
       integer, parameter :: MPI_LXOR    = 8
       integer, parameter :: MPI_BXOR    = 9
       integer, parameter :: MPI_MAXLOC  = 10
       integer, parameter :: MPI_MINLOC  = 11
       integer, parameter :: MPI_REPLACE = 12
       integer, parameter :: MPI_NO_OP   = 13

       integer, parameter :: MPI_ADDRESS_KIND = 8
       integer, parameter :: MPI_COUNT_KIND = MPI_ADDRESS_KIND
       integer, parameter :: MPI_OFFSET_KIND = 8

       integer, parameter :: MPI_BSEND_OVERHEAD = 1

       integer, parameter :: MPI_MAX_OBJECT_NAME = 255
       integer, parameter :: MPI_MAX_INFO_KEY    = 255
       integer, parameter :: MPI_MAX_INFO_VAL    = 1024

       integer, parameter :: MPI_LOCK_SHARED     = 54
       integer, parameter :: MPI_LOCK_EXCLUSIVE  = 55

       integer, parameter :: MPI_WIN_NULL        = -1

       integer, parameter :: MPI_WIN_FLAVOR_CREATE   = 1
       integer, parameter :: MPI_WIN_FLAVOR_ALLOCATE = 2
       integer, parameter :: MPI_WIN_FLAVOR_DYNAMIC  = 3
       integer, parameter :: MPI_WIN_FLAVOR_SHARED   = 4

       integer, parameter :: MPI_WIN_UNIFIED  = 0
       integer, parameter :: MPI_WIN_SEPARATE = 1

       integer, parameter :: MPI_MODE_NOCHECK   =  1
       integer, parameter :: MPI_MODE_NOPRECEDE =  2
       integer, parameter :: MPI_MODE_NOPUT     =  4
       integer, parameter :: MPI_MODE_NOSTORE   =  8
       integer, parameter :: MPI_MODE_NOSUCCEED = 16

       logical, parameter :: MPI_SUBARRAYS_SUPPORTED = .FALSE.
       logical, parameter :: MPI_ASYNC_PROTECTS_NONBLOCKING = .FALSE.

       integer(KIND=MPI_ADDRESS_KIND), external :: MPI_AINT_ADD
       integer(KIND=MPI_ADDRESS_KIND), external :: MPI_AINT_DIFF

       integer, parameter :: MPI_MESSAGE_NULL = -1
       integer, parameter :: MPI_MESSAGE_NO_PROC = -2

       ! necessary for ROMIO's tests
       integer, parameter :: MPI_ORDER_C = 56
       integer, parameter :: MPI_ORDER_FORTRAN = 57
       integer, parameter :: MPI_DISTRIBUTE_BLOCK = 121
       integer, parameter :: MPI_DISTRIBUTE_CYCLIC = 122
       integer, parameter :: MPI_DISTRIBUTE_NONE = 123
       integer, parameter :: MPI_DISTRIBUTE_DFLT_DARG = -49767

       include 'mpiof.h'

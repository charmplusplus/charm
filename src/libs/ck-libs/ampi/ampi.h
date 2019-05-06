#ifndef _MPI_H
#define _MPI_H

#include <stdlib.h> /* for redefinition of exit() below */
#include <inttypes.h> /* for intptr_t */
#include "conv-config.h"
#include "charm-api.h" /* for CLINKAGE */

#ifdef __cplusplus
#if CMK_CUDA
#include "hapi.h"
#endif
#endif

/* NON-standard define: this lets people #ifdef on
   AMPI, e.g. to portably use AMPI extensions to the MPI standard. */
#define AMPI

/* Macro to define the AMPI'fied name of an MPI function, plus the AMPI'fied
 * PMPI name. When building Charm++/AMPI on top of MPI, we rename the user's
 * MPI_ calls to use AMPI_. */
#if CMK_CONVERSE_MPI
  #define AMPI_API_DEF(return_type, function_name, ...) \
    return_type A##function_name(__VA_ARGS__);          \
    return_type AP##function_name(__VA_ARGS__);
#else
  #define AMPI_API_DEF(return_type, function_name, ...) \
    return_type function_name(__VA_ARGS__);             \
    return_type P##function_name(__VA_ARGS__);
#endif

#ifdef __cplusplus
# define AMPI_API_STATIC_CAST(type, obj)      (static_cast<type>(obj))
# define AMPI_API_REINTERPRET_CAST(type, obj) (reinterpret_cast<type>(obj))
#else
# define AMPI_API_STATIC_CAST(type, obj)      ((type)(obj))
# define AMPI_API_REINTERPRET_CAST(type, obj) ((type)(obj))
#endif

CLINKAGE void AMPI_Exit(int exitCode);
/* Allow applications to terminate cleanly with exit():
 * In C++ applications, this can conflict with user-defined
 * exit routines inside namespaces, such as Foo::exit(), so
 * we allow turning off AMPI's renaming of exit() with
 * -DAMPI_RENAME_EXIT=0. Same for 'atexit' below... */
#ifndef AMPI_RENAME_EXIT
#define AMPI_RENAME_EXIT 1
#endif
#if AMPI_RENAME_EXIT
#define exit(status) AMPI_Exit(status)
#endif

/* Notify AMPI when atexit() is used in order to prevent running MPI_Finalize()
   in a function registered with atexit. Only applies when including mpi.h. */
CLINKAGE void ampiMarkAtexit(void);
#ifndef AMPI_RENAME_ATEXIT
#define AMPI_RENAME_ATEXIT 1
#endif
#if AMPI_RENAME_ATEXIT
#define atexit(...) do {atexit(__VA_ARGS__); atexit(ampiMarkAtexit);} while(0)
#endif

/*
Silently rename the user's main routine.
This is needed so we can call the routine as a new thread,
instead of as an actual "main".
*/
#ifdef __cplusplus /* C++ version-- rename "main" as "AMPI_Main_cpp" */
#  define main AMPI_Main_cpp
int AMPI_Main_cpp(int argc,char **argv); /* prototype for C++ main routine */
int AMPI_Main_cpp(void); /* prototype for C++ main routines without args, as in autoconf tests */

extern "C" {
#else /* C version-- rename "main" as "AMPI_Main" */
#  define main AMPI_Main
#endif

int AMPI_Main(); /* declaration for C main routine (not a strict prototype!) */
void AMPI_Main_c(int argc,char **argv); /* C wrapper for calling AMPI_Main() from C++ */

typedef void (*MPI_MainFn) (int,char**);

typedef int MPI_Datatype;
typedef intptr_t MPI_Aint;
typedef int MPI_Fint;
typedef MPI_Aint MPI_Count;
typedef long long int MPI_Offset;
#define HAVE_MPI_OFFSET 1 /* ROMIO requires us to set this */

/********************** MPI-1.1 prototypes and defines ***************************/
/* MPI-1 Errors */
#define MPI_SUCCESS                     0
#define MPI_ERR_BUFFER                  1
#define MPI_ERR_COUNT                   2
#define MPI_ERR_TYPE                    3
#define MPI_ERR_TAG                     4
#define MPI_ERR_COMM                    5
#define MPI_ERR_RANK                    6
#define MPI_ERR_REQUEST                 7
#define MPI_ERR_ROOT                    8
#define MPI_ERR_GROUP                   9
#define MPI_ERR_OP                     10
#define MPI_ERR_TOPOLOGY               11
#define MPI_ERR_DIMS                   12
#define MPI_ERR_ARG                    13
#define MPI_ERR_UNKNOWN                14
#define MPI_ERR_TRUNCATE               15
#define MPI_ERR_OTHER                  16
#define MPI_ERR_INTERN                 17
#define MPI_ERR_IN_STATUS              18
#define MPI_ERR_PENDING                19
/* MPI-2 Errors */
#define MPI_ERR_ACCESS                 20
#define MPI_ERR_AMODE                  21
#define MPI_ERR_ASSERT                 22
#define MPI_ERR_BAD_FILE               23
#define MPI_ERR_BASE                   24
#define MPI_ERR_CONVERSION             25
#define MPI_ERR_DISP                   26
#define MPI_ERR_DUP_DATAREP            27
#define MPI_ERR_FILE_EXISTS            28
#define MPI_ERR_FILE_IN_USE            29
#define MPI_ERR_FILE                   30
#define MPI_ERR_INFO_KEY               31
#define MPI_ERR_INFO_NOKEY             32
#define MPI_ERR_INFO_VALUE             33
#define MPI_ERR_INFO                   34
#define MPI_ERR_IO                     35
#define MPI_ERR_KEYVAL                 36
#define MPI_ERR_LOCKTYPE               37
#define MPI_ERR_NAME                   38
#define MPI_ERR_NO_MEM                 39
#define MPI_ERR_NOT_SAME               40
#define MPI_ERR_NO_SPACE               41
#define MPI_ERR_NO_SUCH_FILE           42
#define MPI_ERR_PORT                   43
#define MPI_ERR_QUOTA                  44
#define MPI_ERR_READ_ONLY              45
#define MPI_ERR_RMA_CONFLICT           46
#define MPI_ERR_RMA_SYNC               47
#define MPI_ERR_SERVICE                48
#define MPI_ERR_SIZE                   49
#define MPI_ERR_SPAWN                  50
#define MPI_ERR_UNSUPPORTED_DATAREP    51
#define MPI_ERR_UNSUPPORTED_OPERATION  52
#define MPI_ERR_WIN                    53
#define MPI_ERR_LASTCODE               53
#define MPI_LASTUSEDCODE               53
/* 0=MPI_SUCCESS<MPI_ERRs(...)<MPI_ERR<=MPI_ERR_LASTCODE<=MPI_LASTUSEDCODE */

#define MPI_MAX_PROCESSOR_NAME         256
#define MPI_MAX_ERROR_STRING           256
#define MPI_MAX_LIBRARY_VERSION_STRING 256

#define MPI_VERSION    2
#define MPI_SUBVERSION 2

#define MPI_THREAD_SINGLE     1
#define MPI_THREAD_FUNNELED   2
#define MPI_THREAD_SERIALIZED 3
#define MPI_THREAD_MULTIPLE   4

/* these values have to match values in ampif.h */
/* base types */
#define MPI_DATATYPE_NULL       -1
#define MPI_DOUBLE               0
#define MPI_INT                  1
#define MPI_INTEGER              MPI_INT
#define MPI_FLOAT                2
#define MPI_LOGICAL              3
#define MPI_C_BOOL               4
#define MPI_CHAR                 5
#define MPI_BYTE                 6
#define MPI_PACKED               7
#define MPI_SHORT                8
#define MPI_LONG                 9
#define MPI_UNSIGNED_CHAR       10
#define MPI_UNSIGNED_SHORT      11
#define MPI_UNSIGNED            12
#define MPI_UNSIGNED_LONG       13
#define MPI_LONG_DOUBLE         14
/* mpi-2+ types */
#define MPI_LONG_LONG_INT       15
#define MPI_LONG_LONG           MPI_LONG_LONG_INT
#define MPI_OFFSET              MPI_LONG_LONG
#define MPI_SIGNED_CHAR         16
#define MPI_UNSIGNED_LONG_LONG  17
#define MPI_WCHAR               18
#define MPI_INT8_T              19
#define MPI_INT16_T             20
#define MPI_INT32_T             21
#define MPI_INT64_T             22
#define MPI_UINT8_T             23
#define MPI_UINT16_T            24
#define MPI_UINT32_T            25
#define MPI_UINT64_T            26
#define MPI_AINT                27
#define MPI_COUNT               MPI_AINT
#define MPI_LB                  28
#define MPI_UB                  29
/*
 * AMPI_MAX_BASIC_TYPE is defined in ddt.h
 * and is tied to the above values, if the above
 * indexes change or values are added/deleted
 * you may need to change AMPI_MAX_BASIC_TYPE
 */
/* tuple types */
#define MPI_FLOAT_INT           30
#define MPI_DOUBLE_INT          31
#define MPI_LONG_INT            32
#define MPI_2INT                33
#define MPI_SHORT_INT           34
#define MPI_LONG_DOUBLE_INT     35
#define MPI_2FLOAT              36
#define MPI_2DOUBLE             37
/* mpi-2+ types */
#define MPI_COMPLEX             38
#define MPI_FLOAT_COMPLEX       39
#define MPI_DOUBLE_COMPLEX      40
#define MPI_LONG_DOUBLE_COMPLEX 41
/*
 * AMPI_MAX_PREDEFINED_TYPE is defined in ddt.h
 * and is tied to the above values, if the above
 * indexes change or values are added/deleted
 * you may need to change AMPI_MAX_PREDEFINED_TYPE
 */

#define MPI_ANY_TAG        MPI_TAG_UB_VALUE+1
#define MPI_REQUEST_NULL   (-1)
#define MPI_GROUP_NULL     (-1)
#define MPI_GROUP_EMPTY       0
#define MPI_COMM_NULL      (-1)
#define MPI_PROC_NULL      (-2)
#define MPI_ROOT           (-3)
#define MPI_ANY_SOURCE     (-1)
#define MPI_KEYVAL_INVALID (-1)
#define MPI_INFO_NULL      (-1)

#define MPI_IN_PLACE    AMPI_API_REINTERPRET_CAST(void *, -1L)

#define MPI_BOTTOM      AMPI_API_REINTERPRET_CAST(void *, -2L)
#define MPI_UNDEFINED   (-32766)

#define MPI_IDENT       0
#define MPI_SIMILAR     1
#define MPI_CONGRUENT   2
#define MPI_UNEQUAL     3

#define MPI_COMM_TYPE_SHARED   1
#define AMPI_COMM_TYPE_HOST    2
#define AMPI_COMM_TYPE_PROCESS 3
#define AMPI_COMM_TYPE_WTH     4

typedef int MPI_Op;

typedef void (MPI_User_function)(void *invec, void *inoutvec,
                                 int *len, MPI_Datatype *datatype);

#define MPI_OP_NULL -1
#define MPI_MAX      0
#define MPI_MIN      1
#define MPI_SUM      2
#define MPI_PROD     3
#define MPI_LAND     4
#define MPI_BAND     5
#define MPI_LOR      6
#define MPI_BOR      7
#define MPI_LXOR     8
#define MPI_BXOR     9
#define MPI_MAXLOC  10
#define MPI_MINLOC  11
#define MPI_REPLACE 12
#define MPI_NO_OP   13
/*
 * AMPI_MAX_PREDEFINED_OP is defined in ampiimpl.h
 * and is tied to the above values, if the above
 * indexes change or values are added/deleted
 * you may need to change AMPI_MAX_PREDEFINED_TYPE
 */

#define MPI_UNWEIGHTED 0
#define MPI_CART       1
#define MPI_GRAPH      2
#define MPI_DIST_GRAPH 3

/* This is one less than the system-tags defined in ampiimpl.h.
 * This is so that the tags used by the system don't clash with user-tags.
 * The MPI 3.1 standard requires this to be at least 32767 (2^15 -1).
 */
#define MPI_TAG_UB_VALUE  1073741824

/* These are the builtin MPI keyvals, plus some AMPI specific ones. */
#define MPI_TAG_UB             -2
#define MPI_HOST               -3
#define MPI_IO                 -4
#define MPI_WTIME_IS_GLOBAL    -5
#define MPI_APPNUM             -6
#define MPI_UNIVERSE_SIZE      -7
#define MPI_WIN_BASE           -8
#define MPI_WIN_SIZE           -9
#define MPI_WIN_DISP_UNIT     -10
#define MPI_WIN_MODEL         -11
#define MPI_WIN_CREATE_FLAVOR -12
#define AMPI_MY_WTH           -13
#define AMPI_NUM_WTHS         -14
#define AMPI_MY_PROCESS       -15
#define AMPI_NUM_PROCESSES    -16

/** Communicators give a communication context to a set of processors.
    An intercommunicator can be used for point to point messaging between two groups.
    An intracommunicator can be used to send messages within a single group. */
typedef int MPI_Comm;

/** Groups represent an set of processors 0...n-1. They can be created locally */
typedef int MPI_Group;

typedef int MPI_Info;

#define MPI_COMM_SELF               AMPI_API_STATIC_CAST(MPI_Comm, 1000000) /*MPI_COMM_SELF is the first split comm */
#define MPI_COMM_FIRST_SPLIT        AMPI_API_STATIC_CAST(MPI_Comm, 1000000) /*Communicator from MPI_Comm_split */
#define MPI_COMM_FIRST_GROUP        AMPI_API_STATIC_CAST(MPI_Comm, 2000000) /*Communicator from MPI_Comm_group */
#define MPI_COMM_FIRST_CART         AMPI_API_STATIC_CAST(MPI_Comm, 3000000) /*Communicator from MPI_Cart_create */
#define MPI_COMM_FIRST_GRAPH        AMPI_API_STATIC_CAST(MPI_Comm, 4000000) /*Communicator from MPI_Graph_create */
#define MPI_COMM_FIRST_DIST_GRAPH   AMPI_API_STATIC_CAST(MPI_Comm, 5000000) /*Communicator from MPI_Dist_Graph_create */
#define MPI_COMM_FIRST_INTER        AMPI_API_STATIC_CAST(MPI_Comm, 6000000) /*Communicator from MPI_Intercomm_create*/
#define MPI_COMM_FIRST_INTRA        AMPI_API_STATIC_CAST(MPI_Comm, 7000000) /*Communicator from MPI_Intercomm_merge*/
#define MPI_COMM_FIRST_RESVD        AMPI_API_STATIC_CAST(MPI_Comm, 8000000) /*Communicator reserved for now*/
#define MPI_COMM_WORLD              AMPI_API_STATIC_CAST(MPI_Comm, 9000000) /*Start of universe*/
#define MPI_MAX_COMM_WORLDS  8
extern MPI_Comm MPI_COMM_UNIVERSE[MPI_MAX_COMM_WORLDS];

#define MPI_INFO_ENV                AMPI_API_STATIC_CAST(MPI_Info, 0)
#define AMPI_INFO_LB_SYNC           AMPI_API_STATIC_CAST(MPI_Info, 1)
#define AMPI_INFO_LB_ASYNC          AMPI_API_STATIC_CAST(MPI_Info, 2)
#define AMPI_INFO_CHKPT_IN_MEMORY   AMPI_API_STATIC_CAST(MPI_Info, 3)

/* the size of MPI_Status must conform to MPI_STATUS_SIZE in ampif.h */
struct AmpiMsg;
typedef int MPI_Request;
typedef struct {
  int MPI_TAG, MPI_SOURCE, MPI_COMM, MPI_LENGTH, MPI_ERROR, MPI_CANCEL; /* FIXME: MPI_ERROR is never used */
  struct AmpiMsg *msg;
} MPI_Status;

#define MPI_STATUS_IGNORE   AMPI_API_REINTERPRET_CAST(MPI_Status *, 0)
#define MPI_STATUSES_IGNORE AMPI_API_REINTERPRET_CAST(MPI_Status *, 0)

/* type for MPI messages used in MPI_Mprobe, MPI_Mrecv, MPI_Improbe, MPI_Imrecv */
typedef int MPI_Message;
#define MPI_MESSAGE_NULL    -1
#define MPI_MESSAGE_NO_PROC -2

typedef int MPI_Errhandler;
#define MPI_ERRHANDLER_NULL  0
#define MPI_ERRORS_RETURN    1
#define MPI_ERRORS_ARE_FATAL 2

typedef void (MPI_Comm_errhandler_fn)(MPI_Comm *, int *, ...);
typedef void (MPI_Comm_errhandler_function)(MPI_Comm *, int *, ...);
typedef int  (MPI_Comm_copy_attr_function)(MPI_Comm oldcomm, int keyval,
                                           void *extra_state, void *attribute_val_in,
                                           void *attribute_val_out, int *flag);
typedef int  (MPI_Comm_delete_attr_function)(MPI_Comm comm, int keyval,
                                             void *attribute_val, void *extra_state);

typedef void (MPI_Handler_function)(MPI_Comm *, int *, ...);
typedef int  (MPI_Copy_function)(MPI_Comm oldcomm, int keyval,
                                 void *extra_state, void *attribute_val_in,
                                 void *attribute_val_out, int *flag);
typedef int  (MPI_Delete_function)(MPI_Comm comm, int keyval,
                                   void *attribute_val, void *extra_state);

#define MPI_COMM_NULL_COPY_FN   MPI_comm_null_copy_fn
#define MPI_COMM_NULL_DELETE_FN MPI_comm_null_delete_fn
#define MPI_COMM_DUP_FN         MPI_comm_dup_fn

#define MPI_NULL_COPY_FN   MPI_comm_null_copy_fn
#define MPI_NULL_DELETE_FN MPI_comm_null_delete_fn
#define MPI_DUP_FN         MPI_comm_dup_fn

int MPI_COMM_NULL_COPY_FN   ( MPI_Comm, int, void *, void *, void *, int * );
int MPI_COMM_NULL_DELETE_FN ( MPI_Comm, int, void *, void * );
int MPI_COMM_DUP_FN         ( MPI_Comm, int, void *, void *, void *, int * );

typedef int MPI_Type_copy_attr_function(MPI_Datatype oldtype, int type_keyval, void *extra_state,
                                        void *attribute_val_in, void *attribute_val_out, int *flag);
typedef int MPI_Type_delete_attr_function(MPI_Datatype datatype, int type_keyval,
                                          void *attribute_val, void *extra_state);

#define MPI_TYPE_NULL_DELETE_FN MPI_type_null_delete_fn
#define MPI_TYPE_NULL_COPY_FN   MPI_type_null_copy_fn
#define MPI_TYPE_DUP_FN         MPI_type_dup_fn

int MPI_TYPE_NULL_COPY_FN   ( MPI_Datatype, int, void *, void *, void *, int * );
int MPI_TYPE_NULL_DELETE_FN ( MPI_Datatype, int, void *, void * );
int MPI_TYPE_DUP_FN         ( MPI_Datatype, int, void *, void *, void *, int * );

typedef int MPI_Grequest_query_function(void *extra_state, MPI_Status *status);
typedef int MPI_Grequest_free_function(void *extra_state);
typedef int MPI_Grequest_cancel_function(void *extra_state, int complete);

#include "pup_c.h"

typedef void (*MPI_PupFn)(pup_er, void*);
typedef void (*MPI_MigrateFn)(void);

/* for the datatype decoders */
#define MPI_COMBINER_NAMED            1
#define MPI_COMBINER_CONTIGUOUS       2
#define MPI_COMBINER_VECTOR           3
#define MPI_COMBINER_HVECTOR          4
#define MPI_COMBINER_HVECTOR_INTEGER  5
#define MPI_COMBINER_INDEXED          6
#define MPI_COMBINER_HINDEXED         7
#define MPI_COMBINER_HINDEXED_INTEGER 8
#define MPI_COMBINER_STRUCT           9
#define MPI_COMBINER_STRUCT_INTEGER   10
#define MPI_COMBINER_DARRAY           11
#define MPI_COMBINER_RESIZED          12
#define MPI_COMBINER_SUBARRAY         13
#define MPI_COMBINER_INDEXED_BLOCK    14
#define MPI_COMBINER_HINDEXED_BLOCK   15

#define MPI_BSEND_OVERHEAD 0

/* When AMPI is built on top of MPI, rename user's MPI_* calls to AMPI_* */
#if CMK_CONVERSE_MPI
/***pt2pt***/
#define  MPI_Send  AMPI_Send
#define PMPI_Send APMPI_Send
#define  MPI_Ssend  AMPI_Ssend
#define PMPI_Ssend APMPI_Ssend
#define  MPI_Recv  AMPI_Recv
#define PMPI_Recv APMPI_Recv
#define  MPI_Mrecv  AMPI_Mrecv
#define PMPI_Mrecv APMPI_Mrecv
#define  MPI_Get_count  AMPI_Get_count
#define PMPI_Get_count APMPI_Get_count
#define  MPI_Bsend  AMPI_Bsend
#define PMPI_Bsend APMPI_Bsend
#define  MPI_Rsend  AMPI_Rsend
#define PMPI_Rsend APMPI_Rsend
#define  MPI_Buffer_attach  AMPI_Buffer_attach
#define PMPI_Buffer_attach APMPI_Buffer_attach
#define  MPI_Buffer_detach  AMPI_Buffer_detach
#define PMPI_Buffer_detach APMPI_Buffer_detach
#define  MPI_Isend  AMPI_Isend
#define PMPI_Isend APMPI_Isend
#define  MPI_Ibsend  AMPI_Ibsend
#define PMPI_Ibsend APMPI_Ibsend
#define  MPI_Issend  AMPI_Issend
#define PMPI_Issend APMPI_Issend
#define  MPI_Irsend  AMPI_Irsend
#define PMPI_Irsend APMPI_Irsend
#define  MPI_Irecv  AMPI_Irecv
#define PMPI_Irecv APMPI_Irecv
#define  MPI_Imrecv  AMPI_Imrecv
#define PMPI_Imrecv APMPI_Imrecv
#define  MPI_Waitany  AMPI_Waitany
#define PMPI_Waitany APMPI_Waitany
#define  MPI_Test  AMPI_Test
#define PMPI_Test APMPI_Test
#define  MPI_Wait  AMPI_Wait
#define PMPI_Wait APMPI_Wait
#define  MPI_Testany  AMPI_Testany
#define PMPI_Testany APMPI_Testany
#define  MPI_Waitall  AMPI_Waitall
#define PMPI_Waitall APMPI_Waitall
#define  MPI_Testall  AMPI_Testall
#define PMPI_Testall APMPI_Testall
#define  MPI_Waitsome  AMPI_Waitsome
#define PMPI_Waitsome APMPI_Waitsome
#define  MPI_Testsome  AMPI_Testsome
#define PMPI_Testsome APMPI_Testsome
#define  MPI_Request_get_status  AMPI_Request_get_status
#define PMPI_Request_get_status APMPI_Request_get_status
#define  MPI_Request_free  AMPI_Request_free
#define PMPI_Request_free APMPI_Request_free
#define  MPI_Grequest_start  AMPI_Grequest_start
#define PMPI_Grequest_start APMPI_Grequest_start
#define  MPI_Grequest_complete  AMPI_Grequest_complete
#define PMPI_Grequest_complete APMPI_Grequest_complete
#define  MPI_Cancel  AMPI_Cancel
#define PMPI_Cancel APMPI_Cancel
#define  MPI_Test_cancelled  AMPI_Test_cancelled
#define PMPI_Test_cancelled APMPI_Test_cancelled
#define  MPI_Status_set_cancelled  AMPI_Status_set_cancelled
#define PMPI_Status_set_cancelled APMPI_Status_set_cancelled
#define  MPI_Iprobe AMPI_Iprobe
#define PMPI_Iprobe APMPI_Iprobe
#define  MPI_Probe AMPI_Probe
#define PMPI_Probe APMPI_Probe
#define  MPI_Improbe AMPI_Improbe
#define PMPI_Improbe APMPI_Improbe
#define  MPI_Mprobe AMPI_Mprobe
#define PMPI_Mprobe APMPI_Mprobe
#define  MPI_Send_init  AMPI_Send_init
#define PMPI_Send_init APMPI_Send_init
#define  MPI_Ssend_init  AMPI_Ssend_init
#define PMPI_Ssend_init APMPI_Ssend_init
#define  MPI_Rsend_init  AMPI_Rsend_init
#define PMPI_Rsend_init APMPI_Rsend_init
#define  MPI_Bsend_init  AMPI_Bsend_init
#define PMPI_Bsend_init APMPI_Bsend_init
#define  MPI_Recv_init  AMPI_Recv_init
#define PMPI_Recv_init APMPI_Recv_init
#define  MPI_Start  AMPI_Start
#define PMPI_Start APMPI_Start
#define  MPI_Startall  AMPI_Startall
#define PMPI_Startall APMPI_Startall
#define  MPI_Sendrecv  AMPI_Sendrecv
#define PMPI_Sendrecv APMPI_Sendrecv
#define  MPI_Sendrecv_replace  AMPI_Sendrecv_replace
#define PMPI_Sendrecv_replace APMPI_Sendrecv_replace

/***datatypes***/
#define  MPI_Type_contiguous  AMPI_Type_contiguous
#define PMPI_Type_contiguous APMPI_Type_contiguous
#define  MPI_Type_vector  AMPI_Type_vector
#define PMPI_Type_vector APMPI_Type_vector
#define  MPI_Type_create_hvector  AMPI_Type_create_hvector
#define PMPI_Type_create_hvector APMPI_Type_create_hvector
#define  MPI_Type_hvector  AMPI_Type_hvector
#define PMPI_Type_hvector APMPI_Type_hvector
#define  MPI_Type_indexed  AMPI_Type_indexed
#define PMPI_Type_indexed APMPI_Type_indexed
#define  MPI_Type_create_hindexed  AMPI_Type_create_hindexed
#define PMPI_Type_create_hindexed APMPI_Type_create_hindexed
#define  MPI_Type_create_indexed_block  AMPI_Type_create_indexed_block
#define PMPI_Type_create_indexed_block APMPI_Type_create_indexed_block
#define  MPI_Type_create_hindexed_block  AMPI_Type_create_hindexed_block
#define PMPI_Type_create_hindexed_block APMPI_Type_create_hindexed_block
#define  MPI_Type_hindexed  AMPI_Type_hindexed
#define PMPI_Type_hindexed APMPI_Type_hindexed
#define  MPI_Type_create_struct  AMPI_Type_create_struct
#define PMPI_Type_create_struct APMPI_Type_create_struct
#define  MPI_Type_struct  AMPI_Type_struct
#define PMPI_Type_struct APMPI_Type_struct
#define  MPI_Type_get_envelope  AMPI_Type_get_envelope
#define PMPI_Type_get_envelope APMPI_Type_get_envelope
#define  MPI_Type_get_contents  AMPI_Type_get_contents
#define PMPI_Type_get_contents APMPI_Type_get_contents
#define  MPI_Type_commit  AMPI_Type_commit
#define PMPI_Type_commit APMPI_Type_commit
#define  MPI_Type_free  AMPI_Type_free
#define PMPI_Type_free APMPI_Type_free
#define  MPI_Type_get_extent  AMPI_Type_get_extent
#define PMPI_Type_get_extent APMPI_Type_get_extent
#define  MPI_Type_get_extent_x  AMPI_Type_get_extent_x
#define PMPI_Type_get_extent_x APMPI_Type_get_extent_x
#define  MPI_Type_extent  AMPI_Type_extent
#define PMPI_Type_extent APMPI_Type_extent
#define  MPI_Type_get_true_extent  AMPI_Type_get_true_extent
#define PMPI_Type_get_true_extent APMPI_Type_get_true_extent
#define  MPI_Type_get_true_extent_x  AMPI_Type_get_true_extent_x
#define PMPI_Type_get_true_extent_X APMPI_Type_get_true_extent_x
#define  MPI_Type_size  AMPI_Type_size
#define PMPI_Type_size APMPI_Type_size
#define  MPI_Type_size_x  AMPI_Type_size_x
#define PMPI_Type_size_x APMPI_Type_size_x
#define  MPI_Type_lb  AMPI_Type_lb
#define PMPI_Type_lb APMPI_Type_lb
#define  MPI_Type_ub  AMPI_Type_ub
#define PMPI_Type_ub APMPI_Type_ub
#define  MPI_Type_set_name  AMPI_Type_set_name
#define PMPI_Type_set_name APMPI_Type_set_name
#define  MPI_Type_get_name  AMPI_Type_get_name
#define PMPI_Type_get_name APMPI_Type_get_name
#define  MPI_Type_dup  AMPI_Type_dup
#define PMPI_Type_dup APMPI_Type_dup
#define  MPI_Type_create_resized  AMPI_Type_create_resized
#define PMPI_Type_create_resized APMPI_Type_create_resized
#define  MPI_Type_set_attr  AMPI_Type_set_attr
#define PMPI_Type_set_attr APMPI_Type_set_attr
#define  MPI_Type_get_attr  AMPI_Type_get_attr
#define PMPI_Type_get_attr APMPI_Type_get_attr
#define  MPI_Type_delete_attr  AMPI_Type_delete_attr
#define PMPI_Type_delete_attr APMPI_Type_delete_attr
#define  MPI_Type_create_keyval  AMPI_Type_create_keyval
#define PMPI_Type_create_keyval APMPI_Type_create_keyval
#define  MPI_Type_free_keyval  AMPI_Type_free_keyval
#define PMPI_Type_free_keyval APMPI_Type_free_keyval
#define  MPI_Type_create_darray  AMPI_Type_create_darray
#define PMPI_Type_create_darray APMPI_Type_create_darray
#define  MPI_Type_create_subarray  AMPI_Type_create_subarray
#define PMPI_Type_create_subarray APMPI_Type_create_subarray
#define  MPI_Get_address  AMPI_Get_address
#define PMPI_Get_address APMPI_Get_address
#define  MPI_Address  AMPI_Address
#define PMPI_Address APMPI_Address
#define  MPI_Status_set_elements  AMPI_Status_set_elements
#define PMPI_Status_set_elements APMPI_Status_set_elements
#define  MPI_Status_set_elements_x  AMPI_Status_set_elements_x
#define PMPI_Status_set_elements_x APMPI_Status_set_elements_x
#define  MPI_Get_elements  AMPI_Get_elements
#define PMPI_Get_elements APMPI_Get_elements
#define  MPI_Get_elements_x  AMPI_Get_elements_x
#define PMPI_Get_elements_x APMPI_Get_elements_x
#define  MPI_Pack  AMPI_Pack
#define PMPI_Pack APMPI_Pack
#define  MPI_Unpack  AMPI_Unpack
#define PMPI_Unpack APMPI_Unpack
#define  MPI_Pack_size  AMPI_Pack_size
#define PMPI_Pack_size APMPI_Pack_size

/***collectives***/
#define  MPI_Barrier  AMPI_Barrier
#define PMPI_Barrier APMPI_Barrier
#define  MPI_Ibarrier  AMPI_Ibarrier
#define PMPI_Ibarrier APMPI_Ibarrier
#define  MPI_Bcast  AMPI_Bcast
#define PMPI_Bcast APMPI_Bcast
#define  MPI_Ibcast  AMPI_Ibcast
#define PMPI_Ibcast APMPI_Ibcast
#define  MPI_Gather  AMPI_Gather
#define PMPI_Gather APMPI_Gather
#define  MPI_Igather  AMPI_Igather
#define PMPI_Igather APMPI_Igather
#define  MPI_Gatherv  AMPI_Gatherv
#define PMPI_Gatherv APMPI_Gatherv
#define  MPI_Igatherv  AMPI_Igatherv
#define PMPI_Igatherv APMPI_Igatherv
#define  MPI_Scatter  AMPI_Scatter
#define PMPI_Scatter APMPI_Scatter
#define  MPI_Iscatter  AMPI_Iscatter
#define PMPI_Iscatter APMPI_Iscatter
#define  MPI_Scatterv  AMPI_Scatterv
#define PMPI_Scatterv APMPI_Scatterv
#define  MPI_Iscatterv  AMPI_Iscatterv
#define PMPI_Iscatterv APMPI_Iscatterv
#define  MPI_Allgather  AMPI_Allgather
#define PMPI_Allgather APMPI_Allgather
#define  MPI_Iallgather  AMPI_Iallgather
#define PMPI_Iallgather APMPI_Iallgather
#define  MPI_Allgatherv  AMPI_Allgatherv
#define PMPI_Allgatherv APMPI_Allgatherv
#define  MPI_Iallgatherv  AMPI_Iallgatherv
#define PMPI_Iallgatherv APMPI_Iallgatherv
#define  MPI_Alltoall  AMPI_Alltoall
#define PMPI_Alltoall APMPI_Alltoall
#define  MPI_Ialltoall  AMPI_Ialltoall
#define PMPI_Ialltoall APMPI_Ialltoall
#define  MPI_Alltoallv  AMPI_Alltoallv
#define PMPI_Alltoallv APMPI_Alltoallv
#define  MPI_Ialltoallv  AMPI_Ialltoallv
#define PMPI_Ialltoallv APMPI_Ialltoallv
#define  MPI_Alltoallw  AMPI_Alltoallw
#define PMPI_Alltoallw APMPI_Alltoallw
#define  MPI_Ialltoallw  AMPI_Ialltoallw
#define PMPI_Ialltoallw APMPI_Ialltoallw
#define  MPI_Reduce  AMPI_Reduce
#define PMPI_Reduce APMPI_Reduce
#define  MPI_Ireduce  AMPI_Ireduce
#define PMPI_Ireduce APMPI_Ireduce
#define  MPI_Allreduce  AMPI_Allreduce
#define PMPI_Allreduce APMPI_Allreduce
#define  MPI_Iallreduce  AMPI_Iallreduce
#define PMPI_Iallreduce APMPI_Iallreduce
#define  MPI_Reduce_local  AMPI_Reduce_local
#define PMPI_Reduce_local APMPI_Reduce_local
#define  MPI_Reduce_scatter_block  AMPI_Reduce_scatter_block
#define PMPI_Reduce_scatter_block APMPI_Reduce_scatter_block
#define  MPI_Ireduce_scatter_block  AMPI_Ireduce_scatter_block
#define PMPI_Ireduce_scatter_block APMPI_Ireduce_scatter_block
#define  MPI_Reduce_scatter  AMPI_Reduce_scatter
#define PMPI_Reduce_scatter APMPI_Reduce_scatter
#define  MPI_Ireduce_scatter  AMPI_Ireduce_scatter
#define PMPI_Ireduce_scatter APMPI_Ireduce_scatter
#define  MPI_Scan  AMPI_Scan
#define PMPI_Scan APMPI_Scan
#define  MPI_Iscan  AMPI_Iscan
#define PMPI_Iscan APMPI_Iscan
#define  MPI_Exscan  AMPI_Exscan
#define PMPI_Exscan APMPI_Exscan
#define  MPI_Iexscan  AMPI_Iexscan
#define PMPI_Iexscan APMPI_Iexscan

/***neighborhood collectives***/
#define  MPI_Neighbor_alltoall  AMPI_Neighbor_alltoall
#define PMPI_Neighbor_alltoall APMPI_Neighbor_alltoall
#define  MPI_Ineighbor_alltoall  AMPI_Ineighbor_alltoall
#define PMPI_Ineighbor_alltoall APMPI_Ineighbor_alltoall
#define  MPI_Neighbor_alltoallv  AMPI_Neighbor_alltoallv
#define PMPI_Neighbor_alltoallv APMPI_Neighbor_alltoallv
#define  MPI_Ineighbor_alltoallv  AMPI_Ineighbor_alltoallv
#define PMPI_Ineighbor_alltoallv APMPI_Ineighbor_alltoallv
#define  MPI_Neighbor_alltoallw  AMPI_Neighbor_alltoallw
#define PMPI_Neighbor_alltoallw APMPI_Neighbor_alltoallw
#define  MPI_Ineighbor_alltoallw  AMPI_Ineighbor_alltoallw
#define PMPI_Ineighbor_alltoallw APMPI_Ineighbor_alltoallw
#define  MPI_Neighbor_allgather  AMPI_Neighbor_allgather
#define PMPI_Neighbor_allgather APMPI_Neighbor_allgather
#define  MPI_Ineighbor_allgather  AMPI_Ineighbor_allgather
#define PMPI_Ineighbor_allgather APMPI_Ineighbor_allgather
#define  MPI_Neighbor_allgatherv  AMPI_Neighbor_allgatherv
#define PMPI_Neighbor_allgatherv APMPI_Neighbor_allgatherv
#define  MPI_Ineighbor_allgatherv  AMPI_Ineighbor_allgatherv
#define PMPI_Ineighbor_allgatherv APMPI_Ineighbor_allgatherv

/***ops***/
#define  MPI_Op_create  AMPI_Op_create
#define PMPI_Op_create APMPI_Op_create
#define  MPI_Op_free  AMPI_Op_free
#define PMPI_Op_free APMPI_Op_free
#define  MPI_Op_commutative  AMPI_Op_commutative
#define PMPI_Op_commutative APMPI_Op_commutative

/***groups***/
#define  MPI_Group_size  AMPI_Group_size
#define PMPI_Group_size APMPI_Group_size
#define  MPI_Group_rank  AMPI_Group_rank
#define PMPI_Group_rank APMPI_Group_rank
#define  MPI_Group_translate_ranks  AMPI_Group_translate_ranks
#define PMPI_Group_translate_ranks APMPI_Group_translate_ranks
#define  MPI_Group_compare  AMPI_Group_compare
#define PMPI_Group_compare APMPI_Group_compare
#define  MPI_Comm_group  AMPI_Comm_group
#define PMPI_Comm_group APMPI_Comm_group
#define  MPI_Group_union  AMPI_Group_union
#define PMPI_Group_union APMPI_Group_union
#define  MPI_Group_intersection  AMPI_Group_intersection
#define PMPI_Group_intersection APMPI_Group_intersection
#define  MPI_Group_difference  AMPI_Group_difference
#define PMPI_Group_difference APMPI_Group_difference
#define  MPI_Group_incl  AMPI_Group_incl
#define PMPI_Group_incl APMPI_Group_incl
#define  MPI_Group_excl  AMPI_Group_excl
#define PMPI_Group_excl APMPI_Group_excl
#define  MPI_Group_range_incl  AMPI_Group_range_incl
#define PMPI_Group_range_incl APMPI_Group_range_incl
#define  MPI_Group_range_excl  AMPI_Group_range_excl
#define PMPI_Group_range_excl APMPI_Group_range_excl
#define  MPI_Group_free  AMPI_Group_free
#define PMPI_Group_free APMPI_Group_free

/***communicators***/
#define  MPI_Intercomm_create  AMPI_Intercomm_create
#define PMPI_Intercomm_create APMPI_Intercomm_create
#define  MPI_Intercomm_merge  AMPI_Intercomm_merge
#define PMPI_Intercomm_merge APMPI_Intercomm_merge
#define  MPI_Comm_create  AMPI_Comm_create
#define PMPI_Comm_create APMPI_Comm_create
#define  MPI_Comm_create_group AMPI_Comm_create_group
#define PMPI_Comm_create_group APMPI_Comm_create_group
#define  MPI_Comm_size  AMPI_Comm_size
#define PMPI_Comm_size APMPI_Comm_size
#define  MPI_Comm_rank  AMPI_Comm_rank
#define PMPI_Comm_rank APMPI_Comm_rank
#define  MPI_Comm_compare  AMPI_Comm_compare
#define PMPI_Comm_compare APMPI_Comm_compare
#define  MPI_Comm_split  AMPI_Comm_split
#define PMPI_Comm_split APMPI_Comm_split
#define  MPI_Comm_split_type  AMPI_Comm_split_type
#define PMPI_Comm_split_type APMPI_Comm_split_type
#define  MPI_Comm_dup  AMPI_Comm_dup
#define PMPI_Comm_dup APMPI_Comm_dup
#define  MPI_Comm_idup  AMPI_Comm_idup
#define PMPI_Comm_idup APMPI_Comm_idup
#define  MPI_Comm_dup_with_info  AMPI_Comm_dup_with_info
#define PMPI_Comm_dup_with_info APMPI_Comm_dup_with_info
#define  MPI_Comm_idup_with_info  AMPI_Comm_idup_with_info
#define PMPI_Comm_idup_with_info APMPI_Comm_idup_with_info
#define  MPI_Comm_free  AMPI_Comm_free
#define PMPI_Comm_free APMPI_Comm_free
#define  MPI_Comm_test_inter  AMPI_Comm_test_inter
#define PMPI_Comm_test_inter APMPI_Comm_test_inter
#define  MPI_Comm_remote_size  AMPI_Comm_remote_size
#define PMPI_Comm_remote_size APMPI_Comm_remote_size
#define  MPI_Comm_remote_group  AMPI_Comm_remote_group
#define PMPI_Comm_remote_group APMPI_Comm_remote_group
#define  MPI_Comm_set_name  AMPI_Comm_set_name
#define PMPI_Comm_set_name APMPI_Comm_set_name
#define  MPI_Comm_get_name  AMPI_Comm_get_name
#define PMPI_Comm_get_name APMPI_Comm_get_name
#define  MPI_Comm_set_info  AMPI_Comm_set_info
#define PMPI_Comm_set_info APMPI_Comm_set_info
#define  MPI_Comm_get_info  AMPI_Comm_get_info
#define PMPI_Comm_get_info APMPI_Comm_get_info
#define  MPI_Comm_call_errhandler  AMPI_Comm_call_errhandler
#define PMPI_Comm_call_errhandler APMPI_Comm_call_errhandler
#define  MPI_Comm_create_errhandler  AMPI_Comm_create_errhandler
#define PMPI_Comm_create_errhandler APMPI_Comm_create_errhandler
#define  MPI_Comm_set_errhandler  AMPI_Comm_set_errhandler
#define PMPI_Comm_set_errhandler APMPI_Comm_set_errhandler
#define  MPI_Comm_get_errhandler  AMPI_Comm_get_errhandler
#define PMPI_Comm_get_errhandler APMPI_Comm_get_errhandler
#define  MPI_Comm_free_errhandler  AMPI_Comm_free_errhandler
#define PMPI_Comm_free_errhandler APMPI_Comm_free_errhandler
#define  MPI_Comm_create_keyval  AMPI_Comm_create_keyval
#define PMPI_Comm_create_keyval APMPI_Comm_create_keyval
#define  MPI_Comm_free_keyval  AMPI_Comm_free_keyval
#define PMPI_Comm_free_keyval APMPI_Comm_free_keyval
#define  MPI_Comm_set_attr  AMPI_Comm_set_attr
#define PMPI_Comm_set_attr APMPI_Comm_set_attr
#define  MPI_Comm_get_attr  AMPI_Comm_get_attr
#define PMPI_Comm_get_attr APMPI_Comm_get_attr
#define  MPI_Comm_delete_attr AMPI_Comm_delete_attr
#define PMPI_Comm_delete_attr APMPI_Comm_delete_attr

/***keyvals/attributes***/
#define  MPI_Keyval_create  AMPI_Keyval_create
#define PMPI_Keyval_create APMPI_Keyval_create
#define  MPI_Keyval_free  AMPI_Keyval_free
#define PMPI_Keyval_free APMPI_Keyval_free
#define  MPI_Attr_put  AMPI_Attr_put
#define PMPI_Attr_put APMPI_Attr_put
#define  MPI_Attr_get  AMPI_Attr_get
#define PMPI_Attr_get APMPI_Attr_get
#define  MPI_Attr_delete  AMPI_Attr_delete
#define PMPI_Attr_delete APMPI_Attr_delete

/***topologies***/
#define  MPI_Cart_create  AMPI_Cart_create
#define PMPI_Cart_create APMPI_Cart_create
#define  MPI_Graph_create  AMPI_Graph_create
#define PMPI_Graph_create APMPI_Graph_create
#define  MPI_Dist_graph_create_adjacent AMPI_Dist_graph_create_adjacent
#define PMPI_Dist_graph_create_adjacent APMPI_Dist_graph_create_adjacent
#define  MPI_Dist_graph_create AMPI_Dist_graph_create
#define PMPI_Dist_graph_create APMPI_Dist_graph_create
#define  MPI_Topo_test  AMPI_Topo_test
#define PMPI_Topo_test APMPI_Topo_test
#define  MPI_Cart_map  AMPI_Cart_map
#define PMPI_Cart_map APMPI_Cart_map
#define  MPI_Graph_map  AMPI_Graph_map
#define PMPI_Graph_map APMPI_Graph_map
#define  MPI_Cartdim_get  AMPI_Cartdim_get
#define PMPI_Cartdim_get APMPI_Cartdim_get
#define  MPI_Cart_get  AMPI_Cart_get
#define PMPI_Cart_get APMPI_Cart_get
#define  MPI_Cart_rank  AMPI_Cart_rank
#define PMPI_Cart_rank APMPI_Cart_rank
#define  MPI_Cart_coords  AMPI_Cart_coords
#define PMPI_Cart_coords APMPI_Cart_coords
#define  MPI_Cart_shift  AMPI_Cart_shift
#define PMPI_Cart_shift APMPI_Cart_shift
#define  MPI_Graphdims_get  AMPI_Graphdims_get
#define PMPI_Graphdims_get APMPI_Graphdims_get
#define  MPI_Graph_get  AMPI_Graph_get
#define PMPI_Graph_get APMPI_Graph_get
#define  MPI_Graph_neighbors_count  AMPI_Graph_neighbors_count
#define PMPI_Graph_neighbors_count APMPI_Graph_neighbors_count
#define  MPI_Graph_neighbors  AMPI_Graph_neighbors
#define PMPI_Graph_neighbors APMPI_Graph_neighbors
#define  MPI_Dist_graph_neighbors_count AMPI_Dist_graph_neighbors_count
#define PMPI_Dist_graph_neighbors_count APMPI_Dist_graph_neighbors_count
#define  MPI_Dist_graph_neighbors AMPI_Dist_graph_neighbors
#define PMPI_Dist_graph_neighbors APMPI_Dist_graph_neighbors
#define  MPI_Dims_create  AMPI_Dims_create
#define PMPI_Dims_create APMPI_Dims_create
#define  MPI_Cart_sub  AMPI_Cart_sub
#define PMPI_Cart_sub APMPI_Cart_sub

/***environment management***/
#define  MPI_Errhandler_create  AMPI_Errhandler_create
#define PMPI_Errhandler_create APMPI_Errhandler_create
#define  MPI_Errhandler_set  AMPI_Errhandler_set
#define PMPI_Errhandler_set APMPI_Errhandler_set
#define  MPI_Errhandler_get  AMPI_Errhandler_get
#define PMPI_Errhandler_get APMPI_Errhandler_get
#define  MPI_Errhandler_free  AMPI_Errhandler_free
#define PMPI_Errhandler_free APMPI_Errhandler_free
#define  MPI_Add_error_code  AMPI_Add_error_code
#define PMPI_Add_error_code APMPI_Add_error_code
#define  MPI_Add_error_class  AMPI_Add_error_class
#define PMPI_Add_error_class APMPI_Add_error_class
#define  MPI_Add_error_string  AMPI_Add_error_string
#define PMPI_Add_error_string APMPI_Add_error_string
#define  MPI_Error_class  AMPI_Error_class
#define PMPI_Error_class APMPI_Error_class
#define  MPI_Error_string  AMPI_Error_string
#define PMPI_Error_string APMPI_Error_string
#define  MPI_Get_version  AMPI_Get_version
#define PMPI_Get_version APMPI_Get_version
#define  MPI_Get_library_version  AMPI_Get_library_version
#define PMPI_Get_library_version APMPI_Get_library_version
#define  MPI_Get_processor_name  AMPI_Get_processor_name
#define PMPI_Get_processor_name APMPI_Get_processor_name
#define  MPI_Wtime  AMPI_Wtime
#define PMPI_Wtime APMPI_Wtime
#define  MPI_Wtick  AMPI_Wtick
#define PMPI_Wtick APMPI_Wtick
#define  MPI_Is_thread_main  AMPI_Is_thread_main
#define PMPI_Is_thread_main APMPI_Is_thread_main
#define  MPI_Query_thread  AMPI_Query_thread
#define PMPI_Query_thread APMPI_Query_thread
#define  MPI_Init_thread  AMPI_Init_thread
#define PMPI_Init_thread APMPI_Init_thread
#define  MPI_Init  AMPI_Init
#define PMPI_Init APMPI_Init
#define  MPI_Initialized  AMPI_Initialized
#define PMPI_Initialized APMPI_Initialized
#define  MPI_Finalize  AMPI_Finalize
#define PMPI_Finalize APMPI_Finalize
#define  MPI_Finalized  AMPI_Finalized
#define PMPI_Finalized APMPI_Finalized
#define  MPI_Abort  AMPI_Abort
#define PMPI_Abort APMPI_Abort
#define  MPI_Pcontrol  AMPI_Pcontrol
#define PMPI_Pcontrol APMPI_Pcontrol

/***windows/rma***/
#define  MPI_Win_create  AMPI_Win_create
#define PMPI_Win_create APMPI_Win_create
#define  MPI_Win_free  AMPI_Win_free
#define PMPI_Win_free APMPI_Win_free
#define  MPI_Win_create_errhandler  AMPI_Win_create_errhandler
#define PMPI_Win_create_errhandler APMPI_Win_create_errhandler
#define  MPI_Win_call_errhandler  AMPI_Win_call_errhandler
#define PMPI_Win_call_errhandler APMPI_Win_call_errhandler
#define  MPI_Win_get_errhandler  AMPI_Win_get_errhandler
#define PMPI_Win_get_errhandler APMPI_Win_get_errhandler
#define  MPI_Win_set_errhandler  AMPI_Win_set_errhandler
#define PMPI_Win_set_errhandler APMPI_Win_set_errhandler
#define  MPI_Win_create_keyval  AMPI_Win_create_keyval
#define PMPI_Win_create_keyval APMPI_Win_create_keyval
#define  MPI_Win_free_keyval  AMPI_Win_free_keyval
#define PMPI_Win_free_keyval APMPI_Win_free_keyval
#define  MPI_Win_delete_attr  AMPI_Win_delete_attr
#define PMPI_Win_delete_attr APMPI_Win_delete_attr
#define  MPI_Win_get_attr  AMPI_Win_get_attr
#define PMPI_Win_get_attr APMPI_Win_get_attr
#define  MPI_Win_set_attr  AMPI_Win_set_attr
#define PMPI_Win_set_attr APMPI_Win_set_attr
#define  MPI_Win_get_group  AMPI_Win_get_group
#define PMPI_Win_get_group APMPI_Win_get_group
#define  MPI_Win_set_name  AMPI_Win_set_name
#define PMPI_Win_set_name APMPI_Win_set_name
#define  MPI_Win_get_name  AMPI_Win_get_name
#define PMPI_Win_get_name APMPI_Win_get_name
#define  MPI_Win_set_info  AMPI_Win_set_info
#define PMPI_Win_set_info APMPI_Win_set_info
#define  MPI_Win_get_info  AMPI_Win_get_info
#define PMPI_Win_get_info APMPI_Win_get_info
#define  MPI_Win_fence  AMPI_Win_fence
#define PMPI_Win_fence APMPI_Win_fence
#define  MPI_Win_lock  AMPI_Win_lock
#define PMPI_Win_lock APMPI_Win_lock
#define  MPI_Win_unlock  AMPI_Win_unlock
#define PMPI_Win_unlock APMPI_Win_unlock
#define  MPI_Win_post  AMPI_Win_post
#define PMPI_Win_post APMPI_Win_post
#define  MPI_Win_wait  AMPI_Win_wait
#define PMPI_Win_wait APMPI_Win_wait
#define  MPI_Win_start  AMPI_Win_start
#define PMPI_Win_start APMPI_Win_start
#define  MPI_Win_complete  AMPI_Win_complete
#define PMPI_Win_complete APMPI_Win_complete
#define  MPI_Win_test  AMPI_Win_test
#define PMPI_Win_test APMPI_Win_test
#define  MPI_Alloc_mem  AMPI_Alloc_mem
#define PMPI_Alloc_mem APMPI_Alloc_mem
#define  MPI_Free_mem  AMPI_Free_mem
#define PMPI_Free_mem APMPI_Free_mem
#define  MPI_Put  AMPI_Put
#define PMPI_Put APMPI_Put
#define  MPI_Get  AMPI_Get
#define PMPI_Get APMPI_Get
#define  MPI_Accumulate  AMPI_Accumulate
#define PMPI_Accumulate APMPI_Accumulate
#define  MPI_Get_accumulate  AMPI_Get_accumulate
#define PMPI_Get_accumulate APMPI_Get_accumulate
#define  MPI_Rput  AMPI_Rput
#define PMPI_Rput APMPI_Rput
#define  MPI_Rget  AMPI_Rget
#define PMPI_Rget APMPI_Rget
#define  MPI_Raccumulate  AMPI_Raccumulate
#define PMPI_Raccumulate APMPI_Raccumulate
#define  MPI_Rget_accumulate  AMPI_Rget_accumulate
#define PMPI_Rget_accumulate APMPI_Rget_accumulate
#define  MPI_Fetch_and_op  AMPI_Fetch_and_op
#define PMPI_Fetch_and_op APMPI_Fetch_and_op
#define  MPI_Compare_and_swap  AMPI_Compare_and_swap
#define PMPI_Compare_and_swap APMPI_Compare_and_swap

/***infos***/
#define  MPI_Info_create  AMPI_Info_create
#define PMPI_Info_create APMPI_Info_create
#define  MPI_Info_set  AMPI_Info_set
#define PMPI_Info_set APMPI_Info_set
#define  MPI_Info_delete  AMPI_Info_delete
#define PMPI_Info_delete APMPI_Info_delete
#define  MPI_Info_get  AMPI_Info_get
#define PMPI_Info_get APMPI_Info_get
#define  MPI_Info_get_valuelen  AMPI_Info_get_valuelen
#define PMPI_Info_get_valuelen APMPI_Info_get_valuelen
#define  MPI_Info_get_nkeys  AMPI_Info_get_nkeys
#define PMPI_Info_get_nkeys APMPI_Info_get_nkeys
#define  MPI_Info_get_nthkey  AMPI_Info_get_nthkey
#define PMPI_Info_get_nthkey APMPI_Info_get_nthkey
#define  MPI_Info_dup  AMPI_Info_dup
#define PMPI_Info_dup APMPI_Info_dup
#define  MPI_Info_free  AMPI_Info_free
#define PMPI_Info_free APMPI_Info_free

/***MPIX***/
#define  MPIX_Grequest_start  AMPIX_Grequest_start
#define PMPIX_Grequest_start APMPIX_Grequest_start
#define  MPIX_Grequest_class_create  AMPIX_Grequest_class_create
#define PMPIX_Grequest_class_create APMPIX_Grequest_class_create
#define  MPIX_Grequest_class_allocate  AMPIX_Grequest_class_allocate
#define PMPIX_Grequest_class_allocate APMPIX_Grequest_class_allocate

#define  MPI_Pack_external  AMPI_Pack_external
#define PMPI_Pack_external APMPI_Pack_external
#define  MPI_Pack_external_size  AMPI_Pack_external_size
#define PMPI_Pack_external_size APMPI_Pack_external_size
#define  MPI_Unpack_external  AMPI_Unpack_external
#define PMPI_Unpack_external APMPI_Unpack_external

#define  MPI_File_call_errhandler  AMPI_File_call_errhandler
#define PMPI_File_call_errhandler APMPI_File_call_errhandler
#define  MPI_File_create_errhandler  AMPI_File_create_errhandler
#define PMPI_File_create_errhandler APMPI_File_create_errhandler
#define  MPI_File_get_errhandler  AMPI_File_get_errhandler
#define PMPI_File_get_errhandler APMPI_File_get_errhandler
#define  MPI_File_set_errhandler  AMPI_File_set_errhandler
#define PMPI_File_set_errhandler APMPI_File_set_errhandler

#define  MPI_Close_port  AMPI_Close_port
#define PMPI_Close_port APMPI_Close_port
#define  MPI_Comm_accept  AMPI_Comm_accept
#define PMPI_Comm_accept APMPI_Comm_accept
#define  MPI_Comm_connect  AMPI_Comm_connect
#define PMPI_Comm_connect APMPI_Comm_connect
#define  MPI_Comm_disconnect  AMPI_Comm_disconnect
#define PMPI_Comm_disconnect APMPI_Comm_disconnect
#define  MPI_Comm_get_parent  AMPI_Comm_get_parent
#define PMPI_Comm_get_parent APMPI_Comm_get_parent
#define  MPI_Comm_join  AMPI_Comm_join
#define PMPI_Comm_join APMPI_Comm_join
#define  MPI_Comm_spawn_multiple  AMPI_Comm_spawn_multiple
#define PMPI_Comm_spawn_multiple APMPI_Comm_spawn_multiple
#define  MPI_Lookup_name  AMPI_Lookup_name
#define PMPI_Lookup_name APMPI_Lookup_name
#define  MPI_Open_port  AMPI_Open_port
#define PMPI_Open_port APMPI_Open_port
#define  MPI_Publish_name  AMPI_Publish_name
#define PMPI_Publish_name APMPI_Publish_name
#define  MPI_Unpublish_name  AMPI_Unpublish_name
#define PMPI_Unpublish_name APMPI_Unpublish_name
#define  MPI_Comm_spawn  AMPI_Comm_spawn
#define PMPI_Comm_spawn APMPI_Comm_spawn

#define  MPI_Win_allocate  AMPI_Win_allocate
#define PMPI_Win_allocate APMPI_Win_allocate
#define  MPI_Win_allocate_shared  AMPI_Win_allocate_shared
#define PMPI_Win_allocate_shared APMPI_Win_allocate_shared
#define  MPI_Win_attach  AMPI_Win_attach
#define PMPI_Win_attach APMPI_Win_attach
#define  MPI_Win_create_dynamic  AMPI_Win_create_dynamic
#define PMPI_Win_create_dynamic APMPI_Win_create_dynamic
#define  MPI_Win_detach  AMPI_Win_detach
#define PMPI_Win_detach APMPI_Win_detach
#define  MPI_Win_flush  AMPI_Win_flush
#define PMPI_Win_flush APMPI_Win_flush
#define  MPI_Win_flush_all  AMPI_Win_flush_all
#define PMPI_Win_flush_all APMPI_Win_flush_all
#define  MPI_Win_flush_local  AMPI_Win_flush_local
#define PMPI_Win_flush_local APMPI_Win_flush_local
#define  MPI_Win_flush_local_all  AMPI_Win_flush_local_all
#define PMPI_Win_flush_local_all APMPI_Win_flush_local_all
#define  MPI_Win_lock_all  AMPI_Win_lock_all
#define PMPI_Win_lock_all APMPI_Win_lock_all
#define  MPI_Win_shared_query  AMPI_Win_shared_query
#define PMPI_Win_shared_query APMPI_Win_shared_query
#define  MPI_Win_sync  AMPI_Win_sync
#define PMPI_Win_sync APMPI_Win_sync
#define  MPI_Win_unlock_all  AMPI_Win_unlock_all
#define PMPI_Win_unlock_all APMPI_Win_unlock_all

#define  MPI_CONVERSION_FN_NULL  AMPI_CONVERSION_FN_NULL
#define PMPI_CONVERSION_FN_NULL APMPI_CONVERSION_FN_NULL
#define  MPI_File_iread_all  AMPI_File_iread_all
#define PMPI_File_iread_all APMPI_File_iread_all
#define  MPI_File_iread_at_all  AMPI_File_iread_at_all
#define PMPI_File_iread_at_all APMPI_File_iread_at_all
#define  MPI_File_iwrite_all  AMPI_File_iwrite_all
#define PMPI_File_iwrite_all APMPI_File_iwrite_all
#define  MPI_File_iwrite_at_all  AMPI_File_iwrite_at_all
#define PMPI_File_iwrite_at_all APMPI_File_iwrite_at_all

#define  MPI_Status_f082f  AMPI_Status_f082f
#define PMPI_Status_f082f APMPI_Status_f082f
#define  MPI_Status_f2f08  AMPI_Status_f2f08
#define PMPI_Status_f2f08 APMPI_Status_f2f08
#define  MPI_Type_create_f90_complex  AMPI_Type_create_f90_complex
#define PMPI_Type_create_f90_complex APMPI_Type_create_f90_complex
#define  MPI_Type_create_f90_integer  AMPI_Type_create_f90_integer
#define PMPI_Type_create_f90_integer APMPI_Type_create_f90_integer
#define  MPI_Type_create_f90_real  AMPI_Type_create_f90_real
#define PMPI_Type_create_f90_real APMPI_Type_create_f90_real
#define  MPI_Type_match_size  AMPI_Type_match_size
#define PMPI_Type_match_size APMPI_Type_match_size
#define  MPI_Message_c2f  AMPI_Message_c2f
#define PMPI_Message_c2f APMPI_Message_c2f
#define  MPI_Message_f2c  AMPI_Message_f2c
#define PMPI_Message_f2c APMPI_Message_f2c
#define  MPI_Status_c2f  AMPI_Status_c2f
#define PMPI_Status_c2f APMPI_Status_c2f
#define  MPI_Status_c2f08  AMPI_Status_c2f08
#define PMPI_Status_c2f08 APMPI_Status_c2f08
#define  MPI_Status_f082c  AMPI_Status_f082c
#define PMPI_Status_f082c APMPI_Status_f082c
#define  MPI_Status_f2c  AMPI_Status_f2c
#define PMPI_Status_f2c APMPI_Status_f2c

#define  MPI_T_category_changed  AMPI_T_category_changed
#define PMPI_T_category_changed APMPI_T_category_changed
#define  MPI_T_category_get_categories  AMPI_T_category_get_categories
#define PMPI_T_category_get_categories APMPI_T_category_get_categories
#define  MPI_T_category_get_cvars  AMPI_T_category_get_cvars
#define PMPI_T_category_get_cvars APMPI_T_category_get_cvars
#define  MPI_T_category_get_index  AMPI_T_category_get_index
#define PMPI_T_category_get_index APMPI_T_category_get_index
#define  MPI_T_category_get_info  AMPI_T_category_get_info
#define PMPI_T_category_get_info APMPI_T_category_get_info
#define  MPI_T_category_get_num  AMPI_T_category_get_num
#define PMPI_T_category_get_num APMPI_T_category_get_num
#define  MPI_T_category_get_pvars  AMPI_T_category_get_pvars
#define PMPI_T_category_get_pvars APMPI_T_category_get_pvars
#define  MPI_T_cvar_get_index  AMPI_T_cvar_get_index
#define PMPI_T_cvar_get_index APMPI_T_cvar_get_index
#define  MPI_T_cvar_get_info  AMPI_T_cvar_get_info
#define PMPI_T_cvar_get_info APMPI_T_cvar_get_info
#define  MPI_T_cvar_get_num  AMPI_T_cvar_get_num
#define PMPI_T_cvar_get_num APMPI_T_cvar_get_num
#define  MPI_T_cvar_handle_alloc  AMPI_T_cvar_handle_alloc
#define PMPI_T_cvar_handle_alloc APMPI_T_cvar_handle_alloc
#define  MPI_T_cvar_handle_free  AMPI_T_cvar_handle_free
#define PMPI_T_cvar_handle_free APMPI_T_cvar_handle_free
#define  MPI_T_cvar_read  AMPI_T_cvar_read
#define PMPI_T_cvar_read APMPI_T_cvar_read
#define  MPI_T_cvar_write  AMPI_T_cvar_write
#define PMPI_T_cvar_write APMPI_T_cvar_write
#define  MPI_T_enum_get_info  AMPI_T_enum_get_info
#define PMPI_T_enum_get_info APMPI_T_enum_get_info
#define  MPI_T_enum_get_item  AMPI_T_enum_get_item
#define PMPI_T_enum_get_item APMPI_T_enum_get_item
#define  MPI_T_finalize  AMPI_T_finalize
#define PMPI_T_finalize APMPI_T_finalize
#define  MPI_T_init_thread  AMPI_T_init_thread
#define PMPI_T_init_thread APMPI_T_init_thread
#define  MPI_T_pvar_get_index  AMPI_T_pvar_get_index
#define PMPI_T_pvar_get_index APMPI_T_pvar_get_index
#define  MPI_T_pvar_get_info  AMPI_T_pvar_get_info
#define PMPI_T_pvar_get_info APMPI_T_pvar_get_info
#define  MPI_T_pvar_get_num  AMPI_T_pvar_get_num
#define PMPI_T_pvar_get_num APMPI_T_pvar_get_num
#define  MPI_T_pvar_handle_alloc  AMPI_T_pvar_handle_alloc
#define PMPI_T_pvar_handle_alloc APMPI_T_pvar_handle_alloc
#define  MPI_T_pvar_handle_free  AMPI_T_pvar_handle_free
#define PMPI_T_pvar_handle_free APMPI_T_pvar_handle_free
#define  MPI_T_pvar_read  AMPI_T_pvar_read
#define PMPI_T_pvar_read APMPI_T_pvar_read
#define  MPI_T_pvar_readreset  AMPI_T_pvar_readreset
#define PMPI_T_pvar_readreset APMPI_T_pvar_readreset
#define  MPI_T_pvar_reset  AMPI_T_pvar_reset
#define PMPI_T_pvar_reset APMPI_T_pvar_reset
#define  MPI_T_pvar_session_create  AMPI_T_pvar_session_create
#define PMPI_T_pvar_session_create APMPI_T_pvar_session_create
#define  MPI_T_pvar_session_free  AMPI_T_pvar_session_free
#define PMPI_T_pvar_session_free APMPI_T_pvar_session_free
#define  MPI_T_pvar_start  AMPI_T_pvar_start
#define PMPI_T_pvar_start APMPI_T_pvar_start
#define  MPI_T_pvar_stop  AMPI_T_pvar_stop
#define PMPI_T_pvar_stop APMPI_T_pvar_stop
#define  MPI_T_pvar_write  AMPI_T_pvar_write
#define PMPI_T_pvar_write APMPI_T_pvar_write

#endif //CMK_CONVERSE_MPI

/***pt2pt***/
AMPI_API_DEF(int, MPI_Send, const void *msg, int count, MPI_Datatype type, int dest,
              int tag, MPI_Comm comm)
AMPI_API_DEF(int, MPI_Ssend, const void *msg, int count, MPI_Datatype type, int dest,
               int tag, MPI_Comm comm)
AMPI_API_DEF(int, MPI_Recv, void *msg, int count, MPI_Datatype type, int src, int tag,
              MPI_Comm comm, MPI_Status *status)
AMPI_API_DEF(int, MPI_Mrecv, void* buf, int count, MPI_Datatype datatype, MPI_Message *message,
                  MPI_Status *status)
AMPI_API_DEF(int, MPI_Get_count, const MPI_Status *sts, MPI_Datatype dtype, int *count)
AMPI_API_DEF(int, MPI_Bsend, const void *buf, int count, MPI_Datatype datatype,
             int dest, int tag,MPI_Comm comm)
AMPI_API_DEF(int, MPI_Rsend, const void *buf, int count, MPI_Datatype datatype,
             int dest, int tag,MPI_Comm comm)
AMPI_API_DEF(int, MPI_Buffer_attach, void *buffer, int size)
AMPI_API_DEF(int, MPI_Buffer_detach, void *buffer, int *size)
AMPI_API_DEF(int, MPI_Isend, const void *buf, int count, MPI_Datatype datatype, int dest,
               int tag, MPI_Comm comm, MPI_Request *request)
AMPI_API_DEF(int, MPI_Ibsend, const void *buf, int count, MPI_Datatype datatype, int dest,
               int tag, MPI_Comm comm, MPI_Request *request)
AMPI_API_DEF(int, MPI_Issend, const void *buf, int count, MPI_Datatype datatype, int dest,
             int tag, MPI_Comm comm, MPI_Request *request)
AMPI_API_DEF(int, MPI_Irsend, const void *buf, int count, MPI_Datatype datatype, int dest,
             int tag, MPI_Comm comm, MPI_Request *request)
AMPI_API_DEF(int, MPI_Irecv, void *buf, int count, MPI_Datatype datatype, int src,
               int tag, MPI_Comm comm, MPI_Request *request)
AMPI_API_DEF(int, MPI_Imrecv, void* buf, int count, MPI_Datatype datatype, MPI_Message *message,
                  MPI_Request *request)
AMPI_API_DEF(int, MPI_Wait, MPI_Request *request, MPI_Status *sts)
AMPI_API_DEF(int, MPI_Test, MPI_Request *request, int *flag, MPI_Status *sts)
AMPI_API_DEF(int, MPI_Waitany, int count, MPI_Request *request, int *index, MPI_Status *sts)
AMPI_API_DEF(int, MPI_Testany, int count, MPI_Request *request, int *index, int *flag, MPI_Status *status)
AMPI_API_DEF(int, MPI_Waitall, int count, MPI_Request *request, MPI_Status *sts)
AMPI_API_DEF(int, MPI_Testall, int count, MPI_Request *request, int *flag, MPI_Status *sts)
AMPI_API_DEF(int, MPI_Waitsome, int incount, MPI_Request *array_of_requests, int *outcount,
                  int *array_of_indices, MPI_Status *array_of_statuses)
AMPI_API_DEF(int, MPI_Testsome, int incount, MPI_Request *array_of_requests, int *outcount,
                  int *array_of_indices, MPI_Status *array_of_statuses)
AMPI_API_DEF(int, MPI_Request_get_status, MPI_Request request, int *flag, MPI_Status *sts)
AMPI_API_DEF(int, MPI_Request_free, MPI_Request *request)
AMPI_API_DEF(int, MPI_Grequest_start, MPI_Grequest_query_function *query_fn, MPI_Grequest_free_function *free_fn,\
                  MPI_Grequest_cancel_function *cancel_fn, void *extra_state, MPI_Request *request)
AMPI_API_DEF(int, MPI_Grequest_complete, MPI_Request request)
AMPI_API_DEF(int, MPI_Cancel, MPI_Request *request)
AMPI_API_DEF(int, MPI_Test_cancelled, const MPI_Status *status, int *flag) /* FIXME: always returns success */
AMPI_API_DEF(int, MPI_Status_set_cancelled, MPI_Status *status, int flag)
AMPI_API_DEF(int, MPI_Iprobe, int src, int tag, MPI_Comm comm, int *flag, MPI_Status *sts)
AMPI_API_DEF(int, MPI_Probe, int source, int tag, MPI_Comm comm, MPI_Status *sts)
AMPI_API_DEF(int, MPI_Improbe, int source, int tag, MPI_Comm comm, int *flag,
                  MPI_Message *message, MPI_Status *status)
AMPI_API_DEF(int, MPI_Mprobe, int source, int tag, MPI_Comm comm, MPI_Message *message,
                  MPI_Status *status)
AMPI_API_DEF(int, MPI_Send_init, const void *buf, int count, MPI_Datatype type, int dest, int tag,
                  MPI_Comm comm, MPI_Request *req)
AMPI_API_DEF(int, MPI_Ssend_init, const void *buf, int count, MPI_Datatype type, int dest, int tag,
                    MPI_Comm comm, MPI_Request *req)
AMPI_API_DEF(int, MPI_Rsend_init, const void *buf, int count, MPI_Datatype type, int dest, int tag,
                  MPI_Comm comm, MPI_Request *req)
AMPI_API_DEF(int, MPI_Bsend_init, const void *buf, int count, MPI_Datatype type, int dest, int tag,
                  MPI_Comm comm, MPI_Request *req)
AMPI_API_DEF(int, MPI_Recv_init, void *buf, int count, MPI_Datatype type, int src, int tag,
                   MPI_Comm comm, MPI_Request *req)
AMPI_API_DEF(int, MPI_Start, MPI_Request *reqnum)
AMPI_API_DEF(int, MPI_Startall, int count, MPI_Request *array_of_requests)
AMPI_API_DEF(int, MPI_Sendrecv, const void *sbuf, int scount, MPI_Datatype stype, int dest,
                  int stag, void *rbuf, int rcount, MPI_Datatype rtype,
                  int src, int rtag, MPI_Comm comm, MPI_Status *sts)
AMPI_API_DEF(int, MPI_Sendrecv_replace, void* buf, int count, MPI_Datatype datatype,
                          int dest, int sendtag, int source, int recvtag,
                          MPI_Comm comm, MPI_Status *status)

/***datatypes***/
AMPI_API_DEF(int, MPI_Type_contiguous, int count, MPI_Datatype oldtype,
                         MPI_Datatype *newtype)
AMPI_API_DEF(int, MPI_Type_vector, int count, int blocklength, int stride,
                     MPI_Datatype oldtype, MPI_Datatype *newtype)
AMPI_API_DEF(int, MPI_Type_create_hvector, int count, int blocklength, MPI_Aint stride,
                             MPI_Datatype oldtype, MPI_Datatype *newtype)
AMPI_API_DEF(int, MPI_Type_hvector, int count, int blocklength, MPI_Aint stride,
                      MPI_Datatype oldtype, MPI_Datatype *newtype)
AMPI_API_DEF(int, MPI_Type_indexed, int count, const int* arrBlength, const int* arrDisp,
                      MPI_Datatype oldtype, MPI_Datatype *newtype)
AMPI_API_DEF(int, MPI_Type_create_hindexed, int count, const int* arrBlength, const MPI_Aint* arrDisp,
                              MPI_Datatype oldtype, MPI_Datatype *newtype)
AMPI_API_DEF(int, MPI_Type_create_indexed_block, int count, int Blength, const int *arrDisp,
                                   MPI_Datatype oldtype, MPI_Datatype *newtype)
AMPI_API_DEF(int, MPI_Type_create_hindexed_block, int count, int Blength, const MPI_Aint *arrDisp,
                                    MPI_Datatype oldtype, MPI_Datatype *newtype)
AMPI_API_DEF(int, MPI_Type_hindexed, int count, int* arrBlength, MPI_Aint* arrDisp,
                       MPI_Datatype oldtype, MPI_Datatype *newtype)
AMPI_API_DEF(int, MPI_Type_create_struct, int count, const int* arrBLength, const MPI_Aint* arrDisp,
                            const MPI_Datatype *oldType, MPI_Datatype *newType)
AMPI_API_DEF(int, MPI_Type_struct, int count, int* arrBLength, MPI_Aint* arrDisp,
                     MPI_Datatype *oldType, MPI_Datatype *newType)
AMPI_API_DEF(int, MPI_Type_get_envelope, MPI_Datatype datatype, int *num_integers, int *num_addresses,
                           int *num_datatypes, int *combiner)
AMPI_API_DEF(int, MPI_Type_get_contents, MPI_Datatype datatype, int max_integers, int max_addresses,
                           int max_datatypes, int array_of_integers[], MPI_Aint array_of_addresses[],
                           MPI_Datatype array_of_datatypes[])
AMPI_API_DEF(int, MPI_Type_commit, MPI_Datatype *datatype)
AMPI_API_DEF(int, MPI_Type_free, MPI_Datatype *datatype)
AMPI_API_DEF(int, MPI_Type_get_extent, MPI_Datatype datatype, MPI_Aint *lb, MPI_Aint *extent)
AMPI_API_DEF(int, MPI_Type_get_extent_x, MPI_Datatype datatype, MPI_Count *lb, MPI_Count *extent)
AMPI_API_DEF(int, MPI_Type_extent, MPI_Datatype datatype, MPI_Aint *extent)
AMPI_API_DEF(int, MPI_Type_get_true_extent, MPI_Datatype datatype, MPI_Aint *true_lb, MPI_Aint *true_extent)
AMPI_API_DEF(int, MPI_Type_get_true_extent_x, MPI_Datatype datatype, MPI_Count *true_lb, MPI_Count *true_extent)
AMPI_API_DEF(int, MPI_Type_size, MPI_Datatype datatype, int *size)
AMPI_API_DEF(int, MPI_Type_size_x, MPI_Datatype datatype, MPI_Count *size)
AMPI_API_DEF(int, MPI_Type_lb, MPI_Datatype datatype, MPI_Aint* displacement)
AMPI_API_DEF(int, MPI_Type_ub, MPI_Datatype datatype, MPI_Aint* displacement)
AMPI_API_DEF(int, MPI_Type_set_name, MPI_Datatype datatype, const char *name)
AMPI_API_DEF(int, MPI_Type_get_name, MPI_Datatype datatype, char *name, int *resultlen)
AMPI_API_DEF(int, MPI_Type_dup, MPI_Datatype oldtype, MPI_Datatype *newtype)
AMPI_API_DEF(int, MPI_Type_create_resized, MPI_Datatype oldtype, MPI_Aint lb, MPI_Aint extent, MPI_Datatype *newtype)
AMPI_API_DEF(int, MPI_Type_set_attr, MPI_Datatype datatype, int type_keyval, void *attribute_val)
AMPI_API_DEF(int, MPI_Type_get_attr, MPI_Datatype datatype, int type_keyval, void *attribute_val, int *flag)
AMPI_API_DEF(int, MPI_Type_delete_attr, MPI_Datatype datatype, int type_keyval)
AMPI_API_DEF(int, MPI_Type_create_keyval, MPI_Type_copy_attr_function *type_copy_attr_fn,
                            MPI_Type_delete_attr_function *type_delete_attr_fn,
                            int *type_keyval, void *extra_state)
AMPI_API_DEF(int, MPI_Type_free_keyval, int *type_keyval)
AMPI_API_DEF(int, MPI_Type_create_darray, int size, int rank, int ndims,
  const int array_of_gsizes[], const int array_of_distribs[],
  const int array_of_dargs[], const int array_of_psizes[],
  int order, MPI_Datatype oldtype, MPI_Datatype *newtype)
AMPI_API_DEF(int, MPI_Type_create_subarray, int ndims, const int array_of_sizes[],
  const int array_of_subsizes[], const int array_of_starts[], int order,
  MPI_Datatype oldtype, MPI_Datatype *newtype)
AMPI_API_DEF(int, MPI_Get_address, const void* location, MPI_Aint *address)
AMPI_API_DEF(int, MPI_Address, void* location, MPI_Aint *address)
AMPI_API_DEF(int, MPI_Status_set_elements, MPI_Status *status, MPI_Datatype datatype, int count)
AMPI_API_DEF(int, MPI_Status_set_elements_x, MPI_Status *status, MPI_Datatype datatype, MPI_Count count)
AMPI_API_DEF(int, MPI_Get_elements, const MPI_Status *status, MPI_Datatype datatype, int *count)
AMPI_API_DEF(int, MPI_Get_elements_x, const MPI_Status *status, MPI_Datatype datatype, MPI_Count *count)
AMPI_API_DEF(int, MPI_Pack, const void *inbuf, int incount, MPI_Datatype dtype, void *outbuf,
              int outsize, int *position, MPI_Comm comm)
AMPI_API_DEF(int, MPI_Unpack, const void *inbuf, int insize, int *position, void *outbuf,
                int outcount, MPI_Datatype dtype, MPI_Comm comm)
AMPI_API_DEF(int, MPI_Pack_size, int incount,MPI_Datatype datatype,MPI_Comm comm,int *sz)
#define MPI_Aint_add(addr, disp) ((MPI_Aint)((char*)(addr) + (disp)))
#define MPI_Aint_diff(addr1, addr2) ((MPI_Aint)((char*)(addr1) - (char*)(addr2)))

/***collectives***/
AMPI_API_DEF(int, MPI_Barrier, MPI_Comm comm)
AMPI_API_DEF(int, MPI_Ibarrier, MPI_Comm comm, MPI_Request *request)
AMPI_API_DEF(int, MPI_Bcast, void *buf, int count, MPI_Datatype type, int root, MPI_Comm comm)
AMPI_API_DEF(int, MPI_Ibcast, void *buf, int count, MPI_Datatype type, int root, MPI_Comm comm,
                MPI_Request *request)
AMPI_API_DEF(int, MPI_Gather, const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                void *recvbuf, int recvcount, MPI_Datatype recvtype,
                int root, MPI_Comm comm)
AMPI_API_DEF(int, MPI_Igather, const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                 void *recvbuf, int recvcount, MPI_Datatype recvtype,
                 int root, MPI_Comm comm, MPI_Request *request)
AMPI_API_DEF(int, MPI_Gatherv, const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                 void *recvbuf, const int *recvcounts, const int *displs,
                 MPI_Datatype recvtype, int root, MPI_Comm comm)
AMPI_API_DEF(int, MPI_Igatherv, const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                  void *recvbuf, const int *recvcounts, const int *displs,
                  MPI_Datatype recvtype, int root, MPI_Comm comm,
                  MPI_Request *request)
AMPI_API_DEF(int, MPI_Scatter, const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                 void *recvbuf, int recvcount, MPI_Datatype recvtype,
                 int root, MPI_Comm comm)
AMPI_API_DEF(int, MPI_Iscatter, const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                  void *recvbuf, int recvcount, MPI_Datatype recvtype,
                  int root, MPI_Comm comm, MPI_Request *request)
AMPI_API_DEF(int, MPI_Scatterv, const void *sendbuf, const int *sendcounts, const int *displs, MPI_Datatype sendtype,
                  void *recvbuf, int recvcount, MPI_Datatype recvtype,
                  int root, MPI_Comm comm)
AMPI_API_DEF(int, MPI_Iscatterv, const void *sendbuf, const int *sendcounts, const int *displs, MPI_Datatype sendtype,
                   void *recvbuf, int recvcount, MPI_Datatype recvtype,
                   int root, MPI_Comm comm, MPI_Request *request)
AMPI_API_DEF(int, MPI_Allgather, const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                   void *recvbuf, int recvcount, MPI_Datatype recvtype,
                   MPI_Comm comm)
AMPI_API_DEF(int, MPI_Iallgather, const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                    void *recvbuf, int recvcount, MPI_Datatype recvtype,
                    MPI_Comm comm, MPI_Request* request)
AMPI_API_DEF(int, MPI_Allgatherv, const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                    void *recvbuf, const int *recvcounts, const int *displs,
                    MPI_Datatype recvtype, MPI_Comm comm)
AMPI_API_DEF(int, MPI_Iallgatherv, const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                     void *recvbuf, const int *recvcounts, const int *displs,
                     MPI_Datatype recvtype, MPI_Comm comm, MPI_Request *request)
AMPI_API_DEF(int, MPI_Alltoall, const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                  void *recvbuf, int recvcount, MPI_Datatype recvtype,
                  MPI_Comm comm)
AMPI_API_DEF(int, MPI_Ialltoall, const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                   void *recvbuf, int recvcount, MPI_Datatype recvtype,
                   MPI_Comm comm, MPI_Request *request)
AMPI_API_DEF(int, MPI_Alltoallv, const void *sendbuf, const int *sendcounts, const int *sdispls,
                   MPI_Datatype sendtype, void *recvbuf, const int *recvcounts,
                   const int *rdispls, MPI_Datatype recvtype, MPI_Comm comm)
AMPI_API_DEF(int, MPI_Ialltoallv, void *sendbuf, int *sendcounts, int *sdispls,
                    MPI_Datatype sendtype, void *recvbuf, int *recvcounts,
                    int *rdispls, MPI_Datatype recvtype, MPI_Comm comm,
                    MPI_Request *request)
AMPI_API_DEF(int, MPI_Alltoallw, const void *sendbuf, const int *sendcounts, const int *sdispls,
                   const MPI_Datatype *sendtypes, void *recvbuf, const int *recvcounts,
                   const int *rdispls, const MPI_Datatype *recvtypes, MPI_Comm comm)
AMPI_API_DEF(int, MPI_Ialltoallw, const void *sendbuf, const int *sendcounts, const int *sdispls,
                    const MPI_Datatype *sendtypes, void *recvbuf, const int *recvcounts,
                    const int *rdispls, const MPI_Datatype *recvtypes, MPI_Comm comm,
                    MPI_Request *request)
AMPI_API_DEF(int, MPI_Reduce, const void *inbuf, void *outbuf, int count, MPI_Datatype type,
                MPI_Op op, int root, MPI_Comm comm)
AMPI_API_DEF(int, MPI_Ireduce, const void *sendbuf, void *recvbuf, int count, MPI_Datatype type,
                 MPI_Op op, int root, MPI_Comm comm, MPI_Request *request)
AMPI_API_DEF(int, MPI_Allreduce, const void *inbuf, void *outbuf, int count, MPI_Datatype type,
                   MPI_Op op, MPI_Comm comm)
AMPI_API_DEF(int, MPI_Iallreduce, const void *inbuf, void *outbuf, int count, MPI_Datatype type,
                    MPI_Op op, MPI_Comm comm, MPI_Request *request)
AMPI_API_DEF(int, MPI_Reduce_local, const void *inbuf, void *outbuf, int count,
                      MPI_Datatype datatype, MPI_Op op)
AMPI_API_DEF(int, MPI_Reduce_scatter_block, const void* sendbuf, void* recvbuf, int count,
                              MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
AMPI_API_DEF(int, MPI_Ireduce_scatter_block, const void* sendbuf, void* recvbuf, int count,
                              MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, MPI_Request *request)
AMPI_API_DEF(int, MPI_Reduce_scatter, const void* sendbuf, void* recvbuf, const int *recvcounts,
                        MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
AMPI_API_DEF(int, MPI_Ireduce_scatter, const void* sendbuf, void* recvbuf, const int *recvcounts,
                        MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, MPI_Request *request)
AMPI_API_DEF(int, MPI_Scan, const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype,
              MPI_Op op, MPI_Comm comm )
AMPI_API_DEF(int, MPI_Iscan, const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype,
              MPI_Op op, MPI_Comm comm, MPI_Request *request)
AMPI_API_DEF(int, MPI_Exscan, const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype,
                MPI_Op op, MPI_Comm comm)
AMPI_API_DEF(int, MPI_Iexscan, const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype,
                MPI_Op op, MPI_Comm comm, MPI_Request *request)

/***neighborhood collectives***/
AMPI_API_DEF(int, MPI_Neighbor_alltoall, const void* sendbuf, int sendcount, MPI_Datatype sendtype,
                           void* recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm)
AMPI_API_DEF(int, MPI_Ineighbor_alltoall, const void* sendbuf, int sendcount, MPI_Datatype sendtype,
                            void* recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm,
                            MPI_Request* request)
AMPI_API_DEF(int, MPI_Neighbor_alltoallv, const void* sendbuf, const int* sendcounts, const int* sdispls,
                            MPI_Datatype sendtype, void* recvbuf, const int* recvcounts, const int* rdispls,
                            MPI_Datatype recvtype, MPI_Comm comm)
AMPI_API_DEF(int, MPI_Ineighbor_alltoallv, const void* sendbuf, const int* sendcounts, const int* sdispls,
                             MPI_Datatype sendtype, void* recvbuf, const int* recvcounts, const int* rdispls,
                             MPI_Datatype recvtype, MPI_Comm comm, MPI_Request* request)
AMPI_API_DEF(int, MPI_Neighbor_alltoallw, const void* sendbuf, const int* sendcounts, const MPI_Aint* sdipls,
                            const MPI_Datatype* sendtypes, void* recvbuf, const int* recvcounts, const MPI_Aint* rdispls,
                            const MPI_Datatype* recvtypes, MPI_Comm comm)
AMPI_API_DEF(int, MPI_Ineighbor_alltoallw, const void* sendbuf, const int* sendcounts, const MPI_Aint* sdispls,
                             const MPI_Datatype* sendtypes, void* recvbuf, const int* recvcounts, const MPI_Aint* rdispls,
                             const MPI_Datatype* recvtypes, MPI_Comm comm, MPI_Request* request)
AMPI_API_DEF(int, MPI_Neighbor_allgather, const void* sendbuf, int sendcount, MPI_Datatype sendtype,
                            void* recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm)
AMPI_API_DEF(int, MPI_Ineighbor_allgather, const void* sendbuf, int sendcount, MPI_Datatype sendtype,
                             void* recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm,
                             MPI_Request *request)
AMPI_API_DEF(int, MPI_Neighbor_allgatherv, const void* sendbuf, int sendcount, MPI_Datatype sendtype,
                             void* recvbuf, const int* recvcounts, const int* displs, MPI_Datatype recvtype,
                             MPI_Comm comm)
AMPI_API_DEF(int, MPI_Ineighbor_allgatherv, const void* sendbuf, int sendcount, MPI_Datatype sendtype,
                              void* recvbuf, const int* recvcounts, const int* displs, MPI_Datatype recvtype,
                              MPI_Comm comm, MPI_Request *request)

/***ops***/
AMPI_API_DEF(int, MPI_Op_create, MPI_User_function *function, int commute, MPI_Op *op)
AMPI_API_DEF(int, MPI_Op_free, MPI_Op *op)
AMPI_API_DEF(int, MPI_Op_commutative, MPI_Op op, int* commute)

/***groups***/
AMPI_API_DEF(int, MPI_Group_size, MPI_Group group, int *size)
AMPI_API_DEF(int, MPI_Group_rank, MPI_Group group, int *rank)
AMPI_API_DEF(int, MPI_Group_translate_ranks, MPI_Group group1, int n, const int *ranks1, MPI_Group group2, int *ranks2)
AMPI_API_DEF(int, MPI_Group_compare, MPI_Group group1,MPI_Group group2, int *result)
AMPI_API_DEF(int, MPI_Comm_group, MPI_Comm comm, MPI_Group *group)
AMPI_API_DEF(int, MPI_Group_union, MPI_Group group1, MPI_Group group2, MPI_Group *newgroup)
AMPI_API_DEF(int, MPI_Group_intersection, MPI_Group group1, MPI_Group group2, MPI_Group *newgroup)
AMPI_API_DEF(int, MPI_Group_difference, MPI_Group group1, MPI_Group group2, MPI_Group *newgroup)
AMPI_API_DEF(int, MPI_Group_incl, MPI_Group group, int n, const int *ranks, MPI_Group *newgroup)
AMPI_API_DEF(int, MPI_Group_excl, MPI_Group group, int n, const int *ranks, MPI_Group *newgroup)
AMPI_API_DEF(int, MPI_Group_range_incl, MPI_Group group, int n, int ranges[][3], MPI_Group *newgroup)
AMPI_API_DEF(int, MPI_Group_range_excl, MPI_Group group, int n, int ranges[][3], MPI_Group *newgroup)
AMPI_API_DEF(int, MPI_Group_free, MPI_Group *group)

/***communicators***/
AMPI_API_DEF(int, MPI_Intercomm_create, MPI_Comm local_comm, int local_leader, MPI_Comm peer_comm,
                          int remote_leader, int tag, MPI_Comm *newintercomm)
AMPI_API_DEF(int, MPI_Intercomm_merge, MPI_Comm intercomm, int high, MPI_Comm *newintracomm)
AMPI_API_DEF(int, MPI_Comm_create, MPI_Comm comm, MPI_Group group, MPI_Comm* newcomm)
AMPI_API_DEF(int, MPI_Comm_create_group, MPI_Comm comm, MPI_Group group, int tag, MPI_Comm *newcomm)
AMPI_API_DEF(int, MPI_Comm_size, MPI_Comm comm, int *size)
AMPI_API_DEF(int, MPI_Comm_rank, MPI_Comm comm, int *rank)
AMPI_API_DEF(int, MPI_Comm_compare, MPI_Comm comm1,MPI_Comm comm2, int *result)
AMPI_API_DEF(int, MPI_Comm_split, MPI_Comm src, int color, int key, MPI_Comm *dest)
AMPI_API_DEF(int, MPI_Comm_split_type, MPI_Comm src, int split_type, int key, MPI_Info info, MPI_Comm *dest)
AMPI_API_DEF(int, MPI_Comm_dup, MPI_Comm src, MPI_Comm *dest)
AMPI_API_DEF(int, MPI_Comm_idup, MPI_Comm comm, MPI_Comm *newcomm, MPI_Request *request)
AMPI_API_DEF(int, MPI_Comm_dup_with_info, MPI_Comm src, MPI_Info info, MPI_Comm *dest)
AMPI_API_DEF(int, MPI_Comm_idup_with_info, MPI_Comm src, MPI_Info info, MPI_Comm *dest, MPI_Request *request)
AMPI_API_DEF(int, MPI_Comm_free, MPI_Comm *comm)
AMPI_API_DEF(int, MPI_Comm_test_inter, MPI_Comm comm, int *flag)
AMPI_API_DEF(int, MPI_Comm_remote_size, MPI_Comm comm, int *size)
AMPI_API_DEF(int, MPI_Comm_remote_group, MPI_Comm comm, MPI_Group *group)
AMPI_API_DEF(int, MPI_Comm_set_name, MPI_Comm comm, const char *name)
AMPI_API_DEF(int, MPI_Comm_get_name, MPI_Comm comm, char *comm_name, int *resultlen)
AMPI_API_DEF(int, MPI_Comm_set_info, MPI_Comm comm, MPI_Info info)
AMPI_API_DEF(int, MPI_Comm_get_info, MPI_Comm comm, MPI_Info *info)
AMPI_API_DEF(int, MPI_Comm_call_errhandler, MPI_Comm comm, int errorcode)
AMPI_API_DEF(int, MPI_Comm_create_errhandler, MPI_Comm_errhandler_fn *function, MPI_Errhandler *errhandler)
AMPI_API_DEF(int, MPI_Comm_set_errhandler, MPI_Comm comm, MPI_Errhandler errhandler)
AMPI_API_DEF(int, MPI_Comm_get_errhandler, MPI_Comm comm, MPI_Errhandler *errhandler)
AMPI_API_DEF(int, MPI_Comm_free_errhandler, MPI_Errhandler *errhandler)
AMPI_API_DEF(int, MPI_Comm_create_keyval, MPI_Comm_copy_attr_function *copy_fn, MPI_Comm_delete_attr_function *delete_fn,
                            int *keyval, void* extra_state)
AMPI_API_DEF(int, MPI_Comm_free_keyval, int *keyval)
AMPI_API_DEF(int, MPI_Comm_set_attr, MPI_Comm comm, int keyval, void* attribute_val)
AMPI_API_DEF(int, MPI_Comm_get_attr, MPI_Comm comm, int keyval, void *attribute_val, int *flag)
AMPI_API_DEF(int, MPI_Comm_delete_attr, MPI_Comm comm, int keyval)

/***keyvals/attributes***/
AMPI_API_DEF(int, MPI_Keyval_create, MPI_Copy_function *copy_fn, MPI_Delete_function *delete_fn,
                       int *keyval, void* extra_state)
AMPI_API_DEF(int, MPI_Keyval_free, int *keyval)
AMPI_API_DEF(int, MPI_Attr_put, MPI_Comm comm, int keyval, void* attribute_val)
AMPI_API_DEF(int, MPI_Attr_get, MPI_Comm comm, int keyval, void *attribute_val, int *flag)
AMPI_API_DEF(int, MPI_Attr_delete, MPI_Comm comm, int keyval)

/***topologies***/
AMPI_API_DEF(int, MPI_Cart_create, MPI_Comm comm_old, int ndims, const int *dims,
                     const int *periods, int reorder, MPI_Comm *comm_cart)
AMPI_API_DEF(int, MPI_Graph_create, MPI_Comm comm_old, int nnodes, const int *index,
                      const int *edges, int reorder, MPI_Comm *comm_graph)
AMPI_API_DEF(int, MPI_Dist_graph_create_adjacent, MPI_Comm comm_old, int indegree, const int sources[],
                                    const int sourceweights[], int outdegree,
                                    const int destinations[], const int destweights[],
                                    MPI_Info info, int reorder, MPI_Comm *comm_dist_graph)
AMPI_API_DEF(int, MPI_Dist_graph_create, MPI_Comm comm_old, int n, const int sources[], const int degrees[],
                           const int destintations[], const int weights[], MPI_Info info,
                           int reorder, MPI_Comm *comm_dist_graph)
AMPI_API_DEF(int, MPI_Topo_test, MPI_Comm comm, int *status)
AMPI_API_DEF(int, MPI_Cart_map, MPI_Comm comm, int ndims, const int *dims, const int *periods,
                  int *newrank)
AMPI_API_DEF(int, MPI_Graph_map, MPI_Comm comm, int nnodes, const int *index, const int *edges,
                   int *newrank)
AMPI_API_DEF(int, MPI_Cartdim_get, MPI_Comm comm, int *ndims)
AMPI_API_DEF(int, MPI_Cart_get, MPI_Comm comm, int maxdims, int *dims, int *periods, int *coords)
AMPI_API_DEF(int, MPI_Cart_rank, MPI_Comm comm, const int *coords, int *rank)
AMPI_API_DEF(int, MPI_Cart_coords, MPI_Comm comm, int rank, int maxdims, int *coords)
AMPI_API_DEF(int, MPI_Cart_shift, MPI_Comm comm, int direction, int disp, int *rank_source, int *rank_dest)
AMPI_API_DEF(int, MPI_Graphdims_get, MPI_Comm comm, int *nnodes, int *nedges)
AMPI_API_DEF(int, MPI_Graph_get, MPI_Comm comm, int maxindex, int maxedges, int *index, int *edges)
AMPI_API_DEF(int, MPI_Graph_neighbors_count, MPI_Comm comm, int rank, int *nneighbors)
AMPI_API_DEF(int, MPI_Graph_neighbors, MPI_Comm comm, int rank, int maxneighbors, int *neighbors)
AMPI_API_DEF(int, MPI_Dims_create, int nnodes, int ndims, int *dims)
AMPI_API_DEF(int, MPI_Cart_sub, MPI_Comm comm, const int *remain_dims, MPI_Comm *newcomm)
AMPI_API_DEF(int, MPI_Dist_graph_neighbors, MPI_Comm comm, int maxindegree, int sources[], int sourceweights[],
                              int maxoutdegree, int destinations[], int destweights[])
AMPI_API_DEF(int, MPI_Dist_graph_neighbors_count, MPI_Comm comm, int *indegree, int *outdegree, int *weighted)

/***environment management***/
AMPI_API_DEF(int, MPI_Errhandler_create, MPI_Handler_function *function, MPI_Errhandler *errhandler)
AMPI_API_DEF(int, MPI_Errhandler_set, MPI_Comm comm, MPI_Errhandler errhandler)
AMPI_API_DEF(int, MPI_Errhandler_get, MPI_Comm comm, MPI_Errhandler *errhandler)
AMPI_API_DEF(int, MPI_Errhandler_free, MPI_Errhandler *errhandler)
AMPI_API_DEF(int, MPI_Add_error_code, int errorclass, int *errorcode)
AMPI_API_DEF(int, MPI_Add_error_class, int *errorclass)
AMPI_API_DEF(int, MPI_Add_error_string, int errorcode, const char *errorstring)
AMPI_API_DEF(int, MPI_Error_class, int errorcode, int *errorclass)
AMPI_API_DEF(int, MPI_Error_string, int errorcode, char *string, int *resultlen)
AMPI_API_DEF(int, MPI_Get_version, int *version, int *subversion)
AMPI_API_DEF(int, MPI_Get_library_version, char *version, int *resultlen)
AMPI_API_DEF(int, MPI_Get_processor_name, char *name, int *resultlen)
AMPI_API_DEF(double, MPI_Wtime, void)
AMPI_API_DEF(double, MPI_Wtick, void)
AMPI_API_DEF(int, MPI_Is_thread_main, int *flag)
AMPI_API_DEF(int, MPI_Query_thread, int *provided)
AMPI_API_DEF(int, MPI_Init_thread, int *argc, char*** argv, int required, int *provided)
AMPI_API_DEF(int, MPI_Init, int *argc, char*** argv)
AMPI_API_DEF(int, MPI_Initialized, int *isInit)
AMPI_API_DEF(int, MPI_Finalize, void)
AMPI_API_DEF(int, MPI_Finalized, int *finalized)
AMPI_API_DEF(int, MPI_Abort, MPI_Comm comm, int errorcode)
AMPI_API_DEF(int, MPI_Pcontrol, const int level, ...)

/*********************One sided communication routines *****************/
/*  MPI_Win : an index into a list in ampiParent (just like MPI_Group) */
/* name length for COMM, TYPE and WIN */
#define MPI_MAX_OBJECT_NAME  255
#define MPI_MAX_INFO_KEY     255
#define MPI_MAX_INFO_VAL    1024
#define MPI_LOCK_SHARED       54
#define MPI_LOCK_EXCLUSIVE    55
#define MPI_WIN_NULL          -1

#define MPI_WIN_FLAVOR_CREATE   1
#define MPI_WIN_FLAVOR_ALLOCATE 2
#define MPI_WIN_FLAVOR_DYNAMIC  3
#define MPI_WIN_FLAVOR_SHARED   4

#define MPI_WIN_UNIFIED  0
#define MPI_WIN_SEPARATE 1

#define MPI_MODE_NOCHECK    1
#define MPI_MODE_NOPRECEDE  2
#define MPI_MODE_NOPUT      4
#define MPI_MODE_NOSTORE    8
#define MPI_MODE_NOSUCCEED 16

typedef int MPI_Win;

typedef void (MPI_Win_errhandler_fn)(MPI_Win *, int *, ...);
typedef void (MPI_Win_errhandler_function)(MPI_Win *, int *, ...);

typedef int  (MPI_Win_copy_attr_function)(MPI_Win oldwin, int keyval,
                                          void *extra_state, void *attribute_val_in,
                                          void *attribute_val_out, int *flag);
typedef int  (MPI_Win_delete_attr_function)(MPI_Win win, int keyval,
                                            void *attribute_val, void *extra_state);

#define MPI_WIN_NULL_DELETE_FN MPI_win_null_delete_fn
#define MPI_WIN_NULL_COPY_FN   MPI_win_null_copy_fn
#define MPI_WIN_DUP_FN         MPI_win_dup_fn

int MPI_WIN_NULL_COPY_FN   ( MPI_Win, int, void *, void *, void *, int * );
int MPI_WIN_NULL_DELETE_FN ( MPI_Win, int, void *, void * );
int MPI_WIN_DUP_FN         ( MPI_Win, int, void *, void *, void *, int * );

/***windows/rma***/
AMPI_API_DEF(int, MPI_Win_create, void *base, MPI_Aint size, int disp_unit,
                    MPI_Info info, MPI_Comm comm, MPI_Win *newwin)
AMPI_API_DEF(int, MPI_Win_free, MPI_Win *win)
AMPI_API_DEF(int, MPI_Win_create_errhandler, MPI_Win_errhandler_function *win_errhandler_fn,
                               MPI_Errhandler *errhandler)
AMPI_API_DEF(int, MPI_Win_call_errhandler, MPI_Win win, int errorcode)
AMPI_API_DEF(int, MPI_Win_get_errhandler, MPI_Win win, MPI_Errhandler *errhandler)
AMPI_API_DEF(int, MPI_Win_set_errhandler, MPI_Win win, MPI_Errhandler errhandler)
AMPI_API_DEF(int, MPI_Win_create_keyval, MPI_Win_copy_attr_function *copy_fn,
                           MPI_Win_delete_attr_function *delete_fn,
                           int *keyval, void *extra_state)
AMPI_API_DEF(int, MPI_Win_free_keyval, int *keyval)
AMPI_API_DEF(int, MPI_Win_delete_attr, MPI_Win win, int key)
AMPI_API_DEF(int, MPI_Win_get_attr, MPI_Win win, int win_keyval, void *attribute_val, int *flag)
AMPI_API_DEF(int, MPI_Win_set_attr, MPI_Win win, int win_keyval, void *attribute_val)
AMPI_API_DEF(int, MPI_Win_get_group, MPI_Win win, MPI_Group *group)
AMPI_API_DEF(int, MPI_Win_set_name, MPI_Win win, const char *name)
AMPI_API_DEF(int, MPI_Win_get_name, MPI_Win win, char *name, int *length)
AMPI_API_DEF(int, MPI_Win_set_info, MPI_Win win, MPI_Info info)
AMPI_API_DEF(int, MPI_Win_get_info, MPI_Win win, MPI_Info *info)
AMPI_API_DEF(int, MPI_Win_fence, int assertion, MPI_Win win)
AMPI_API_DEF(int, MPI_Win_lock, int lock_type, int rank, int assert, MPI_Win win)
AMPI_API_DEF(int, MPI_Win_unlock, int rank, MPI_Win win)
AMPI_API_DEF(int, MPI_Win_post, MPI_Group group, int assertion, MPI_Win win)
AMPI_API_DEF(int, MPI_Win_wait, MPI_Win win)
AMPI_API_DEF(int, MPI_Win_start, MPI_Group group, int assertion, MPI_Win win)
AMPI_API_DEF(int, MPI_Win_complete, MPI_Win win)
AMPI_API_DEF(int, MPI_Win_test, MPI_Win win, int *flag)
AMPI_API_DEF(int, MPI_Alloc_mem, MPI_Aint size, MPI_Info info, void *baseptr)
AMPI_API_DEF(int, MPI_Free_mem, void *base)
AMPI_API_DEF(int, MPI_Put, const void *orgaddr, int orgcnt, MPI_Datatype orgtype, int rank,
             MPI_Aint targdisp, int targcnt, MPI_Datatype targtype, MPI_Win win)
AMPI_API_DEF(int, MPI_Get, void *orgaddr, int orgcnt, MPI_Datatype orgtype, int rank,
             MPI_Aint targdisp, int targcnt, MPI_Datatype targtype, MPI_Win win)
AMPI_API_DEF(int, MPI_Accumulate, const void *orgaddr, int orgcnt, MPI_Datatype orgtype, int rank,
                    MPI_Aint targdisp, int targcnt, MPI_Datatype targtype,
                    MPI_Op op, MPI_Win win)
AMPI_API_DEF(int, MPI_Get_accumulate, const void *orgaddr, int orgcnt, MPI_Datatype orgtype,
                        void *resaddr, int rescnt, MPI_Datatype restype,
                        int rank, MPI_Aint targdisp, int targcnt,
                        MPI_Datatype targtype, MPI_Op op, MPI_Win win)
AMPI_API_DEF(int, MPI_Rput, const void *orgaddr, int orgcnt, MPI_Datatype orgtype, int targrank,
             MPI_Aint targdisp, int targcnt, MPI_Datatype targtype, MPI_Win win,
             MPI_Request *request)
AMPI_API_DEF(int, MPI_Rget, void *orgaddr, int orgcnt, MPI_Datatype orgtype, int rank,
              MPI_Aint targdisp, int targcnt, MPI_Datatype targtype, MPI_Win win,
              MPI_Request *request)
AMPI_API_DEF(int, MPI_Raccumulate, const void *orgaddr, int orgcnt, MPI_Datatype orgtype, int rank,
                     MPI_Aint targdisp, int targcnt, MPI_Datatype targtype,
                     MPI_Op op, MPI_Win win, MPI_Request *request)
AMPI_API_DEF(int, MPI_Rget_accumulate, const void *orgaddr, int orgcnt, MPI_Datatype orgtype,
                         void *resaddr, int rescnt, MPI_Datatype restype,
                         int rank, MPI_Aint targdisp, int targcnt,
                         MPI_Datatype targtype, MPI_Op op, MPI_Win win,
                         MPI_Request *request)
AMPI_API_DEF(int, MPI_Fetch_and_op, const void *orgaddr, void *resaddr, MPI_Datatype type,
                      int rank, MPI_Aint targdisp, MPI_Op op, MPI_Win win)
AMPI_API_DEF(int, MPI_Compare_and_swap, const void *orgaddr, const void *compaddr, void *resaddr,
                          MPI_Datatype type, int rank, MPI_Aint targdisp,
                          MPI_Win win)

/***infos***/
AMPI_API_DEF(int, MPI_Info_create, MPI_Info *info)
AMPI_API_DEF(int, MPI_Info_set, MPI_Info info, const char *key, const char *value)
AMPI_API_DEF(int, MPI_Info_delete, MPI_Info info, const char *key)
AMPI_API_DEF(int, MPI_Info_get, MPI_Info info, const char *key, int valuelen, char *value, int *flag)
AMPI_API_DEF(int, MPI_Info_get_valuelen, MPI_Info info, const char *key, int *valuelen, int *flag)
AMPI_API_DEF(int, MPI_Info_get_nkeys, MPI_Info info, int *nkeys)
AMPI_API_DEF(int, MPI_Info_get_nthkey, MPI_Info info, int n, char *key)
AMPI_API_DEF(int, MPI_Info_dup, MPI_Info info, MPI_Info *newinfo)
AMPI_API_DEF(int, MPI_Info_free, MPI_Info *info)


/***MPIX***/
typedef int MPIX_Grequest_class;
typedef int MPIX_Grequest_poll_function(void *extra_state, MPI_Status *status);
typedef int MPIX_Grequest_wait_function(int count, void **array_of_states,
  double timeout, MPI_Status *status);

AMPI_API_DEF(int, MPIX_Grequest_start, MPI_Grequest_query_function *query_fn,
  MPI_Grequest_free_function *free_fn, MPI_Grequest_cancel_function *cancel_fn,
  MPIX_Grequest_poll_function *poll_fn, void *extra_state, MPI_Request *request)
AMPI_API_DEF(int, MPIX_Grequest_class_create, MPI_Grequest_query_function *query_fn,
  MPI_Grequest_free_function *free_fn, MPI_Grequest_cancel_function *cancel_fn,
  MPIX_Grequest_poll_function *poll_fn, MPIX_Grequest_wait_function *wait_fn,
  MPIX_Grequest_class *greq_class)
AMPI_API_DEF(int, MPIX_Grequest_class_allocate, MPIX_Grequest_class greq_class,
  void *extra_state, MPI_Request *request)


/***Fortran-C bindings***/
#define MPI_Comm_c2f(comm) (MPI_Fint)(comm)
#define MPI_Comm_f2c(comm) (MPI_Comm)(comm)
#define MPI_Type_c2f(datatype) (MPI_Fint)(datatype)
#define MPI_Type_f2c(datatype) (MPI_Datatype)(datatype)
#define MPI_Group_c2f(group) (MPI_Fint)(group)
#define MPI_Group_f2c(group) (MPI_Group)(group)
#define MPI_Info_c2f(info) (MPI_Fint)(info)
#define MPI_Info_f2c(info) (MPI_Info)(info)
#define MPI_Request_f2c(request) (MPI_Request)(request)
#define MPI_Request_c2f(request) (MPI_Fint)(request)
#define MPI_Op_c2f(op) (MPI_Fint)(op)
#define MPI_Op_f2c(op) (MPI_Op)(op)
#define MPI_Errhandler_c2f(errhandler) (MPI_Fint)(errhandler)
#define MPI_Errhandler_f2c(errhandler) (MPI_Errhandler)(errhandler)
#define MPI_Win_c2f(win) (MPI_Fint)(win)
#define MPI_Win_f2c(win) (MPI_Win)(win)

#include "mpio.h"

/*** AMPI Extensions ***/
int AMPI_Migrate(MPI_Info hints);
int AMPI_Load_start_measure(void);
int AMPI_Load_stop_measure(void);
int AMPI_Load_reset_measure(void);
int AMPI_Load_set_value(double value);
int AMPI_Migrate_to_pe(int dest);
int AMPI_Set_migratable(int mig);
int AMPI_Register_pup(MPI_PupFn fn, void *data, int *idx);
int AMPI_Get_pup_data(int idx, void *data);
int AMPI_Register_main(MPI_MainFn mainFn, const char *name);
int AMPI_Register_about_to_migrate(MPI_MigrateFn fn);
int AMPI_Register_just_migrated(MPI_MigrateFn fn);
int AMPI_Iget(MPI_Aint orgdisp, int orgcnt, MPI_Datatype orgtype, int rank,
              MPI_Aint targdisp, int targcnt, MPI_Datatype targtype,
              MPI_Win win, MPI_Request *request);
int AMPI_Iget_wait(MPI_Request *request, MPI_Status *status, MPI_Win win);
int AMPI_Iget_free(MPI_Request *request, MPI_Status *status, MPI_Win win);
int AMPI_Iget_data(void *data, MPI_Status status);
int AMPI_Type_is_contiguous(MPI_Datatype datatype, int *flag);
#if CMK_FAULT_EVAC
int AMPI_Evacuate(void);
#endif
int AMPI_Yield(void);
int AMPI_Suspend(void);
int AMPI_Resume(int dest, MPI_Comm comm);
int AMPI_Print(const char *str);
int AMPI_Trace_begin(void);
int AMPI_Trace_end(void);
int AMPI_Alltoall_medium(void *sendbuf, int sendcount, MPI_Datatype sendtype,
                         void *recvbuf, int recvcount, MPI_Datatype recvtype,
                         MPI_Comm comm);
int AMPI_Alltoall_long(void *sendbuf, int sendcount, MPI_Datatype sendtype,
                       void *recvbuf, int recvcount, MPI_Datatype recvtype,
                       MPI_Comm comm);

#if CMK_BIGSIM_CHARM
int AMPI_Set_start_event(MPI_Comm comm);
int AMPI_Set_end_event(void);
void beginTraceBigSim(char* msg);
void endTraceBigSim(char* msg, char* param);
#endif

#ifdef __cplusplus
#if CMK_CUDA
int AMPI_GPU_Iinvoke_wr(hapiWorkRequest *to_call, MPI_Request *request);
int AMPI_GPU_Iinvoke(cudaStream_t stream, MPI_Request *request);
int AMPI_GPU_Invoke_wr(hapiWorkRequest *to_call);
int AMPI_GPU_Invoke(cudaStream_t stream);
#endif
#endif

/* Execute this shell command (just like "system()") */
int AMPI_System(const char *cmd);

/* Determine approximate depth of stack at the point of this call */
extern long ampiCurrentStackUsage(void);

// Functions and constants unsupported in AMPI

#if defined __cplusplus && __cplusplus >= 201402L
# define AMPI_UNIMPLEMENTED [[deprecated("currently unimplemented in AMPI")]]
#elif defined __GNUC__ || defined __clang__
# define AMPI_UNIMPLEMENTED __attribute__((deprecated("currently unimplemented in AMPI")))
#elif defined _MSC_VER
# define AMPI_UNIMPLEMENTED __declspec(deprecated("currently unimplemented in AMPI"))
#else
# define AMPI_UNIMPLEMENTED
#endif

#define AMPI_API_DEF_NOIMPL(return_type, function_name, ...) \
  AMPI_UNIMPLEMENTED AMPI_API_DEF(return_type, function_name, __VA_ARGS__)

// MPI-2 Constants
#define MPI_ARGV_NULL (char **)0
#define MPI_ARGVS_NULL (char ***)0
#define MPI_MAX_PORT_NAME 256
#define MPI_ORDER_C 56 /* same as in ROMIO */
#define MPI_ORDER_FORTRAN 57 /* same as in ROMIO */
#define MPI_TYPECLASS_INTEGER -1
#define MPI_TYPECLASS_REAL -2
#define MPI_TYPECLASS_COMPLEX -3
#define MPI_DISTRIBUTE_BLOCK 121  /* same as in ROMIO */
#define MPI_DISTRIBUTE_CYCLIC 122  /* same as in ROMIO */
#define MPI_DISTRIBUTE_NONE 123 /* same as in ROMIO */
#define MPI_DISTRIBUTE_DFLT_DARG -49767 /* same as in ROMIO */
#define MPI_INTEGER1 MPI_CHAR
#define MPI_INTEGER2 MPI_SHORT
#define MPI_INTEGER4 MPI_INT
#define MPI_INTEGER8 MPI_LONG_LONG_INT
#define MPI_REAL4 MPI_FLOAT
#define MPI_REAL8 MPI_DOUBLE
#define MPI_REAL16 MPI_LONG_DOUBLE
#define MPI_COMPLEX8 MPI_FLOAT_COMPLEX
#define MPI_COMPLEX16 MPI_DOUBLE_COMPLEX
#define MPI_COMPLEX32 MPI_LONG_DOUBLE_COMPLEX
#define MPI_C_FLOAT_COMPLEX MPI_FLOAT_COMPLEX
#define MPI_C_DOUBLE_COMPLEX MPI_DOUBLE_COMPLEX
#define MPI_C_LONG_DOUBLE_COMPLEX MPI_LONG_DOUBLE_COMPLEX
#define MPI_CXX_BOOL MPI_C_BOOL
#define MPI_CXX_FLOAT_COMPLEX MPI_C_FLOAT_COMPLEX
#define MPI_CXX_DOUBLE_COMPLEX MPI_C_DOUBLE_COMPLEX
#define MPI_CXX_LONG_DOUBLE_COMPLEX MPI_C_LONG_DOUBLE_COMPLEX

// MPI-2 Routines
#define MPI_Win_free_errhandler (void*)

// MPI-3 Constants
#define MPI_ERR_RMA_RANGE 54
#define MPI_ERR_RMA_ATTACH 55
#define MPI_ERR_RMA_SHARED 56
#define MPI_ERR_RMA_FLAVOR 57

// MPI_T interface
typedef int MPI_T_enum;
typedef int MPI_T_cvar_handle;
typedef int MPI_T_pvar_handle;
typedef int MPI_T_pvar_session;
#define MPI_T_ENUM_NULL (-1)
#define MPI_T_CVAR_HANDLE_NULL (-1)
#define MPI_T_PVAR_HANDLE_NULL (-1)
#define MPI_T_PVAR_SESSION_NULL (-1)
#define MPI_T_VERBOSITY_USER_BASIC 1
#define MPI_T_VERBOSITY_USER_DETAIL 2
#define MPI_T_VERBOSITY_USER_ALL 3
#define MPI_T_VERBOSITY_TUNER_BASIC 4
#define MPI_T_VERBOSITY_TUNER_DETAIL 5
#define MPI_T_VERBOSITY_TUNER_ALL 6
#define MPI_T_VERBOSITY_MPIDEV_BASIC 7
#define MPI_T_VERBOSITY_MPIDEV_DETAIL 8
#define MPI_T_VERBOSITY_MPIDEV_ALL 9
#define MPI_T_BIND_NO_OBJECT 1
#define MPI_T_BIND_MPI_COMM 2
#define MPI_T_BIND_MPI_DATATYPE 3
#define MPI_T_BIND_MPI_ERRHANDLER 4
#define MPI_T_BIND_MPI_FILE 5
#define MPI_T_BIND_MPI_GROUP 6
#define MPI_T_BIND_MPI_OP 7
#define MPI_T_BIND_MPI_REQUEST 8
#define MPI_T_BIND_MPI_WIN 9
#define MPI_T_BIND_MPI_MESSAGE 10
#define MPI_T_BIND_MPI_INFO 11
#define MPI_T_SCOPE_CONSTANT 1
#define MPI_T_SCOPE_READONLY 2
#define MPI_T_SCOPE_LOCAL 3
#define MPI_T_SCOPE_GROUP 4
#define MPI_T_SCOPE_GROUP_EQ 5
#define MPI_T_SCOPE_ALL 6
#define MPI_T_SCOPE_ALL_EQ 7
#define MPI_T_PVAR_ALL_HANDLES (-1)
#define MPI_T_PVAR_CLASS_STATE 1
#define MPI_T_PVAR_CLASS_LEVEL 2
#define MPI_T_PVAR_CLASS_SIZE 3
#define MPI_T_PVAR_CLASS_PERCENTAGE 4
#define MPI_T_PVAR_CLASS_HIGHWATERMARK 5
#define MPI_T_PVAR_CLASS_LOWWATERMARK 6
#define MPI_T_PVAR_CLASS_COUNTER 7
#define MPI_T_PVAR_CLASS_AGGREGATE 8
#define MPI_T_PVAR_CLASS_TIMER 9
#define MPI_T_PVAR_CLASS_GENERIC 10
#define MPI_T_ERR_MEMORY 58
#define MPI_T_ERR_NOT_INITIALIZED 59
#define MPI_T_ERR_CANNOT_INIT 60
#define MPI_T_ERR_INVALID_INDEX 61
#define MPI_T_ERR_INVALID_ITEM 62
#define MPI_T_ERR_INVALID_NAME 63
#define MPI_T_ERR_INVALID_HANDLE 64
#define MPI_T_ERR_OUT_OF_HANDLES 65
#define MPI_T_ERR_OUT_OF_SESSIONS 66
#define MPI_T_ERR_INVALID_SESSION 67
#define MPI_T_ERR_CVAR_SET_NOT_NOW 68
#define MPI_T_ERR_CVAR_SET_NEVER 69
#define MPI_T_ERR_CVAR_READ 70
#define MPI_T_ERR_CVAR_WRITE 71
#define MPI_T_ERR_PVAR_START 72
#define MPI_T_ERR_PVAR_STOP 73
#define MPI_T_ERR_PVAR_READ 74
#define MPI_T_ERR_PVAR_WRITE 75
#define MPI_T_ERR_PVAR_RESET 76
#define MPI_T_ERR_PVAR_READRESET 77
#define MPI_T_ERR_PVAR_NO_STARTSTOP 78
#define MPI_T_ERR_PVAR_NO_WRITE 79
#define MPI_T_ERR_PVAR_NO_ATOMIC 80

// MPIX FT interface (MPICH extensions needed to compile tests/ampi/mpich-test/)
#define MPIX_ERR_PROC_FAILED 77
#define MPIX_ERR_PROC_FAILED_PENDING 78
#define MPIX_ERR_REVOKED 79
#define MPIX_Comm_agree (void*)
#define MPIX_Comm_failure_ack (void*)
#define MPIX_Comm_shrink (void*)
#define MPIX_Comm_failure_get_acked (void*)
#define MPIX_Comm_revoke (void*)


/* MPI 3.1 standards compliance overview.
 * This list contains all MPI functions not supported in AMPI currently.
*/

/* A.2.1 Point-to-Point Communication C Bindings */

/* A.2.2 Datatypes C Bindings */

AMPI_API_DEF_NOIMPL(int, MPI_Pack_external, const char datarep[], const void *inbuf, int incount, MPI_Datatype datatype, void *outbuf, MPI_Aint outsize, MPI_Aint *position)
AMPI_API_DEF_NOIMPL(int, MPI_Pack_external_size, const char datarep[], int incount, MPI_Datatype datatype, MPI_Aint *size)
AMPI_API_DEF_NOIMPL(int, MPI_Unpack_external, const char datarep[], const void *inbuf, MPI_Aint insize, MPI_Aint *position, void *outbuf, int outcount, MPI_Datatype datatype)

/* A.2.3 Collective Communication C Bindings */

/* A.2.4 Groups, Contexts, Communicators, and Caching C Bindings */

/* A.2.6 MPI Environmental Management C Bindings */

typedef void (MPI_File_errhandler_function)(MPI_File *file, int *err, ...);

AMPI_API_DEF_NOIMPL(int, MPI_File_call_errhandler, MPI_File fh, int errorcode)
AMPI_API_DEF_NOIMPL(int, MPI_File_create_errhandler, MPI_File_errhandler_function *file_errhandler_fn, MPI_Errhandler *errhandler)
AMPI_API_DEF_NOIMPL(int, MPI_File_get_errhandler, MPI_File file, MPI_Errhandler *errhandler)
AMPI_API_DEF_NOIMPL(int, MPI_File_set_errhandler, MPI_File file, MPI_Errhandler errhandler)


/* A.2.7 The Info Object C Bindings */

/* A.2.8 Process Creation and Management C Bindings */

AMPI_API_DEF_NOIMPL(int, MPI_Close_port, const char *port_name)
AMPI_API_DEF_NOIMPL(int, MPI_Comm_accept, const char *port_name, MPI_Info info, int root, MPI_Comm comm, MPI_Comm *newcomm)
AMPI_API_DEF_NOIMPL(int, MPI_Comm_connect, const char *port_name, MPI_Info info, int root, MPI_Comm comm, MPI_Comm *newcomm)
AMPI_API_DEF_NOIMPL(int, MPI_Comm_disconnect, MPI_Comm *comm)
AMPI_API_DEF_NOIMPL(int, MPI_Comm_get_parent, MPI_Comm *parent)
AMPI_API_DEF_NOIMPL(int, MPI_Comm_join, int fd, MPI_Comm *intercomm)
AMPI_API_DEF_NOIMPL(int, MPI_Comm_spawn, const char *command, char *argv[], int maxprocs, MPI_Info info, int root, MPI_Comm comm, MPI_Comm *intercomm, int array_of_errcodes[])
AMPI_API_DEF_NOIMPL(int, MPI_Comm_spawn_multiple, int count, char *array_of_commands[], char **array_of_argv[], const int array_of_maxprocs[], const MPI_Info array_of_info[], int root, MPI_Comm comm, MPI_Comm *intercomm, int array_of_errcodes[])
AMPI_API_DEF_NOIMPL(int, MPI_Lookup_name, const char *service_name, MPI_Info info, char *port_name)
AMPI_API_DEF_NOIMPL(int, MPI_Open_port, MPI_Info info, char *port_name)
AMPI_API_DEF_NOIMPL(int, MPI_Publish_name, const char *service_name, MPI_Info info, const char *port_name)
AMPI_API_DEF_NOIMPL(int, MPI_Unpublish_name, const char *service_name, MPI_Info info, const char *port_name)


/* A.2.9 One-Sided Communications C Bindings */

AMPI_API_DEF_NOIMPL(int, MPI_Win_allocate, MPI_Aint size, int disp_unit, MPI_Info info, MPI_Comm comm, void *baseptr, MPI_Win *win)
AMPI_API_DEF_NOIMPL(int, MPI_Win_allocate_shared, MPI_Aint size, int disp_unit, MPI_Info info, MPI_Comm comm, void *baseptr, MPI_Win *win)
AMPI_API_DEF_NOIMPL(int, MPI_Win_attach, MPI_Win win, void *base, MPI_Aint size)
AMPI_API_DEF_NOIMPL(int, MPI_Win_create_dynamic, MPI_Info info, MPI_Comm comm, MPI_Win *win)
AMPI_API_DEF_NOIMPL(int, MPI_Win_detach, MPI_Win win, const void *base)
AMPI_API_DEF_NOIMPL(int, MPI_Win_flush, int rank, MPI_Win win)
AMPI_API_DEF_NOIMPL(int, MPI_Win_flush_all, MPI_Win win)
AMPI_API_DEF_NOIMPL(int, MPI_Win_flush_local, int rank, MPI_Win win)
AMPI_API_DEF_NOIMPL(int, MPI_Win_flush_local_all, MPI_Win win)
AMPI_API_DEF_NOIMPL(int, MPI_Win_lock_all, int assert, MPI_Win win)
AMPI_API_DEF_NOIMPL(int, MPI_Win_shared_query, MPI_Win win, int rank, MPI_Aint *size, int *disp_unit, void *baseptr)
AMPI_API_DEF_NOIMPL(int, MPI_Win_sync, MPI_Win win)
AMPI_API_DEF_NOIMPL(int, MPI_Win_unlock_all, MPI_Win win)


/* A.2.10 External Interfaces C Bindings */

/* A.2.11 I/O C Bindings */

AMPI_API_DEF_NOIMPL(int, MPI_CONVERSION_FN_NULL, void *userbuf, MPI_Datatype datatype, int count, void *filebuf, MPI_Offset position, void *extra_state)
AMPI_API_DEF_NOIMPL(int, MPI_File_iread_all, MPI_File fh, void *buf, int count, MPI_Datatype datatype, MPI_Request *request)
AMPI_API_DEF_NOIMPL(int, MPI_File_iread_at_all, MPI_File fh, MPI_Offset offset, void *buf, int count, MPI_Datatype datatype, MPI_Request *request)
AMPI_API_DEF_NOIMPL(int, MPI_File_iwrite_all, MPI_File fh, const void *buf, int count, MPI_Datatype datatype, MPI_Request *request)
AMPI_API_DEF_NOIMPL(int, MPI_File_iwrite_at_all, MPI_File fh, MPI_Offset offset, const void *buf, int count, MPI_Datatype datatype, MPI_Request *request)
// AMPI_API_DEF_NOIMPL(int, MPI_Register_datarep, const char *datarep, MPI_Datarep_conversion_function *read_conversion_fn, MPI_Datarep_conversion_function *write_conversion_fn, MPI_Datarep_extent_function *dtype_file_extent_fn, void *extra_state) //Provided by ROMIO


/* A.2.12 Language Bindings C Bindings */

typedef struct {
  MPI_Fint count_lo, count_hi_and_cancelled, MPI_SOURCE, MPI_TAG, MPI_ERROR;
} MPI_F08_status;

AMPI_API_DEF_NOIMPL(int, MPI_Status_f082f, MPI_F08_status *f08_status, MPI_Fint *f_status)
AMPI_API_DEF_NOIMPL(int, MPI_Status_f2f08, MPI_Fint *f_status, MPI_F08_status *f08_status)
AMPI_API_DEF_NOIMPL(int, MPI_Type_create_f90_complex, int p, int r, MPI_Datatype *newtype)
AMPI_API_DEF_NOIMPL(int, MPI_Type_create_f90_integer, int r, MPI_Datatype *newtype)
AMPI_API_DEF_NOIMPL(int, MPI_Type_create_f90_real, int p, int r, MPI_Datatype *newtype)
AMPI_API_DEF_NOIMPL(int, MPI_Type_match_size, int typeclass, int size, MPI_Datatype *datatype)
AMPI_API_DEF_NOIMPL(MPI_Fint, MPI_Message_c2f, MPI_Message message)
AMPI_API_DEF_NOIMPL(MPI_Message, MPI_Message_f2c, MPI_Fint message)
AMPI_API_DEF_NOIMPL(int, MPI_Status_c2f, const MPI_Status *c_status, MPI_Fint *f_status)
AMPI_API_DEF_NOIMPL(int, MPI_Status_c2f08, const MPI_Status *c_status, MPI_F08_status *f08_status)
AMPI_API_DEF_NOIMPL(int, MPI_Status_f082c, const MPI_F08_status *f08_status, MPI_Status *c_status)
AMPI_API_DEF_NOIMPL(int, MPI_Status_f2c, const MPI_Fint *f_status, MPI_Status *c_status)


/* A.2.13 Tools / Profiling Interface C Bindings */

/* A.2.14 Tools / MPI Tool Information Interface C Bindings */

AMPI_API_DEF_NOIMPL(int, MPI_T_category_changed, int *stamp)
AMPI_API_DEF_NOIMPL(int, MPI_T_category_get_categories, int cat_index, int len, int indices[])
AMPI_API_DEF_NOIMPL(int, MPI_T_category_get_cvars, int cat_index, int len, int indices[])
AMPI_API_DEF_NOIMPL(int, MPI_T_category_get_index, const char *name, int *cat_index)
AMPI_API_DEF_NOIMPL(int, MPI_T_category_get_info, int cat_index, char *name, int *name_len, char *desc, int *desc_len, int *num_cvars, int *num_pvars, int *num_categories)
AMPI_API_DEF_NOIMPL(int, MPI_T_category_get_num, int *num_cat)
AMPI_API_DEF_NOIMPL(int, MPI_T_category_get_pvars, int cat_index, int len, int indices[])
AMPI_API_DEF_NOIMPL(int, MPI_T_cvar_get_index, const char *name, int *cvar_index)
AMPI_API_DEF_NOIMPL(int, MPI_T_cvar_get_info, int cvar_index, char *name, int *name_len, int *verbosity, MPI_Datatype *datatype, MPI_T_enum *enumtype, char *desc, int *desc_len, int *bind, int *scope)
AMPI_API_DEF_NOIMPL(int, MPI_T_cvar_get_num, int *num_cvar)
AMPI_API_DEF_NOIMPL(int, MPI_T_cvar_handle_alloc, int cvar_index, void *obj_handle, MPI_T_cvar_handle *handle, int *count)
AMPI_API_DEF_NOIMPL(int, MPI_T_cvar_handle_free, MPI_T_cvar_handle *handle)
AMPI_API_DEF_NOIMPL(int, MPI_T_cvar_read, MPI_T_cvar_handle handle, void* buf)
AMPI_API_DEF_NOIMPL(int, MPI_T_cvar_write, MPI_T_cvar_handle handle, const void* buf)
AMPI_API_DEF_NOIMPL(int, MPI_T_enum_get_info, MPI_T_enum enumtype, int *num, char *name, int *name_len)
AMPI_API_DEF_NOIMPL(int, MPI_T_enum_get_item, MPI_T_enum enumtype, int index, int *value, char *name, int *name_len)
AMPI_API_DEF_NOIMPL(int, MPI_T_finalize, void)
AMPI_API_DEF_NOIMPL(int, MPI_T_init_thread, int required, int *provided)
AMPI_API_DEF_NOIMPL(int, MPI_T_pvar_get_index, const char *name, int var_class, int *pvar_index)
AMPI_API_DEF_NOIMPL(int, MPI_T_pvar_get_info, int pvar_index, char *name, int *name_len, int *verbosity, int *var_class, MPI_Datatype *datatype, MPI_T_enum *enumtype, char *desc, int *desc_len, int *bind, int *readonly, int *continuous, int *atomic)
AMPI_API_DEF_NOIMPL(int, MPI_T_pvar_get_num, int *num_pvar)
AMPI_API_DEF_NOIMPL(int, MPI_T_pvar_handle_alloc, MPI_T_pvar_session session, int pvar_index, void *obj_handle, MPI_T_pvar_handle *handle, int *count)
AMPI_API_DEF_NOIMPL(int, MPI_T_pvar_handle_free, MPI_T_pvar_session session,MPI_T_pvar_handle *handle)
AMPI_API_DEF_NOIMPL(int, MPI_T_pvar_read, MPI_T_pvar_session session, MPI_T_pvar_handle handle,void* buf)
AMPI_API_DEF_NOIMPL(int, MPI_T_pvar_readreset, MPI_T_pvar_session session,MPI_T_pvar_handle handle, void* buf)
AMPI_API_DEF_NOIMPL(int, MPI_T_pvar_reset, MPI_T_pvar_session session, MPI_T_pvar_handle handle)
AMPI_API_DEF_NOIMPL(int, MPI_T_pvar_session_create, MPI_T_pvar_session *session)
AMPI_API_DEF_NOIMPL(int, MPI_T_pvar_session_free, MPI_T_pvar_session *session)
AMPI_API_DEF_NOIMPL(int, MPI_T_pvar_start, MPI_T_pvar_session session, MPI_T_pvar_handle handle)
AMPI_API_DEF_NOIMPL(int, MPI_T_pvar_stop, MPI_T_pvar_session session, MPI_T_pvar_handle handle)
AMPI_API_DEF_NOIMPL(int, MPI_T_pvar_write, MPI_T_pvar_session session, MPI_T_pvar_handle handle, const void* buf)


/* A.2.15 Deprecated C Bindings */

#ifdef __cplusplus
}
#endif

#endif

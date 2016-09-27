#ifndef _MPI_H
#define _MPI_H

#include <inttypes.h> /* for intptr_t */
#include "conv-config.h"

/* NON-standard define: this lets people #ifdef on
   AMPI, e.g. for our bizarre AMPI_Main. */
#define AMPI

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
/* 0=MPI_SUCCESS<MPI_ERRs(...)<MPI_ERR<=MPI_ERR_LASTCODE<=MPI_LASTCODEUSED */

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
#define MPI_COMPLEX              3
#define MPI_LOGICAL              4
#define MPI_C_BOOL               5
#define MPI_CHAR                 6
#define MPI_BYTE                 7
#define MPI_PACKED               8
#define MPI_SHORT                9
#define MPI_LONG                10
#define MPI_UNSIGNED_CHAR       11
#define MPI_UNSIGNED_SHORT      12
#define MPI_UNSIGNED            13
#define MPI_UNSIGNED_LONG       14
#define MPI_LONG_DOUBLE         15
/* tuple types */
#define MPI_FLOAT_INT           16
#define MPI_DOUBLE_INT          17
#define MPI_LONG_INT            18
#define MPI_2INT                19
#define MPI_SHORT_INT           20
#define MPI_LONG_DOUBLE_INT     21
#define MPI_2FLOAT              22
#define MPI_2DOUBLE             23
/* special types */
#define MPI_LB                  24
#define MPI_UB                  25
/* mpi-2+ types */
#define MPI_LONG_LONG_INT       26
#define MPI_LONG_LONG           MPI_LONG_LONG_INT
#define MPI_OFFSET              MPI_LONG_LONG
#define MPI_DOUBLE_COMPLEX      27
#define MPI_SIGNED_CHAR         28
#define MPI_UNSIGNED_LONG_LONG  29
#define MPI_WCHAR               30
#define MPI_INT8_T              31
#define MPI_INT16_T             32
#define MPI_INT32_T             33
#define MPI_INT64_T             34
#define MPI_UINT8_T             35
#define MPI_UINT16_T            36
#define MPI_UINT32_T            37
#define MPI_UINT64_T            38
#define MPI_FLOAT_COMPLEX       39
#define MPI_LONG_DOUBLE_COMPLEX 40
#define MPI_AINT                41

#define MPI_ANY_TAG        (-1)
#define MPI_REQUEST_NULL   (-1)
#define MPI_GROUP_NULL     (-1)
#define MPI_GROUP_EMPTY       0
#define MPI_COMM_NULL      (-1)
#define MPI_TYPE_NULL      (-1)
#define MPI_PROC_NULL      (-2)
#define MPI_ANY_SOURCE     (-1)
#define MPI_KEYVAL_INVALID (-1)
#define MPI_INFO_NULL      (-1)

#define MPI_IN_PLACE    ((void *) -1L)

#define MPI_BOTTOM      0
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

#define MPI_GRAPH 1
#define MPI_CART  2

/* This is one less than the system-tags defined in ampiimpl.h.
 * This is so that the tags used by the system dont clash with user-tags.
 * MPI standard requires this to be at least 2^15.
 */
#define MPI_TAG_UB_VALUE  1073741824

/* These are the builtin MPI keyvals, plus some AMPI specific ones. */
#define MPI_TAG_UB             -1
#define MPI_HOST               -2
#define MPI_IO                 -3
#define MPI_WTIME_IS_GLOBAL    -4
#define MPI_APPNUM             -5
#define MPI_UNIVERSE_SIZE      -6
#define MPI_WIN_BASE           -7
#define MPI_WIN_SIZE           -8
#define MPI_WIN_DISP_UNIT      -9
#define MPI_WIN_MODEL         -10
#define MPI_WIN_CREATE_FLAVOR -11
#define AMPI_MY_PE            -12
#define AMPI_NUM_PES          -13
#define AMPI_MY_NODE          -14
#define AMPI_NUM_NODES        -15

/** Communicators give a communication context to a set of processors.
    An intercommunicator can be used for point to point messaging between two groups.
    An intracommunicator can be used to send messages within a single group. */
typedef int MPI_Comm;

/** Groups represent an set of processors 0...n-1. They can be created locally */
typedef int MPI_Group;

typedef int MPI_Info;

#define MPI_COMM_FIRST_SPLIT (MPI_Comm)(1000000) /*Communicator from MPI_Comm_split */
#define MPI_COMM_FIRST_GROUP (MPI_Comm)(2000000) /*Communicator from MPI_Comm_group */
#define MPI_COMM_FIRST_CART  (MPI_Comm)(3000000) /*Communicator from MPI_Cart_create */
#define MPI_COMM_FIRST_GRAPH (MPI_Comm)(4000000) /*Communicator from MPI_Graph_create */
#define MPI_COMM_FIRST_INTER (MPI_Comm)(5000000) /*Communicator from MPI_Intercomm_create*/
#define MPI_COMM_FIRST_INTRA (MPI_Comm)(6000000) /*Communicator from MPI_Intercomm_merge*/
#define MPI_COMM_FIRST_RESVD (MPI_Comm)(7000000) /*Communicator reserved for now*/
#define MPI_COMM_SELF        (MPI_Comm)(8000000)
#define MPI_COMM_WORLD       (MPI_Comm)(9000000) /*Start of universe*/
#define MPI_MAX_COMM_WORLDS  8
extern MPI_Comm MPI_COMM_UNIVERSE[MPI_MAX_COMM_WORLDS];

/* the size of MPI_Status must conform to MPI_STATUS_SIZE in ampif.h */
struct AmpiMsg;
typedef int MPI_Request;
typedef struct {
  int MPI_TAG, MPI_SOURCE, MPI_COMM, MPI_LENGTH, MPI_ERROR; /* FIXME: MPI_ERROR is never used */
  struct AmpiMsg *msg;
} MPI_Status;

#define stsempty(sts)       (sts).MPI_TAG=(sts).MPI_SOURCE=(sts).MPI_COMM=(sts).MPI_LENGTH=0
#define MPI_STATUS_IGNORE   (MPI_Status *)0
#define MPI_STATUSES_IGNORE (MPI_Status *)0

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

#define MPI_TYPE_NULL_DELETE_FN MPI_type_null_delete_fn
#define MPI_TYPE_NULL_COPY_FN   MPI_type_null_copy_fn
#define MPI_TYPE_DUP_FN         MPI_type_dup_fn

int MPI_TYPE_NULL_COPY_FN   ( MPI_Datatype, int, void *, void *, void *, int * );
int MPI_TYPE_NULL_DELETE_FN ( MPI_Datatype, int, void *, void * );
int MPI_TYPE_DUP_FN         ( MPI_Datatype, int, void *, void *, void *, int * );

#include "pup_c.h"

typedef void (*MPI_PupFn)(pup_er, void*);
typedef void (*MPI_MigrateFn)(void);

/********************** MPI-2 prototypes and defines ***************************/
/* for the datatype decoders */
#define MPI_COMBINER_NAMED           1
#define MPI_COMBINER_CONTIGUOUS      2
#define MPI_COMBINER_VECTOR          3
#define MPI_COMBINER_HVECTOR         4
#define MPI_COMBINER_INDEXED         5
#define MPI_COMBINER_HINDEXED        6
#define MPI_COMBINER_STRUCT          7
#define MPI_COMBINER_DARRAY          8
#define MPI_COMBINER_RESIZED         9
#define MPI_COMBINER_SUBARRAY       10
#define MPI_COMBINER_INDEXED_BLOCK  11
#define MPI_COMBINER_HINDEXED_BLOCK 12

/********************** MPI-1.1 Functions ***************************/
/***pt2pt***/
#define MPI_BSEND_OVERHEAD 0
#define MPI_Send AMPI_Send
int AMPI_Send(void *msg, int count, MPI_Datatype type, int dest,
              int tag, MPI_Comm comm);
#define MPI_Ssend AMPI_Ssend
int AMPI_Ssend(void *msg, int count, MPI_Datatype type, int dest,
               int tag, MPI_Comm comm);
#define MPI_Recv AMPI_Recv
int AMPI_Recv(void *msg, int count, MPI_Datatype type, int src, int tag,
              MPI_Comm comm, MPI_Status *status);
#define MPI_Get_count AMPI_Get_count
int AMPI_Get_count(MPI_Status *sts, MPI_Datatype dtype, int *count);
#define AMPI_Bsend AMPI_Send
#define AMPI_Rsend AMPI_Send
#define MPI_Bsend AMPI_Bsend
#define MPI_Rsend AMPI_Rsend /* FIXME: MPI_Rsend can be posted only after recv */
#define MPI_Buffer_attach(buf,len) /*LIE: emtpy*/ /*Silly: default send is buffering in Charm++*/
#define MPI_Buffer_detach(buf,len) /*LIE: emtpy*/
#define MPI_Isend AMPI_Isend
int AMPI_Isend(void *buf, int count, MPI_Datatype datatype, int dest,
               int tag, MPI_Comm comm, MPI_Request *request);
#define MPI_Ibsend AMPI_Isend
#define MPI_Issend AMPI_Issend
int AMPI_Issend(void *buf, int count, MPI_Datatype datatype, int dest,
                int tag, MPI_Comm comm, MPI_Request *request);
#define MPI_Irsend AMPI_Isend /* FIXME: see MPI_Rsend */
#define MPI_Irecv AMPI_Irecv
int AMPI_Irecv(void *buf, int count, MPI_Datatype datatype, int src,
               int tag, MPI_Comm comm, MPI_Request *request);
#define MPI_Wait AMPI_Wait
int AMPI_Wait(MPI_Request *request, MPI_Status *sts);
#define MPI_Test AMPI_Test
int AMPI_Test(MPI_Request *request, int *flag, MPI_Status *sts);
#define MPI_Waitany AMPI_Waitany
int AMPI_Waitany(int count, MPI_Request *request, int *index, MPI_Status *sts);
#define MPI_Testany AMPI_Testany
int AMPI_Testany(int count, MPI_Request *request, int *index, int *flag, MPI_Status *status);
#define MPI_Waitall AMPI_Waitall
int AMPI_Waitall(int count, MPI_Request *request, MPI_Status *sts);
#define MPI_Testall AMPI_Testall
int AMPI_Testall(int count, MPI_Request *request, int *flag, MPI_Status *sts);
#define MPI_Waitsome AMPI_Waitsome
int AMPI_Waitsome(int incount, MPI_Request *array_of_requests, int *outcount,
                  int *array_of_indices, MPI_Status *array_of_statuses);
#define MPI_Testsome AMPI_Testsome
int AMPI_Testsome(int incount, MPI_Request *array_of_requests, int *outcount,
                  int *array_of_indices, MPI_Status *array_of_statuses);
#define MPI_Request_get_status AMPI_Request_get_status
int AMPI_Request_get_status(MPI_Request request, int *flag, MPI_Status *sts);
#define MPI_Request_free AMPI_Request_free
int AMPI_Request_free(MPI_Request *request);
#define MPI_Cancel AMPI_Cancel
int AMPI_Cancel(MPI_Request *request);
#define MPI_Test_cancelled AMPI_Test_cancelled
int AMPI_Test_cancelled(MPI_Status *status, int *flag); /* FIXME: always returns success */
#define MPI_Status_set_cancelled AMPI_Status_set_cancelled
int AMPI_Status_set_cancelled(MPI_Status *status, int flag);
#define MPI_Iprobe AMPI_Iprobe
int AMPI_Iprobe(int src, int tag, MPI_Comm comm, int *flag, MPI_Status *sts);
#define MPI_Probe AMPI_Probe
int AMPI_Probe(int source, int tag, MPI_Comm comm, MPI_Status *sts);
#define MPI_Send_init AMPI_Send_init
int AMPI_Send_init(void *buf, int count, MPI_Datatype type, int dest, int tag,
                  MPI_Comm comm, MPI_Request *req);
#define MPI_Ssend_init AMPI_Ssend_init
int AMPI_Ssend_init(void *buf, int count, MPI_Datatype type, int dest, int tag,
                    MPI_Comm comm, MPI_Request *req);
#define MPI_Bsend_init AMPI_Send_init
#define MPI_Rsend_init AMPI_Send_init /* FIXME: see MPI_Rsend */
#define MPI_Recv_init AMPI_Recv_init
int AMPI_Recv_init(void *buf, int count, MPI_Datatype type, int src, int tag,
                   MPI_Comm comm, MPI_Request *req);
#define MPI_Start AMPI_Start
int AMPI_Start(MPI_Request *reqnum);
#define MPI_Startall AMPI_Startall
int AMPI_Startall(int count, MPI_Request *array_of_requests);
#define MPI_Sendrecv AMPI_Sendrecv
int AMPI_Sendrecv(void *sbuf, int scount, MPI_Datatype stype, int dest,
                  int stag, void *rbuf, int rcount, MPI_Datatype rtype,
                  int src, int rtag, MPI_Comm comm, MPI_Status *sts);
#define MPI_Sendrecv_replace AMPI_Sendrecv_replace
int AMPI_Sendrecv_replace(void* buf, int count, MPI_Datatype datatype,
                          int dest, int sendtag, int source, int recvtag,
                          MPI_Comm comm, MPI_Status *status);
#define MPI_Type_contiguous AMPI_Type_contiguous
int AMPI_Type_contiguous(int count, MPI_Datatype oldtype,
                         MPI_Datatype *newtype);
#define MPI_Type_vector AMPI_Type_vector
int AMPI_Type_vector(int count, int blocklength, int stride,
                     MPI_Datatype oldtype, MPI_Datatype *newtype);
#define MPI_Type_create_hvector AMPI_Type_create_hvector
int AMPI_Type_create_hvector(int count, int blocklength, MPI_Aint stride,
                             MPI_Datatype oldtype, MPI_Datatype *newtype);
#define MPI_Type_hvector AMPI_Type_hvector
int AMPI_Type_hvector(int count, int blocklength, MPI_Aint stride,
                      MPI_Datatype oldtype, MPI_Datatype *newtype);
#define MPI_Type_indexed AMPI_Type_indexed
int AMPI_Type_indexed(int count, int* arrBlength, int* arrDisp,
                      MPI_Datatype oldtype, MPI_Datatype *newtype);
#define MPI_Type_create_hindexed AMPI_Type_create_hindexed
int AMPI_Type_create_hindexed(int count, int* arrBlength, MPI_Aint* arrDisp,
                              MPI_Datatype oldtype, MPI_Datatype *newtype);
#define MPI_Type_create_indexed_block AMPI_Type_create_indexed_block
int AMPI_Type_create_indexed_block(int count, int Blength, MPI_Aint *arr,
                                   MPI_Datatype oldtype, MPI_Datatype *newtype);
#define MPI_Type_create_hindexed_block AMPI_Type_create_hindexed_block
int AMPI_Type_create_hindexed_block(int count, int Blength, MPI_Aint *arr,
                                    MPI_Datatype oldtype, MPI_Datatype *newtype);
#define MPI_Type_hindexed AMPI_Type_hindexed
int AMPI_Type_hindexed(int count, int* arrBlength, MPI_Aint* arrDisp,
                       MPI_Datatype oldtype, MPI_Datatype *newtype);
#define MPI_Type_create_struct AMPI_Type_create_struct
int AMPI_Type_create_struct(int count, int* arrBLength, MPI_Aint* arrDisp,
                            MPI_Datatype *oldType, MPI_Datatype *newType);
#define MPI_Type_struct AMPI_Type_struct
int AMPI_Type_struct(int count, int* arrBLength, MPI_Aint* arrDisp,
                     MPI_Datatype *oldType, MPI_Datatype *newType);
#define MPI_Type_get_envelope AMPI_Type_get_envelope
int AMPI_Type_get_envelope(MPI_Datatype datatype, int *num_integers, int *num_addresses,
                           int *num_datatypes, int *combiner);
#define MPI_Type_get_contents AMPI_Type_get_contents
int AMPI_Type_get_contents(MPI_Datatype datatype, int max_integers, int max_addresses,
                           int max_datatypes, int array_of_integers[], MPI_Aint array_of_addresses[],
                           MPI_Datatype array_of_datatypes[]);
#define MPI_Type_commit AMPI_Type_commit
int AMPI_Type_commit(MPI_Datatype *datatype);
#define MPI_Type_free AMPI_Type_free
int AMPI_Type_free(MPI_Datatype *datatype);
#define MPI_Type_get_extent AMPI_Type_get_extent
int AMPI_Type_get_extent(MPI_Datatype datatype, MPI_Aint *lb, MPI_Aint *extent);
#define MPI_Type_extent AMPI_Type_extent
int AMPI_Type_extent(MPI_Datatype datatype, MPI_Aint *extent);
#define MPI_Type_size AMPI_Type_size
int AMPI_Type_size(MPI_Datatype datatype, int *size);
#define MPI_Type_lb AMPI_Type_lb
int AMPI_Type_lb(MPI_Datatype datatype, MPI_Aint* displacement);
#define MPI_Type_ub AMPI_Type_ub
int AMPI_Type_ub(MPI_Datatype datatype, MPI_Aint* displacement);
#define MPI_Type_set_name AMPI_Type_set_name
int AMPI_Type_set_name(MPI_Datatype datatype, const char *name);
#define MPI_Type_get_name AMPI_Type_get_name
int AMPI_Type_get_name(MPI_Datatype datatype, char *name, int *resultlen);
#define MPI_Type_create_resized AMPI_Type_create_resized
int AMPI_Type_create_resized(MPI_Datatype oldtype, MPI_Aint lb, MPI_Aint extent, MPI_Datatype *newtype);
#define MPI_Get_address AMPI_Get_address
int AMPI_Get_address(const void* location, MPI_Aint *address);
#define MPI_Address AMPI_Address
int AMPI_Address(void* location, MPI_Aint *address);
#define MPI_Status_set_elements AMPI_Status_set_elements
int AMPI_Status_set_elements(MPI_Status *status, MPI_Datatype datatype, int count);
#define MPI_Get_elements AMPI_Get_elements
int AMPI_Get_elements(MPI_Status *status, MPI_Datatype datatype, int *count);
#define MPI_Pack AMPI_Pack
int AMPI_Pack(void *inbuf, int incount, MPI_Datatype dtype, void *outbuf,
              int outsize, int *position, MPI_Comm comm);
#define MPI_Unpack AMPI_Unpack
int AMPI_Unpack(void *inbuf, int insize, int *position, void *outbuf,
                int outcount, MPI_Datatype dtype, MPI_Comm comm);
#define MPI_Pack_size AMPI_Pack_size
int AMPI_Pack_size(int incount,MPI_Datatype datatype,MPI_Comm comm,int *sz);
#define MPI_Aint_add(addr, disp) ((MPI_Aint)((char*)(addr) + (disp)))
#define MPI_Aint_diff(addr1, addr2) ((MPI_Aint)((char*)(addr1) - (char*)(addr2)))

/***collective***/
#define MPI_Barrier AMPI_Barrier
int AMPI_Barrier(MPI_Comm comm);
#define MPI_Ibarrier AMPI_Ibarrier
int AMPI_Ibarrier(MPI_Comm comm, MPI_Request *request);
#define MPI_Bcast AMPI_Bcast
int AMPI_Bcast(void *buf, int count, MPI_Datatype type, int root, MPI_Comm comm);
#define MPI_Ibcast AMPI_Ibcast
int AMPI_Ibcast(void *buf, int count, MPI_Datatype type, int root, MPI_Comm comm,
                MPI_Request *request);
#define MPI_Gather AMPI_Gather
int AMPI_Gather(void *sendbuf, int sendcount, MPI_Datatype sendtype,
                void *recvbuf, int recvcount, MPI_Datatype recvtype,
                int root, MPI_Comm comm);
#define MPI_Igather AMPI_Igather
int AMPI_Igather(void *sendbuf, int sendcount, MPI_Datatype sendtype,
                 void *recvbuf, int recvcount, MPI_Datatype recvtype,
                 int root, MPI_Comm comm, MPI_Request *request);
#define MPI_Gatherv AMPI_Gatherv
int AMPI_Gatherv(void *sendbuf, int sendcount, MPI_Datatype sendtype,
                 void *recvbuf, int *recvcounts, int *displs,
                 MPI_Datatype recvtype, int root, MPI_Comm comm);
#define MPI_Igatherv AMPI_Igatherv
int AMPI_Igatherv(void *sendbuf, int sendcount, MPI_Datatype sendtype,
                  void *recvbuf, int *recvcounts, int *displs,
                  MPI_Datatype recvtype, int root, MPI_Comm comm,
                  MPI_Request *request);
#define MPI_Scatter AMPI_Scatter
int AMPI_Scatter(void *sendbuf, int sendcount, MPI_Datatype sendtype,
                 void *recvbuf, int recvcount, MPI_Datatype recvtype,
                 int root, MPI_Comm comm);
#define MPI_Iscatter AMPI_Iscatter
int AMPI_Iscatter(void *sendbuf, int sendcount, MPI_Datatype sendtype,
                  void *recvbuf, int recvcount, MPI_Datatype recvtype,
                  int root, MPI_Comm comm, MPI_Request *request);
#define MPI_Scatterv AMPI_Scatterv
int AMPI_Scatterv(void *sendbuf, int *sendcounts, int *displs, MPI_Datatype sendtype,
                  void *recvbuf, int recvcount, MPI_Datatype recvtype,
                  int root, MPI_Comm comm);
#define MPI_Iscatterv AMPI_Iscatterv
int AMPI_Iscatterv(void *sendbuf, int *sendcounts, int *displs, MPI_Datatype sendtype,
                   void *recvbuf, int recvcount, MPI_Datatype recvtype,
                   int root, MPI_Comm comm, MPI_Request *request);
#define MPI_Allgather AMPI_Allgather
int AMPI_Allgather(void *sendbuf, int sendcount, MPI_Datatype sendtype,
                   void *recvbuf, int recvcount, MPI_Datatype recvtype,
                   MPI_Comm comm);
#define MPI_Iallgather AMPI_Iallgather
int AMPI_Iallgather(void *sendbuf, int sendcount, MPI_Datatype sendtype,
                    void *recvbuf, int recvcount, MPI_Datatype recvtype,
                    MPI_Comm comm, MPI_Request* request);
#define MPI_Allgatherv AMPI_Allgatherv
int AMPI_Allgatherv(void *sendbuf, int sendcount, MPI_Datatype sendtype,
                    void *recvbuf, int *recvcounts, int *displs,
                    MPI_Datatype recvtype, MPI_Comm comm) ;
#define MPI_Iallgatherv AMPI_Iallgatherv
int AMPI_Iallgatherv(void *sendbuf, int sendcount, MPI_Datatype sendtype,
                     void *recvbuf, int *recvcounts, int *displs,
                     MPI_Datatype recvtype, MPI_Comm comm, MPI_Request *request);
#define MPI_Alltoall AMPI_Alltoall
int AMPI_Alltoall(void *sendbuf, int sendcount, MPI_Datatype sendtype,
                  void *recvbuf, int recvcount, MPI_Datatype recvtype,
                  MPI_Comm comm);
#define MPI_Ialltoall AMPI_Ialltoall
int AMPI_Ialltoall(void *sendbuf, int sendcount, MPI_Datatype sendtype,
                   void *recvbuf, int recvcount, MPI_Datatype recvtype,
                   MPI_Comm comm, MPI_Request *request);
#define MPI_Alltoallv AMPI_Alltoallv
int AMPI_Alltoallv(void *sendbuf, int *sendcounts, int *sdispls,
                   MPI_Datatype sendtype, void *recvbuf, int *recvcounts,
                   int *rdispls, MPI_Datatype recvtype, MPI_Comm comm);
#define MPI_Ialltoallv AMPI_Ialltoallv
int AMPI_Ialltoallv(void *sendbuf, int *sendcounts, int *sdispls,
                    MPI_Datatype sendtype, void *recvbuf, int *recvcounts,
                    int *rdispls, MPI_Datatype recvtype, MPI_Comm comm,
                    MPI_Request *request);
#define MPI_Alltoallw AMPI_Alltoallw
int AMPI_Alltoallw(void *sendbuf, int *sendcounts, int *sdispls,
                   MPI_Datatype *sendtypes, void *recvbuf, int *recvcounts,
                   int *rdispls, MPI_Datatype *recvtypes, MPI_Comm comm);
#define MPI_Ialltoallw AMPI_Ialltoallw
int AMPI_Ialltoallw(void *sendbuf, int *sendcounts, int *sdispls,
                    MPI_Datatype *sendtypes, void *recvbuf, int *recvcounts,
                    int *rdispls, MPI_Datatype *recvtypes, MPI_Comm comm,
                    MPI_Request *request);
#define MPI_Reduce AMPI_Reduce
int AMPI_Reduce(void *inbuf, void *outbuf, int count, MPI_Datatype type,
                MPI_Op op, int root, MPI_Comm comm);
#define MPI_Ireduce AMPI_Ireduce
int AMPI_Ireduce(void *sendbuf, void *recvbuf, int count, MPI_Datatype type,
                 MPI_Op op, int root, MPI_Comm comm, MPI_Request *request);
#define MPI_Allreduce AMPI_Allreduce
int AMPI_Allreduce(void *inbuf, void *outbuf, int count, MPI_Datatype type,
                   MPI_Op op, MPI_Comm comm);
#define MPI_Iallreduce AMPI_Iallreduce
int AMPI_Iallreduce(void *inbuf, void *outbuf, int count, MPI_Datatype type,
                    MPI_Op op, MPI_Comm comm, MPI_Request *request);
#define MPI_Reduce_local AMPI_Reduce_local
int AMPI_Reduce_local(void *inbuf, void *outbuf, int count,
                      MPI_Datatype datatype, MPI_Op op);
#define MPI_Reduce_scatter_block AMPI_Reduce_scatter_block
int AMPI_Reduce_scatter_block(void* sendbuf, void* recvbuf, int count,
                              MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);
#define MPI_Reduce_scatter AMPI_Reduce_scatter
int AMPI_Reduce_scatter(void* sendbuf, void* recvbuf, int *recvcounts,
                        MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);
#define MPI_Scan AMPI_Scan
int AMPI_Scan(void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype,
              MPI_Op op, MPI_Comm comm );
#define MPI_Exscan AMPI_Exscan
int AMPI_Exscan(void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype,
                MPI_Op op, MPI_Comm comm);
#define MPI_Neighbor_alltoall AMPI_Neighbor_alltoall
int AMPI_Neighbor_alltoall(void* sendbuf, int sendcount, MPI_Datatype sendtype,
                           void* recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm);
#define MPI_Ineighbor_alltoall AMPI_Ineighbor_alltoall
int AMPI_Ineighbor_alltoall(void* sendbuf, int sendcount, MPI_Datatype sendtype,
                            void* recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm,
                            MPI_Request* request);
#define MPI_Neighbor_alltoallv AMPI_Neighbor_alltoallv
int AMPI_Neighbor_alltoallv(void* sendbuf, int* sendcounts, int* sdispls,
                            MPI_Datatype sendtype, void* recvbuf, int* recvcounts, int* rdispls,
                            MPI_Datatype recvtype, MPI_Comm comm);
#define MPI_Ineighbor_alltoallv AMPI_Ineighbor_alltoallv
int AMPI_Ineighbor_alltoallv(void* sendbuf, int* sendcounts, int* sdispls,
                             MPI_Datatype sendtype, void* recvbuf, int* recvcounts, int* rdispls,
                             MPI_Datatype recvtype, MPI_Comm comm, MPI_Request* request);
#define MPI_Neighbor_alltoallw AMPI_Neighbor_alltoallw
int AMPI_Neighbor_alltoallw(void* sendbuf, int* sendcounts, MPI_Aint* sdipls,
                            MPI_Datatype* sendtypes, void* recvbuf, int* recvcounts, MPI_Aint* rdispls,
                            MPI_Datatype* recvtypes, MPI_Comm comm);
#define MPI_Ineighbor_alltoallw AMPI_Ineighbor_alltoallw
int AMPI_Ineighbor_alltoallw(void* sendbuf, int* sendcounts, MPI_Aint* sdispls,
                             MPI_Datatype* sendtypes, void* recvbuf, int* recvcounts, MPI_Aint* rdispls,
                             MPI_Datatype* recvtypes, MPI_Comm comm, MPI_Request* request);
#define MPI_Neighbor_allgather AMPI_Neighbor_allgather
int AMPI_Neighbor_allgather(void* sendbuf, int sendcount, MPI_Datatype sendtype,
                            void* recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm);
#define MPI_Ineighbor_allgather AMPI_Ineighbor_allgather
int AMPI_Ineighbor_allgather(void* sendbuf, int sendcount, MPI_Datatype sendtype,
                             void* recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm,
                             MPI_Request *request);
#define MPI_Neighbor_allgatherv AMPI_Neighbor_allgatherv
int AMPI_Neighbor_allgatherv(void* sendbuf, int sendcount, MPI_Datatype sendtype,
                             void* recvbuf, int* recvcounts, int* displs, MPI_Datatype recvtype,
                             MPI_Comm comm);
#define MPI_Ineighbor_allgatherv AMPI_Ineighbor_allgatherv
int AMPI_Ineighbor_allgatherv(void* sendbuf, int sendcount, MPI_Datatype sendtype,
                              void* recvbuf, int* recvcounts, int* displs, MPI_Datatype recvtype,
                              MPI_Comm comm, MPI_Request *request);
#define MPI_Op_create AMPI_Op_create
int AMPI_Op_create(MPI_User_function *function, int commute, MPI_Op *op);
#define MPI_Op_free AMPI_Op_free
int AMPI_Op_free(MPI_Op *op);
#define MPI_Op_commutative AMPI_Op_commutative
int AMPI_Op_commutative(MPI_Op op, int* commute);

/***groups,contexts and communicators***/
#define MPI_Group_size AMPI_Group_size
int AMPI_Group_size(MPI_Group group, int *size);
#define MPI_Group_rank AMPI_Group_rank
int AMPI_Group_rank(MPI_Group group, int *rank);
#define MPI_Group_translate_ranks AMPI_Group_translate_ranks
int AMPI_Group_translate_ranks(MPI_Group group1, int n, int *ranks1, MPI_Group group2, int *ranks2);
#define MPI_Group_compare AMPI_Group_compare
int AMPI_Group_compare(MPI_Group group1,MPI_Group group2, int *result);
#define MPI_Comm_group AMPI_Comm_group
int AMPI_Comm_group(MPI_Comm comm, MPI_Group *group);
#define MPI_Group_union AMPI_Group_union
int AMPI_Group_union(MPI_Group group1, MPI_Group group2, MPI_Group *newgroup);
#define MPI_Group_intersection AMPI_Group_intersection
int AMPI_Group_intersection(MPI_Group group1, MPI_Group group2, MPI_Group *newgroup);
#define MPI_Group_difference AMPI_Group_difference
int AMPI_Group_difference(MPI_Group group1, MPI_Group group2, MPI_Group *newgroup);
#define MPI_Group_incl AMPI_Group_incl
int AMPI_Group_incl(MPI_Group group, int n, int *ranks, MPI_Group *newgroup);
#define MPI_Group_excl AMPI_Group_excl
int AMPI_Group_excl(MPI_Group group, int n, int *ranks, MPI_Group *newgroup);
#define MPI_Group_range_incl AMPI_Group_range_incl
int AMPI_Group_range_incl(MPI_Group group, int n, int ranges[][3], MPI_Group *newgroup);
#define MPI_Group_range_excl AMPI_Group_range_excl
int AMPI_Group_range_excl(MPI_Group group, int n, int ranges[][3], MPI_Group *newgroup);
#define MPI_Group_free AMPI_Group_free
int AMPI_Group_free(MPI_Group *group);

#define MPI_Intercomm_create AMPI_Intercomm_create
int AMPI_Intercomm_create(MPI_Comm local_comm, int local_leader, MPI_Comm peer_comm,
                          int remote_leader, int tag, MPI_Comm *newintercomm);
#define MPI_Intercomm_merge AMPI_Intercomm_merge
int AMPI_Intercomm_merge(MPI_Comm intercomm, int high, MPI_Comm *newintracomm);

#define MPI_Comm_create AMPI_Comm_create
int AMPI_Comm_create(MPI_Comm comm, MPI_Group group, MPI_Comm* newcomm);
#define MPI_Comm_size AMPI_Comm_size
int AMPI_Comm_size(MPI_Comm comm, int *size);
#define MPI_Comm_rank AMPI_Comm_rank
int AMPI_Comm_rank(MPI_Comm comm, int *rank);
#define MPI_Comm_compare AMPI_Comm_compare
int AMPI_Comm_compare(MPI_Comm comm1,MPI_Comm comm2, int *result);
#define MPI_Comm_split AMPI_Comm_split
int AMPI_Comm_split(MPI_Comm src, int color, int key, MPI_Comm *dest);
#define MPI_Comm_split_type AMPI_Comm_split_type
int AMPI_Comm_split_type(MPI_Comm src, int split_type, int key, MPI_Info info, MPI_Comm *dest);
#define MPI_Comm_dup AMPI_Comm_dup
int AMPI_Comm_dup(MPI_Comm src, MPI_Comm *dest);
#define MPI_Comm_free AMPI_Comm_free
int AMPI_Comm_free(MPI_Comm *comm);
#define MPI_Comm_test_inter AMPI_Comm_test_inter
int AMPI_Comm_test_inter(MPI_Comm comm, int *flag);
#define MPI_Comm_remote_size AMPI_Comm_remote_size
int AMPI_Comm_remote_size(MPI_Comm comm, int *size);
#define MPI_Comm_remote_group AMPI_Comm_remote_group
int AMPI_Comm_remote_group(MPI_Comm comm, MPI_Group *group);
#define MPI_Comm_set_name AMPI_Comm_set_name
int AMPI_Comm_set_name(MPI_Comm comm, const char *name);
#define MPI_Comm_get_name AMPI_Comm_get_name
int AMPI_Comm_get_name(MPI_Comm comm, char *comm_name, int *resultlen);
#define MPI_Comm_set_info AMPI_Comm_set_info
int AMPI_Comm_set_info(MPI_Comm comm, MPI_Info info);
#define MPI_Comm_get_info AMPI_Comm_get_info
int AMPI_Comm_get_info(MPI_Comm comm, MPI_Info *info);
#define MPI_Comm_call_errhandler AMPI_Comm_call_errhandler
int AMPI_Comm_call_errhandler(MPI_Comm comm, int errorcode);
#define MPI_Comm_create_errhandler AMPI_Comm_create_errhandler
int AMPI_Comm_create_errhandler(MPI_Comm_errhandler_fn *function, MPI_Errhandler *errhandler);
#define MPI_Comm_set_errhandler AMPI_Comm_set_errhandler
int AMPI_Comm_set_errhandler(MPI_Comm comm, MPI_Errhandler errhandler);
#define MPI_Comm_get_errhandler AMPI_Comm_get_errhandler
int AMPI_Comm_get_errhandler(MPI_Comm comm, MPI_Errhandler *errhandler);
#define MPI_Comm_free_errhandler AMPI_Comm_free_errhandler
int AMPI_Comm_free_errhandler(MPI_Errhandler *errhandler);
#define MPI_Comm_create_keyval AMPI_Comm_create_keyval
int AMPI_Comm_create_keyval(MPI_Comm_copy_attr_function *copy_fn, MPI_Comm_delete_attr_function *delete_fn,
                            int *keyval, void* extra_state);
#define MPI_Comm_free_keyval AMPI_Comm_free_keyval
int AMPI_Comm_free_keyval(int *keyval);
#define MPI_Comm_set_attr AMPI_Comm_set_attr
int AMPI_Comm_set_attr(MPI_Comm comm, int keyval, void* attribute_val);
#define MPI_Comm_get_attr AMPI_Comm_get_attr
int AMPI_Comm_get_attr(MPI_Comm comm, int keyval, void *attribute_val, int *flag);
#define MPI_Comm_delete_attr AMPI_Comm_delete_attr
int AMPI_Comm_delete_attr(MPI_Comm comm, int keyval);

#define MPI_Keyval_create AMPI_Keyval_create
int AMPI_Keyval_create(MPI_Copy_function *copy_fn, MPI_Delete_function *delete_fn,
                       int *keyval, void* extra_state);
#define MPI_Keyval_free AMPI_Keyval_free
int AMPI_Keyval_free(int *keyval);
#define MPI_Attr_put AMPI_Attr_put
int AMPI_Attr_put(MPI_Comm comm, int keyval, void* attribute_val);
#define MPI_Attr_get AMPI_Attr_get
int AMPI_Attr_get(MPI_Comm comm, int keyval, void *attribute_val, int *flag);
#define MPI_Attr_delete AMPI_Attr_delete
int AMPI_Attr_delete(MPI_Comm comm, int keyval);

/***topologies***/
#define MPI_Cart_create AMPI_Cart_create
int AMPI_Cart_create(MPI_Comm comm_old, int ndims, int *dims, int *periods,
                     int reorder, MPI_Comm *comm_cart);
#define MPI_Graph_create AMPI_Graph_create
int AMPI_Graph_create(MPI_Comm comm_old, int nnodes, int *index, int *edges,
                      int reorder, MPI_Comm *comm_graph);
#define MPI_Topo_test AMPI_Topo_test
int AMPI_Topo_test(MPI_Comm comm, int *status);
#define MPI_Cart_map AMPI_Cart_map
int AMPI_Cart_map(MPI_Comm comm, int ndims, int *dims, int *periods, int *newrank);
#define MPI_Graph_map AMPI_Graph_map
int AMPI_Graph_map(MPI_Comm comm, int nnodes, int *index, int *edges, int *newrank);
#define MPI_Cartdim_get AMPI_Cartdim_get
int AMPI_Cartdim_get(MPI_Comm comm, int *ndims);
#define MPI_Cart_get AMPI_Cart_get
int AMPI_Cart_get(MPI_Comm comm, int maxdims, int *dims, int *periods, int *coords);
#define MPI_Cart_rank AMPI_Cart_rank
int AMPI_Cart_rank(MPI_Comm comm, int *coords, int *rank);
#define MPI_Cart_coords AMPI_Cart_coords
int AMPI_Cart_coords(MPI_Comm comm, int rank, int maxdims, int *coords);
#define MPI_Cart_shift AMPI_Cart_shift
int AMPI_Cart_shift(MPI_Comm comm, int direction, int disp, int *rank_source, int *rank_dest);
#define MPI_Graphdims_get AMPI_Graphdims_get
int AMPI_Graphdims_get(MPI_Comm comm, int *nnodes, int *nedges);
#define MPI_Graph_get AMPI_Graph_get
int AMPI_Graph_get(MPI_Comm comm, int maxindex, int maxedges, int *index, int *edges);
#define MPI_Graph_neighbors_count AMPI_Graph_neighbors_count
int AMPI_Graph_neighbors_count(MPI_Comm comm, int rank, int *nneighbors);
#define MPI_Graph_neighbors AMPI_Graph_neighbors
int AMPI_Graph_neighbors(MPI_Comm comm, int rank, int maxneighbors, int *neighbors);
#define MPI_Dims_create AMPI_Dims_create
int AMPI_Dims_create(int nnodes, int ndims, int *dims);
#define MPI_Cart_sub AMPI_Cart_sub
int AMPI_Cart_sub(MPI_Comm comm, int *remain_dims, MPI_Comm *newcomm);

/***environment management***/
#define MPI_Get_version AMPI_Get_version
int AMPI_Get_version(int *version, int *subversion);
#define MPI_Get_library_version AMPI_Get_library_version
int AMPI_Get_library_version(char *version, int *resultlen);
#define MPI_Get_processor_name AMPI_Get_processor_name
int AMPI_Get_processor_name(char *name, int *resultlen);
#define MPI_Errhandler_create AMPI_Errhandler_create
int AMPI_Errhandler_create(MPI_Handler_function *function, MPI_Errhandler *errhandler);
#define MPI_Errhandler_set AMPI_Errhandler_set
int AMPI_Errhandler_set(MPI_Comm comm, MPI_Errhandler errhandler);
#define MPI_Errhandler_get AMPI_Errhandler_get
int AMPI_Errhandler_get(MPI_Comm comm, MPI_Errhandler *errhandler);
#define MPI_Errhandler_free AMPI_Errhandler_free
int AMPI_Errhandler_free(MPI_Errhandler *errhandler);
#define MPI_Add_error_code AMPI_Add_error_code
int AMPI_Add_error_code(int errorclass, int *errorcode);
#define MPI_Add_error_class AMPI_Add_error_class
int AMPI_Add_error_class(int *errorclass);
#define MPI_Add_error_string AMPI_Add_error_string
int AMPI_Add_error_string(int errorcode, const char *errorstring);
#define MPI_Error_class AMPI_Error_class
int AMPI_Error_class(int errorcode, int *errorclass);
#define MPI_Error_string AMPI_Error_string
int AMPI_Error_string(int errorcode, char *string, int *resultlen);
#define MPI_Wtime AMPI_Wtime
double AMPI_Wtime(void);
#define MPI_Wtick AMPI_Wtick
double AMPI_Wtick(void);
#define MPI_Is_thread_main AMPI_Is_thread_main
int AMPI_Is_thread_main(int *flag);
#define MPI_Query_thread AMPI_Query_thread
int AMPI_Query_thread(int *provided);
#define MPI_Init_thread AMPI_Init_thread
int AMPI_Init_thread(int *argc, char*** argv, int required, int *provided);
#define MPI_Init AMPI_Init
int AMPI_Init(int *argc, char*** argv);
#define MPI_Initialized AMPI_Initialized
int AMPI_Initialized(int *isInit);
#define MPI_Finalize AMPI_Finalize
int AMPI_Finalize(void);
#define MPI_Finalized AMPI_Finalized
int AMPI_Finalized(int *);
#define MPI_Abort AMPI_Abort
int AMPI_Abort(MPI_Comm comm, int errorcode);

/*** Profiling ***/
#define MPI_Pcontrol AMPI_Pcontrol
int AMPI_Pcontrol(const int level, ...);

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

#define MPI_Win_create AMPI_Win_create
int AMPI_Win_create(void *base, MPI_Aint size, int disp_unit,
                    MPI_Info info, MPI_Comm comm, MPI_Win *newwin);
#define MPI_Win_free AMPI_Win_free
int AMPI_Win_free(MPI_Win *win);
#define MPI_Win_create_errhandler AMPI_Win_create_errhandler
int AMPI_Win_create_errhandler(MPI_Win_errhandler_function *win_errhandler_fn,
                               MPI_Errhandler *errhandler);
#define MPI_Win_get_errhandler AMPI_Win_get_errhandler
int AMPI_Win_get_errhandler(MPI_Win win, MPI_Errhandler *errhandler);
#define MPI_Win_set_errhandler AMPI_Win_set_errhandler
int AMPI_Win_set_errhandler(MPI_Win win, MPI_Errhandler errhandler);
#define MPI_Win_create_keyval AMPI_Win_create_keyval
int AMPI_Win_create_keyval(MPI_Win_copy_attr_function *copy_fn,
                           MPI_Win_delete_attr_function *delete_fn,
                           int *keyval, void *extra_state);
#define MPI_Win_free_keyval AMPI_Win_free_keyval
int AMPI_Win_free_keyval(int *keyval);
#define MPI_Win_delete_attr AMPI_Win_delete_attr
int AMPI_Win_delete_attr(MPI_Win win, int key);
#define MPI_Win_get_attr AMPI_Win_get_attr
int AMPI_Win_get_attr(MPI_Win win, int win_keyval, void *attribute_val, int *flag);
#define MPI_Win_set_attr AMPI_Win_set_attr
int AMPI_Win_set_attr(MPI_Win win, int win_keyval, void *attribute_val);
#define MPI_Win_get_group AMPI_Win_get_group
int AMPI_Win_get_group(MPI_Win win, MPI_Group *group);
#define MPI_Win_set_name AMPI_Win_set_name
int AMPI_Win_set_name(MPI_Win win, const char *name);
#define MPI_Win_get_name AMPI_Win_get_name
int AMPI_Win_get_name(MPI_Win win, char *name, int *length);
#define MPI_Win_set_info AMPI_Win_set_info
int AMPI_Win_set_info(MPI_Win win, MPI_Info info);
#define MPI_Win_get_info AMPI_Win_get_info
int AMPI_Win_get_info(MPI_Win win, MPI_Info *info);
#define MPI_Win_fence AMPI_Win_fence
int AMPI_Win_fence(int assertion, MPI_Win win);
#define MPI_Win_lock AMPI_Win_lock
int AMPI_Win_lock(int lock_type, int rank, int assert, MPI_Win win);
#define MPI_Win_unlock AMPI_Win_unlock
int AMPI_Win_unlock(int rank, MPI_Win win);
#define MPI_Win_post AMPI_Win_post
int AMPI_Win_post(MPI_Group group, int assertion, MPI_Win win);
#define MPI_Win_wait AMPI_Win_wait
int AMPI_Win_wait(MPI_Win win);
#define MPI_Win_start AMPI_Win_start
int AMPI_Win_start(MPI_Group group, int assertion, MPI_Win win);
#define MPI_Win_complete AMPI_Win_complete
int AMPI_Win_complete(MPI_Win win);
#define MPI_Alloc_mem AMPI_Alloc_mem
int AMPI_Alloc_mem(MPI_Aint size, MPI_Info info, void *baseptr);
#define MPI_Free_mem AMPI_Free_mem
int AMPI_Free_mem(void *base);
#define MPI_Put AMPI_Put
int AMPI_Put(void *orgaddr, int orgcnt, MPI_Datatype orgtype, int rank,
             MPI_Aint targdisp, int targcnt, MPI_Datatype targtype, MPI_Win win);
#define MPI_Get AMPI_Get
int AMPI_Get(void *orgaddr, int orgcnt, MPI_Datatype orgtype, int rank,
             MPI_Aint targdisp, int targcnt, MPI_Datatype targtype, MPI_Win win);
#define MPI_Accumulate AMPI_Accumulate
int AMPI_Accumulate(void *orgaddr, int orgcnt, MPI_Datatype orgtype, int rank,
                    MPI_Aint targdisp, int targcnt, MPI_Datatype targtype,
                    MPI_Op op, MPI_Win win);
#define MPI_Get_accumulate AMPI_Get_accumulate
int AMPI_Get_accumulate(void *orgaddr, int orgcnt, MPI_Datatype orgtype,
                        void *resaddr, int rescnt, MPI_Datatype restype,
                        int rank, MPI_Aint targdisp, int targcnt,
                        MPI_Datatype targtype, MPI_Op op, MPI_Win win);
#define MPI_Fetch_and_op AMPI_Fetch_and_op
int AMPI_Fetch_and_op(void *orgaddr, void *resaddr, MPI_Datatype type,
                      int rank, MPI_Aint targdisp, MPI_Op op, MPI_Win win);
#define MPI_Compare_and_swap AMPI_Compare_and_swap
int AMPI_Compare_and_swap(void *orgaddr, void *compaddr, void *resaddr,
                          MPI_Datatype type, int rank, MPI_Aint targdisp,
                          MPI_Win win);

#define MPI_Info_create AMPI_Info_create
int AMPI_Info_create(MPI_Info *info);
#define MPI_Info_set AMPI_Info_set
int AMPI_Info_set(MPI_Info info, const char *key, const char *value);
#define MPI_Info_delete AMPI_Info_delete
int AMPI_Info_delete(MPI_Info info, const char *key);
#define MPI_Info_get AMPI_Info_get
int AMPI_Info_get(MPI_Info info, const char *key, int valuelen, char *value, int *flag);
#define MPI_Info_get_valuelen AMPI_Info_get_valuelen
int AMPI_Info_get_valuelen(MPI_Info info, const char *key, int *valuelen, int *flag);
#define MPI_Info_get_nkeys AMPI_Info_get_nkeys
int AMPI_Info_get_nkeys(MPI_Info info, int *nkeys);
#define MPI_Info_get_nthkey AMPI_Info_get_nthkey
int AMPI_Info_get_nthkey(MPI_Info info, int n, char *key);
#define MPI_Info_dup AMPI_Info_dup
int AMPI_Info_dup(MPI_Info info, MPI_Info *newinfo);
#define MPI_Info_free AMPI_Info_free
int AMPI_Info_free(MPI_Info *info);

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
int AMPI_Load_set_value(double value);
int AMPI_Migrate_to_pe(int dest);
int AMPI_Comm_set_migratable(MPI_Comm comm, int mig);
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
int AMPI_Evacuate(void);
int AMPI_Yield(MPI_Comm comm);
int AMPI_Suspend(MPI_Comm comm);
int AMPI_Resume(int dest, MPI_Comm comm);
int AMPI_Print(const char *str);
int AMPI_Trace_begin(void);
int AMPI_Trace_end(void);
int AMPI_Alltoall_iget(void *sendbuf, int sendcount, MPI_Datatype sendtype,
                       void *recvbuf, int recvcount, MPI_Datatype recvtype,
                       MPI_Comm comm);
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

#if CMK_CUDA
typedef struct workRequest workRequest;
int AMPI_GPU_Iinvoke(workRequest *to_call, MPI_Request *request);
int AMPI_GPU_Invoke(workRequest *to_call);
#endif

/* Execute this shell command (just like "system()") */
int AMPI_System(const char *cmd);

extern int traceRegisterFunction(const char *name, int idx);
extern void traceBeginFuncProj(const char *, const char *, int);
extern void traceEndFuncProj(const char *);
extern void traceBeginFuncIndexProj(int, const char *, int);
extern void traceEndFuncIndexProj(int);

/* Determine approximate depth of stack at the point of this call */
extern long ampiCurrentStackUsage(void);

#define AMPI_Trace_register_function_id(x, id) traceRegisterFunction(x, id);
#define AMPI_Trace_begin_function_id(id) traceBeginFuncIndexProj(id, __FILE__, __LINE__);
#define AMPI_Trace_end_function_id(id) traceEndFuncIndexProj(id);

#define AMPI_Trace_register_function_name(x) traceRegisterFunction(x, -999);
#define AMPI_Trace_begin_function_name(name) traceBeginFuncProj(name, __FILE__, __LINE__);
#define AMPI_Trace_end_function_name(name) traceEndFuncProj(name);

#include "ampiProjections.h"
#ifdef __cplusplus
}
#endif

#endif

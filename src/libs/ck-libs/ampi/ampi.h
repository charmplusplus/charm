/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#ifndef _MPI_H
#define _MPI_H

#ifdef __cplusplus
extern "C" {
#endif

/*NON-standard define: this lets people #ifdef on
AMPI, e.g. for our bizarre MPI_Main.
*/
#define AMPI

/*
Silently rename the user's main routine to "MPI_Main" or "MPI_Main_cpp".
This is needed so we can call the routine as a new thread.
*/
#ifdef __cplusplus
#  define main MPI_Main_cpp
#else
#  define main MPI_Main
#endif

/* MPI prototypes and #defines here */
#define MPI_SUCCESS 0
/* Somebody needs to define MPI_ERRs here */

/* these values have to match values in ampif.h */
#define MPI_DOUBLE 0
#define MPI_INT 1
#define MPI_FLOAT 2
#define MPI_COMPLEX 3
#define MPI_LOGICAL 4
#define MPI_CHAR 5
#define MPI_BYTE 6
#define MPI_PACKED 7
#define MPI_SHORT  8
#define MPI_LONG  9
#define MPI_UNSIGNED_CHAR  10
#define MPI_UNSIGNED_SHORT  11
#define MPI_UNSIGNED    12
#define MPI_UNSIGNED_LONG  13
#define MPI_LONG_DOUBLE  14
#define MPI_FLOAT_INT 15
#define MPI_DOUBLE_INT 16
#define MPI_LONG_INT 17
#define MPI_2INT 18
#define MPI_SHORT_INT 19
#define MPI_LONG_DOUBLE_INT 20
#define MPI_2FLOAT 21
#define MPI_2DOUBLE 22

#define MPI_ANY_SOURCE   (-1)
#define MPI_ANY_TAG      (-1)
#define MPI_REQUEST_NULL (-1)
#define MPI_GROUP_NULL   (-1)
#define MPI_COMM_NULL    (-1)
#define MPI_TYPE_NULL    (-1)

#define MPI_UNDEFINED    (-32766)

#define MPI_IDENT	0
#define MPI_SIMILAR	1
#define MPI_UNEQUAL	2

#define MPI_MAX 1
#define MPI_MIN 2
#define MPI_SUM 3
#define MPI_PROD 4
#define MPI_MAXLOC 5
#define MPI_MINLOC 6

/* This is one less than the system-tags defined in ampiimpl.h.
 * This is so that the tags used by the system dont clash with user-tags.
 * MPI standard requires this to be at least 2^15.
 */
#define MPI_TAG_UB  1073741824

typedef int MPI_Comm;
typedef int MPI_Group;

#define MPI_COMM_FIRST_SPLIT (MPI_Comm)(1000000) /*Communicator from a "split"*/
#define MPI_COMM_FIRST_GROUP (MPI_Comm)(2000000) /*Communicator from a process group*/
#define MPI_COMM_FIRST_RESVD (MPI_Comm)(3000000) /*Communicator reserved for now*/

#define MPI_COMM_WORLD (MPI_Comm)(8000000) /*Start of universe*/
#define MPI_MAX_COMM_WORLDS 8
extern MPI_Comm MPI_COMM_UNIVERSE[MPI_MAX_COMM_WORLDS];


typedef int MPI_Op;
typedef int MPI_Request;
typedef struct {
  int MPI_TAG, MPI_SOURCE, MPI_COMM, MPI_LENGTH;
} MPI_Status;

typedef int MPI_Datatype;
typedef int MPI_Aint;


#include "pup_c.h"

typedef void (*MPI_PupFn)(pup_er, void*);

/***pt2pt***/
int MPI_Send(void *msg, int count, MPI_Datatype type, int dest,
             int tag, MPI_Comm comm);
int MPI_Recv(void *msg, int count, int type, int src, int tag,
             MPI_Comm comm, MPI_Status *status);
int MPI_Get_count(MPI_Status *sts, MPI_Datatype dtype, int *count);
#define MPI_Bsend MPI_Send
int MPI_Ssend(void *msg, int count, MPI_Datatype type, int dest,
             int tag, MPI_Comm comm);
//MPI_Rsend
#define MPI_Buffer_attach(buf,len) /*LIE: emtpy*/ /*Silly: default send is buffering in Charm++*/
#define MPI_Buffer_detach(buf,len) /*LIE: emtpy*/
int MPI_Isend(void *buf, int count, MPI_Datatype datatype, int dest,
              int tag, MPI_Comm comm, MPI_Request *request);
#define MPI_Ibsend MPI_Isend
int MPI_Issend(void *buf, int count, MPI_Datatype datatype, int dest,
              int tag, MPI_Comm comm, MPI_Request *request);
int MPI_Irecv(void *buf, int count, MPI_Datatype datatype, int src,
              int tag, MPI_Comm comm, MPI_Request *request);
int MPI_Wait(MPI_Request *request, MPI_Status *sts);
int MPI_Waitany(int count, MPI_Request *request, int *index,MPI_Status *sts);
int MPI_Waitall(int count, MPI_Request *request, MPI_Status *sts);
int MPI_Test(MPI_Request *request, int *flag, MPI_Status *sts);
int MPI_Testall(int count, MPI_Request *request, int *flag, MPI_Status *sts);
//MPI_Testsome
//MPI_Testany
//MPI_Request_free
//MPI_Cancel
//MPI_Test_cancel
int MPI_Iprobe(int src, int tag, MPI_Comm comm, int *flag, MPI_Status *sts);
int MPI_Probe(int source, int tag, MPI_Comm comm, MPI_Status *sts);
int MPI_Send_init(void *buf, int count, int type, int dest, int tag,
                  MPI_Comm comm, MPI_Request *req);
//MPI_Bsend_init
//MPI_Ssend_init
//MPI_Rsend_init
int MPI_Recv_init(void *buf, int count, int type, int src, int tag,
                  MPI_Comm comm, MPI_Request *req);
int MPI_Start(MPI_Request *reqnum);
//MPI_Startall
int MPI_Sendrecv(void *sbuf, int scount, int stype, int dest,
                 int stag, void *rbuf, int rcount, int rtype,
                 int src, int rtag, MPI_Comm comm, MPI_Status *sts);
//MPI_Sendrecv_replace
int MPI_Type_contiguous(int count, MPI_Datatype oldtype,
                         MPI_Datatype *newtype);
int MPI_Type_vector(int count, int blocklength, int stride,
                     MPI_Datatype oldtype, MPI_Datatype *newtype);
int MPI_Type_hvector(int count, int blocklength, MPI_Aint stride,
                      MPI_Datatype oldtype, MPI_Datatype *newtype);
int MPI_Type_indexed(int count, int* arrBlength, int* arrDisp,
                      MPI_Datatype oldtype, MPI_Datatype *newtype);
int MPI_Type_hindexed(int count, int* arrBlength, MPI_Aint* arrDisp,
                       MPI_Datatype oldtype, MPI_Datatype *newtype);
int  MPI_Type_struct(int count, int* arrBLength, MPI_Aint* arrDisp,
                      MPI_Datatype *oldType, MPI_Datatype *newType);
int MPI_Type_commit(MPI_Datatype *datatype);
int MPI_Type_free(MPI_Datatype *datatype);
int  MPI_Type_extent(MPI_Datatype datatype, MPI_Aint *extent);
int  MPI_Type_size(MPI_Datatype datatype, int *size);
//MPI_Type_lb
//MPI_Type_ub
//MPI_Get_elements
//MPI_Address
int MPI_Pack(void *inbuf, int incount, MPI_Datatype dtype, void *outbuf,
              int outsize, int *position, MPI_Comm comm);
int MPI_Unpack(void *inbuf, int insize, int *position, void *outbuf,
              int outcount, MPI_Datatype dtype, MPI_Comm comm);
int MPI_Pack_size(int incount,MPI_Datatype datatype,MPI_Comm comm,int *sz);

/***collective***/
int MPI_Barrier(MPI_Comm comm);
int MPI_Bcast(void *buf, int count, int type, int root,
              MPI_Comm comm);
int MPI_Gather(void *sendbuf, int sendcount, MPI_Datatype sendtype,
               void *recvbuf, int recvcount, MPI_Datatype recvtype,
               int root, MPI_Comm comm);
int MPI_Gatherv(void *sendbuf, int sendcount, MPI_Datatype sendtype,
                void *recvbuf, int *recvcounts, int *displs,
                MPI_Datatype recvtype, int root, MPI_Comm comm);
int MPI_Scatter(void *sendbuf, int sendcount, MPI_Datatype sendtype,
                void *recvbuf, int recvcount, MPI_Datatype recvtype,
                int root, MPI_Comm comm);
int MPI_Scatterv(void *sendbuf, int *sendcounts, int *displs, MPI_Datatype sendtype,
                 void *recvbuf, int recvcount, MPI_Datatype recvtype,
                 int root, MPI_Comm comm);
int MPI_Allgather(void *sendbuf, int sendcount, MPI_Datatype sendtype,
                  void *recvbuf, int recvcount, MPI_Datatype recvtype,
                  MPI_Comm comm);
int MPI_Allgatherv(void *sendbuf, int sendcount, MPI_Datatype sendtype,
                   void *recvbuf, int *recvcounts, int *displs,
                   MPI_Datatype recvtype, MPI_Comm comm) ;
//MPI_Scatter
//MPI_Scatterv
int MPI_Alltoall(void *sendbuf, int sendcount, MPI_Datatype sendtype,
                 void *recvbuf, int recvcount, MPI_Datatype recvtype,
                 MPI_Comm comm);
int MPI_Alltoallv(void *sendbuf, int *sendcounts, int *sdispls,
                  MPI_Datatype sendtype, void *recvbuf, int *recvcounts,
                  int *rdispls, MPI_Datatype recvtype, MPI_Comm comm);
int MPI_Reduce(void *inbuf, void *outbuf, int count, int type,
               MPI_Op op, int root, MPI_Comm comm);
int MPI_Allreduce(void *inbuf, void *outbuf, int count, int type,
                  MPI_Op op, MPI_Comm comm);
//MPI_Reduce_scatter
//MPI_Scan
//MPI_Op_create
//MPI_Op_free

/***groups,contexts and communicators***/
int MPI_Group_size(MPI_Group group, int *size);
int MPI_Group_rank(MPI_Group group, int *rank);
int MPI_Group_translate_ranks (MPI_Group group1, int n, int *ranks1, MPI_Group group2, int *ranks2);
int MPI_Group_compare(MPI_Group group1,MPI_Group group2, int *result);

int MPI_Comm_group(MPI_Comm comm, MPI_Group *group);
int MPI_Group_union(MPI_Group group1, MPI_Group group2, MPI_Group *newgroup);
int MPI_Group_intersection(MPI_Group group1, MPI_Group group2, MPI_Group *newgroup);
int MPI_Group_difference(MPI_Group group1, MPI_Group group2, MPI_Group *newgroup);

int MPI_Group_incl(MPI_Group group, int n, int *ranks, MPI_Group *newgroup);
int MPI_Group_excl(MPI_Group group, int n, int *ranks, MPI_Group *newgroup);
int MPI_Group_range_incl(MPI_Group group, int n, int ranges[][3], MPI_Group *newgroup);
int MPI_Group_range_excl(MPI_Group group, int n, int ranges[][3], MPI_Group *newgroup);
int MPI_Group_free(MPI_Group *group);
int MPI_Comm_create(MPI_Comm comm, MPI_Group group, MPI_Comm* newcomm);

int MPI_Comm_size(MPI_Comm comm, int *size);
int MPI_Comm_rank(MPI_Comm comm, int *rank);
int MPI_Comm_dup(MPI_Comm src, MPI_Comm *dest);
int MPI_Comm_split(MPI_Comm src, int color, int key, MPI_Comm *dest);
int MPI_Comm_free(MPI_Comm *comm);
//MPI_Comm_test_inter
//MPI_Comm_remote_size
//MPI_Comm_remote_group
//MPI_Intercomm_create
//MPI_Intercomm_merge
//MPI_Keyval_create
//MPI_Keyval_free
//MPI_Attr_put
//MPI_Attr_get
//MPI_Attr_delete

/***topologies***/
//MPI_Cart_create
//MPI_Dims_create
//MPI_Graph_create
//MPI_Topo_test
//MPI_Cartdim_get
//MPI_Cart_get
//MPI_Cart_rank
//MPI_Cart_coords
//MPI_Cart_shift
//MPI_Cart_sub
//MPI_Cart_map
//MPI_Graphdims_get
//MPI_Graph_get
//MPI_Graph_neighbors_count
//MPI_Graph_neighbors
//MPI_Graph_map

/***environment management***/
//MPI_Get_processor_name
//MPI_Errhandler_create
//MPI_Errhandler_set
//MPI_Errhandler_get
//MPI_Errhandler_free
int MPI_Error_string(int errorcode, char *string, int *resultlen);
//MPI_Error_class
double MPI_Wtime(void);
//double MPI_Wtick(void);
int MPI_Init(int *argc, char*** argv); /* FORTRAN VERSION MISSING */
int MPI_Initialized(int *isInit); /* FORTRAN VERSION MISSING */
int MPI_Finalize(void);
int MPI_Abort(MPI_Comm comm, int errorcode);

/***extras***/
void MPI_Print(char *str);
int MPI_Register(void *, MPI_PupFn);
void MPI_Migrate(void);
void MPI_Checkpoint(char *dname);
void MPI_Restart(char *dname);
void *MPI_Get_userdata(int);

/*Create a new threads array and attach to it*/
typedef void (*MPI_MainFn) (int,char**);
void MPI_Register_main(MPI_MainFn mainFn, const char *name);

/*Attach a new AMPI to each existing threads array element*/
void MPI_Attach(const char *name);

int MPI_Ireduce(void *sendbuf, void *recvbuf, int count, int type, MPI_Op op, int root, MPI_Comm comm, MPI_Request *request);

#ifdef __cplusplus
}
#endif

#endif

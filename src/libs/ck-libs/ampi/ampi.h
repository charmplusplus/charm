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

#define MPI_COMM_WORLD (0)
#define MPI_ANY_SOURCE (-1)
#define MPI_ANY_TAG (-1)
#define MPI_REQUEST_NULL (-1)

#define MPI_TYPE_NULL  (-1)

#define MPI_MAX 1
#define MPI_MIN 2
#define MPI_SUM 3
#define MPI_PROD 4

typedef int MPI_Comm;
typedef int MPI_Op;
typedef int MPI_Request;
typedef struct {
  int MPI_TAG, MPI_SOURCE, MPI_COMM, MPI_LENGTH;
} MPI_Status;

typedef int MPI_Datatype;
typedef int*  MPI_Aint ;

#include "pup_c.h"

typedef void (*MPI_PupFn)(pup_er, void*);

int MPI_Init(int *argc, char*** argv);
int MPI_Comm_rank(MPI_Comm comm, int *rank);
int MPI_Comm_size(MPI_Comm comm, int *size);
int MPI_Finalize(void);
int MPI_Send(void *msg, int count, MPI_Datatype type, int dest,
             int tag, MPI_Comm comm);
int MPI_Ssend(void *msg, int count, MPI_Datatype type, int dest,
             int tag, MPI_Comm comm);
int MPI_Recv(void *msg, int count, int type, int src, int tag,
             MPI_Comm comm, MPI_Status *status);
int MPI_Sendrecv(void *sbuf, int scount, int stype, int dest,
                 int stag, void *rbuf, int rcount, int rtype,
                 int src, int rtag, MPI_Comm comm, MPI_Status *sts);
int MPI_Barrier(MPI_Comm comm);
int MPI_Bcast(void *buf, int count, int type, int root,
              MPI_Comm comm);
int MPI_Reduce(void *inbuf, void *outbuf, int count, int type,
               MPI_Op op, int root, MPI_Comm comm);
int MPI_Allreduce(void *inbuf, void *outbuf, int count, int type,
                  MPI_Op op, MPI_Comm comm);
double MPI_Wtime(void);
int MPI_Start(MPI_Request *reqnum);
int MPI_Probe(int source, int tag, MPI_Comm comm, MPI_Status *sts);
int MPI_Iprobe(int src, int tag, MPI_Comm comm, int *flag, MPI_Status *sts);
int MPI_Waitall(int count, MPI_Request *request, MPI_Status *sts);
int MPI_Waitany(int count, MPI_Request *request, int *index,MPI_Status *sts);
int MPI_Wait(MPI_Request *request, MPI_Status *sts);
int MPI_Test(MPI_Request *request, int *flag, MPI_Status *sts);
int MPI_Testall(int count, MPI_Request *request, int *flag, MPI_Status *sts);
int MPI_Get_count(MPI_Status *sts, MPI_Datatype dtype, int *count);
int MPI_Recv_init(void *buf, int count, int type, int src, int tag,
                  MPI_Comm comm, MPI_Request *req);
int MPI_Send_init(void *buf, int count, int type, int dest, int tag,
                  MPI_Comm comm, MPI_Request *req);

int MPI_Type_contiguous(int count, MPI_Datatype oldtype, 
                         MPI_Datatype *newtype);
int MPI_Type_vector(int count, int blocklength, int stride, 
                     MPI_Datatype oldtype, MPI_Datatype *newtype);
int MPI_Type_hvector(int count, int blocklength, int stride, 
                      MPI_Datatype oldtype, MPI_Datatype *newtype);
int MPI_Type_indexed(int count, int* arrBlength, int* arrDisp, 
                      MPI_Datatype oldtype, MPI_Datatype *newtype);
int MPI_Type_hindexed(int count, int* arrBlength, int* arrDisp, 
                       MPI_Datatype oldtype, MPI_Datatype *newtype);
int  MPI_Type_struct(int count, int* arrBLength, int* arrDisp, 
                      MPI_Datatype *oldType, MPI_Datatype *newType);
int MPI_Type_commit(MPI_Datatype *datatype);
int MPI_Type_free(MPI_Datatype *datatype);
int  MPI_Type_extent(MPI_Datatype datatype, MPI_Aint extent);
int  MPI_Type_size(MPI_Datatype datatype, MPI_Aint size);

int MPI_Pack(void *inbuf, int incount, MPI_Datatype dtype, void *outbuf,
              int outsize, int *position, MPI_Comm comm);
int MPI_Unpack(void *inbuf, int insize, int *position, void *outbuf,
              int outcount, MPI_Datatype dtype, MPI_Comm comm);
int MPI_Pack_size(int incount,MPI_Datatype datatype,MPI_Comm comm,int *sz);

int MPI_Isend(void *buf, int count, MPI_Datatype datatype, int dest, 
              int tag, MPI_Comm comm, MPI_Request *request);
int MPI_Issend(void *buf, int count, MPI_Datatype datatype, int dest, 
              int tag, MPI_Comm comm, MPI_Request *request);
int MPI_Irecv(void *buf, int count, MPI_Datatype datatype, int src, 
              int tag, MPI_Comm comm, MPI_Request *request);
int MPI_Allgatherv(void *sendbuf, int sendcount, MPI_Datatype sendtype, 
                   void *recvbuf, int *recvcounts, int *displs, 
                   MPI_Datatype recvtype, MPI_Comm comm) ;
int MPI_Allgather(void *sendbuf, int sendcount, MPI_Datatype sendtype,
                  void *recvbuf, int recvcount, MPI_Datatype recvtype,
                  MPI_Comm comm);
int MPI_Gatherv(void *sendbuf, int sendcount, MPI_Datatype sendtype,
                void *recvbuf, int *recvcounts, int *displs,
                MPI_Datatype recvtype, int root, MPI_Comm comm);
int MPI_Gather(void *sendbuf, int sendcount, MPI_Datatype sendtype,
               void *recvbuf, int recvcount, MPI_Datatype recvtype, 
               int root, MPI_Comm comm);
int MPI_Alltoallv(void *sendbuf, int *sendcounts, int *sdispls,
                  MPI_Datatype sendtype, void *recvbuf, int *recvcounts,
                  int *rdispls, MPI_Datatype recvtype, MPI_Comm comm);
int MPI_Alltoall(void *sendbuf, int sendcount, MPI_Datatype sendtype, 
                 void *recvbuf, int recvcount, MPI_Datatype recvtype, 
                 MPI_Comm comm);
int MPI_Comm_dup(MPI_Comm comm, MPI_Comm *newcomm);
int MPI_Comm_free(MPI_Comm *comm);
int MPI_Abort(MPI_Comm comm, int errorcode);

void MPI_Print(char *str);
int MPI_Register(void *, MPI_PupFn);
void MPI_Migrate(void);
void MPI_Checkpoint(char *dirname);
void *MPI_Get_userdata(int);

/*Create a new threads array and attach to it*/
typedef void (*MPI_MainFn) (int,char**);
void MPI_Register_main(MPI_MainFn mainFn, const char *name);

/*Attach a new AMPI to each existing threads array element*/
void MPI_Attach(const char *name);

#ifdef __cplusplus
}
#endif

#endif

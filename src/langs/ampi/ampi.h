/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#ifndef _AMPI_H
#define _AMPI_H

/*UDT_MOD1	Begin*/
/* #include	"ddt.h" */
/*UDT_MOD1	End*/

#ifdef __cplusplus
extern "C" {
#endif

/* MPI prototypes and #defines here */

/* these values have to match values in ampif.h */
#define AMPI_DOUBLE 0
#define AMPI_INT 1
#define AMPI_FLOAT 2
#define AMPI_COMPLEX 3
#define AMPI_LOGICAL 4
#define AMPI_CHAR 5
#define AMPI_BYTE 6
#define AMPI_PACKED 7

/*UDT_MOD1	Begin*/
#define	AMPI_SHORT	8
#define	AMPI_LONG	9
#define AMPI_UNSIGNED_CHAR	10
#define	AMPI_UNSIGNED_SHORT	11
#define	AMPI_UNSIGNED		12
#define	AMPI_UNSIGNED_LONG	13
#define	AMPI_LONG_DOUBLE	14
/*UDT_MOD1 End*/

#define AMPI_COMM_WORLD 0
#define AMPI_ANY_SOURCE (-1)
#define AMPI_ANY_TAG (-1)
#define AMPI_REQUEST_NULL (-1)

#define AMPI_TYPE_NULL	(-1)

#define AMPI_MAX 1
#define AMPI_MIN 2
#define AMPI_SUM 3
#define AMPI_PROD 4

typedef int AMPI_Comm;
typedef int AMPI_Op;
typedef int AMPI_Request;
typedef struct {
  int one, two, three;
} AMPI_Status;

/*UDT_MOD1	Begin */
/* typedef DDT_Type AMPI_Datatype; */
typedef int AMPI_Datatype;
typedef	int*	AMPI_Aint ;
/*UDT_MOD1 End*/

int AMPI_Init(int *argc, char*** argv);
int AMPI_Comm_rank(AMPI_Comm comm, int *rank);
int AMPI_Comm_size(AMPI_Comm comm, int *size);
int AMPI_Finalize(void);
int AMPI_Send(void *msg, int count, AMPI_Datatype type, int dest,
             int tag, AMPI_Comm comm);
int AMPI_Recv(void *msg, int count, int type, int src, int tag,
             AMPI_Comm comm, AMPI_Status *status);
int AMPI_Sendrecv(void *sbuf, int scount, int stype, int dest,
                 int stag, void *rbuf, int rcount, int rtype,
                 int src, int rtag, AMPI_Comm comm, AMPI_Status *sts);
int AMPI_Barrier(AMPI_Comm comm);
int AMPI_Bcast(void *buf, int count, int type, int root,
              AMPI_Comm comm);
int AMPI_Reduce(void *inbuf, void *outbuf, int count, int type,
               AMPI_Op op, int root, AMPI_Comm comm);
int AMPI_Allreduce(void *inbuf, void *outbuf, int count, int type,
                  AMPI_Op op, AMPI_Comm comm);
double AMPI_Wtime(void);
int AMPI_Start(AMPI_Request *reqnum);
int AMPI_Waitall(int count, AMPI_Request *request, AMPI_Status *sts);
int AMPI_Recv_init(void *buf, int count, int type, int src, int tag,
                  AMPI_Comm comm, AMPI_Request *req);
int AMPI_Send_init(void *buf, int count, int type, int dest, int tag,
                  AMPI_Comm comm, AMPI_Request *req);

/*UDT_MOD1 Begin*/
int AMPI_Type_Contiguous(int count, AMPI_Datatype oldtype, AMPI_Datatype *newtype);
int AMPI_Type_Vector(int count, int blocklength, int stride, AMPI_Datatype oldtype, AMPI_Datatype *newtype);
int AMPI_Type_HVector(int count, int blocklength, int stride, AMPI_Datatype oldtype, AMPI_Datatype *newtype);
int AMPI_Type_Indexed(int count, int* arrBlength, int* arrDisp, AMPI_Datatype oldtype, AMPI_Datatype *newtype);
int AMPI_Type_HIndexed(int count, int* arrBlength, int* arrDisp, AMPI_Datatype oldtype, AMPI_Datatype *newtype);
int	AMPI_Type_Struct(int count, int* arrBLength, int* arrDisp, AMPI_Datatype *oldType, AMPI_Datatype *newType);
int AMPI_Type_commit(AMPI_Datatype *datatype);
int AMPI_Type_free(AMPI_Datatype *datatype);
void  AMPI_Type_Extent(AMPI_Datatype datatype, AMPI_Aint extent);
void  AMPI_Type_Size(AMPI_Datatype datatype, AMPI_Aint size);

/*UDT_MOD1 end*/
int AMPI_Isend(void *buf, int count, AMPI_Datatype datatype, int dest, 
              int tag, AMPI_Comm comm, AMPI_Request *request);
int AMPI_Irecv(void *buf, int count, AMPI_Datatype datatype, int src, 
              int tag, AMPI_Comm comm, AMPI_Request *request);
int AMPI_Allgatherv(void *sendbuf, int sendcount, AMPI_Datatype sendtype, 
                   void *recvbuf, int *recvcounts, int *displs, 
                   AMPI_Datatype recvtype, AMPI_Comm comm) ;
int AMPI_Allgather(void *sendbuf, int sendcount, AMPI_Datatype sendtype,
                  void *recvbuf, int recvcount, AMPI_Datatype recvtype,
                  AMPI_Comm comm);
int AMPI_Gatherv(void *sendbuf, int sendcount, AMPI_Datatype sendtype,
                void *recvbuf, int *recvcounts, int *displs,
                AMPI_Datatype recvtype, int root, AMPI_Comm comm);
int AMPI_Gather(void *sendbuf, int sendcount, AMPI_Datatype sendtype,
               void *recvbuf, int recvcount, AMPI_Datatype recvtype, 
               int root, AMPI_Comm comm);
int AMPI_Alltoallv(void *sendbuf, int *sendcounts, int *sdispls,
                  AMPI_Datatype sendtype, void *recvbuf, int *recvcounts,
                  int *rdispls, AMPI_Datatype recvtype, AMPI_Comm comm);
int AMPI_Alltoall(void *sendbuf, int sendcount, AMPI_Datatype sendtype, 
                 void *recvbuf, int recvcount, AMPI_Datatype recvtype, 
                 AMPI_Comm comm);
int AMPI_Comm_dup(AMPI_Comm comm, AMPI_Comm *newcomm);
int AMPI_Comm_free(AMPI_Comm *comm);
int AMPI_Abort(AMPI_Comm comm, int errorcode);
#ifdef __cplusplus
}
#endif

#endif

#include "ampi.h"

extern "C" 
void ampi_init_(int *ierr)
{
  *ierr = AMPI_Init(0,0);
}

extern "C" 
void ampi_comm_rank_(int *comm, int *rank, int *ierr)
{
  *ierr = AMPI_Comm_rank(*comm, rank);
}

extern "C" 
void ampi_comm_size_(int *comm, int *size, int *ierr)
{
  *ierr = AMPI_Comm_size(*comm, size);
}

extern "C" 
void ampi_finalize_(int *ierr)
{
  *ierr = AMPI_Finalize();
}

extern "C" 
void ampi_send_(void *msg, int *count, int *type, int *dest, int *tag, 
                int *comm, int *ierr)
{
  *ierr = AMPI_Send(msg, *count, *type, *dest, *tag, *comm);
}

extern "C" 
void ampi_recv_(void *msg, int *count, int *type, int *src, int *tag, 
                int *comm, int *status, int *ierr)
{
  *ierr = AMPI_Recv(msg, *count, *type, *src, *tag, *comm, 
                    (AMPI_Status*) status);
}

extern "C" 
void ampi_sendrecv_(void *sndbuf, int *sndcount, int *sndtype, 
                    int *dest, int *sndtag, void *rcvbuf, 
                    int *rcvcount, int *rcvtype, int *src, 
                    int *rcvtag, int *comm, int *status, int *ierr)
{
  ampi_send_(sndbuf, sndcount, sndtype, dest, sndtag, comm, ierr);
  ampi_recv_(rcvbuf, rcvcount, rcvtype, src, rcvtag, comm, status, ierr);
}

extern "C" 
void ampi_barrier_(int *comm, int *ierr)
{
  *ierr = AMPI_Comm(*comm);
}

extern "C" 
void ampi_bcast_(void *buf, int *count, int *type, int *root, int *comm, 
                 int *ierr)
{
  *ierr = AMPI_Bcast(buf, *count, *type, *root, *comm);
}

extern "C" 
void ampi_reduce_(void *inbuf, void *outbuf, int *count, int *type,
                  int *op, int *root, int *comm, int *ierr)
{
  *ierr = AMPI_Reduce(inbuf, outbuf, *count, *type, *op, *root, *comm);
}

extern "C" 
void ampi_allreduce_(void *inbuf,void *outbuf,int *count,int *type,
                     int *op, int *comm, int *ierr)
{
  *ierr = AMPI_Allreduce(inbuf, outbuf, *count, *type, *op, *comm);
}

extern "C" 
double ampi_wtime_(void)
{
  return AMPI_Wtime();
}

extern "C" 
void ampi_start_(int *reqnum, int *ierr)
{
  *ierr = AMPI_Start((AMPI_Request*) reqnum);
}

extern "C" 
void ampi_waitall_(int *count, int *request, int *status, int *ierr)
{
  *ierr = AMPI_Waitall(*count, (AMPI_Request*) request, (AMPI_Status*) status);
}

extern "C" 
void ampi_recv_init_(void *buf, int *count, int *type, int *srcpe,
                     int *tag, int *comm, int *req, int *ierr)
{
  *ierr = AMPI_Recv_init(buf,*count,*type,*srcpe,*tag,*comm,(AMPI_Request*)req);
}

extern "C" 
void ampi_send_init_(void *buf, int *count, int *type, int *destpe,
                     int *tag, int *comm, int *req, int *ierr)
{
  *ierr = AMPI_Send_init(buf,*count,*type,*destpe,*tag,*comm,
                         (AMPI_Request*)req);
}

extern "C" 
void ampi_type_contiguous_(int *count, int *oldtype, int *newtype, 
                           int *ierr)
{
  *ierr = AMPI_Type_contiguous(*count, *oldtype, newtype);
}

extern  "C"  
void ampi_type_vector_(int *count, int *blocklength, int *stride, 
                       int *oldtype, int*  newtype, int *ierr)
{
  *ierr = AMPI_Type_vector(*count, *blocklength, *stride, *oldtype, newtype);
}

extern  "C"  
void ampi_type_hvector_(int *count, int *blocklength, int *stride, 
                        int* oldtype, int* newtype, int* ierr)
{
  *ierr = AMPI_Type_hvector(*count, *blocklength, *stride, *oldtype, newtype);
}

extern  "C"  
void ampi_type_indexed_(int *count, int* arrBlength, int* arrDisp, 
                        int* oldtype, int*  newtype, int* ierr)
{
  *ierr = AMPI_Type_indexed(*count, arrBlength, arrDisp, *oldtype, newtype);
}

extern  "C"  
void ampi_type_hindexed_(int* count, int* arrBlength, int* arrDisp, 
                         int* oldtype, int* newtype, int* ierr)
{
  *ierr = AMPI_Type_hindexed(*count, arrBlength, arrDisp, *oldtype, newtype);
}

extern  "C"  
void ampi_type_struct_(int* count, int* arrBlength, int* arrDisp, 
                       int* oldtype, int* newtype, int* ierr)
{
  *ierr = AMPI_Type_struct(*count, arrBlength, arrDisp, oldtype, newtype);
}


extern "C" 
void ampi_type_commit_(int *type, int *ierr)
{
  *ierr = AMPI_Type_commit(type);
}

extern "C" 
void ampi_type_free_(int *type, int *ierr)
{
  *ierr = AMPI_Type_free(type);
}

extern "C"
void  ampi_type_extent_(int* type, int* extent, int* ierr)
{
  *ierr = AMPI_Type_extent(*type, extent);
}

extern "C"
void  ampi_type_size_(int* type, int* size, int* ierr)
{
  *ierr = AMPI_Type_size(*type, size);
}

extern "C" 
void ampi_isend_(void *buf, int *count, int *datatype, int *dest,
                 int *tag, int *comm, int *request, int *ierr)
{
  *ierr = AMPI_Isend(buf, *count, *datatype, *dest, *tag, *comm, request);
}

extern "C" 
void ampi_irecv_(void *buf, int *count, int *datatype, int *src,
                 int *tag, int *comm, int *request, int *ierr)
{
  *ierr = AMPI_Irecv(buf, *count, *datatype, *src, *tag, *comm, request);
}

extern "C"
void ampi_allgatherv_(void *sendbuf, int *sendcount, int *sendtype,
                     void *recvbuf, int *recvcounts, int *displs,
                     int *recvtype, int *comm, int *ierr)
{
  *ierr = AMPI_Allgatherv(sendbuf, *sendcount, *sendtype, recvbuf, recvcounts,
                          displs, *recvtype, *comm);
}

extern "C"
void ampi_allgather_(void *sendbuf, int *sendcount, int *sendtype,
                     void *recvbuf, int *recvcount, int *recvtype,
                     int *comm, int *ierr)
{
  *ierr = AMPI_Allgather(sendbuf, *sendcount, *sendtype, recvbuf, *recvcount,
                         *recvtype, *comm);
}

extern "C"
void ampi_gatherv_(void *sendbuf, int *sendcount, int *sendtype,
                   void *recvbuf, int *recvcounts, int *displs,
                   int *recvtype, int *root, int *comm, int *ierr)
{
  *ierr = AMPI_Gatherv(sendbuf, *sendcount, *sendtype, recvbuf, recvcounts,
                       displs, *recvtype, *root, *comm);
}

extern "C"
void ampi_gather_(void *sendbuf, int *sendcount, int *sendtype,
                  void *recvbuf, int *recvcount, int *recvtype,
                  int *root, int *comm, int *ierr)
{
  *ierr = AMPI_Gather(sendbuf, *sendcount, *sendtype, recvbuf, *recvcount, 
                      *recvtype, *root, *comm);
}

extern "C"
void ampi_alltoallv_(void *sendbuf, int *sendcounts, int *sdispls,
                     int *sendtype, void *recvbuf, int *recvcounts,
                     int *rdispls, int *recvtype, int *comm, int *ierr)
{
  *ierr = AMPI_Alltoallv(sendbuf, sendcounts, sdispls, *sendtype, recvbuf,
                         recvcounts, rdispls, *recvtype, *comm);
}

extern "C"
void ampi_alltoall_(void *sendbuf, int *sendcount, int *sendtype,
                    void *recvbuf, int *recvcount, int *recvtype,
                    int *comm, int *ierr)
{
  *ierr = AMPI_Alltoall(sendbuf, *sendcount, *sendtype, recvbuf, *recvcount,
                        *recvtype, *comm);
}

extern "C"
void ampi_comm_dup_(int *comm, int *newcomm, int *ierr)
{
  *newcomm = *comm;
  *ierr = 0;
}

extern "C"
void ampi_comm_free_(int *comm, int *ierr)
{
  *ierr = 0;
}

extern "C"
void ampi_abort_(int *comm, int *errorcode, int *ierr)
{
  *ierr = AMPI_Abort(*comm, *errorcode);
}

#include "ampi.h"
#include "ampiimpl.h"

CtvExtern(ampi *, ampiPtr);

extern "C" void 
#if CMK_FORTRAN_USES_ALLCAPS
AMPI_INIT
#else
ampi_init_
#endif
  (int *ierr)
{
  *ierr = AMPI_Init(0,0);
}

extern "C" void 
#if CMK_FORTRAN_USES_ALLCAPS
AMPI_COMM_RANK
#else
ampi_comm_rank_
#endif
  (int *comm, int *rank, int *ierr)
{
  *ierr = AMPI_Comm_rank(*comm, rank);
}

extern "C" void 
#if CMK_FORTRAN_USES_ALLCAPS
AMPI_COMM_SIZE
#else
ampi_comm_size_
#endif
  (int *comm, int *size, int *ierr)
{
  *ierr = AMPI_Comm_size(*comm, size);
}

extern "C" void 
#if CMK_FORTRAN_USES_ALLCAPS
AMPI_FINALIZE
#else
ampi_finalize_
#endif
  (int *ierr)
{
  *ierr = AMPI_Finalize();
}

extern "C" void 
#if CMK_FORTRAN_USES_ALLCAPS
AMPI_SEND
#else
ampi_send_
#endif
  (void *msg, int *count, int *type, int *dest, int *tag, int *comm, int *ierr)
{
  *ierr = AMPI_Send(msg, *count, *type, *dest, *tag, *comm);
}

extern "C" void 
#if CMK_FORTRAN_USES_ALLCAPS
AMPI_RECV
#else
ampi_recv_
#endif
  (void *msg, int *count, int *type, int *src, int *tag, int *comm, 
   int *status, int *ierr)
{
  *ierr = AMPI_Recv(msg, *count, *type, *src, *tag, *comm, 
                    (AMPI_Status*) status);
}

extern "C" void 
#if CMK_FORTRAN_USES_ALLCAPS
AMPI_SENDRECV
#else
ampi_sendrecv_
#endif
  (void *sndbuf, int *sndcount, int *sndtype, 
   int *dest, int *sndtag, void *rcvbuf, 
   int *rcvcount, int *rcvtype, int *src, 
   int *rcvtag, int *comm, int *status, int *ierr)
{
  *ierr = AMPI_Sendrecv(sndbuf, *sndcount, *sndtype, *dest, *sndtag,
                        rcvbuf, *rcvcount, *rcvtype, *src, *rcvtag,
			*comm, (AMPI_Status*) status);
}

extern "C" void 
#if CMK_FORTRAN_USES_ALLCAPS
AMPI_BARRIER
#else
ampi_barrier_
#endif
  (int *comm, int *ierr)
{
  *ierr = AMPI_Comm(*comm);
}

extern "C" void 
#if CMK_FORTRAN_USES_ALLCAPS
AMPI_BCAST
#else
ampi_bcast_
#endif
  (void *buf, int *count, int *type, int *root, int *comm, 
   int *ierr)
{
  *ierr = AMPI_Bcast(buf, *count, *type, *root, *comm);
}

extern "C" void 
#if CMK_FORTRAN_USES_ALLCAPS
AMPI_REDUCE
#else
ampi_reduce_
#endif
  (void *inbuf, void *outbuf, int *count, int *type,
   int *op, int *root, int *comm, int *ierr)
{
  *ierr = AMPI_Reduce(inbuf, outbuf, *count, *type, *op, *root, *comm);
}

extern "C" void 
#if CMK_FORTRAN_USES_ALLCAPS
AMPI_ALLREDUCE
#else
ampi_allreduce_
#endif
  (void *inbuf,void *outbuf,int *count,int *type,
   int *op, int *comm, int *ierr)
{
  *ierr = AMPI_Allreduce(inbuf, outbuf, *count, *type, *op, *comm);
}

extern "C" double 
#if CMK_FORTRAN_USES_ALLCAPS
AMPI_WTIME
#else
ampi_wtime_
#endif
  (void)
{
  return AMPI_Wtime();
}

extern "C" void 
#if CMK_FORTRAN_USES_ALLCAPS
AMPI_START
#else
ampi_start_
#endif
  (int *reqnum, int *ierr)
{
  *ierr = AMPI_Start((AMPI_Request*) reqnum);
}

extern "C" void 
#if CMK_FORTRAN_USES_ALLCAPS
AMPI_WAITALL
#else
ampi_waitall_
#endif
  (int *count, int *request, int *status, int *ierr)
{
  *ierr = AMPI_Waitall(*count, (AMPI_Request*) request, (AMPI_Status*) status);
}

extern "C" void 
#if CMK_FORTRAN_USES_ALLCAPS
AMPI_RECV_INIT
#else
ampi_recv_init_
#endif
  (void *buf, int *count, int *type, int *srcpe,
   int *tag, int *comm, int *req, int *ierr)
{
  *ierr = AMPI_Recv_init(buf,*count,*type,*srcpe,*tag,*comm,(AMPI_Request*)req);
}

extern "C" void 
#if CMK_FORTRAN_USES_ALLCAPS
AMPI_SEND_INIT
#else
ampi_send_init_
#endif
  (void *buf, int *count, int *type, int *destpe,
   int *tag, int *comm, int *req, int *ierr)
{
  *ierr = AMPI_Send_init(buf,*count,*type,*destpe,*tag,*comm,
                         (AMPI_Request*)req);
}

extern "C" void 
#if CMK_FORTRAN_USES_ALLCAPS
AMPI_TYPE_CONTIGUOUS
#else
ampi_type_contiguous_
#endif
  (int *count, int *oldtype, int *newtype, 
   int *ierr)
{
  *ierr = AMPI_Type_contiguous(*count, *oldtype, newtype);
}

extern  "C"  void 
#if CMK_FORTRAN_USES_ALLCAPS
AMPI_TYPE_VECTOR
#else
ampi_type_vector_
#endif
  (int *count, int *blocklength, int *stride, 
   int *oldtype, int*  newtype, int *ierr)
{
  *ierr = AMPI_Type_vector(*count, *blocklength, *stride, *oldtype, newtype);
}

extern  "C"  void 
#if CMK_FORTRAN_USES_ALLCAPS
AMPI_TYPE_HVECTOR
#else
ampi_type_hvector_
#endif
  (int *count, int *blocklength, int *stride, 
   int* oldtype, int* newtype, int* ierr)
{
  *ierr = AMPI_Type_hvector(*count, *blocklength, *stride, *oldtype, newtype);
}

extern  "C"  void 
#if CMK_FORTRAN_USES_ALLCAPS
AMPI_TYPE_INDEXED
#else
ampi_type_indexed_
#endif
  (int *count, int* arrBlength, int* arrDisp, 
   int* oldtype, int*  newtype, int* ierr)
{
  *ierr = AMPI_Type_indexed(*count, arrBlength, arrDisp, *oldtype, newtype);
}

extern  "C"  void 
#if CMK_FORTRAN_USES_ALLCAPS
AMPI_TYPE_HINDEXED
#else
ampi_type_hindexed_
#endif
  (int* count, int* arrBlength, int* arrDisp, 
   int* oldtype, int* newtype, int* ierr)
{
  *ierr = AMPI_Type_hindexed(*count, arrBlength, arrDisp, *oldtype, newtype);
}

extern  "C"  void 
#if CMK_FORTRAN_USES_ALLCAPS
AMPI_TYPE_STRUCT
#else
ampi_type_struct_
#endif
  (int* count, int* arrBlength, int* arrDisp, 
   int* oldtype, int* newtype, int* ierr)
{
  *ierr = AMPI_Type_struct(*count, arrBlength, arrDisp, oldtype, newtype);
}


extern "C" void 
#if CMK_FORTRAN_USES_ALLCAPS
AMPI_TYPE_COMMIT
#else
ampi_type_commit_
#endif
  (int *type, int *ierr)
{
  *ierr = AMPI_Type_commit(type);
}

extern "C" void 
#if CMK_FORTRAN_USES_ALLCAPS
AMPI_TYPE_FREE
#else
ampi_type_free_
#endif
  (int *type, int *ierr)
{
  *ierr = AMPI_Type_free(type);
}

extern "C" void  
#if CMK_FORTRAN_USES_ALLCAPS
AMPI_TYPE_EXTENT
#else
ampi_type_extent_
#endif
  (int* type, int* extent, int* ierr)
{
  *ierr = AMPI_Type_extent(*type, extent);
}

extern "C" void  
#if CMK_FORTRAN_USES_ALLCAPS
AMPI_TYPE_SIZE
#else
ampi_type_size_
#endif
  (int* type, int* size, int* ierr)
{
  *ierr = AMPI_Type_size(*type, size);
}

extern "C" void 
#if CMK_FORTRAN_USES_ALLCAPS
AMPI_ISEND
#else
ampi_isend_
#endif
  (void *buf, int *count, int *datatype, int *dest,
   int *tag, int *comm, int *request, int *ierr)
{
  *ierr = AMPI_Isend(buf, *count, *datatype, *dest, *tag, *comm, request);
}

extern "C" void 
#if CMK_FORTRAN_USES_ALLCAPS
AMPI_IRECV
#else
ampi_irecv_
#endif
  (void *buf, int *count, int *datatype, int *src,
   int *tag, int *comm, int *request, int *ierr)
{
  *ierr = AMPI_Irecv(buf, *count, *datatype, *src, *tag, *comm, request);
}

extern "C" void 
#if CMK_FORTRAN_USES_ALLCAPS
AMPI_ALLGATHERV
#else
ampi_allgatherv_
#endif
  (void *sendbuf, int *sendcount, int *sendtype,
   void *recvbuf, int *recvcounts, int *displs,
   int *recvtype, int *comm, int *ierr)
{
  *ierr = AMPI_Allgatherv(sendbuf, *sendcount, *sendtype, recvbuf, recvcounts,
                          displs, *recvtype, *comm);
}

extern "C" void 
#if CMK_FORTRAN_USES_ALLCAPS
AMPI_ALLGATHER
#else
ampi_allgather_
#endif
  (void *sendbuf, int *sendcount, int *sendtype,
   void *recvbuf, int *recvcount, int *recvtype,
   int *comm, int *ierr)
{
  *ierr = AMPI_Allgather(sendbuf, *sendcount, *sendtype, recvbuf, *recvcount,
                         *recvtype, *comm);
}

extern "C" void 
#if CMK_FORTRAN_USES_ALLCAPS
AMPI_GATHERV
#else
ampi_gatherv_
#endif
  (void *sendbuf, int *sendcount, int *sendtype,
   void *recvbuf, int *recvcounts, int *displs,
   int *recvtype, int *root, int *comm, int *ierr)
{
  *ierr = AMPI_Gatherv(sendbuf, *sendcount, *sendtype, recvbuf, recvcounts,
                       displs, *recvtype, *root, *comm);
}

extern "C" void 
#if CMK_FORTRAN_USES_ALLCAPS
AMPI_GATHER
#else
ampi_gather_
#endif
  (void *sendbuf, int *sendcount, int *sendtype,
   void *recvbuf, int *recvcount, int *recvtype,
   int *root, int *comm, int *ierr)
{
  *ierr = AMPI_Gather(sendbuf, *sendcount, *sendtype, recvbuf, *recvcount, 
                      *recvtype, *root, *comm);
}

extern "C" void 
#if CMK_FORTRAN_USES_ALLCAPS
AMPI_ALLTOALLV
#else
ampi_alltoallv_
#endif
  (void *sendbuf, int *sendcounts, int *sdispls,
   int *sendtype, void *recvbuf, int *recvcounts,
   int *rdispls, int *recvtype, int *comm, int *ierr)
{
  *ierr = AMPI_Alltoallv(sendbuf, sendcounts, sdispls, *sendtype, recvbuf,
                         recvcounts, rdispls, *recvtype, *comm);
}

extern "C" void 
#if CMK_FORTRAN_USES_ALLCAPS
AMPI_ALLTOALL
#else
ampi_alltoall_
#endif
  (void *sendbuf, int *sendcount, int *sendtype,
   void *recvbuf, int *recvcount, int *recvtype,
   int *comm, int *ierr)
{
  *ierr = AMPI_Alltoall(sendbuf, *sendcount, *sendtype, recvbuf, *recvcount,
                        *recvtype, *comm);
}

extern "C" void 
#if CMK_FORTRAN_USES_ALLCAPS
AMPI_COMM_DUP
#else
ampi_comm_dup_
#endif
  (int *comm, int *newcomm, int *ierr)
{
  *newcomm = *comm;
  *ierr = 0;
}

extern "C" void 
#if CMK_FORTRAN_USES_ALLCAPS
AMPI_COMM_FREE
#else
ampi_comm_free_
#endif
  (int *comm, int *ierr)
{
  *ierr = 0;
}

extern "C" void 
#if CMK_FORTRAN_USES_ALLCAPS
AMPI_ABORT
#else
ampi_abort_
#endif
  (int *comm, int *errorcode, int *ierr)
{
  *ierr = AMPI_Abort(*comm, *errorcode);
}

extern "C" void 
#if CMK_FORTRAN_USES_ALLCAPS
AMPI_PRINT
#else
ampi_print_
#endif
  (char *str, int len)
{
  char *tmpstr = new char[len+1];
  memcpy(tmpstr,str,len);
  tmpstr[len] = '\0';
  AMPI_Print(tmpstr);
  delete[] tmpstr;
}

extern "C" void
#if CMK_FORTRAN_USES_ALLCAPS
AMPI_MIGRATE
#else
ampi_migrate_
#endif
  (void)
{
  AMPI_Migrate();
}

extern "C" int
#if CMK_FORTRAN_USES_ALLCAPS
AMPI_REGISTER
#else
ampi_register_
#endif
  (void *d, AMPI_PupFn f)
{
  return AMPI_Register(d,f);
}

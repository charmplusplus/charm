#include "ampi.h"
#include "ampiimpl.h"

FDECL {

#define ampi_init_universe FTN_NAME( AMPI_INIT_UNIVERSE , ampi_init_universe )
#define ampi_comm_rank FTN_NAME( AMPI_COMM_RANK , ampi_comm_rank )
#define ampi_comm_size FTN_NAME( AMPI_COMM_SIZE , ampi_comm_size )
#define ampi_finalize FTN_NAME( AMPI_FINALIZE , ampi_finalize )
#define ampi_send FTN_NAME( AMPI_SEND , ampi_send )
#define ampi_ssend FTN_NAME( AMPI_SSEND , ampi_ssend )
#define ampi_recv FTN_NAME( AMPI_RECV , ampi_recv )
#define ampi_probe FTN_NAME( AMPI_PROBE , ampi_probe )
#define ampi_iprobe FTN_NAME( AMPI_IPROBE , ampi_iprobe )
#define ampi_isend FTN_NAME( AMPI_ISEND , ampi_isend )
#define ampi_issend FTN_NAME( AMPI_ISSEND , ampi_issend )
#define ampi_irecv FTN_NAME( AMPI_IRECV , ampi_irecv )
#define ampi_sendrecv FTN_NAME( AMPI_SENDRECV , ampi_sendrecv )
#define ampi_barrier FTN_NAME( AMPI_BARRIER , ampi_barrier )
#define ampi_bcast FTN_NAME( AMPI_BCAST , ampi_bcast )
#define ampi_reduce FTN_NAME( AMPI_REDUCE , ampi_reduce )
#define ampi_allreduce FTN_NAME( AMPI_ALLREDUCE , ampi_allreduce )
#define ampi_wtime FTN_NAME( AMPI_WTIME , ampi_wtime )
#define ampi_start FTN_NAME( AMPI_START , ampi_start )
#define ampi_waitall FTN_NAME( AMPI_WAITALL , ampi_waitall )
#define ampi_waitany FTN_NAME( AMPI_WAITANY , ampi_waitany )
#define ampi_wait FTN_NAME( AMPI_WAIT , ampi_wait )
#define ampi_testall FTN_NAME( AMPI_TESTALL , ampi_testall )
#define ampi_test FTN_NAME( AMPI_TEST , ampi_test )
#define ampi_send_init FTN_NAME( AMPI_SEND_INIT , ampi_send_init )
#define ampi_recv_init FTN_NAME( AMPI_RECV_INIT , ampi_recv_init )
#define ampi_type_contiguous FTN_NAME( AMPI_TYPE_CONTIGUOUS , ampi_type_contiguous )
#define ampi_type_vector FTN_NAME( AMPI_TYPE_VECTOR , ampi_type_vector )
#define ampi_type_hvector FTN_NAME( AMPI_TYPE_HVECTOR , ampi_type_hvector )
#define ampi_type_indexed FTN_NAME( AMPI_TYPE_INDEXED , ampi_type_indexed )
#define ampi_type_hindexed FTN_NAME( AMPI_TYPE_HINDEXED , ampi_type_hindexed )
#define ampi_type_struct FTN_NAME( AMPI_TYPE_STRUCT , ampi_type_struct )
#define ampi_type_commit FTN_NAME( AMPI_TYPE_COMMIT , ampi_type_commit )
#define ampi_type_free FTN_NAME( AMPI_TYPE_FREE , ampi_type_free )
#define ampi_type_extent FTN_NAME( AMPI_TYPE_EXTENT , ampi_type_extent )
#define ampi_type_size FTN_NAME( AMPI_TYPE_SIZE , ampi_type_size )
#define ampi_pack FTN_NAME( AMPI_PACK , ampi_pack )
#define ampi_unpack FTN_NAME( AMPI_UNPACK , ampi_unpack )
#define ampi_pack_size FTN_NAME( AMPI_PACK_SIZE , ampi_pack_size )
#define ampi_allgatherv FTN_NAME( AMPI_ALLGATHERV , ampi_allgatherv )
#define ampi_allgather FTN_NAME( AMPI_ALLGATHER , ampi_allgather )
#define ampi_gatherv FTN_NAME( AMPI_GATHERV , ampi_gatherv )
#define ampi_gather FTN_NAME( AMPI_GATHER , ampi_gather )
#define ampi_alltoallv FTN_NAME( AMPI_ALLTOALLV , ampi_alltoallv )
#define ampi_alltoall FTN_NAME( AMPI_ALLTOALL , ampi_alltoall )
#define ampi_comm_dup FTN_NAME( AMPI_COMM_DUP , ampi_comm_dup )
#define ampi_comm_free FTN_NAME( AMPI_COMM_FREE , ampi_comm_free )
#define ampi_abort FTN_NAME( AMPI_ABORT , ampi_abort )
#define ampi_print FTN_NAME( AMPI_PRINT , ampi_print )
#define ampi_migrate FTN_NAME( AMPI_MIGRATE , ampi_migrate )
#define ampi_register FTN_NAME( AMPI_REGISTER , ampi_register )
#define ampi_get_count FTN_NAME( AMPI_GET_COUNT , ampi_get_count )

extern int AMPI_COMM_UNIVERSE[AMPI_MAX_COMM];

void ampi_init_universe(int *unicomm)
{
  for(int i=0;i<ampi_ncomms; i++)
  {
    unicomm[i] = AMPI_COMM_UNIVERSE[i];
  }
}

void ampi_comm_rank(int *comm, int *rank, int *ierr)
{
  *ierr = AMPI_Comm_rank(*comm, rank);
}

void ampi_comm_size(int *comm, int *size, int *ierr)
{
  *ierr = AMPI_Comm_size(*comm, size);
}

void ampi_finalize(int *ierr)
{
  *ierr = AMPI_Finalize();
}

void ampi_send(void *msg, int *count, int *type, int *dest, 
  int *tag, int *comm, int *ierr)
{
  *ierr = AMPI_Send(msg, *count, *type, *dest, *tag, *comm);
}

void ampi_ssend(void *msg, int *count, int *type, int *dest, 
  int *tag, int *comm, int *ierr)
{
  *ierr = AMPI_Ssend(msg, *count, *type, *dest, *tag, *comm);
}

void ampi_recv(void *msg, int *count, int *type, int *src, 
  int *tag, int *comm, int *status, int *ierr)
{
  *ierr = AMPI_Recv(msg, *count, *type, *src, *tag, *comm, 
                    (AMPI_Status*) status);
}

void ampi_probe(int *src, int *tag, int *comm, int *status, int *ierr)
{
  *ierr = AMPI_Probe(*src, *tag, *comm, (AMPI_Status*) status);
}

void ampi_iprobe(int *src,int *tag,int *comm,int *flag,int *status,int *ierr)
{
  *ierr = AMPI_Iprobe(*src, *tag, *comm, flag, (AMPI_Status*) status);
}

void ampi_sendrecv(void *sndbuf, int *sndcount, int *sndtype, 
  int *dest, int *sndtag, void *rcvbuf, 
  int *rcvcount, int *rcvtype, int *src, 
  int *rcvtag, int *comm, int *status, int *ierr)
{
  *ierr = AMPI_Sendrecv(sndbuf, *sndcount, *sndtype, *dest, *sndtag,
                        rcvbuf, *rcvcount, *rcvtype, *src, *rcvtag,
			*comm, (AMPI_Status*) status);
}

void ampi_barrier(int *comm, int *ierr)
{
  *ierr = AMPI_Barrier(*comm);
}

void ampi_bcast(void *buf, int *count, int *type, int *root, int *comm, 
   int *ierr)
{
  *ierr = AMPI_Bcast(buf, *count, *type, *root, *comm);
}

void ampi_reduce(void *inbuf, void *outbuf, int *count, int *type,
   int *op, int *root, int *comm, int *ierr)
{
  *ierr = AMPI_Reduce(inbuf, outbuf, *count, *type, *op, *root, *comm);
}

void ampi_allreduce(void *inbuf,void *outbuf,int *count,int *type,
   int *op, int *comm, int *ierr)
{
  *ierr = AMPI_Allreduce(inbuf, outbuf, *count, *type, *op, *comm);
}

double ampi_wtime(void)
{
  return AMPI_Wtime();
}

void ampi_start(int *reqnum, int *ierr)
{
  *ierr = AMPI_Start((AMPI_Request*) reqnum);
}

void ampi_waitall(int *count, int *request, int *status, int *ierr)
{
  *ierr = AMPI_Waitall(*count, (AMPI_Request*) request, (AMPI_Status*) status);
}

void ampi_waitany(int *count, int *request, int *index, int *status, int *ierr)
{
  *ierr = AMPI_Waitany(*count, (AMPI_Request*) request, index, 
                       (AMPI_Status*) status);
}

void ampi_wait(int *request, int *status, int *ierr)
{
  *ierr = AMPI_Wait((AMPI_Request*) request, (AMPI_Status*) status);
}

void ampi_testall(int *count, int *request, int *flag, int *status, int *ierr)
{
  *ierr = AMPI_Testall(*count, (AMPI_Request*) request, flag, 
      (AMPI_Status*) status);
}

void ampi_test(int *request, int *flag, int *status, int *ierr)
{
  *ierr = AMPI_Test((AMPI_Request*) request, flag, (AMPI_Status*) status);
}

void ampi_recv_init(void *buf, int *count, int *type, int *srcpe,
   int *tag, int *comm, int *req, int *ierr)
{
  *ierr = AMPI_Recv_init(buf,*count,*type,*srcpe,*tag,*comm,(AMPI_Request*)req);
}

void ampi_send_init(void *buf, int *count, int *type, int *destpe,
   int *tag, int *comm, int *req, int *ierr)
{
  *ierr = AMPI_Send_init(buf,*count,*type,*destpe,*tag,*comm,
                         (AMPI_Request*)req);
}

void ampi_type_contiguous(int *count, int *oldtype, int *newtype, int *ierr)
{
  *ierr = AMPI_Type_contiguous(*count, *oldtype, newtype);
}

void ampi_type_vector(int *count, int *blocklength, int *stride, 
   int *oldtype, int*  newtype, int *ierr)
{
  *ierr = AMPI_Type_vector(*count, *blocklength, *stride, *oldtype, newtype);
}

void ampi_type_hvector(int *count, int *blocklength, int *stride, 
   int* oldtype, int* newtype, int* ierr)
{
  *ierr = AMPI_Type_hvector(*count, *blocklength, *stride, *oldtype, newtype);
}

void ampi_type_indexed(int *count, int* arrBlength, int* arrDisp, 
   int* oldtype, int*  newtype, int* ierr)
{
  *ierr = AMPI_Type_indexed(*count, arrBlength, arrDisp, *oldtype, newtype);
}

void ampi_type_hindexed(int* count, int* arrBlength, int* arrDisp, 
   int* oldtype, int* newtype, int* ierr)
{
  *ierr = AMPI_Type_hindexed(*count, arrBlength, arrDisp, *oldtype, newtype);
}

void ampi_type_struct(int* count, int* arrBlength, int* arrDisp, 
   int* oldtype, int* newtype, int* ierr)
{
  *ierr = AMPI_Type_struct(*count, arrBlength, arrDisp, oldtype, newtype);
}


void ampi_type_commit(int *type, int *ierr)
{
  *ierr = AMPI_Type_commit(type);
}

void ampi_type_free(int *type, int *ierr)
{
  *ierr = AMPI_Type_free(type);
}

void  ampi_type_extent(int* type, int* extent, int* ierr)
{
  *ierr = AMPI_Type_extent(*type, extent);
}

void  ampi_type_size(int* type, int* size, int* ierr)
{
  *ierr = AMPI_Type_size(*type, size);
}

void ampi_pack(void *inbuf, int *incount, int *datatype, void *outbuf, 
    int *outsize, int *position, int *comm, int *ierr)
{
  *ierr = AMPI_Pack(inbuf, *incount, (AMPI_Datatype)*datatype, outbuf, 
      *outsize, position, *comm);
}

void ampi_unpack(void *inbuf, int *insize, int *position, void *outbuf, 
    int *outcount, int *datatype, int *comm, int *ierr)
{
  *ierr = AMPI_Unpack(inbuf, *insize, position, outbuf, *outcount, 
      (AMPI_Datatype) *datatype, (AMPI_Comm) *comm);
}

void ampi_pack_size(int *incount, int *datatype, int *comm, int *size, int *ierr)
{
  *ierr = AMPI_Pack_size(*incount, (AMPI_Datatype) *datatype, *comm, size);
}

void ampi_isend(void *buf, int *count, int *datatype, int *dest,
   int *tag, int *comm, int *request, int *ierr)
{
  *ierr = AMPI_Isend(buf, *count, *datatype, *dest, *tag, *comm, request);
}

void ampi_issend(void *buf, int *count, int *datatype, int *dest,
   int *tag, int *comm, int *request, int *ierr)
{
  *ierr = AMPI_Issend(buf, *count, *datatype, *dest, *tag, *comm, request);
}

void ampi_irecv(void *buf, int *count, int *datatype, int *src,
   int *tag, int *comm, int *request, int *ierr)
{
  *ierr = AMPI_Irecv(buf, *count, *datatype, *src, *tag, *comm, request);
}

void ampi_allgatherv(void *sendbuf, int *sendcount, int *sendtype,
   void *recvbuf, int *recvcounts, int *displs,
   int *recvtype, int *comm, int *ierr)
{
  *ierr = AMPI_Allgatherv(sendbuf, *sendcount, *sendtype, recvbuf, recvcounts,
                          displs, *recvtype, *comm);
}

void ampi_allgather(void *sendbuf, int *sendcount, int *sendtype,
   void *recvbuf, int *recvcount, int *recvtype,
   int *comm, int *ierr)
{
  *ierr = AMPI_Allgather(sendbuf, *sendcount, *sendtype, recvbuf, *recvcount,
                         *recvtype, *comm);
}

void ampi_gatherv(void *sendbuf, int *sendcount, int *sendtype,
   void *recvbuf, int *recvcounts, int *displs,
   int *recvtype, int *root, int *comm, int *ierr)
{
  *ierr = AMPI_Gatherv(sendbuf, *sendcount, *sendtype, recvbuf, recvcounts,
                       displs, *recvtype, *root, *comm);
}

void ampi_gather(void *sendbuf, int *sendcount, int *sendtype,
   void *recvbuf, int *recvcount, int *recvtype,
   int *root, int *comm, int *ierr)
{
  *ierr = AMPI_Gather(sendbuf, *sendcount, *sendtype, recvbuf, *recvcount, 
                      *recvtype, *root, *comm);
}

void ampi_alltoallv(void *sendbuf, int *sendcounts, int *sdispls,
   int *sendtype, void *recvbuf, int *recvcounts,
   int *rdispls, int *recvtype, int *comm, int *ierr)
{
  *ierr = AMPI_Alltoallv(sendbuf, sendcounts, sdispls, *sendtype, recvbuf,
                         recvcounts, rdispls, *recvtype, *comm);
}

void ampi_alltoall(void *sendbuf, int *sendcount, int *sendtype,
   void *recvbuf, int *recvcount, int *recvtype,
   int *comm, int *ierr)
{
  *ierr = AMPI_Alltoall(sendbuf, *sendcount, *sendtype, recvbuf, *recvcount,
                        *recvtype, *comm);
}

void ampi_comm_dup(int *comm, int *newcomm, int *ierr)
{
  *newcomm = *comm;
  *ierr = 0;
}

void ampi_comm_free(int *comm, int *ierr)
{
  *ierr = 0;
}

void ampi_abort(int *comm, int *errorcode, int *ierr)
{
  *ierr = AMPI_Abort(*comm, *errorcode);
}

void ampi_get_count(int *sts, int *dtype, int *cnt, int *ierr)
{
  *ierr = AMPI_Get_count((AMPI_Status*) sts, *dtype, cnt);
}

void ampi_print(char *str, int len)
{
  char *tmpstr = new char[len+1];
  memcpy(tmpstr,str,len);
  tmpstr[len] = '\0';
  AMPI_Print(tmpstr);
  delete[] tmpstr;
}

void ampi_migrate(void)
{
  AMPI_Migrate();
}

int ampi_register(void *d, AMPI_PupFn f)
{
  return AMPI_Register(d,f);
}

} // extern "C"

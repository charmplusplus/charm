#include "ampi.h"
#include "ampiimpl.h"

extern "C" {

#if CMK_FORTRAN_USES_ALLCAPS

#define ampi_init_universe           AMPI_INIT_UNIVERSE
#define ampi_comm_rank               AMPI_COMM_RANK
#define ampi_comm_size               AMPI_COMM_SIZE
#define ampi_finalize                AMPI_FINALIZE
#define ampi_send                    AMPI_SEND
#define ampi_recv                    AMPI_RECV
#define ampi_probe                   AMPI_PROBE
#define ampi_iprobe                  AMPI_IPROBE
#define ampi_isend                   AMPI_ISEND
#define ampi_irecv                   AMPI_IRECV
#define ampi_sendrecv                AMPI_SENDRECV
#define ampi_barrier                 AMPI_BARRIER
#define ampi_bcast                   AMPI_BCAST
#define ampi_reduce                  AMPI_REDUCE
#define ampi_allreduce               AMPI_ALLREDUCE
#define ampi_wtime                   AMPI_WTIME
#define ampi_start                   AMPI_START
#define ampi_waitall                 AMPI_WAITALL
#define ampi_testall                 AMPI_TESTALL
#define ampi_test                    AMPI_TEST
#define ampi_send_init               AMPI_SEND_INIT
#define ampi_recv_init               AMPI_RECV_INIT
#define ampi_type_contiguous         AMPI_TYPE_CONTIGUOUS
#define ampi_type_vector             AMPI_TYPE_VECTOR
#define ampi_type_hvector            AMPI_TYPE_HVECTOR
#define ampi_type_indexed            AMPI_TYPE_INDEXED
#define ampi_type_hindexed           AMPI_TYPE_HINDEXED
#define ampi_type_struct             AMPI_TYPE_STRUCT
#define ampi_type_commit             AMPI_TYPE_COMMIT
#define ampi_type_free               AMPI_TYPE_FREE
#define ampi_type_extent             AMPI_TYPE_EXTENT
#define ampi_type_size               AMPI_TYPE_SIZE
#define ampi_pack                    AMPI_PACK
#define ampi_unpack                  AMPI_UNPACK
#define ampi_pack_size               AMPI_PACK_SIZE
#define ampi_allgatherv              AMPI_ALLGATHERV
#define ampi_allgather               AMPI_ALLGATHER
#define ampi_gatherv                 AMPI_GATHERV
#define ampi_gather                  AMPI_GATHER
#define ampi_alltoallv               AMPI_ALLTOALLV
#define ampi_alltoall                AMPI_ALLTOALL
#define ampi_comm_dup                AMPI_COMM_DUP
#define ampi_comm_free               AMPI_COMM_FREE
#define ampi_abort                   AMPI_ABORT
#define ampi_print                   AMPI_PRINT
#define ampi_migrate                 AMPI_MIGRATE
#define ampi_register                AMPI_REGISTER
#define ampi_get_count               AMPI_GET_COUNT

#else

#define ampi_init_universe           FNAME(ampi_init_universe)
#define ampi_comm_rank               FNAME(ampi_comm_rank)
#define ampi_comm_size               FNAME(ampi_comm_size)
#define ampi_finalize                FNAME(ampi_finalize)
#define ampi_send                    FNAME(ampi_send)
#define ampi_recv                    FNAME(ampi_recv)
#define ampi_probe                   FNAME(ampi_probe)
#define ampi_iprobe                  FNAME(ampi_iprobe)
#define ampi_isend                   FNAME(ampi_isend)
#define ampi_irecv                   FNAME(ampi_irecv)
#define ampi_sendrecv                FNAME(ampi_sendrecv)
#define ampi_barrier                 FNAME(ampi_barrier)
#define ampi_bcast                   FNAME(ampi_bcast)
#define ampi_reduce                  FNAME(ampi_reduce)
#define ampi_allreduce               FNAME(ampi_allreduce)
#define ampi_wtime                   FNAME(ampi_wtime)
#define ampi_start                   FNAME(ampi_start)
#define ampi_waitall                 FNAME(ampi_waitall)
#define ampi_testall                 FNAME(ampi_testall)
#define ampi_test                    FNAME(ampi_test)
#define ampi_send_init               FNAME(ampi_send_init)
#define ampi_recv_init               FNAME(ampi_recv_init)
#define ampi_type_contiguous         FNAME(ampi_type_contiguous)
#define ampi_type_vector             FNAME(ampi_type_vector)
#define ampi_type_hvector            FNAME(ampi_type_hvector)
#define ampi_type_indexed            FNAME(ampi_type_indexed)
#define ampi_type_hindexed           FNAME(ampi_type_hindexed)
#define ampi_type_struct             FNAME(ampi_type_struct)
#define ampi_type_commit             FNAME(ampi_type_commit)
#define ampi_type_free               FNAME(ampi_type_free)
#define ampi_type_extent             FNAME(ampi_type_extent)
#define ampi_type_size               FNAME(ampi_type_size)
#define ampi_pack                    FNAME(ampi_pack)
#define ampi_unpack                  FNAME(ampi_unpack)
#define ampi_pack_size               FNAME(ampi_pack_size)
#define ampi_allgatherv              FNAME(ampi_allgatherv)
#define ampi_allgather               FNAME(ampi_allgather)
#define ampi_gatherv                 FNAME(ampi_gatherv)
#define ampi_gather                  FNAME(ampi_gather)
#define ampi_alltoallv               FNAME(ampi_alltoallv)
#define ampi_alltoall                FNAME(ampi_alltoall)
#define ampi_comm_dup                FNAME(ampi_comm_dup)
#define ampi_comm_free               FNAME(ampi_comm_free)
#define ampi_abort                   FNAME(ampi_abort)
#define ampi_print                   FNAME(ampi_print)
#define ampi_migrate                 FNAME(ampi_migrate)
#define ampi_register                FNAME(ampi_register)
#define ampi_get_count               FNAME(ampi_get_count)

#endif

extern int AMPI_COMM_UNIVERSE[AMPI_MAX_COMM];

void  ampi_init_universe(int *unicomm)
{
  for(int i=0;i<ampimain::ncomms; i++)
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

void ampi_testall(int *count, int *request, int *flag, int *status, int *ierr)
{
  *ierr = AMPI_TestAll(*count, (AMPI_Request*) request, flag, 
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
  *ierr = AMPI_Get_count(sts, *dtype, cnt);
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

#include "ampi.h"
#include "ampiimpl.h"

FDECL {

#define mpi_init_universe FTN_NAME( MPI_INIT_UNIVERSE , mpi_init_universe )
#define mpi_comm_rank FTN_NAME( MPI_COMM_RANK , mpi_comm_rank )
#define mpi_comm_size FTN_NAME( MPI_COMM_SIZE , mpi_comm_size )
#define mpi_finalize FTN_NAME( MPI_FINALIZE , mpi_finalize )
#define mpi_send FTN_NAME( MPI_SEND , mpi_send )
#define mpi_ssend FTN_NAME( MPI_SSEND , mpi_ssend )
#define mpi_recv FTN_NAME( MPI_RECV , mpi_recv )
#define mpi_probe FTN_NAME( MPI_PROBE , mpi_probe )
#define mpi_iprobe FTN_NAME( MPI_IPROBE , mpi_iprobe )
#define mpi_isend FTN_NAME( MPI_ISEND , mpi_isend )
#define mpi_issend FTN_NAME( MPI_ISSEND , mpi_issend )
#define mpi_irecv FTN_NAME( MPI_IRECV , mpi_irecv )
#define mpi_sendrecv FTN_NAME( MPI_SENDRECV , mpi_sendrecv )
#define mpi_barrier FTN_NAME( MPI_BARRIER , mpi_barrier )
#define mpi_bcast FTN_NAME( MPI_BCAST , mpi_bcast )
#define mpi_reduce FTN_NAME( MPI_REDUCE , mpi_reduce )
#define mpi_allreduce FTN_NAME( MPI_ALLREDUCE , mpi_allreduce )
#define mpi_wtime FTN_NAME( MPI_WTIME , mpi_wtime )
#define mpi_start FTN_NAME( MPI_START , mpi_start )
#define mpi_waitall FTN_NAME( MPI_WAITALL , mpi_waitall )
#define mpi_waitany FTN_NAME( MPI_WAITANY , mpi_waitany )
#define mpi_wait FTN_NAME( MPI_WAIT , mpi_wait )
#define mpi_testall FTN_NAME( MPI_TESTALL , mpi_testall )
#define mpi_test FTN_NAME( MPI_TEST , mpi_test )
#define mpi_send_init FTN_NAME( MPI_SEND_INIT , mpi_send_init )
#define mpi_recv_init FTN_NAME( MPI_RECV_INIT , mpi_recv_init )
#define mpi_type_contiguous FTN_NAME( MPI_TYPE_CONTIGUOUS , mpi_type_contiguous )
#define mpi_type_vector FTN_NAME( MPI_TYPE_VECTOR , mpi_type_vector )
#define mpi_type_hvector FTN_NAME( MPI_TYPE_HVECTOR , mpi_type_hvector )
#define mpi_type_indexed FTN_NAME( MPI_TYPE_INDEXED , mpi_type_indexed )
#define mpi_type_hindexed FTN_NAME( MPI_TYPE_HINDEXED , mpi_type_hindexed )
#define mpi_type_struct FTN_NAME( MPI_TYPE_STRUCT , mpi_type_struct )
#define mpi_type_commit FTN_NAME( MPI_TYPE_COMMIT , mpi_type_commit )
#define mpi_type_free FTN_NAME( MPI_TYPE_FREE , mpi_type_free )
#define mpi_type_extent FTN_NAME( MPI_TYPE_EXTENT , mpi_type_extent )
#define mpi_type_size FTN_NAME( MPI_TYPE_SIZE , mpi_type_size )
#define mpi_pack FTN_NAME( MPI_PACK , mpi_pack )
#define mpi_unpack FTN_NAME( MPI_UNPACK , mpi_unpack )
#define mpi_pack_size FTN_NAME( MPI_PACK_SIZE , mpi_pack_size )
#define mpi_allgatherv FTN_NAME( MPI_ALLGATHERV , mpi_allgatherv )
#define mpi_allgather FTN_NAME( MPI_ALLGATHER , mpi_allgather )
#define mpi_gatherv FTN_NAME( MPI_GATHERV , mpi_gatherv )
#define mpi_gather FTN_NAME( MPI_GATHER , mpi_gather )
#define mpi_scatterv FTN_NAME( MPI_SCATTERV , mpi_scatterv )
#define mpi_scatter FTN_NAME( MPI_SCATTER , mpi_scatter )
#define mpi_alltoallv FTN_NAME( MPI_ALLTOALLV , mpi_alltoallv )
#define mpi_alltoall FTN_NAME( MPI_ALLTOALL , mpi_alltoall )
#define mpi_comm_dup FTN_NAME( MPI_COMM_DUP , mpi_comm_dup )
#define mpi_comm_free FTN_NAME( MPI_COMM_FREE , mpi_comm_free )
#define mpi_comm_group FTN_NAME( MPI_COMM_GROUP, mpi_comm_group)
#define mpi_group_size FTN_NAME( MPI_GROUP_SIZE, mpi_group_size)
#define mpi_group_rank FTN_NAME( MPI_GROUP_RANK, mpi_group_rank)
#define mpi_group_translate_ranks FTN_NAME(MPI_GROUP_TRANSLATE_RANKS, mpi_group_translate_ranks)
#define mpi_group_compare FTN_NAME(MPI_GROUP_COMPARE, mpi_group_compare)
#define mpi_group_union FTN_NAME(MPI_GROUP_UNION, mpi_group_union)
#define mpi_group_intersection FTN_NAME(MPI_GROUP_INTERSECTION, mpi_group_intersection)
#define mpi_group_difference FTN_NAME(MPI_GROUP_DIFFERENCE, mpi_group_difference)
#define mpi_group_incl FTN_NAME(MPI_GROUP_INCL, mpi_group_incl)
#define mpi_group_excl FTN_NAME(MPI_GROUP_EXCL, mpi_group_excl)
#define mpi_group_range_incl FTN_NAME(MPI_GROUP_RANGE_INCL, mpi_group_range_incl)
#define mpi_group_range_excl FTN_NAME(MPI_GROUP_RANGE_EXCL, mpi_group_range_excl)
#define mpi_group_free FTN_NAME(MPI_GROUP_FREE, mpi_group_free)
#define mpi_comm_create FTN_NAME(MPI_COMM_CREATE, mpi_comm_create)
#define mpi_abort FTN_NAME( MPI_ABORT , mpi_abort )
#define mpi_print FTN_NAME( MPI_PRINT , mpi_print )
#define mpi_migrate FTN_NAME( MPI_MIGRATE , mpi_migrate )
#define mpi_register FTN_NAME( MPI_REGISTER , mpi_register )
#define mpi_get_count FTN_NAME( MPI_GET_COUNT , mpi_get_count )

void mpi_init_universe(int *unicomm)
{
  AMPIAPI("mpi_init_universe");
  for(int i=0;i<mpi_nworlds; i++)
  {
    unicomm[i] = MPI_COMM_UNIVERSE[i];
  }
}

void mpi_comm_rank(int *comm, int *rank, int *ierr)
{
  *ierr = MPI_Comm_rank(*comm, rank);
}

void mpi_comm_size(int *comm, int *size, int *ierr)
{
  *ierr = MPI_Comm_size(*comm, size);
}

void mpi_finalize(int *ierr)
{
  *ierr = MPI_Finalize();
}

void mpi_send(void *msg, int *count, int *type, int *dest, 
  int *tag, int *comm, int *ierr)
{
  *ierr = MPI_Send(msg, *count, *type, *dest, *tag, *comm);
}

void mpi_ssend(void *msg, int *count, int *type, int *dest, 
  int *tag, int *comm, int *ierr)
{
  *ierr = MPI_Ssend(msg, *count, *type, *dest, *tag, *comm);
}

void mpi_recv(void *msg, int *count, int *type, int *src,
  int *tag, int *comm, int *status, int *ierr)
{
  *ierr = MPI_Recv(msg, *count, *type, *src, *tag, *comm, 
                    (MPI_Status*) status);
}

void mpi_probe(int *src, int *tag, int *comm, int *status, int *ierr)
{
  *ierr = MPI_Probe(*src, *tag, *comm, (MPI_Status*) status);
}

void mpi_iprobe(int *src,int *tag,int *comm,int *flag,int *status,int *ierr)
{
  *ierr = MPI_Iprobe(*src, *tag, *comm, flag, (MPI_Status*) status);
}

void mpi_sendrecv(void *sndbuf, int *sndcount, int *sndtype, 
  int *dest, int *sndtag, void *rcvbuf, 
  int *rcvcount, int *rcvtype, int *src, 
  int *rcvtag, int *comm, int *status, int *ierr)
{
  *ierr = MPI_Sendrecv(sndbuf, *sndcount, *sndtype, *dest, *sndtag,
                        rcvbuf, *rcvcount, *rcvtype, *src, *rcvtag,
			*comm, (MPI_Status*) status);
}

void mpi_barrier(int *comm, int *ierr)
{
  *ierr = MPI_Barrier(*comm);
}

void mpi_bcast(void *buf, int *count, int *type, int *root, int *comm, 
   int *ierr)
{
  *ierr = MPI_Bcast(buf, *count, *type, *root, *comm);
}

void mpi_reduce(void *inbuf, void *outbuf, int *count, int *type,
   int *op, int *root, int *comm, int *ierr)
{
  *ierr = MPI_Reduce(inbuf, outbuf, *count, *type, *op, *root, *comm);
}

void mpi_allreduce(void *inbuf,void *outbuf,int *count,int *type,
   int *op, int *comm, int *ierr)
{
  *ierr = MPI_Allreduce(inbuf, outbuf, *count, *type, *op, *comm);
}

double mpi_wtime(void)
{
  return MPI_Wtime();
}

void mpi_start(int *reqnum, int *ierr)
{
  *ierr = MPI_Start((MPI_Request*) reqnum);
}

void mpi_waitall(int *count, int *request, int *status, int *ierr)
{
  *ierr = MPI_Waitall(*count, (MPI_Request*) request, (MPI_Status*) status);
}

void mpi_waitany(int *count, int *request, int *index, int *status, int *ierr)
{
  *ierr = MPI_Waitany(*count, (MPI_Request*) request, index, 
                       (MPI_Status*) status);
}

void mpi_wait(int *request, int *status, int *ierr)
{
  *ierr = MPI_Wait((MPI_Request*) request, (MPI_Status*) status);
}

void mpi_testall(int *count, int *request, int *flag, int *status, int *ierr)
{
  *ierr = MPI_Testall(*count, (MPI_Request*) request, flag, 
      (MPI_Status*) status);
}

void mpi_test(int *request, int *flag, int *status, int *ierr)
{
  *ierr = MPI_Test((MPI_Request*) request, flag, (MPI_Status*) status);
}

void mpi_recv_init(void *buf, int *count, int *type, int *srcpe,
   int *tag, int *comm, int *req, int *ierr)
{
  *ierr = MPI_Recv_init(buf,*count,*type,*srcpe,*tag,*comm,(MPI_Request*)req);
}

void mpi_send_init(void *buf, int *count, int *type, int *destpe,
   int *tag, int *comm, int *req, int *ierr)
{
  *ierr = MPI_Send_init(buf,*count,*type,*destpe,*tag,*comm,
                         (MPI_Request*)req);
}

void mpi_type_contiguous(int *count, int *oldtype, int *newtype, int *ierr)
{
  *ierr = MPI_Type_contiguous(*count, *oldtype, newtype);
}

void mpi_type_vector(int *count, int *blocklength, int *stride,
   int *oldtype, int*  newtype, int *ierr)
{
  *ierr = MPI_Type_vector(*count, *blocklength, *stride, *oldtype, newtype);
}

void mpi_type_hvector(int *count, int *blocklength, int *stride,
   int* oldtype, int* newtype, int* ierr)
{
  *ierr = MPI_Type_hvector(*count, *blocklength, *stride, *oldtype, newtype);
}

void mpi_type_indexed(int *count, int* arrBlength, int* arrDisp,
   int* oldtype, int*  newtype, int* ierr)
{
  *ierr = MPI_Type_indexed(*count, arrBlength, arrDisp, *oldtype, newtype);
}

void mpi_type_hindexed(int* count, int* arrBlength, int* arrDisp,
   int* oldtype, int* newtype, int* ierr)
{
  *ierr = MPI_Type_hindexed(*count, arrBlength, arrDisp, *oldtype, newtype);
}

void mpi_type_struct(int* count, int* arrBlength, int* arrDisp,
   int* oldtype, int* newtype, int* ierr)
{
  *ierr = MPI_Type_struct(*count, arrBlength, arrDisp, oldtype, newtype);
}


void mpi_type_commit(int *type, int *ierr)
{
  *ierr = MPI_Type_commit(type);
}

void mpi_type_free(int *type, int *ierr)
{
  *ierr = MPI_Type_free(type);
}

void  mpi_type_extent(int* type, int* extent, int* ierr)
{
  *ierr = MPI_Type_extent(*type, extent);
}

void  mpi_type_size(int* type, int* size, int* ierr)
{
  *ierr = MPI_Type_size(*type, size);
}

void mpi_pack(void *inbuf, int *incount, int *datatype, void *outbuf,
    int *outsize, int *position, int *comm, int *ierr)
{
  *ierr = MPI_Pack(inbuf, *incount, (MPI_Datatype)*datatype, outbuf,
      *outsize, position, *comm);
}

void mpi_unpack(void *inbuf, int *insize, int *position, void *outbuf,
    int *outcount, int *datatype, int *comm, int *ierr)
{
  *ierr = MPI_Unpack(inbuf, *insize, position, outbuf, *outcount,
      (MPI_Datatype) *datatype, (MPI_Comm) *comm);
}

void mpi_pack_size(int *incount, int *datatype, int *comm, int *size, int *ierr)
{
  *ierr = MPI_Pack_size(*incount, (MPI_Datatype) *datatype, *comm, size);
}

void mpi_isend(void *buf, int *count, int *datatype, int *dest,
   int *tag, int *comm, int *request, int *ierr)
{
  *ierr = MPI_Isend(buf, *count, *datatype, *dest, *tag, *comm, request);
}

void mpi_issend(void *buf, int *count, int *datatype, int *dest,
   int *tag, int *comm, int *request, int *ierr)
{
  *ierr = MPI_Issend(buf, *count, *datatype, *dest, *tag, *comm, request);
}

void mpi_irecv(void *buf, int *count, int *datatype, int *src,
   int *tag, int *comm, int *request, int *ierr)
{
  *ierr = MPI_Irecv(buf, *count, *datatype, *src, *tag, *comm, request);
}

void mpi_allgatherv(void *sendbuf, int *sendcount, int *sendtype,
   void *recvbuf, int *recvcounts, int *displs,
   int *recvtype, int *comm, int *ierr)
{
  *ierr = MPI_Allgatherv(sendbuf, *sendcount, *sendtype, recvbuf, recvcounts,
                          displs, *recvtype, *comm);
}

void mpi_allgather(void *sendbuf, int *sendcount, int *sendtype,
   void *recvbuf, int *recvcount, int *recvtype,
   int *comm, int *ierr)
{
  *ierr = MPI_Allgather(sendbuf, *sendcount, *sendtype, recvbuf, *recvcount,
                         *recvtype, *comm);
}

void mpi_gatherv(void *sendbuf, int *sendcount, int *sendtype,
   void *recvbuf, int *recvcounts, int *displs,
   int *recvtype, int *root, int *comm, int *ierr)
{
  *ierr = MPI_Gatherv(sendbuf, *sendcount, *sendtype, recvbuf, recvcounts,
                       displs, *recvtype, *root, *comm);
}

void mpi_gather(void *sendbuf, int *sendcount, int *sendtype,
   void *recvbuf, int *recvcount, int *recvtype,
   int *root, int *comm, int *ierr)
{
  *ierr = MPI_Gather(sendbuf, *sendcount, *sendtype, recvbuf, *recvcount,
                      *recvtype, *root, *comm);
}

void mpi_scatterv(void *sendbuf, int *sendcounts, int *displs, int *sendtype,
   void *recvbuf, int *recvcount, int *recvtype, int *root, int *comm, int *ierr)
{
  *ierr = MPI_Scatterv(sendbuf, sendcounts, displs, *sendtype, recvbuf, *recvcount,
                       *recvtype, *root, *comm);
}

void mpi_scatter(void *sendbuf, int *sendcount, int *sendtype,
   void *recvbuf, int *recvcount, int *recvtype,
   int *root, int *comm, int *ierr)
{
  *ierr = MPI_Scatter(sendbuf, *sendcount, *sendtype, recvbuf, *recvcount,
                      *recvtype, *root, *comm);
}

void mpi_alltoallv(void *sendbuf, int *sendcounts, int *sdispls,
   int *sendtype, void *recvbuf, int *recvcounts,
   int *rdispls, int *recvtype, int *comm, int *ierr)
{
  *ierr = MPI_Alltoallv(sendbuf, sendcounts, sdispls, *sendtype, recvbuf,
                         recvcounts, rdispls, *recvtype, *comm);
}

void mpi_alltoall(void *sendbuf, int *sendcount, int *sendtype,
   void *recvbuf, int *recvcount, int *recvtype,
   int *comm, int *ierr)
{
  *ierr = MPI_Alltoall(sendbuf, *sendcount, *sendtype, recvbuf, *recvcount,
                        *recvtype, *comm);
}

void mpi_comm_dup(int *comm, int *newcomm, int *ierr)
{
  AMPIAPI("ampi_com_dup");
  *newcomm = *comm;
  *ierr = 0;
}

void mpi_comm_free(int *comm, int *ierr)
{
  AMPIAPI("ampi_comm_free");
  *ierr = 0;
}

void mpi_group_size(int* group, int* size, int* ierror){
  *ierror = MPI_Group_size(*group, size);
}
void mpi_group_rank(int* group, int* rank, int* ierror){
  *ierror = MPI_Group_rank(*group, rank);
}
void mpi_group_translate_ranks(int* group1, int* n, int* ranks1, int* group2, int* ranks2, int* ierror){
  *ierror = MPI_Group_translate_ranks(*group1, *n, ranks1, *group2, ranks2);
}
void mpi_group_compare(int* group1, int* group2, int* result, int* ierror){
  *ierror = MPI_Group_compare(*group1, *group2, result);
}
void mpi_comm_group(int* comm, int* group, int* ierror){
  *ierror = MPI_Comm_group(*comm, group);
}
void mpi_group_union(int* group1, int* group2, int* newgroup, int* ierror){
  *ierror = MPI_Group_union(*group1, *group2, newgroup);
}
void mpi_group_intersection(int* group1, int* group2, int* newgroup, int* ierror){
  *ierror = MPI_Group_intersection(*group1, *group2, newgroup);
}
void mpi_group_difference(int* group1, int* group2, int* newgroup, int* ierror){
  *ierror = MPI_Group_difference(*group1, *group2, newgroup);
}
void mpi_group_incl(int* group, int* n, int* ranks, int* newgroup, int* ierror){
  *ierror = MPI_Group_incl(*group, *n, ranks, newgroup);
}
void mpi_group_excl(int* group, int* n, int* ranks, int* newgroup, int* ierror){
  *ierror = MPI_Group_excl(*group, *n, ranks, newgroup);
}
void mpi_group_range_incl(int* group, int* n, int ranges[][3], int* newgroup, int* ierror){
  *ierror = MPI_Group_range_incl(*group, *n, ranges, newgroup);
}
void mpi_group_range_excl(int* group,int*  n, int ranges[][3], int* newgroup, int* ierror){
  *ierror = MPI_Group_range_excl(*group, *n, ranges, newgroup);
}
void mpi_group_free(int*  group, int*  ierror){
  *ierror = 0;
}
void mpi_comm_create(int*  comm, int*  group, int*  newcomm, int*  ierror){
  *ierror = MPI_Comm_create(*comm, *group, newcomm);
}

void mpi_abort(int *comm, int *errorcode, int *ierr)
{
  *ierr = MPI_Abort(*comm, *errorcode);
}

void mpi_get_count(int *sts, int *dtype, int *cnt, int *ierr)
{
  *ierr = MPI_Get_count((MPI_Status*) sts, *dtype, cnt);
}

void mpi_print(char *str, int len)
{
  char *tmpstr = new char[len+1];
  memcpy(tmpstr,str,len);
  tmpstr[len] = '\0';
  MPI_Print(tmpstr);
  delete[] tmpstr;
}

void mpi_migrate(void)
{
  MPI_Migrate();
}

int mpi_register(void *d, MPI_PupFn f)
{
  return MPI_Register(d,f);
}

} // extern "C"


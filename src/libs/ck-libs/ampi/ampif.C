#include "ampi.h"
#include "ampiimpl.h"

FDECL {
#define mpi_send FTN_NAME( MPI_SEND , mpi_send )
#define mpi_recv FTN_NAME( MPI_RECV , mpi_recv )
#define mpi_get_count FTN_NAME( MPI_GET_COUNT , mpi_get_count )
#define mpi_isend FTN_NAME( MPI_ISEND , mpi_isend )
#define mpi_bsend FTN_NAME( MPI_BSEND , mpi_bsend )
#define mpi_rsend FTN_NAME( MPI_RSEND , mpi_rsend )
#define mpi_ssend FTN_NAME( MPI_SSEND , mpi_ssend )
#define mpi_issend FTN_NAME( MPI_ISSEND , mpi_issend )
#define mpi_irecv FTN_NAME( MPI_IRECV , mpi_irecv )
#define mpi_wait FTN_NAME( MPI_WAIT , mpi_wait )
#define mpi_test FTN_NAME( MPI_TEST , mpi_test )
#define mpi_waitany FTN_NAME( MPI_WAITANY , mpi_waitany )
#define mpi_testany FTN_NAME( MPI_TESTANY , mpi_testany )
#define mpi_waitall FTN_NAME( MPI_WAITALL , mpi_waitall )
#define mpi_testall FTN_NAME( MPI_TESTALL , mpi_testall )
#define mpi_waitsome FTN_NAME( MPI_WAITSOME , mpi_waitsome )
#define mpi_testsome FTN_NAME( MPI_TESTSOME , mpi_testsome )
#define mpi_request_free FTN_NAME(MPI_REQUEST_FREE , mpi_request_free)
#define mpi_cancel FTN_NAME(MPI_CANCEL, mpi_cancel)
#define mpi_iprobe FTN_NAME( MPI_IPROBE , mpi_iprobe )
#define mpi_probe FTN_NAME( MPI_PROBE , mpi_probe )
#define mpi_send_init FTN_NAME( MPI_SEND_INIT , mpi_send_init )
#define mpi_recv_init FTN_NAME( MPI_RECV_INIT , mpi_recv_init )
#define mpi_start FTN_NAME( MPI_START , mpi_start )
#define mpi_startall FTN_NAME( MPI_STARTALL , mpi_startall )
#define mpi_sendrecv FTN_NAME( MPI_SENDRECV , mpi_sendrecv )
#define mpi_sendrecv_replace FTN_NAME( MPI_SENDRECV_REPLACE , mpi_sendrecv_replace )
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
#define mpi_type_lb FTN_NAME( MPI_TYPE_LB , mpi_type_lb )
#define mpi_type_ub FTN_NAME( MPI_TYPE_UB , mpi_type_ub )
#define mpi_address FTN_NAME( MPI_ADDRESS , mpi_address )
#define mpi_get_elements FTN_NAME( MPI_GET_ELEMENTS , mpi_get_elements )
#define mpi_pack FTN_NAME( MPI_PACK , mpi_pack )
#define mpi_unpack FTN_NAME( MPI_UNPACK , mpi_unpack )
#define mpi_pack_size FTN_NAME( MPI_PACK_SIZE , mpi_pack_size )

#define mpi_barrier FTN_NAME( MPI_BARRIER , mpi_barrier )
#define mpi_bcast FTN_NAME( MPI_BCAST , mpi_bcast )
#define mpi_gather FTN_NAME( MPI_GATHER , mpi_gather )
#define mpi_gatherv FTN_NAME( MPI_GATHERV , mpi_gatherv )
#define mpi_scatter FTN_NAME( MPI_SCATTER , mpi_scatter )
#define mpi_scatterv FTN_NAME( MPI_SCATTERV , mpi_scatterv )
#define mpi_allgather FTN_NAME( MPI_ALLGATHER , mpi_allgather )
#define mpi_iallgather FTN_NAME( MPI_IALLGATHER , mpi_iallgather )
#define mpi_allgatherv FTN_NAME( MPI_ALLGATHERV , mpi_allgatherv )
#define mpi_alltoall FTN_NAME( MPI_ALLTOALL , mpi_alltoall )
#define mpi_ialltoall FTN_NAME( MPI_IALLTOALL , mpi_ialltoall )
#define mpi_alltoallv FTN_NAME( MPI_ALLTOALLV , mpi_alltoallv )
#define mpi_reduce FTN_NAME( MPI_REDUCE , mpi_reduce )
#define mpi_ireduce FTN_NAME( MPI_IREDUCE , mpi_ireduce )
#define mpi_allreduce FTN_NAME( MPI_ALLREDUCE , mpi_allreduce )
#define mpi_iallreduce FTN_NAME( MPI_IALLREDUCE , mpi_iallreduce )
#define mpi_reduce_scatter FTN_NAME( MPI_REDUCE_SCATTER , mpi_reduce_scatter )
#define mpi_scan FTN_NAME( MPI_SCAN , mpi_scan )
#define mpi_op_create FTN_NAME( MPI_OP_CREATE , mpi_op_create )
#define mpi_op_free FTN_NAME( MPI_OP_FREE , mpi_op_free )

#define mpi_group_size FTN_NAME( MPI_GROUP_SIZE, mpi_group_size)
#define mpi_group_rank FTN_NAME( MPI_GROUP_RANK, mpi_group_rank)
#define mpi_group_translate_ranks FTN_NAME(MPI_GROUP_TRANSLATE_RANKS, mpi_group_translate_ranks)
#define mpi_group_compare FTN_NAME(MPI_GROUP_COMPARE, mpi_group_compare)

#define mpi_comm_group FTN_NAME( MPI_COMM_GROUP, mpi_comm_group)
#define mpi_group_union FTN_NAME(MPI_GROUP_UNION, mpi_group_union)
#define mpi_group_intersection FTN_NAME(MPI_GROUP_INTERSECTION, mpi_group_intersection)
#define mpi_group_difference FTN_NAME(MPI_GROUP_DIFFERENCE, mpi_group_difference)
#define mpi_group_incl FTN_NAME(MPI_GROUP_INCL, mpi_group_incl)
#define mpi_group_excl FTN_NAME(MPI_GROUP_EXCL, mpi_group_excl)
#define mpi_group_range_incl FTN_NAME(MPI_GROUP_RANGE_INCL, mpi_group_range_incl)
#define mpi_group_range_excl FTN_NAME(MPI_GROUP_RANGE_EXCL, mpi_group_range_excl)
#define mpi_group_free FTN_NAME(MPI_GROUP_FREE, mpi_group_free)
#define mpi_comm_create FTN_NAME(MPI_COMM_CREATE, mpi_comm_create)

#define mpi_comm_rank FTN_NAME( MPI_COMM_RANK , mpi_comm_rank )
#define mpi_comm_size FTN_NAME( MPI_COMM_SIZE , mpi_comm_size )
#define mpi_comm_dup FTN_NAME( MPI_COMM_DUP , mpi_comm_dup )
#define mpi_comm_split FTN_NAME( MPI_COMM_SPLIT , mpi_comm_split )
#define mpi_comm_free FTN_NAME( MPI_COMM_FREE , mpi_comm_free )
#define mpi_comm_test_inter FTN_NAME( MPI_COMM_TEST_INTER , mpi_comm_test_inter )
#define mpi_comm_remote_size FTN_NAME ( MPI_COMM_REMOTE_SIZE , mpi_comm_remote_size )
#define mpi_comm_remote_group FTN_NAME ( MPI_COMM_REMOTE_GROUP , mpi_comm_remote_group )
#define mpi_intercomm_create FTN_NAME ( MPI_INTERCOMM_CREATE , mpi_intercomm_create )
#define mpi_intercomm_merge FTN_NAME ( MPI_INTERCOMM_MERGE , mpi_intercomm_merge )
#define mpi_keyval_create FTN_NAME ( MPI_KEYVAL_CREATE , mpi_keyval_create )
#define mpi_keyval_free FTN_NAME ( MPI_KEYVAL_FREE , mpi_keyval_free )
#define mpi_attr_put FTN_NAME ( MPI_ATTR_PUT , mpi_attr_put )
#define mpi_attr_get FTN_NAME ( MPI_ATTR_GET , mpi_attr_get )
#define mpi_attr_delete FTN_NAME ( MPI_ATTR_DELETE , mpi_attr_delete )

#define mpi_cart_create FTN_NAME ( MPI_CART_CREATE , mpi_cart_create )
#define mpi_graph_create FTN_NAME ( MPI_GRAPH_CREATE , mpi_graph_create )
#define mpi_topo_test FTN_NAME ( MPI_TOPO_TEST , mpi_topo_test )
#define mpi_cart_map FTN_NAME ( MPI_CART_MAP , mpi_cart_map )
#define mpi_graph_map FTN_NAME ( MPI_GRAPH_MAP , mpi_graph_map )
#define mpi_cartdim_get FTN_NAME ( MPI_CARTDIM_GET , mpi_cartdim_get )
#define mpi_cart_get FTN_NAME ( MPI_CART_GET , mpi_cart_get )
#define mpi_cart_rank FTN_NAME ( MPI_CART_RANK , mpi_cart_rank )
#define mpi_cart_coords FTN_NAME ( MPI_CART_COORDS , mpi_cart_coords )
#define mpi_cart_shift FTN_NAME ( MPI_CART_SHIFT , mpi_cart_shift )
#define mpi_graphdims_get FTN_NAME ( MPI_GRAPHDIMS_GET , mpi_graphdims_get )
#define mpi_graph_get FTN_NAME ( MPI_GRAPH_GET , mpi_graph_get )
#define mpi_graph_neighbors_count FTN_NAME ( MPI_GRAPH_NEIGHBORS_COUNT , mpi_graph_neighbors_count )
#define mpi_graph_neighbors FTN_NAME ( MPI_GRAPH_NEIGHBORS , mpi_graph_neighbors )
#define mpi_dims_create FTN_NAME ( MPI_DIMS_CREATE , mpi_dims_create )
#define mpi_cart_sub FTN_NAME ( MPI_CART_SUB , mpi_cart_sub )

#define mpi_get_processor_name FTN_NAME ( MPI_GET_PROCESSOR_NAME , mpi_get_processor_name )
#define mpi_errhandler_create FTN_NAME( MPI_ERRHANDLER_CREATE , mpi_errhandler_create )
#define mpi_errhandler_set FTN_NAME( MPI_ERRHANDLER_SET , mpi_errhandler_set )
#define mpi_errhandler_get FTN_NAME( MPI_ERRHANDLER_GET , mpi_errhandler_get )
#define mpi_errhandler_free FTN_NAME( MPI_ERRHANDLER_FREE , mpi_errhandler_free )
#define mpi_error_string FTN_NAME( MPI_ERROR_STRING , mpi_error_string )
#define mpi_error_class FTN_NAME( MPI_ERROR_CLASS , mpi_error_class )
#define mpi_wtime FTN_NAME( MPI_WTIME , mpi_wtime )
#define mpi_wtick FTN_NAME( MPI_WTICK , mpi_wtick )
#define mpi_init FTN_NAME( MPI_INIT , mpi_init )
#define mpi_initialized FTN_NAME( MPI_INITIALIZED , mpi_initialized )
#define mpi_init_universe FTN_NAME( MPI_INIT_UNIVERSE , mpi_init_universe )
#define mpi_finalize FTN_NAME( MPI_FINALIZE , mpi_finalize )
#define mpi_finalized FTN_NAME( MPI_FINALIZED , mpi_finalized )
#define mpi_abort FTN_NAME( MPI_ABORT , mpi_abort )

#define mpi_yield FTN_NAME ( MPI_YIELD , mpi_yield )
#define mpi_resume FTN_NAME ( MPI_RESUME, mpi_resume )
#define mpi_print FTN_NAME( MPI_PRINT , mpi_print )
#define mpi_Start_measure FTN_NAME( MPI_START_MEASURE, mpi_start_measure)
#define mpi_Stop_measure FTN_NAME( MPI_STOP_MEASURE, mpi_stop_measure)
#define mpi_set_load FTN_NAME( MPI_SET_LOAD, mpi_set_load)
#define mpi_register FTN_NAME( MPI_REGISTER , mpi_register )
#define mpi_migrate FTN_NAME( MPI_MIGRATE , mpi_migrate )
#define mpi_migrateto FTN_NAME( MPI_MIGRATETO , mpi_migrateto )
#define mpi_async_migrate FTN_NAME( MPI_ASYNC_MIGRATE , mpi_async_migrate )
#define mpi_allow_migrate FTN_NAME( MPI_ALLOW_MIGRATE , mpi_allow_migrate )
#define mpi_setmigratable FTN_NAME (MPI_SETMIGRATABLE , mpi_setmigratable )
#define mpi_checkpoint FTN_NAME( MPI_CHECKPOINT , mpi_checkpoint )
#define mpi_memcheckpoint FTN_NAME( MPI_MEMCHECKPOINT , mpi_memcheckpoint )

#define mpi_get_argc FTN_NAME( MPI_GET_ARGC , mpi_get_argc )
#define mpi_get_argv FTN_NAME( MPI_GET_ARGV , mpi_get_argv )

/* MPI-2 */
#define mpi_type_get_envelope FTN_NAME ( MPI_TYPE_GET_ENVELOPE , mpi_type_get_envelope )
#define mpi_type_get_contents FTN_NAME ( MPI_TYPE_GET_CONTENTS , mpi_type_get_contents )

#define mpi_win_create FTN_NAME ( MPI_WIN_CREATE , mpi_win_create )
#define mpi_win_free  FTN_NAME ( MPI_WIN_FREE  , mpi_win_free )
#define mpi_win_delete_attr  FTN_NAME ( MPI_WIN_DELETE_ATTR  , mpi_win_delete_attr )
#define mpi_win_get_group  FTN_NAME ( MPI_WIN_GET_GROUP  , mpi_win_get_group )
#define mpi_win_set_name  FTN_NAME ( MPI_WIN_SET_NAME  , mpi_win_set_name )
#define mpi_win_get_name  FTN_NAME ( MPI_WIN_GET_NAME  , mpi_win_get_name )
#define mpi_win_fence  FTN_NAME ( MPI_WIN_FENCE  , mpi_win_fence )
#define mpi_win_lock  FTN_NAME ( MPI_WIN_LOCK  , mpi_win_lock )
#define mpi_win_unlock  FTN_NAME ( MPI_WIN_UNLOCK  , mpi_win_unlock )
#define mpi_win_post  FTN_NAME ( MPI_WIN_POST  , mpi_win_post )
#define mpi_win_wait  FTN_NAME ( MPI_WIN_WAIT  , mpi_win_wait )
#define mpi_win_start  FTN_NAME ( MPI_WIN_START  , mpi_win_start )
#define mpi_win_complete  FTN_NAME ( MPI_WIN_COMPLETE  , mpi_win_complete )
#define mpi_alloc_mem  FTN_NAME ( MPI_ALLOC_MEM  , mpi_alloc_mem )
#define mpi_free_mem  FTN_NAME ( MPI_FREE_MEM  , mpi_free_mem )
#define mpi_put  FTN_NAME ( MPI_PUT  , mpi_put )
#define mpi_get  FTN_NAME ( MPI_GET  , mpi_get )
#define mpi_accumulate  FTN_NAME ( MPI_ACCUMULATE  , mpi_accumulate )

#define mpi_info_create FTN_NAME ( MPI_INFO_CREATE , mpi_info_create )
#define mpi_info_set FTN_NAME ( MPI_INFO_SET , mpi_info_set )
#define mpi_info_delete FTN_NAME ( MPI_INFO_DELETE , mpi_info_delete )
#define mpi_info_get FTN_NAME ( MPI_INFO_GET , mpi_info_get )
#define mpi_info_get_valuelen FTN_NAME ( MPI_INFO_GET_VALUELEN , mpi_info_get_valuelen )
#define mpi_info_get_nkeys FTN_NAME ( MPI_INFO_GET_NKEYS , mpi_info_get_nkeys )
#define mpi_info_get_nthkey FTN_NAME ( MPI_INFO_GET_NTHKEYS , mpi_info_get_nthkey )
#define mpi_info_dup FTN_NAME ( MPI_INFO_DUP , mpi_info_dup )
#define mpi_info_free FTN_NAME ( MPI_INFO_FREE , mpi_info_free )

#define REDUCERF(caps, nocaps) \
void FTN_NAME(caps, nocaps)(void *iv, void *iov, int *len, MPI_Datatype *dt){ \
  caps(iv, iov, len, dt); \
}

#define mpi_info_maxmemory FTN_NAME (MPI_INFO_MAXMEMORY, mpi_info_maxmemory)
#define mpi_info_memory FTN_NAME (MPI_INFO_MEMORY, mpi_info_memory)    

#if !CMK_FORTRAN_USES_ALLCAPS
REDUCERF(MPI_MAX    , mpi_max)
REDUCERF(MPI_MIN    , mpi_min)
REDUCERF(MPI_SUM    , mpi_sum)
REDUCERF(MPI_PROD   , mpi_prod)
REDUCERF(MPI_LAND   , mpi_land)
REDUCERF(MPI_BAND   , mpi_band)
REDUCERF(MPI_LOR    , mpi_lor)
REDUCERF(MPI_BOR    , mpi_bor)
REDUCERF(MPI_LXOR   , mpi_lxor)
REDUCERF(MPI_BXOR   , mpi_bxor)
REDUCERF(MPI_MAXLOC , mpi_maxloc)
REDUCERF(MPI_MINLOC , mpi_minloc)
#endif

typedef MPI_Op  MPI_Op_Array[128];
CtvExtern(MPI_Op_Array, mpi_ops);
CtvExtern(int, mpi_opc);

// must be consistent with mpif.h
#define MPI_OP_FIRST   100

//#define GET_MPI_OP(idx)      (CmiAssert(idx - MPI_OP_FIRST >= 0 && idx - MPI_OP_FIRST < CtvAccess(mpi_opc)), CtvAccess(mpi_ops)[idx - MPI_OP_FIRST])
inline MPI_Op & GET_MPI_OP(int idx)      { MPI_Op *tab=CtvAccess(mpi_ops); return tab[idx - MPI_OP_FIRST]; }

void mpi_init_universe(int *unicomm)
{
  AMPIAPI("mpi_init_universe");
  for(int i=0;i<_mpi_nworlds; i++)
  {
    unicomm[i] = MPI_COMM_UNIVERSE[i];
  }
}

void mpi_init(int *ierr){
  *ierr = AMPI_Init(NULL,NULL);
}

void mpi_initialized(int *isInit, int* ierr)
{
  *ierr = AMPI_Initialized(isInit);
}

void mpi_finalized(int *isFinalized, int* ierr)
{
  *ierr = AMPI_Finalized(isFinalized);
}

void mpi_comm_rank(int *comm, int *rank, int *ierr)
{
  *ierr = AMPI_Comm_rank(*comm, rank);
}

void mpi_comm_size(int *comm, int *size, int *ierr)
{
  *ierr = AMPI_Comm_size(*comm, size);
}

void mpi_finalize(int *ierr)
{
  *ierr = AMPI_Finalize();
}

void mpi_send(void *msg, int *count, int *type, int *dest,
  int *tag, int *comm, int *ierr)
{
  *ierr = AMPI_Send(msg, *count, *type, *dest, *tag, *comm);
}

void mpi_recv(void *msg, int *count, int *type, int *src,
  int *tag, int *comm, int *status, int *ierr)
{
  *ierr = AMPI_Recv(msg, *count, *type, *src, *tag, *comm,
                    (MPI_Status*) status);
}

void mpi_bsend(void *msg, int *count, int *type, int *dest,
  int *tag, int *comm, int *ierr)
{
  *ierr = AMPI_Bsend(msg, *count, *type, *dest, *tag, *comm);
}

void mpi_rsend(void *msg, int *count, int *type, int *dest,
  int *tag, int *comm, int *ierr)
{
  *ierr = AMPI_Rsend(msg, *count, *type, *dest, *tag, *comm);
}

void mpi_ssend(void *msg, int *count, int *type, int *dest,
  int *tag, int *comm, int *ierr)
{
  *ierr = AMPI_Ssend(msg, *count, *type, *dest, *tag, *comm);
}

void mpi_issend(void *buf, int *count, int *datatype, int *dest,
   int *tag, int *comm, int *request, int *ierr)
{
  *ierr = AMPI_Issend(buf, *count, *datatype, *dest, *tag, *comm, (MPI_Request *)request);
}

void mpi_probe(int *src, int *tag, int *comm, int *status, int *ierr)
{
  *ierr = AMPI_Probe(*src, *tag, *comm, (MPI_Status*) status);
}

void mpi_iprobe(int *src,int *tag,int *comm,int *flag,int *status,int *ierr)
{
  *ierr = AMPI_Iprobe(*src, *tag, *comm, flag, (MPI_Status*) status);
}

void mpi_sendrecv(void *sndbuf, int *sndcount, int *sndtype,
  int *dest, int *sndtag, void *rcvbuf,
  int *rcvcount, int *rcvtype, int *src,
  int *rcvtag, int *comm, int *status, int *ierr)
{
  *ierr = AMPI_Sendrecv(sndbuf, *sndcount, *sndtype, *dest, *sndtag,
                        rcvbuf, *rcvcount, *rcvtype, *src, *rcvtag,
			*comm, (MPI_Status*) status);
}

void mpi_sendrecv_replace(void *buf, int* count, int* datatype,
                          int* dest, int* sendtag, int* source, int* recvtag,
                          int* comm, int* status, int *ierr)
{
  *ierr = AMPI_Sendrecv_replace(buf, *count, *datatype, *dest, *sendtag,
                               *source, *recvtag, *comm, (MPI_Status*) status);
}

void mpi_barrier(int *comm, int *ierr)
{
  *ierr = AMPI_Barrier(*comm);
}

void mpi_bcast(void *buf, int *count, int *type, int *root, int *comm, 
   int *ierr)
{
  *ierr = AMPI_Bcast(buf, *count, *type, *root, *comm);
}

void mpi_reduce(void *inbuf, void *outbuf, int *count, int *type,
   int *opc, int *root, int *comm, int *ierr)
{
  MPI_Op op = GET_MPI_OP(*opc);
  if (inbuf == NULL) inbuf = MPI_IN_PLACE;
  if (outbuf == NULL) outbuf = MPI_IN_PLACE;
  *ierr = AMPI_Reduce(inbuf, outbuf, *count, *type, op, *root, *comm);
}

void mpi_allreduce(void *inbuf,void *outbuf,int *count,int *type,
   int *opc, int *comm, int *ierr)
{
  MPI_Op op = GET_MPI_OP(*opc);
  if (inbuf == NULL) inbuf = MPI_IN_PLACE;
  if (outbuf == NULL) outbuf = MPI_IN_PLACE;
  *ierr = AMPI_Allreduce(inbuf, outbuf, *count, *type, op, *comm);
}

double mpi_wtime(void)
{
  return AMPI_Wtime();
}

double mpi_wtick(void)
{
  return AMPI_Wtick();
}

void mpi_start(int *reqnum, int *ierr)
{
  *ierr = AMPI_Start((MPI_Request*) reqnum);
}

void mpi_startall(int *count, int *reqnum, int *ierr)
{
  *ierr = AMPI_Startall(*count, (MPI_Request*) reqnum);
}

void mpi_waitall(int *count, int *request, int *status, int *ierr)
{
  *ierr = AMPI_Waitall(*count, (MPI_Request*) request, (MPI_Status*) status);
}

void mpi_waitany(int *count, int *request, int *index, int *status, int *ierr)
{
  *ierr = AMPI_Waitany(*count, (MPI_Request*) request, index,
                       (MPI_Status*) status);
}

void mpi_wait(int *request, int *status, int *ierr)
{
  *ierr = AMPI_Wait((MPI_Request*) request, (MPI_Status*) status);
}

void mpi_testall(int *count, int *request, int *flag, int *status, int *ierr)
{
  *ierr = AMPI_Testall(*count, (MPI_Request*) request, flag,
      (MPI_Status*) status);
}

void mpi_waitsome(int *incount, int *array_of_requests, int *outcount, int *array_of_indices, int *array_of_statuses, int *ierr)
{
  *ierr = AMPI_Waitsome(*incount, (MPI_Request *)array_of_requests, outcount, array_of_indices, (MPI_Status*) array_of_statuses);
}

void mpi_testsome(int *incount, int *array_of_requests, int *outcount, int *array_of_indices, int *array_of_statuses, int *ierr)
{
  *ierr = AMPI_Testsome(*incount, (MPI_Request *)array_of_requests, outcount, array_of_indices, (MPI_Status*) array_of_statuses);
}

void mpi_testany(int *count, int *request, int *index, int *flag, int *status, int *ierr)
{
  *ierr = AMPI_Testany(*count, (MPI_Request*) request, index, flag,
      (MPI_Status*) status);
}

void mpi_test(int *request, int *flag, int *status, int *ierr)
{
  *ierr = AMPI_Test((MPI_Request*) request, flag, (MPI_Status*) status);
}

void mpi_request_free(int *request, int *ierr)
{
  *ierr = AMPI_Request_free((MPI_Request *)request);
}

void mpi_cancel(int *request, int *ierr)
{
  *ierr = AMPI_Cancel((MPI_Request *)request);
}

void mpi_recv_init(void *buf, int *count, int *type, int *srcpe,
   int *tag, int *comm, int *req, int *ierr)
{
  *ierr = AMPI_Recv_init(buf,*count,*type,*srcpe,*tag,*comm,(MPI_Request*)req);
}

void mpi_send_init(void *buf, int *count, int *type, int *destpe,
   int *tag, int *comm, int *req, int *ierr)
{
  *ierr = AMPI_Send_init(buf,*count,*type,*destpe,*tag,*comm,
                         (MPI_Request*)req);
}

void mpi_type_contiguous(int *count, int *oldtype, int *newtype, int *ierr)
{
  *ierr = AMPI_Type_contiguous(*count, *oldtype, newtype);
}

void mpi_type_vector(int *count, int *blocklength, int *stride,
   int *oldtype, int*  newtype, int *ierr)
{
  *ierr = AMPI_Type_vector(*count, *blocklength, *stride, *oldtype, newtype);
}

void mpi_type_hvector(int *count, int *blocklength, int *stride,
   int* oldtype, int* newtype, int* ierr)
{
  *ierr = AMPI_Type_hvector(*count, *blocklength, *stride, *oldtype, newtype);
}

void mpi_type_indexed(int *count, int* arrBlength, int* arrDisp,
   int* oldtype, int*  newtype, int* ierr)
{
  *ierr = AMPI_Type_indexed(*count, arrBlength, arrDisp, *oldtype, newtype);
}

void mpi_type_hindexed(int* count, int* arrBlength, int* arrDisp,
   int* oldtype, int* newtype, int* ierr)
{
  *ierr = AMPI_Type_hindexed(*count, arrBlength, arrDisp, *oldtype, newtype);
}

void mpi_type_struct(int* count, int* arrBlength, int* arrDisp,
   int* oldtype, int* newtype, int* ierr)
{
  *ierr = AMPI_Type_struct(*count, arrBlength, arrDisp, oldtype, newtype);
}


void mpi_type_commit(int *type, int *ierr)
{
  *ierr = AMPI_Type_commit(type);
}

void mpi_type_free(int *type, int *ierr)
{
  *ierr = AMPI_Type_free(type);
}

void  mpi_type_extent(int* type, int* extent, int* ierr)
{
  *ierr = AMPI_Type_extent(*type, extent);
}

void  mpi_type_size(int* type, int* size, int* ierr)
{
  *ierr = AMPI_Type_size(*type, size);
}

void mpi_type_lb(int* datatype, int* displacement, int* ierr)
{
  *ierr = AMPI_Type_lb(*datatype, displacement);
}

void mpi_type_ub(int* datatype, int* displacement, int* ierr)
{
  *ierr = AMPI_Type_ub(*datatype, displacement);
}

void mpi_address(int* location, int *address, int* ierr)
{
  *ierr = AMPI_Address(location, address);
}

void mpi_get_elements(int *status, int* datatype, int *count, int* ierr)
{
  *ierr = AMPI_Get_elements((MPI_Status*) status, *datatype, count);
}

void mpi_pack(void *inbuf, int *incount, int *datatype, void *outbuf,
    int *outsize, int *position, int *comm, int *ierr)
{
  *ierr = AMPI_Pack(inbuf, *incount, (MPI_Datatype)*datatype, outbuf,
      *outsize, position, *comm);
}

void mpi_unpack(void *inbuf, int *insize, int *position, void *outbuf,
    int *outcount, int *datatype, int *comm, int *ierr)
{
  *ierr = AMPI_Unpack(inbuf, *insize, position, outbuf, *outcount,
      (MPI_Datatype) *datatype, (MPI_Comm) *comm);
}

void mpi_pack_size(int *incount, int *datatype, int *comm, int *size, int *ierr)
{
  *ierr = AMPI_Pack_size(*incount, (MPI_Datatype) *datatype, *comm, size);
}

void mpi_isend(void *buf, int *count, int *datatype, int *dest,
   int *tag, int *comm, int *request, int *ierr)
{
  *ierr = AMPI_Isend(buf, *count, *datatype, *dest, *tag, *comm, (MPI_Request *)request);
}

void mpi_irecv(void *buf, int *count, int *datatype, int *src,
   int *tag, int *comm, int *request, int *ierr)
{
  *ierr = AMPI_Irecv(buf, *count, *datatype, *src, *tag, *comm, (MPI_Request *)request);
}

void mpi_allgatherv(void *sendbuf, int *sendcount, int *sendtype,
   void *recvbuf, int *recvcounts, int *displs,
   int *recvtype, int *comm, int *ierr)
{
  *ierr = AMPI_Allgatherv(sendbuf, *sendcount, *sendtype, recvbuf, recvcounts,
                          displs, *recvtype, *comm);
}

void mpi_allgather(void *sendbuf, int *sendcount, int *sendtype,
   void *recvbuf, int *recvcount, int *recvtype,
   int *comm, int *ierr)
{
  *ierr = AMPI_Allgather(sendbuf, *sendcount, *sendtype, recvbuf, *recvcount,
                         *recvtype, *comm);
}

void mpi_gatherv(void *sendbuf, int *sendcount, int *sendtype,
   void *recvbuf, int *recvcounts, int *displs,
   int *recvtype, int *root, int *comm, int *ierr)
{
  *ierr = AMPI_Gatherv(sendbuf, *sendcount, *sendtype, recvbuf, recvcounts,
                       displs, *recvtype, *root, *comm);
}

void mpi_gather(void *sendbuf, int *sendcount, int *sendtype,
   void *recvbuf, int *recvcount, int *recvtype,
   int *root, int *comm, int *ierr)
{
  *ierr = AMPI_Gather(sendbuf, *sendcount, *sendtype, recvbuf, *recvcount,
                      *recvtype, *root, *comm);
}

void mpi_scatterv(void *sendbuf, int *sendcounts, int *displs, int *sendtype,
   void *recvbuf, int *recvcount, int *recvtype, int *root, int *comm, int *ierr)
{
  *ierr = AMPI_Scatterv(sendbuf, sendcounts, displs, *sendtype, recvbuf, *recvcount,
                       *recvtype, *root, *comm);
}

void mpi_scatter(void *sendbuf, int *sendcount, int *sendtype,
   void *recvbuf, int *recvcount, int *recvtype,
   int *root, int *comm, int *ierr)
{
  *ierr = AMPI_Scatter(sendbuf, *sendcount, *sendtype, recvbuf, *recvcount,
                      *recvtype, *root, *comm);
}

void mpi_alltoallv(void *sendbuf, int *sendcounts, int *sdispls,
   int *sendtype, void *recvbuf, int *recvcounts,
   int *rdispls, int *recvtype, int *comm, int *ierr)
{
  *ierr = AMPI_Alltoallv(sendbuf, sendcounts, sdispls, *sendtype, recvbuf,
                         recvcounts, rdispls, *recvtype, *comm);
}

void mpi_alltoall(void *sendbuf, int *sendcount, int *sendtype,
   void *recvbuf, int *recvcount, int *recvtype,
   int *comm, int *ierr)
{
  *ierr = AMPI_Alltoall(sendbuf, *sendcount, *sendtype, recvbuf, *recvcount,
                       *recvtype, *comm);
}

void mpi_iallgather(void *sendbuf, int* sendcount, int* sendtype,
                    void *recvbuf, int* recvcount, int* recvtype,
                    int* comm, int* request, int* ierr)
{
  *ierr = AMPI_Iallgather(sendbuf, *sendcount, *sendtype, recvbuf, *recvcount,
                         *recvtype, *comm, (MPI_Request *)request);

}

void mpi_ialltoall(void *sendbuf, int* sendcount, int* sendtype,
                 void *recvbuf, int* recvcount, int* recvtype,
                 int* comm, int *request, int* ierr)
{
  *ierr = AMPI_Ialltoall(sendbuf, *sendcount, *sendtype,
                        recvbuf, *recvcount, *recvtype,
                        *comm, (MPI_Request *)request);
}

void mpi_ireduce(void *sendbuf, void *recvbuf, int* count, int* type,
                int* opc, int* root, int* comm, int *request, int* ierr)
{
  MPI_Op op = GET_MPI_OP(*opc);
  *ierr = AMPI_Ireduce(sendbuf, recvbuf, *count, *type,
                      op, *root, *comm, (MPI_Request*) request);
}

void mpi_iallreduce(void *inbuf, void *outbuf, int* count, int* type,
                   int* opc, int* comm, int *request, int* ierr)
{
  MPI_Op op = GET_MPI_OP(*opc);
  *ierr = AMPI_Iallreduce(inbuf, outbuf, *count, *type,
                         op, *comm, (MPI_Request*) request);
}
void mpi_reduce_scatter(void *sendbuf, void *recvbuf, int *recvcounts,
                       int* datatype, int* opc, int* comm, int* ierr)
{
  MPI_Op op = GET_MPI_OP(*opc);
  *ierr = AMPI_Reduce_scatter(sendbuf, recvbuf, recvcounts,
                             *datatype, op, *comm);
}

void mpi_scan(void* sendbuf, void* recvbuf, int* count, int* datatype, int* opc, int* comm, int* ierr)
{
  MPI_Op op = GET_MPI_OP(*opc);
  *ierr = AMPI_Scan(sendbuf,recvbuf,*count,*datatype,op,*comm );
}

void mpi_op_create(int* function, int* commute, int* opc, int* ierr){
  MPI_Op op;
  *ierr = MPI_Op_create((MPI_User_function *)function, *commute, (MPI_Op *)&op);
  GET_MPI_OP(CtvAccess(mpi_opc)++) = op;
  *opc = CtvAccess(mpi_opc)-1;
}

void mpi_op_free(int* opc, int* ierr){
  MPI_Op op = GET_MPI_OP(*opc);
  GET_MPI_OP(*opc) = NULL;
  *ierr = MPI_Op_free((MPI_Op *)&op);
}

void mpi_comm_dup(int *comm, int *newcomm, int *ierr)
{
  *newcomm = *comm;
  *ierr = 0;
}

void mpi_comm_split(int* src, int* color, int* key, int *dest, int *ierr)
{
  *ierr = AMPI_Comm_split(*src, *color, *key, dest);
}

void mpi_comm_free(int *comm, int *ierr)
{
  *ierr = 0;
}

void mpi_comm_test_inter(int* comm, int* flag, int* ierr)
{
  *ierr = AMPI_Comm_test_inter(*comm, flag);
}

void mpi_cart_create(int* comm_old, int* ndims, int *dims, int *periods,
		    int* reorder, int* comm_cart, int* ierr)
{
  *ierr = AMPI_Cart_create(*comm_old, *ndims, dims, periods, *reorder, comm_cart);
}

void mpi_graph_create(int* comm_old, int* nnodes, int *index, int *edges,
		     int* reorder, int* comm_graph, int* ierr)
{
  *ierr = AMPI_Graph_create(*comm_old, *nnodes, index, edges, *reorder, comm_graph);
}

void mpi_topo_test(int* comm, int *status, int* ierr)
{
  *ierr = AMPI_Topo_test(*comm, status);
}

void mpi_cart_map(int* comm, int* ndims, int *dims, int *periods,
                 int *newrank, int* ierr)
{
  *ierr = AMPI_Cart_map(*comm, *ndims, dims, periods, newrank);
}

void mpi_graph_map(int* comm, int* nnodes, int *index, int *edges,
		  int *newrank, int* ierr)
{
  *ierr = AMPI_Graph_map(*comm, *nnodes, index, edges, newrank);
}

void mpi_cartdim_get(int* comm, int *ndims, int* ierr)
{
  *ierr = AMPI_Cartdim_get(*comm, ndims);
}

void mpi_cart_get(int* comm, int* maxdims, int *dims, int *periods,
		 int *coords, int* ierr)
{
  *ierr = AMPI_Cart_get(*comm, *maxdims, dims, periods, coords);
}

void mpi_cart_rank(int* comm, int *coords, int *rank, int* ierr)
{
  *ierr = AMPI_Cart_rank(*comm, coords, rank);
}

void mpi_cart_coords(int* comm, int* rank, int* maxdims, int *coords, int* ierr)
{
  *ierr = AMPI_Cart_coords(*comm, *rank, *maxdims, coords);
}

void mpi_cart_shift(int* comm, int* direction, int* disp, int *rank_source,
		   int *rank_dest, int* ierr)
{
  *ierr = AMPI_Cart_shift(*comm, *direction, *disp, rank_source, rank_dest);
}

void mpi_graphdims_get(int* comm, int *nnodes, int *nedges, int* ierr)
{
  *ierr = AMPI_Graphdims_get(*comm, nnodes, nedges);
}

void mpi_graph_get(int* comm, int *maxindex, int *maxedges, int *index,
		  int *edges, int* ierr)
{
  *ierr = AMPI_Graph_get(*comm, *maxindex, *maxedges, index, edges);
}

void mpi_graph_neighbors_count(int* comm, int *rank, int *nneighbors, int* ierr)
{
  *ierr = AMPI_Graph_neighbors_count(*comm, *rank, nneighbors);
}

void mpi_graph_neighbors(int* comm, int *rank, int *maxneighbors,
			int *neighbors, int* ierr)
{
  *ierr = AMPI_Graph_neighbors(*comm, *rank, *maxneighbors, neighbors);
}

void mpi_dims_create(int *nnodes, int *ndims, int *dims, int* ierr)
{
  *ierr = AMPI_Dims_create(*nnodes, *ndims, dims);
}

void mpi_cart_sub(int* comm, int *remain_dims, int* newcomm, int* ierr)
{
  *ierr = AMPI_Cart_sub(*comm, remain_dims, newcomm);
}

void mpi_get_processor_name(char* name, int *resultlen, int *ierr)
{
  *ierr = AMPI_Get_processor_name(name, resultlen);
}

void mpi_errhandler_create(int *function, int *errhandler, int *ierr){  *ierr = 0;  }
void mpi_errhandler_set(int* comm, int* errhandler, int *ierr){  *ierr = 0;  }
void mpi_errhandler_get(int* comm, int *errhandler, int *ierr){  *ierr = 0;  }
void mpi_errhandler_free(int *errhandler, int *ierr){  *ierr = 0;  }
void mpi_error_string(int* errorcode, char *string, int *resultlen, int *ierr)
{
  *ierr = AMPI_Error_string(*errorcode, string, resultlen);
}
void mpi_error_class(int* errorcode, int *errorclass, int *ierr)
{
  *ierr = AMPI_Error_class(*errorcode, errorclass);
}

void mpi_group_size(int* group, int* size, int* ierror){
  *ierror = AMPI_Group_size(*group, size);
}
void mpi_group_rank(int* group, int* rank, int* ierror){
  *ierror = AMPI_Group_rank(*group, rank);
}
void mpi_group_translate_ranks(int* group1, int* n, int* ranks1, int* group2, int* ranks2, int* ierror){
  *ierror = AMPI_Group_translate_ranks(*group1, *n, ranks1, *group2, ranks2);
}
void mpi_group_compare(int* group1, int* group2, int* result, int* ierror){
  *ierror = AMPI_Group_compare(*group1, *group2, result);
}
void mpi_comm_group(int* comm, int* group, int* ierror){
  *ierror = AMPI_Comm_group(*comm, group);
}
void mpi_group_union(int* group1, int* group2, int* newgroup, int* ierror){
  *ierror = AMPI_Group_union(*group1, *group2, newgroup);
}
void mpi_group_intersection(int* group1, int* group2, int* newgroup, int* ierror){
  *ierror = AMPI_Group_intersection(*group1, *group2, newgroup);
}
void mpi_group_difference(int* group1, int* group2, int* newgroup, int* ierror){
  *ierror = AMPI_Group_difference(*group1, *group2, newgroup);
}
void mpi_group_incl(int* group, int* n, int* ranks, int* newgroup, int* ierror){
  *ierror = AMPI_Group_incl(*group, *n, ranks, newgroup);
}
void mpi_group_excl(int* group, int* n, int* ranks, int* newgroup, int* ierror){
  *ierror = AMPI_Group_excl(*group, *n, ranks, newgroup);
}
void mpi_group_range_incl(int* group, int* n, int ranges[][3], int* newgroup, int* ierror){
  *ierror = AMPI_Group_range_incl(*group, *n, ranges, newgroup);
}
void mpi_group_range_excl(int* group,int*  n, int ranges[][3], int* newgroup, int* ierror){
  *ierror = AMPI_Group_range_excl(*group, *n, ranges, newgroup);
}
void mpi_group_free(int*  group, int*  ierror){
  *ierror = 0;
}
void mpi_comm_create(int*  comm, int*  group, int*  newcomm, int*  ierror){
  *ierror = AMPI_Comm_create(*comm, *group, newcomm);
}

void mpi_abort(int *comm, int *errorcode, int *ierr)
{
  *ierr = AMPI_Abort(*comm, *errorcode);
}

void mpi_get_count(int *sts, int *dtype, int *cnt, int *ierr)
{
  *ierr = AMPI_Get_count((MPI_Status*) sts, *dtype, cnt);
}

void mpi_print(char *str, int *len)
{
  char *buf = new char[*len+1];
  memcpy(buf, str, *len);
  buf[*len] = '\0';
  AMPI_Print(buf);
  delete [] buf;
}

void mpi_migrate(void)
{
  AMPI_Migrate();
}

void mpi_setmigratable(int *comm, int *mig)
{
  AMPI_Setmigratable(*comm, * mig);
}

void mpi_migrateto(int *destPE)
{
  AMPI_Migrateto(*destPE);
}

void mpi_start_measure()           /* turn on auto load instrumentation */
{
  AMPI_Start_measure();
}

void mpi_stop_measure()
{
  AMPI_Stop_measure();
}

void mpi_set_load(double *load)
{
  AMPI_Set_load(*load);
}

void mpi_register(void *d, MPI_PupFn f)
{
  AMPI_Register(d,f);
}

void mpi_get_userdata(int* dn, void *data)
{
  data = AMPI_Get_userdata(*dn); 
}

void mpi_checkpoint(char *dname){
  AMPI_Checkpoint(dname);
}

void mpi_memcheckpoint(){
  AMPI_MemCheckpoint();
}

void mpi_get_argc(int *c, int *ierr)
{
  *c = CkGetArgc();
  *ierr = 0;
}

void mpi_get_argv(int *c, char *str, int *ierr, int len)
{
  char ** argv = CkGetArgv();
  int nc = CkGetArgc();
  if (*c < nc) {
    strncpy(str, argv[*c], strlen(argv[*c]));
    for (int j=strlen(argv[*c]); j<len; j++)  str[j] = ' ';
    *ierr = 0;
  }
  else {
    memset(str, ' ', len);
    *ierr = 1;
  }
}

void mpi_comm_remote_size(int *comm, int *size, int *ierr){
  *ierr = AMPI_Comm_remote_size(*comm, size);
}

void mpi_comm_remote_group(int *comm, int *group, int *ierr){
  *ierr = AMPI_Comm_remote_group(*comm, group);
}

void mpi_intercomm_create(int *local_comm, int *local_leader, int *peer_comm, int *remote_leader, int *tag, int *newintercomm, int *ierr){
  *ierr = AMPI_Intercomm_create(*local_comm, *local_leader, *peer_comm, *remote_leader, *tag, newintercomm);
}

void mpi_intercomm_merge(int *intercomm, int *high, int *newintracomm, int *ierr){
  *ierr = AMPI_Intercomm_merge(*intercomm, *high, newintracomm);
}

void mpi_keyval_create(MPI_Copy_function *copy_fn, MPI_Delete_function *delete_fn, int *keyval, void* extra_state, int *ierr) {
  *ierr = AMPI_Keyval_create(copy_fn, delete_fn, keyval, extra_state);
}

void mpi_keyval_free(int *keyval, int *ierr){
  *ierr = AMPI_Keyval_free(keyval);
}

void mpi_attr_put(int *comm, int *keyval, void* attribute_val, int *ierr){
  *ierr = AMPI_Attr_put(*comm, *keyval, attribute_val);
}

void mpi_attr_get(int *comm, int *keyval, void *attribute_val, int *flag, int *ierr){
  *ierr = AMPI_Attr_get(*comm, *keyval, attribute_val, flag);
}

void mpi_attr_delete(int *comm, int *keyval, int *ierr) {
  *ierr = AMPI_Attr_delete(*comm, *keyval);
}

void mpi_type_get_envelope(int *datatype, int *num_integers, int *num_addresses,
                          int *num_datatypes, int *combiner, int *ierr){
 *ierr = AMPI_Type_get_envelope(*datatype, num_integers, num_addresses, num_datatypes, combiner);
}

void mpi_type_get_contents(int *datatype, int *max_integers, int *max_addresses,
			   int *max_datatypes, int array_of_integers[], int array_of_addresses[],
			   int array_of_datatypes[], int *ierr){
  *ierr = AMPI_Type_get_contents(*datatype, *max_integers, *max_addresses, *max_datatypes, array_of_integers, 
				array_of_addresses, array_of_datatypes);
}


void mpi_win_create(void *base, int *size, int *disp_unit,
		   int *info, int *comm, MPI_Win *newwin, int *ierr) {
  *ierr = AMPI_Win_create(base, *size, *disp_unit, *info, *comm, newwin);
}

void mpi_win_free(int *win, int *ierr) {
  *ierr = AMPI_Win_free(win);
}

void mpi_win_delete_attr(int win, int *key, int *ierr){
  *ierr = AMPI_Win_delete_attr(win, *key);
}

void mpi_win_get_group(int win, int *group, int *ierr){
  *ierr = AMPI_Win_get_group(win, group);
}

void mpi_win_set_name(int win, char *name, int *ierr){
  *ierr = AMPI_Win_set_name(win, name);
}

void mpi_win_get_name(int win, char *name, int *length, int *ierr){
  *ierr = AMPI_Win_get_name(win, name, length);
}

void mpi_win_fence(int *assertion, int win, int *ierr){
  *ierr = AMPI_Win_fence(*assertion, win);
}

void mpi_win_lock(int *lock_type, int *rank, int *assert, int win, int *ierr){
  *ierr = AMPI_Win_lock(*lock_type, *rank, *assert, win);
}

void mpi_win_unlock(int *rank, int win, int *ierr){
  *ierr = AMPI_Win_unlock(*rank, win);
}

void mpi_win_post(int *group, int *assertion, int win, int *ierr){
  *ierr = AMPI_Win_post(*group, *assertion, win);
}

void mpi_win_wait(int win, int *ierr){
  *ierr = AMPI_Win_wait(win);
}

void mpi_win_start(int *group, int *assertion, int win, int *ierr){
  *ierr = AMPI_Win_start(*group, *assertion, win);
}

void mpi_win_complete(int win, int *ierr){
  *ierr = AMPI_Win_complete(win);
}

void mpi_alloc_mem(int *size, int *info, void *baseptr, int *ierr){
  *ierr = AMPI_Alloc_mem(*size, *info, baseptr);
}

void mpi_free_mem(void *base, int *ierr){
  *ierr = AMPI_Free_mem(base);
}

void mpi_put(void *orgaddr, int *orgcnt, int *orgtype, int *rank, 
	    int *targdisp, int *targcnt, int *targtype, int win, int *ierr){
  *ierr = AMPI_Put(orgaddr, *orgcnt, *orgtype, *rank, *targdisp, *targcnt, *targtype, win);
}

void mpi_get(void *orgaddr, int *orgcnt, int *orgtype, int *rank, 
	    int *targdisp, int *targcnt, int *targtype, int win, int *ierr){
  *ierr = AMPI_Get(orgaddr, *orgcnt, *orgtype, *rank, *targdisp, *targcnt, *targtype, win);
}

void mpi_accumulate(void *orgaddr, int *orgcnt, int *orgtype, int *rank,
		   int *targdisp, int *targcnt, int *targtype, 
		   int *opc, int win, int *ierr){
  MPI_Op op = GET_MPI_OP(*opc);
  *ierr = AMPI_Accumulate(orgaddr, *orgcnt, *orgtype, *rank, *targdisp, *targcnt, *targtype, op, win);
}

void mpi_info_create(int* info, int* ierr){
  *ierr = MPI_Info_create(info);
}
void mpi_info_set(int* info, char *key, char *value, int* ierr){
  *ierr = MPI_Info_set(*info, key, value);
}
void mpi_info_delete(int* info, char* key, int* ierr){
  *ierr = MPI_Info_delete(*info, key);
}
void mpi_info_get(int* info, char *key, int *valuelen, char *value, int *flag, int* ierr){
  *ierr = MPI_Info_get(*info, key, *valuelen, value, flag);
}
void mpi_info_get_valuelen(int* info, char *key, int *valuelen, int *flag, int* ierr){
  *ierr = MPI_Info_get_valuelen(*info, key, valuelen, flag);
}
void mpi_info_get_nkeys(int* info, int *nkeys, int* ierr){
  *ierr = MPI_Info_get_nkeys(*info, nkeys);
}
void mpi_info_get_nthkey(int* info, int *n, char *key, int* ierr){
  *ierr = MPI_Info_get_nthkey(*info, *n, key);
}
void mpi_info_dup(int* info, int* newinfo, int* ierr){
  *ierr = MPI_Info_dup(*info, newinfo);
}
void mpi_info_free(int* info, int* ierr){
  *ierr = MPI_Info_free(info);
}

void mpi_info_maxmemory(){
  CkPrintf("MaxMemory %ld\n", CmiMaxMemoryUsage());
}

void mpi_info_memory(){
  CkPrintf("Memory %ld\n", CmiMemoryUsage());
}

#define begintracebigsim FTN_NAME (BEGINTRACEBIGSIM , begintracebigsim)
#define endtracebigsim FTN_NAME (ENDTRACEBIGSIM , endtracebigsim)
void begintracebigsim(char* msg){
  beginTraceBigSim(msg);
}
void endtracebigsim(char* msg, char* param){
  endTraceBigSim(msg, param);
}

} // extern "C"


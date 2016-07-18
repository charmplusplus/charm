/* 

The following code is adapted from alltoall.c in mpich2-1.0.3 

Licensing details should be addresssed, since this is copyrighted. 

*/

#include "ampiimpl.h"
#include "tcharm.h"
#include "ampiEvents.h" /*** for trace generation for projector *****/
#include "ampiProjections.h"


/* This is the default implementation of alltoall. The algorithm is:
   
   Algorithm: MPI_Alltoall

   We use four algorithms for alltoall. For short messages and
   (comm_size >= 8), we use the algorithm by Jehoshua Bruck et al,
   IEEE TPDS, Nov. 1997. It is a store-and-forward algorithm that
   takes lgp steps. Because of the extra communication, the bandwidth
   requirement is (n/2).lgp.beta.

   Cost = lgp.alpha + (n/2).lgp.beta

   where n is the total amount of data a process needs to send to all
   other processes.

   For medium size messages and (short messages for comm_size < 8), we
   use an algorithm that posts all irecvs and isends and then does a
   waitall. We scatter the order of sources and destinations among the
   processes, so that all processes don't try to send/recv to/from the
   same process at the same time.

   For long messages and power-of-two number of processes, we use a
   pairwise exchange algorithm, which takes p-1 steps. We
   calculate the pairs by using an exclusive-or algorithm:
           for (i=1; i<comm_size; i++)
               dest = rank ^ i;
   This algorithm doesn't work if the number of processes is not a power of
   two. For a non-power-of-two number of processes, we use an
   algorithm in which, in step i, each process  receives from (rank-i)
   and sends to (rank+i). 

   Cost = (p-1).alpha + n.beta

   where n is the total amount of data a process needs to send to all
   other processes.

   Possible improvements: 

   End Algorithm: MPI_Alltoall
*/




/////////////////////////////////////////////////////////////////////////////////////////////////////
//   HELPER FUNCTIONS:
/////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef MAX
int MAX(int a, int b){
  if(a>b)
	return a;
  else
	return b;
}
#endif

#if 0
int MPI_Pack_size(int incount, MPI_Datatype type, MPI_Comm comm, int *size)
{
  CkDDT_DataType *ddt = getAmpiInstance(comm)->getDDT()->getType(type);
  int typesize = ddt->getSize();
  *size = incount * typesize;
  return MPI_SUCCESS;
}
#endif

// A simplified version of the mpich MPICH_Localcopy function
// TODO: This should do a memcpy when data is contiguous (see original)

void MPICH_Localcopy(void *sendbuf, int sendcount, MPI_Datatype sendtype,
					 void *recvbuf, int recvcount, MPI_Datatype recvtype)
{
  int rank;
 
  AMPI_Comm_rank (MPI_COMM_WORLD, &rank);
  getAmpiInstance(MPI_COMM_WORLD)->sendrecv ( sendbuf, sendcount, sendtype,
				  rank, MPI_ATA_TAG, 
				  recvbuf, recvcount, recvtype,
				  rank, MPI_ATA_TAG,
				  MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}


inline void MPID_Datatype_get_extent_macro(MPI_Datatype &type, MPI_Aint &extent){
  CkDDT_DataType *ddt = getAmpiInstance(MPI_COMM_WORLD)->getDDT()->getType(type);
  extent = ddt->getExtent();
}

inline void MPID_Datatype_get_size_macro(MPI_Datatype &type, int &size){
  CkDDT_DataType *ddt = getAmpiInstance(MPI_COMM_WORLD)->getDDT()->getType(type);
  size = ddt->getSize();
}


/////////////////////////////////////////////////////////////////////////////////////////////////////
//   LONG MESSAGES
/////////////////////////////////////////////////////////////////////////////////////////////////////


/* Long message. If comm_size is a power-of-two, do a pairwise
   exchange using exclusive-or to create pairs. Else send to
   rank+i, receive from rank-i. */

int AMPI_Alltoall_long(
						void *sendbuf, 
						int sendcount, 
						MPI_Datatype sendtype, 
						void *recvbuf, 
						int recvcount, 
						MPI_Datatype recvtype, 
						MPI_Comm comm )
{

  int          comm_size, i, pof2;
  MPI_Aint     sendtype_extent, recvtype_extent;
 
  int src, dst, rank, nbytes;
  MPI_Status status;
  int sendtype_size;

  if (sendcount == 0) return MPI_SUCCESS;
  
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &comm_size);
 
    
  /* Get extent of send and recv types */
  MPID_Datatype_get_extent_macro(recvtype, recvtype_extent);
  MPID_Datatype_get_extent_macro(sendtype, sendtype_extent);

  MPID_Datatype_get_size_macro(sendtype, sendtype_size);
  nbytes = sendtype_size * sendcount;
  

  /* Make local copy first */
  MPICH_Localcopy(((char *)sendbuf + 
				   rank*sendcount*sendtype_extent), 
				  sendcount, sendtype, 
				  ((char *)recvbuf +
				   rank*recvcount*recvtype_extent),
				  recvcount, recvtype);
  

  /* Is comm_size a power-of-two? */
  i = 1;
  while (i < comm_size)
	i *= 2;
  if (i == comm_size)
	pof2 = 1;
  else 
	pof2 = 0;

  /* Do the pairwise exchanges */
  for (i=1; i<comm_size; i++) {
	if (pof2 == 1) {
	  /* use exclusive-or algorithm */
	  src = dst = rank ^ i;
	}
	else {
	  src = (rank - i + comm_size) % comm_size;
	  dst = (rank + i) % comm_size;
	}

	getAmpiInstance(comm)->sendrecv(((char *)sendbuf +
							   dst*sendcount*sendtype_extent), 
							  sendcount, sendtype, dst,
							  MPI_ATA_TAG, 
							  ((char *)recvbuf +
							   src*recvcount*recvtype_extent),
							  recvcount, recvtype, src,
							  MPI_ATA_TAG, comm, &status);
  }

  return MPI_SUCCESS;
}


/////////////////////////////////////////////////////////////////////////////////////////////////////
// SHORT MESSAGES
/////////////////////////////////////////////////////////////////////////////////////////////////////

#if 0
int AMPI_Alltoall_short(
						 void *sendbuf, 
						 int sendcount, 
						 MPI_Datatype sendtype, 
						 void *recvbuf, 
						 int recvcount, 
						 MPI_Datatype recvtype, 
						 MPI_Comm comm )
{

  int          comm_size, i, pof2;
  MPI_Aint     sendtype_extent, recvtype_extent;
 
  int mpi_errno=MPI_SUCCESS, src, dst, rank, nbytes;
  MPI_Status status;
  void *tmp_buf;
  int sendtype_size, pack_size, block, position, *displs, count;

  MPI_Datatype newtype;
  MPI_Aint recvtype_true_extent, recvbuf_extent, recvtype_true_lb;


  if (sendcount == 0) return MPI_SUCCESS;
  
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &comm_size);
    
  /* Get extent of send and recv types */
  MPID_Datatype_get_extent_macro(recvtype, recvtype_extent);
  MPID_Datatype_get_extent_macro(sendtype, sendtype_extent);

  MPID_Datatype_get_size_macro(sendtype, sendtype_size);
  nbytes = sendtype_size * sendcount;
    
  /* use the indexing algorithm by Jehoshua Bruck et al,
   * IEEE TPDS, Nov. 97 */ 

  /* allocate temporary buffer */
  MPI_Pack_size(recvcount*comm_size, recvtype, comm, &pack_size);
  tmp_buf = malloc(pack_size);
  CkAssert(tmp_buf);

  /* Do Phase 1 of the algorithim. Shift the data blocks on process i
   * upwards by a distance of i blocks. Store the result in recvbuf. */
  MPICH_Localcopy((char *) sendbuf + rank*sendcount*sendtype_extent, 
				  (comm_size - rank)*sendcount, sendtype, recvbuf, 
				  (comm_size - rank)*recvcount, recvtype);
	    
  MPICH_Localcopy(sendbuf, rank*sendcount, sendtype, 
				  (char *) recvbuf + (comm_size-rank)*recvcount*recvtype_extent, 
				  rank*recvcount, recvtype);
	    			
  /* Input data is now stored in recvbuf with datatype recvtype */

  /* Now do Phase 2, the communication phase. It takes
	 ceiling(lg p) steps. In each step i, each process sends to rank+2^i
	 and receives from rank-2^i, and exchanges all data blocks
	 whose ith bit is 1. */

  /* allocate displacements array for indexed datatype used in
	 communication */

  displs = (int*)malloc(comm_size * sizeof(int));
  CkAssert(displs);


  pof2 = 1;
  while (pof2 < comm_size) {
	dst = (rank + pof2) % comm_size;
	src = (rank - pof2 + comm_size) % comm_size;

	/* Exchange all data blocks whose ith bit is 1 */
	/* Create an indexed datatype for the purpose */

	count = 0;
	for (block=1; block<comm_size; block++) {
	  if (block & pof2) {
		displs[count] = block * recvcount;
		count++;
	  }
	}

	mpi_errno = MPI_Type_create_indexed_block(count, recvcount, displs, recvtype, &newtype);

	if (mpi_errno)
	  return mpi_errno;

	mpi_errno = MPI_Type_commit(&newtype);

	if (mpi_errno)
	  return mpi_errno;
	    
	position = 0;
	mpi_errno = MPI_Pack(recvbuf, 1, newtype, tmp_buf, pack_size, 
						  &position, comm);

	getAmpiInstance(comm)->sendrecv(tmp_buf, position, MPI_PACKED, dst,
							  MPI_ATA_TAG, recvbuf, 1, newtype,
							  src, MPI_ATA_TAG, comm,
							  MPI_STATUS_IGNORE);
	    
	if (mpi_errno)
	  return mpi_errno;
	    

	mpi_errno = MPI_Type_free(&newtype);
	   
	if (mpi_errno)
	  return mpi_errno;

	pof2 *= 2;
  }

  free(displs);
  free(tmp_buf);

  /* Rotate blocks in recvbuf upwards by (rank + 1) blocks. Need
   * a temporary buffer of the same size as recvbuf. */
        
  /* get true extent of recvtype */
  mpi_errno = MPI_Type_get_true_extent(recvtype, &recvtype_true_lb,
										&recvtype_true_extent);  

  if (mpi_errno)
	return mpi_errno;

  recvbuf_extent = recvcount * comm_size *
	(MAX(recvtype_true_extent, recvtype_extent));
  tmp_buf = malloc(recvbuf_extent);
  CkAssert(tmp_buf);

  /* adjust for potential negative lower bound in datatype */
  tmp_buf = (void *)((char*)tmp_buf - recvtype_true_lb);

  MPICH_Localcopy((char *) recvbuf + (rank+1)*recvcount*recvtype_extent, 
				  (comm_size - rank - 1)*recvcount, recvtype, tmp_buf, 
				  (comm_size - rank - 1)*recvcount, recvtype);
			
  MPICH_Localcopy(recvbuf, (rank+1)*recvcount, recvtype, 
				  (char *) tmp_buf + (comm_size-rank-1)*recvcount*recvtype_extent, 
				  (rank+1)*recvcount, recvtype);
	
        
  /* Blocks are in the reverse order now (comm_size-1 to 0). 
   * Reorder them to (0 to comm_size-1) and store them in recvbuf. */

  for (i=0; i<comm_size; i++) 
	MPICH_Localcopy((char *) tmp_buf + i*recvcount*recvtype_extent,
					recvcount, recvtype, 
					(char *) recvbuf + (comm_size-i-1)*recvcount*recvtype_extent, 
					recvcount, recvtype); 

  free((char*)tmp_buf + recvtype_true_lb);

}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////
// MEDIUM MESSAGES
/////////////////////////////////////////////////////////////////////////////////////////////////////

int AMPI_Alltoall_medium(
						  void *sendbuf, 
						  int sendcount, 
						  MPI_Datatype sendtype, 
						  void *recvbuf, 
						  int recvcount, 
						  MPI_Datatype recvtype, 
						  MPI_Comm comm )
{

  int          comm_size, i;
  MPI_Aint     sendtype_extent, recvtype_extent;
 
  int mpi_errno=MPI_SUCCESS, dst, rank, nbytes;
  int sendtype_size;

  MPI_Request *reqarray;
  MPI_Status *starray;

  if (sendcount == 0) return MPI_SUCCESS;
  
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &comm_size);
    
  /* Get extent of send and recv types */
  MPID_Datatype_get_extent_macro(recvtype, recvtype_extent);
  MPID_Datatype_get_extent_macro(sendtype, sendtype_extent);

  MPID_Datatype_get_size_macro(sendtype, sendtype_size);
  nbytes = sendtype_size * sendcount;
    
  /* Medium-size message. Use isend/irecv with scattered destinations */

  reqarray = (MPI_Request *) malloc(2*comm_size*sizeof(MPI_Request));
        
  if (!reqarray) 
	return MPI_ERR_OTHER;
        
  starray = (MPI_Status *) malloc(2*comm_size*sizeof(MPI_Status));
  if (!starray) {
        free(reqarray);
	return MPI_ERR_OTHER;
  }
        
  /* do the communication -- post all sends and receives: */
  ampi *ptr = getAmpiInstance(comm);
  for ( i=0; i<comm_size; i++ ) { 
	dst = (rank+i) % comm_size;
    ptr->irecv((char *)recvbuf + dst*recvcount*recvtype_extent, recvcount, recvtype, dst,
               MPI_ATA_TAG, comm, &reqarray[i]);
  }

  for ( i=0; i<comm_size; i++ ) { 
	dst = (rank+i) % comm_size;
	/*mpi_errno = AMPI_Isend((char *)sendbuf + dst*sendcount*sendtype_extent,
	    sendcount, sendtype, dst, MPI_ATA_TAG, comm, &reqarray[i+comm_size]);*/
	ptr->send(MPI_ATA_TAG, getAmpiInstance(comm)->getRank(comm),
              (char *)sendbuf + dst*sendcount*sendtype_extent,
		      sendcount, sendtype, dst, comm);
	reqarray[i+comm_size] = MPI_REQUEST_NULL;
  }

  /* ... then wait for *all* of them to finish: */
  mpi_errno = AMPI_Waitall(2*comm_size,reqarray,starray);

  /* --BEGIN ERROR HANDLING-- */
//   if (mpi_errno == MPI_ERR_IN_STATUS) {
// 	for (int j=0; j<2*comm_size; j++) {
// 	  if (starray[j] != MPI_SUCCESS) 
// 		mpi_errno = starray[j];
// 	}
//   }
  /* --END ERROR HANDLING-- */

  free(starray);
  free(reqarray);
  
  return mpi_errno;
}



/////////////////////////////////////////////////////////////////////////////////////////////////////
// MPICH OLD VERSION -- coming soon, once I figure out how it worked
/////////////////////////////////////////////////////////////////////////////////////////////////////



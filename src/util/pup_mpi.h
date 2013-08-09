/*
PUP -> MPI interface routines


Orion Sky Lawlor, olawlor@acm.org, 2004/9/15
*/
#ifndef __UIUC_CHARM_PUP_MPI_H
#define __UIUC_CHARM_PUP_MPI_H

#include "pup.h"
#include "mpi.h"

#define pup_checkMPI(err) pup_checkMPIerr(err,__FILE__,__LINE__);
inline void pup_checkMPIerr(int mpi_err,const char *file,int line) {
	if (mpi_err!=MPI_SUCCESS) {
		CmiError("MPI Routine returned error %d at %s:%d\n",
			mpi_err,file,line);
		CmiAbort("MPI Routine returned error code");
	}
}

/// Return the number of dt's in the next MPI message from/tag/comm.
inline int MPI_Incoming_pup(MPI_Datatype dt,int from,int tag,MPI_Comm comm) {
	MPI_Status sts;
	pup_checkMPI(MPI_Probe(from,tag,comm,&sts));
	int len; pup_checkMPI(MPI_Get_count(&sts,dt,&len));
	return len;
}

/// MPI_Recv, but using an object T with a pup routine.
template <class T>
inline void MPI_Recv_pup(T &t, int from,int tag,MPI_Comm comm) {
	int len=MPI_Incoming_pup(MPI_BYTE,from,tag,comm);
	MPI_Status sts;
	char *buf=new char[len];
	pup_checkMPI(MPI_Recv(buf,len,MPI_BYTE, from,tag,comm,&sts));
	PUP::fromMemBuf(t,buf,len);
	delete[] buf;
}

/// MPI_Send, but using an object T with a pup routine.
template <class T>
inline void MPI_Send_pup(T &t, int to,int tag,MPI_Comm comm) {
	size_t len=PUP::size(t); char *buf=new char[len];
	PUP::toMemBuf(t,buf,len);
	pup_checkMPI(MPI_Send(buf,len,MPI_BYTE, to,tag,comm));
	delete[] buf;
}

/// MPI_Bcast, but using an object T with a pup routine.
template <class T>
inline void MPI_Bcast_pup(T &t, int root,MPI_Comm comm) {
	int myRank;
	MPI_Comm_rank(comm,&myRank);
	/* Can't do broadcast until everybody knows the size */
	size_t len=0;
	if(myRank == root) len=PUP::size(t); 
	pup_checkMPI(MPI_Bcast(&len,1,MPI_INT,root,comm));
	/* Now pack object and send off */
	char *buf=new char[len];
	if(myRank == root) PUP::toMemBuf(t,buf,len);
	pup_checkMPI(MPI_Bcast(buf,len,MPI_BYTE, root,comm));
	PUP::fromMemBuf(t,buf,len);
	delete [] buf;
}


#endif

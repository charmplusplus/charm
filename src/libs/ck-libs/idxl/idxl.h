/*
IDXL--Index List communication library
C++ include file, for use by libraries.

This is a low-level bare-bones communication library.
The basic primitive is an "Index List", a list of user
array entries to send and receive.  This basic communication
primitive is enough to represent (for example) FEM shared
nodes or an FEM ghost layer; the ghost cells in a stencil-based
CFD computation, etc.

Orion Sky Lawlor, olawlor@acm.org, 1/3/2003
*/

#ifndef __CHARM_IDXL_H
#define __CHARM_IDXL_H

#include "mpi.h"
#include "pup.h"
#include "idxlc.h" /* C interface */
#include "idxl_comm.h" /* basic comm. data structures */
#include "idxl_layout.h" /* user-data description */
#include "tcharmc.h" /* for TCHARM_Get/Set_global */

void IDXL_Abort(const char *callingRoutine,const char *msg,int m0=0,int m1=0,int m2=0);

/**
 * IDXL_Comm is the implementation of the user type IDXL_Comm.
 * It is the representation for an unfinished communication operation.
 * This object's life cycle is:
 *   -constructor
 *   -repeated add/recv/sum calls
 *   -single send call
 *   -repeated recv/sum calls
 *   -recv(msg) until isDone returns true
 */
class IDXL_Comm {
public: //<- Sun CC demands these types be public for use from an inner class
	typedef enum { send_t=17,recv_t,sum_t} op_t;
	
	/// This class represents one communication operation:
	///   a send/sum or send/recv.
	class sto_t { public:
		const IDXL_Side *idx; ///< Indices to read from/write to
		const IDXL_Layout *dtype; ///< Format of user data
		void *data; ///< User data to read from/write to
		op_t op; ///< Operation to perform
		
		sto_t(const IDXL_Side *idx_,const IDXL_Layout *dtype_,void *data_,op_t op_)
			:idx(idx_), dtype(dtype_), data(data_), op(op_) {}
		sto_t(void) {}
	};
	
	/// This class represents an MPI send or receive operation.
	class msg_t { 
		CkVec<char> buf; /* Send/receive buffer */
	public:
		sto_t *sto; /* Indices to send/receive */
		int ll; /* Local processor to communicate with */
		void allocate(int len) {
			buf.resize(len);
		}
		void *getBuf(void) {return &buf[0];}
	};

private:
	/// Stuff to send/receive.
	CkVec<sto_t> sto;
	/// Message buffers for each processor.
	///  Only the first nMsgs elements are used this time,
	///  but the remaining elements stay allocated (lazy deallocation).
	///  This avoids slow memory allocation at runtime.
	CkVec<msg_t *> msg; 
	int nMsgs;
	
	/// List of outgoing MPI requests (needs to be a separate array for MPI)
	///  Length is always nMsgs.
	CkVec<MPI_Request> msgReq;
	CkVec<MPI_Status> msgSts;
	
	int tag; MPI_Comm comm;
	bool isPost; //If true, no more adds are allowed
	bool isDone; //If true, ready to reset
public:
	IDXL_Comm(int tag,int context);
	void reset(int tag,int context);
	~IDXL_Comm();
	
	// prepare to write this field to the message:
	void send(const IDXL_Side *idx,const IDXL_Layout *dtype,const void *src);
	
	// prepare to recv and copy out this field
	void recv(const IDXL_Side *idx,const IDXL_Layout *dtype,void *dest);
	void sum(const IDXL_Side *idx,const IDXL_Layout *dtype,void *srcdest);
	
	// send off our added fields to their destinations & post receives
	void post(void);
	
	// wait until this communication is complete.
	void wait(void);
	
	/// Return true if we've sent off our messages
	bool isPosted(void) { return isPost; }
	bool isComplete(void) {return isDone;}
};

// This is our TCharm global ID:
enum {IDXL_globalID=32};

/**
 * IDXL_Chunk exists for just two reasons: 
 *     - To do communication on IDXL's--the idxl_recv method.
 *     - To keep track of the various registered idxl entities.
 *
 * For IDXL_Chunk to do its job, you're supposed to inherit from it.
 */
class IDXL_Chunk {
  // MPI Communicator to use by default
  int mpi_comm;
  
  // Lists of index lists: 
  
  /// "Static" index list is for system entities like the FEM
  ///  framework that want to allocate and delete the lists themselves.
  /// Stores IDXL_t's from IDXL_STATIC_IDXL_T to IDXL_LAST_IDXL_T
  CkVec<IDXL *> static_idxls;
  
  /// "Dynamic" index list is for ordinary user-created lists,
  ///   where this class manages their allocation and deallocation.
  /// Stores IDXL_t's from IDXL_DYNAMIC_IDXL_T to IDXL_STATIC_IDXL_T
  CkVec<IDXL *> dynamic_idxls;
  
  // Return the next free index in this table, or -1 if none:
  int storeToFreeIndex(CkVec<IDXL *> &inList,IDXL *store) {
  	int i;
	for (i=0;i<inList.size();i++)
		if (inList[i]==NULL) {
			inList[i]=store;
			return i;
		}
	i=inList.size();
	inList.push_back(store);
	return i;
  }
  
  // Lists ongoing communications (FIXME: add list here)
  IDXL_Comm *currentComm; //The ongoing communicator
  
  void init(void);
public:
	IDXL_Chunk(int mpi_comm_);
	IDXL_Chunk(CkMigrateMessage *m);
	void pup(PUP::er &p);
	~IDXL_Chunk();
	
  static IDXL_Chunk *getNULL(void) {
  	return (IDXL_Chunk *)TCHARM_Get_global(IDXL_globalID);
  }
  static IDXL_Chunk *get(const char *callingRoutine);
	
// Manipulate index lists (IDXL's):
  /// Dynamically create a new empty IDXL.  Must eventually call destroy.
  IDXL_t addDynamic(void);
  /// Register this statically-allocated IDXL, possibly at this index
  IDXL_t addStatic(IDXL *idx,IDXL_t at=-1);
  
  /// Check this IDXL for validity
  void check(IDXL_t u,const char *callingRoutine="") const;
  
  /// Find this previously allocated IDXL:
  IDXL &lookup(IDXL_t u,const char *callingRoutine="");
  const IDXL &lookup(IDXL_t u,const char *callingRoutine="") const;
  
  /// Done with this IDXL.  Deallocates if a dynamic IDXL.
  void destroy(IDXL_t t,const char *callingRoutine="");
  
// Manipulate user-data format descriptors:
  IDXL_Layout_List layouts;

// Manipulate ongoing communication:
  IDXL_Comm_t addComm(int tag,int context);
  IDXL_Comm *lookupComm(IDXL_Comm_t uc,const char *callingRoutine="");
  void waitComm(IDXL_Comm *comm);
};

#define IDXLAPI(routineName) TCHARM_API_TRACE(routineName,"IDXL");

#endif




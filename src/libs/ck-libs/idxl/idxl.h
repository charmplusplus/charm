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

#include "pup.h"
#include "tcharm.h"
#include "idxlc.h" /* C interface */
#include "idxl_comm.h" /* basic comm. data structures */
#include "idxl_layout.h" /* user-data description */

class CProxy_IDXL_Chunk;
class IDXL_DataMsg;

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
	typedef enum { add_t=17,recv_t,sum_t} op_t;
	class sto_t { public:
		const IDXL_Side *idx; //Indices to read from/write to
		const IDXL_Layout *dtype; //Format of user data
		void *data; //User data to read from/write to
		op_t op; //Operation to perform
		
		sto_t(const IDXL_Side *idx_,const IDXL_Layout *dtype_,void *data_,op_t op_)
			:idx(idx_), dtype(dtype_), data(data_), op(op_) {}
		sto_t(void)
			:idx(0), dtype(0), data(0), op((op_t)0) {}
	};
	enum {maxSto=20};
	sto_t sto[maxSto]; //Stuff to put in each message
	int nSto, nStoAdd, nStoRecv;
	
	int seqnum, tag, context;
	int nRecv; //Number of messages remaining to be recv'd.
	bool isSent; //If true, no more adds are allowed
	bool beginRecv; //If true, no more recv or sums are allowed
public:
	IDXL_Comm(int seqnum, int tag,int context);
	IDXL_Comm(void) {nSto=nRecv=0; isSent=true; beginRecv=true;}
	
	// prepare to write this field to the message:
	void send(const IDXL_Side *idx,const IDXL_Layout *dtype,const void *src);
	
	// send off our added fields to their destinations in this array
	void flush(int src,const CkArrayID &chunkArray);
	
	// prepare to recv and copy out this field
	void recv(const IDXL_Side *idx,const IDXL_Layout *dtype,void *dest);
	void sum(const IDXL_Side *idx,const IDXL_Layout *dtype,void *srcdest);
	
	// If this is one of our messages, copy out the user data out,
	//  delete the message, and return true. Return false if not ours.
	bool recv(IDXL_DataMsg *msg);
	
	/// Return true if we've sent off our messages
	bool isFlushed(void) { return isSent; }
	
	// Return true if we expect no more messages
	bool isDone(void) const { return isSent && (nRecv<=0); }
};



/**
 * IDXL_Chunk exists for just two reasons: 
 *     - To do communication on IDXL's--the idxl_recv method.
 *     - To keep track of the various registered idxl entities.
 *
 * For IDXL_Chunk to do its job, you're supposed to inherit from it.
 */
class IDXL_Chunk : public TCharmClient1D {
  typedef TCharmClient1D super;
  
  // List of index lists: 
  // first has a static part for stuff we can't throw away (indices 0..STATIC_IDXL-1)
  // then a dynamic part for user-allocated stuff (indices STATIC_IDXL...LAST_IDXL-1)
  enum {FIRST_IDXL=1550000000, STATIC_IDXL=32, LAST_IDXL=64};
  IDXL *idxls[LAST_IDXL];

  // Lists ongoing communications
  CkMsgQ<IDXL_DataMsg> messages; // update messages to be processed
  int updateSeqnum; // sequence number for last update operation
  IDXL_Comm currentComm; //The ongoing communicator
  IDXL_Comm *blockedOnComm; //If non-null, we've blocked the thread on this comm.
  
  void init(void);
protected:
	virtual void setupThreadPrivate(CthThread forThread);
public:
	IDXL_Chunk(const CkArrayID &threadArrayID);
	IDXL_Chunk(CkMigrateMessage *m);
	void pup(PUP::er &p);
	~IDXL_Chunk();
	
	static IDXL_Chunk *lookup(const char *callingRoutine);
	
	void idxl_recv(IDXL_DataMsg *m);
	
// Manipulate index lists (IDXL's):
  /// Dynamically create a new empty IDXL.  Must eventually call destroy.
  IDXL_t addDynamic(void);
  /// Register this statically-allocated IDXL, possibly at this index
  IDXL_t addStatic(IDXL *idx,IDXL_t at=-1);
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
  inline void flushComm(IDXL_Comm *comm) {
  	comm->flush(thisIndex,thisArrayID);
  }
  void waitComm(IDXL_Comm *comm);
  
};

#define IDXLAPI(routineName) TCHARM_API_TRACE(routineName,"IDXL");

/// Get the currently active layout list.
///  In driver, this is IDXL_Chunk's "layouts" member.
///  Elsewhere (e.g., in init), this is a local member.
IDXL_Layout_List &getLayouts(void);

#endif




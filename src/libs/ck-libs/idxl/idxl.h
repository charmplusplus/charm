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
public: //<- Sun CC demands these types be public for use from an inner class
	typedef enum { send_t=17,recv_t,sum_t} op_t;
	class sto_t { public:
		const IDXL_Side *idx; //Indices to read from/write to
		const IDXL_Layout *dtype; //Format of user data
		void *data; //User data to read from/write to
		op_t op; //Operation to perform
		
		sto_t(const IDXL_Side *idx_,const IDXL_Layout *dtype_,void *data_,op_t op_)
			:idx(idx_), dtype(dtype_), data(data_), op(op_) {}
		sto_t(void) {}
	};
	class msg_t { public:
		sto_t *sto; /* Indices to send/receive */
		int ll; /* Local processor to communicate with */
		char *buf; /* Send/receive buffer */
		int bufLen; /* bytes in send/receive buffer */
		void allocate(int len) {
			if (buf) {delete[] buf;buf=NULL;}
			buf=new char[len];
			bufLen=len;
		}
		msg_t() :buf(NULL) {}
		~msg_t() {if (buf) {delete[] buf;}}
	};

private:
	enum {maxSto=20};
	sto_t sto[maxSto]; //Stuff to send/receive
	int nSto;
	
	enum {maxMsg=50};
	msg_t msg[maxMsg]; //Messages to each processor
	MPI_Request msgReq[maxMsg];
	int nMsg;
	
	int tag; MPI_Comm comm;
	bool isPost; //If true, no more adds are allowed
public:
	IDXL_Comm(int tag,int context);
	
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
};



/**
 * IDXL_Chunk exists for just two reasons: 
 *     - To do communication on IDXL's--the idxl_recv method.
 *     - To keep track of the various registered idxl entities.
 *
 * For IDXL_Chunk to do its job, you're supposed to inherit from it.
 */
class IDXL_Chunk {
  
  // List of index lists: 
  // first has a static part for stuff we can't throw away (indices 0..STATIC_IDXL-1)
  // then a dynamic part for user-allocated stuff (indices STATIC_IDXL...LAST_IDXL-1)
  enum {FIRST_IDXL=1550000000, STATIC_IDXL=32, LAST_IDXL=64};
  IDXL *idxls[LAST_IDXL];

  // Lists ongoing communications (FIXME: add list here)
  IDXL_Comm *currentComm; //The ongoing communicator
  
  void init(void);
public:
	IDXL_Chunk(void);
	IDXL_Chunk(CkMigrateMessage *m);
	void pup(PUP::er &p);
	~IDXL_Chunk();
	
  static IDXL_Chunk *get(const char *callingRoutine);
	
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
  void waitComm(IDXL_Comm *comm);
};

#define IDXLAPI(routineName) TCHARM_API_TRACE(routineName,"IDXL");

#endif




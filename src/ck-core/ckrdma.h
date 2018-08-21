/*
 * Charm Onesided API Utility Functions
 */

#ifndef _CKRDMA_H_
#define _CKRDMA_H_

#include "envelope.h"

/*********************************** Zerocopy Direct API **********************************/

#define CK_BUFFER_REG     CMK_BUFFER_REG
#define CK_BUFFER_UNREG   CMK_BUFFER_UNREG
#define CK_BUFFER_PREREG  CMK_BUFFER_PREREG
#define CK_BUFFER_NOREG   CMK_BUFFER_NOREG

#define CK_BUFFER_DEREG     CMK_BUFFER_DEREG
#define CK_BUFFER_NODEREG   CMK_BUFFER_NODEREG

#ifndef CMK_NOCOPY_DIRECT_BYTES

#if defined(_WIN32)
#define CMK_NOCOPY_DIRECT_BYTES 1
/* It is required to declare CMK_NOCOPY_DIRECT_BYTES to 1 instead of 0
 * as this avoids the C2229 error (illegal zero-sized array)
 * for char layerInfo[CMK_NOCOPY_DIRECT_BYTES] which is seen for
 * a 0 sized array on VC++
 */
#else
#define CMK_NOCOPY_DIRECT_BYTES 0
#endif // end of if defined(_WIN32)

#endif // end of ifndef CMK_NOCOPY_DIRECT_BYTES

#ifndef CMK_COMMON_NOCOPY_DIRECT_BYTES
#define CMK_COMMON_NOCOPY_DIRECT_BYTES 0
#endif

#define CkRdmaAlloc CmiRdmaAlloc
#define CkRdmaFree  CmiRdmaFree

// Represents the mode of the zerocopy transfer
// CkNcpyMode::MEMCPY indicates that the PEs are on the logical node and memcpy can be used
// CkNcpyMode::CMA indicates that the PEs are on the same physical node and CMA can be used
// CkNcpyMode::RDMA indicates that the neither MEMCPY or CMA can be used and REMOTE Direct Memory Access needs to be used
enum class CkNcpyMode : char { MEMCPY, CMA, RDMA };

// Represents the completion status of the zerocopy transfer (used as a return value for CkNcpyBuffer::get & CkNcpyBuffer:::put)
// CMA and MEMCPY transfers complete instantly and return CkNcpyStatus::complete
// RDMA transfers use a remote asynchronous call and hence return CkNcpyStatus::incomplete
enum class CkNcpyStatus : char { incomplete, complete };

// P2P_SEND mode is used for EM P2P Send API
// BCAST_SEND mode is used for EM BCAST Send API
// P2P_RECV mode is used for EM P2P Recv API
// BCAST_RECV mode is used for EM BCAST Send API
enum class ncpyEmApiMode : char { P2P_SEND, BCAST_SEND, P2P_RECV, BCAST_RECV };

// Struct passed in a ZC Post Entry Method to allow receiver side to post 
struct CkNcpyBufferPost {
  // regMode
  unsigned short int regMode;

  // deregMode
  unsigned short int deregMode;
};

// Class to represent an Zerocopy buffer
// CkSendBuffer(....) passed by the user internally translates to a CkNcpyBuffer
class CkNcpyBuffer{

  private:

  // bool to indicate registration for current values of ptr and cnt on pe
  bool isRegistered;

  // machine specific information about the buffer
  #ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wpedantic"
  #endif
  char layerInfo[CMK_COMMON_NOCOPY_DIRECT_BYTES + CMK_NOCOPY_DIRECT_BYTES];
  #ifdef __GNUC__
  #pragma GCC diagnostic pop
  #endif

#if CMK_ERROR_CHECKING
  void checkRegModeIsValid() {
    if(regMode < CK_BUFFER_REG || regMode > CK_BUFFER_NOREG) CmiAbort("checkRegModeIsValid: Invalid value for regMode!\n");
  }

  void checkDeregModeIsValid() {
    if(deregMode < CK_BUFFER_DEREG || deregMode > CK_BUFFER_NODEREG) CmiAbort("checkDeregModeIsValid: Invalid value for deregMode!\n");
  }
#endif

  public:
  // pointer to the buffer
  const void *ptr;

  // number of bytes
  size_t cnt;

  // callback to be invoked on the sender/receiver
  CkCallback cb;

  // home pe
  int pe;

  // regMode
  unsigned short int regMode;

  // deregMode
  unsigned short int deregMode;

  // reference pointer
  const void *ref;

  // bcast ack handling pointer
  const void *bcastAckInfo;

  CkNcpyBuffer() : isRegistered(false), ptr(NULL), cnt(0), pe(-1), regMode(CK_BUFFER_REG), deregMode(CK_BUFFER_DEREG), ref(NULL), bcastAckInfo(NULL) {}

  explicit CkNcpyBuffer(const void *ptr_, size_t cnt_, unsigned short int regMode_=CK_BUFFER_REG, unsigned short int deregMode_=CK_BUFFER_DEREG) {
    cb = CkCallback(CkCallback::ignore);
    init(ptr_, cnt_, regMode_, deregMode_);
  }

  explicit CkNcpyBuffer(const void *ptr_, size_t cnt_, CkCallback &cb_, unsigned short int regMode_=CK_BUFFER_REG, unsigned short int deregMode_=CK_BUFFER_DEREG) {
    init(ptr_, cnt_, cb_, regMode_, deregMode_);
  }

  void print() {
    CkPrintf("[%d][%d][%d] CkNcpyBuffer print: ptr:%p, size:%d, pe:%d, regMode=%d, deregMode=%d, ref:%p, bcastAckInfo:%p\n", CmiMyPe(), CmiMyNode(), CmiMyRank(), ptr, cnt, pe, regMode, deregMode, ref, bcastAckInfo);
  }

  void init(const void *ptr_, size_t cnt_, CkCallback &cb_, unsigned short int regMode_=CK_BUFFER_REG, unsigned short int deregMode_=CK_BUFFER_DEREG) {
    cb   = cb_;
    init(ptr_, cnt_, regMode_, deregMode_);
  }

  void init(const void *ptr_, size_t cnt_, unsigned short int regMode_=CK_BUFFER_REG, unsigned short int deregMode_=CK_BUFFER_DEREG) {
    ptr  = ptr_;
    cnt  = cnt_;
    pe   = CkMyPe();
    regMode = regMode_;
    deregMode = deregMode_;

    isRegistered = false;

#if CMK_ERROR_CHECKING
    // Ensure that regMode is valid
    checkRegModeIsValid();

    // Ensure that deregMode is valid
    checkDeregModeIsValid();
#endif

    // Register memory everytime new values are initialized
    if(cnt > 0)
      registerMem();
  }

  void setRef(const void *ref_) {
    ref = ref_;
  }

  const void *getRef() {
    return ref;
  }

  // Register(Pin) the memory for the buffer
  void registerMem()
  {
    // Check that this object is local when registerMem is called
    CkAssert(CkNodeOf(pe) == CkMyNode());

    // Set machine layer information when regMode is not CK_BUFFER_NOREG
    if(regMode != CK_BUFFER_NOREG) {

      CmiSetRdmaCommonInfo(&layerInfo[0], ptr, cnt);

      /* Set the pointer layerInfo unconditionally for layers that don't require pinning (MPI, PAMI)
       * or if regMode is REG, PREREG on layers that require pinning (GNI, Verbs, OFI, UCX) */
#if CMK_REG_REQUIRED
      if(regMode == CK_BUFFER_REG || regMode == CK_BUFFER_PREREG)
#endif
      {
        CmiSetRdmaBufferInfo(layerInfo + CmiGetRdmaCommonInfoSize(), ptr, cnt, regMode);
        isRegistered = true;
      }
    }
  }

  void setMode(unsigned short int regMode_) { regMode = regMode_; }

  void memcpyGet(CkNcpyBuffer &source);
  void memcpyPut(CkNcpyBuffer &destination);

#if CMK_USE_CMA
  void cmaGet(CkNcpyBuffer &source);
  void cmaPut(CkNcpyBuffer &destination);
#endif

  void rdmaGet(CkNcpyBuffer &source);
  void rdmaPut(CkNcpyBuffer &destination);

  CkNcpyStatus get(CkNcpyBuffer &source);
  CkNcpyStatus put(CkNcpyBuffer &destination);

  // Deregister(Unpin) the memory that is registered for the buffer
  void deregisterMem() {
    // Check that this object is local when deregisterMem is called
    CkAssert(CkNodeOf(pe) == CkMyNode());

    if(isRegistered == false)
      return;

#if CMK_REG_REQUIRED
    if(regMode != CK_BUFFER_NOREG) {
      CmiDeregisterMem(ptr, layerInfo + CmiGetRdmaCommonInfoSize(), pe, regMode);
      isRegistered = false;
    }
#endif
  }

  void pup(PUP::er &p) {
    p((char *)&ptr, sizeof(ptr));
    p((char *)&ref, sizeof(ref));
    p((char *)&bcastAckInfo, sizeof(bcastAckInfo));
    p|cnt;
    p|cb;
    p|pe;
    p|regMode;
    p|deregMode;
    p|isRegistered;
    PUParray(p, layerInfo, CMK_COMMON_NOCOPY_DIRECT_BYTES + CMK_NOCOPY_DIRECT_BYTES);
  }

  friend void CkRdmaDirectAckHandler(void *ack);

  friend void CkRdmaEMBcastAckHandler(void *ack);

  friend void constructSourceBufferObject(NcpyOperationInfo *info, CkNcpyBuffer &src);
  friend void constructDestinationBufferObject(NcpyOperationInfo *info, CkNcpyBuffer &dest);

  friend envelope* CkRdmaIssueRgets(envelope *env, ncpyEmApiMode emMode, void *forwardMsg);
  friend void CkRdmaIssueRgets(envelope *env, ncpyEmApiMode emMode, void *forwardMsg, int numops, void **arrPtrs, CkNcpyBufferPost *postStructs);

  friend void readonlyGet(CkNcpyBuffer &src, CkNcpyBuffer &dest, void *refPtr);
  friend void readonlyCreateOnSource(CkNcpyBuffer &src);


  friend void performEmApiNcpyTransfer(CkNcpyBuffer &source, CkNcpyBuffer &dest, int opIndex, int child_count, char *ref, int extraSize, CkNcpyMode ncpyMode, ncpyEmApiMode emMode);

  friend void performEmApiRget(CkNcpyBuffer &source, CkNcpyBuffer &dest, int opIndex, char *ref, int extraSize, ncpyEmApiMode emMode);

  friend void performEmApiCmaTransfer(CkNcpyBuffer &source, CkNcpyBuffer &dest, int child_count, ncpyEmApiMode emMode);

  friend void deregisterMemFromMsg(envelope *env, bool isRecv);
};

// Ack handler for the Zerocopy Direct API
// Invoked on the completion of any RDMA operation calling using the Direct API
void CkRdmaDirectAckHandler(void *ack);

// Method to invoke a callback on a particular pe with a CkNcpyBuffer being passed
// as a part of a CkDataMsg. This method is used to invoke callbacks on specific pes
// after the completion of the Zerocopy Direct API operation
void invokeCallback(void *cb, int pe, CkNcpyBuffer &buff);

// Returns CkNcpyMode::MEMCPY if both the PEs are the same and memcpy can be used
// Returns CkNcpyMode::CMA if both the PEs are in the same physical node and CMA can be used
// Returns CkNcpyMode::RDMA if RDMA needs to be used
CkNcpyMode findTransferMode(int srcPe, int destPe);

void invokeSourceCallback(NcpyOperationInfo *info);

void invokeDestinationCallback(NcpyOperationInfo *info);

// Method to enqueue a message after the completion of an payload transfer
void enqueueNcpyMessage(int destPe, void *msg);

/*********************************** Zerocopy Entry Method API ****************************/
static inline CkNcpyBuffer CkSendBuffer(const void *ptr_, CkCallback &cb_, unsigned short int regMode_=CK_BUFFER_REG, unsigned short int deregMode_=CK_BUFFER_DEREG) {
  return CkNcpyBuffer(ptr_, 0, cb_, regMode_, deregMode_);
}

static inline CkNcpyBuffer CkSendBuffer(const void *ptr_, unsigned short int regMode_=CK_BUFFER_REG, unsigned short int deregMode_=CK_BUFFER_DEREG) {
  return CkNcpyBuffer(ptr_, 0, regMode_, deregMode_);
}

#if CMK_ONESIDED_IMPL

// NOTE: Inside CkRdmaIssueRgets, a large message allocation is made consisting of space
// for the destination or receiver buffers and some additional information required for processing
// and acknowledgment handling. The space for additional information is typically equal to
// sizeof(NcpyEmInfo) + numops * sizeof(NcpyEmBufferInfo)

// This structure is used to store zerocopy information associated with an entry method
// invocation which uses the RDMA mode of transfer in Zerocopy Entry Method API.
// A variable of the structure stores the information in order to access it after the
// completion of the Rget operation (which is an asynchronous call) in order to invoke
// the entry method
struct NcpyEmInfo{
  int numOps; // number of zerocopy operations i.e number of buffers sent using CkSendBuffer
  int counter; // used for tracking the number of completed RDMA operations
  int pe;
  ncpyEmApiMode mode; // used to distinguish between p2p and bcast
  void *msg; // pointer to the Charm++ message which will be enqueued after completion of all Rgets
  void *forwardMsg; // used for the ncpy broadcast api
};


// This structure is used to store the buffer information specific to each buffer being sent
// using the Zerocopy Entry Method API. A variable of the structure stores the information associated
// with each buffer
struct NcpyEmBufferInfo{
  int index;  // Represents the index of the buffer information (from 0,1... numops - 1)
  NcpyOperationInfo ncpyOpInfo; // Stores all the information required for the zerocopy operation
};


/*
 * Extract ncpy buffer information from the metadata message,
 * allocate buffers and issue ncpy calls (either memcpy or cma read or rdma get)
 */
envelope* CkRdmaIssueRgets(envelope *env, ncpyEmApiMode emMode, void *forwardMsg = NULL);

void CkRdmaIssueRgets(envelope *env, ncpyEmApiMode emMode, void *forwardMsg, int numops, void **arrPtrs, CkNcpyBufferPost *postStructs);

void handleEntryMethodApiCompletion(NcpyOperationInfo *info);

void handleReverseEntryMethodApiCompletion(NcpyOperationInfo *info);

// Method called to pack rdma pointers
void CkPackRdmaPtrs(char *msgBuf);

// Method called to pack rdma pointers
void CkUnpackRdmaPtrs(char *msgBuf);

// Determine the number of ncpy ops and the sum of the ncpy buffer sizes
// from the metadata message
void getRdmaNumopsAndBufsize(envelope *env, int &numops, int &bufsize);

// Ack handler function for the nocopy EM API
void CkRdmaEMAckHandler(int destPe, void *ack);

void CkRdmaEMBcastPostAckHandler(void *msg);

struct NcpyBcastRecvPeerAckInfo{
#if CMK_SMP
  std::atomic<int> numPeers;
#else
  int numPeers;
#endif
  void *bcastAckInfo;
  void *msg;
  int peerParentPe;
#if CMK_SMP
    int getNumPeers() const {
       return numPeers.load(std::memory_order_acquire);
    } 
    void setNumPeers(int r) {
       return numPeers.store(r, std::memory_order_release);
    }
    int incNumPeers() {
        return numPeers.fetch_add(1, std::memory_order_release);
    }
    int decNumPeers() {
         return numPeers.fetch_sub(1, std::memory_order_release);
    }
#else
    int getNumPeers() const { return numPeers; }
    void setNumPeers(int r) { numPeers = r; }
    int incNumPeers() { return numPeers++; }
    int decNumPeers() { return numPeers--; }
#endif

};



/***************************** Zerocopy Bcast Entry Method API ****************************/
struct NcpyBcastAckInfo{
  int numChildren;
#if CMK_SMP
  // Counter is an atomic variable in the SMP mode because on the root node, both the
  // worker thread (in the case of a memcpy transfer) and the comm thread (CMA or RDMA transfer)
  // can increment the variable.
  std::atomic<int> counter;
#else
  int counter;
#endif
  bool isRoot;
  int pe;
  int numops;

#if CMK_SMP
  int getCounter() const {
    return counter.load(std::memory_order_acquire);
  }
  void setCounter(int r) {
    return counter.store(r, std::memory_order_release);
  }
  int incCounter() {
    return counter.fetch_add(1, std::memory_order_release);
  }
  int decCounter() {
    return counter.fetch_sub(1, std::memory_order_release);
  }
#else
  int getCounter() const { return counter; }
  void setCounter(int r) { counter = r; }
  int incCounter() { return counter++; }
  int decCounter() { return counter--; }
#endif
};

struct NcpyBcastRootAckInfo : public NcpyBcastAckInfo {
  CkNcpyBuffer src[0];
};

struct NcpyBcastInterimAckInfo : public NcpyBcastAckInfo {
  void *msg;

  // for RECV
  bool isRecv;
  bool isArray;
  void *parentBcastAckInfo;
  int origPe;

};

// Method called on the bcast source to store some information for ack handling
void CkRdmaPrepareBcastMsg(envelope *env);

void CkReplaceSourcePtrsInBcastMsg(envelope *env, NcpyBcastInterimAckInfo *bcastAckInfo, int origPe);

// Method called to extract the parent bcastAckInfo from the received message for ack handling
const void *getParentBcastAckInfo(void *msg, int &srcPe);

// Allocate a NcpyBcastInterimAckInfo and return the pointer
NcpyBcastInterimAckInfo *allocateInterimNodeAckObj(envelope *myEnv, envelope *myChildEnv, int pe);

void forwardMessageToChildNodes(envelope *myChildrenMsg, UChar msgType);

void forwardMessageToPeerNodes(envelope *myMsg, UChar msgType);

void handleBcastEntryMethodApiCompletion(NcpyOperationInfo *info);

void handleBcastReverseEntryMethodApiCompletion(NcpyOperationInfo *info);

void deregisterMemFromMsg(envelope *env, bool isRecv);

void handleMsgUsingCMAPostCompletionForSendBcast(envelope *copyenv, envelope *env, CkNcpyBuffer &source);

void processBcastSendEmApiCompletion(NcpyEmInfo *ncpyEmInfo, int destPe);

// Method called on intermediate nodes after RGET to switch old source pointers with my pointers
void CkReplaceSourcePtrsInBcastMsg(envelope *prevEnv, envelope *env, void *bcastAckInfo, int origPe);

void processBcastRecvEmApiCompletion(NcpyEmInfo *ncpyEmInfo, int destPe);

// Method called on the root node and other intermediate parent nodes on completion of RGET through ZC Bcast
void CkRdmaEMBcastAckHandler(void *ack);

void handleMsgOnChildPostCompletionForRecvBcast(envelope *env);

void handleMsgOnInterimPostCompletionForRecvBcast(envelope *env, NcpyBcastInterimAckInfo *bcastAckInfo, int pe);



/***************************** Zerocopy Readonly Bcast Support ****************************/

/* Support for Zerocopy Broadcast of large readonly variables */
CkpvExtern(int, _numPendingRORdmaTransfers);

struct NcpyROBcastBuffAckInfo {
  const void *ptr;

  int regMode;

  int pe;

  // machine specific information about the buffer
  #ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wpedantic"
  #endif
  char layerInfo[CMK_COMMON_NOCOPY_DIRECT_BYTES + CMK_NOCOPY_DIRECT_BYTES];
  #ifdef __GNUC__
  #pragma GCC diagnostic pop
  #endif
};

struct NcpyROBcastAckInfo {
  int numChildren;
  int counter;
  bool isRoot;
  int numops;
  NcpyROBcastBuffAckInfo buffAckInfo[0];
};

void readonlyUpdateNumops();

void readonlyAllocateOnSource();

void readonlyCreateOnSource(CkNcpyBuffer &src);

void readonlyGet(CkNcpyBuffer &src, CkNcpyBuffer &dest, void *refPtr);

void readonlyGetCompleted(NcpyOperationInfo *ncpyOpInfo);

#if CMK_SMP
void updatePeerCounterAndPush(envelope *env);
#endif

CkArray* getArrayMgrFromMsg(envelope *env);

void sendAckMsgToParent(envelope *env);

void sendRecvDoneMsgToPeers(envelope *env, CkArray *mgr);

#endif /* End of CMK_ONESIDED_IMPL */

#endif

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

  public:
  // pointer to the buffer
  const void *ptr;

  // number of bytes
  size_t cnt;

  // callback to be invoked on the sender/receiver
  CkCallback cb;

  // home pe
  int pe;

  // mode
  unsigned short int mode;

  // reference pointer
  const void *ref;

  CkNcpyBuffer() : isRegistered(false), ptr(NULL), cnt(0), pe(-1), mode(CK_BUFFER_REG), ref(NULL) {}

  explicit CkNcpyBuffer(const void *address, unsigned short int mode_=CK_BUFFER_REG) {
    ptr = address;
    pe = CkMyPe();
    cb = CkCallback(CkCallback::ignore);
    mode = mode_;
    isRegistered = false;
  }

  CkNcpyBuffer(const void *address, CkCallback &cb_, unsigned short int mode_=CK_BUFFER_REG) {
    ptr = address;
    pe = CkMyPe();
    cb = cb_;
    mode = mode_;
    isRegistered = false;
  }

  CkNcpyBuffer(const void *ptr_, size_t cnt_, CkCallback &cb_, unsigned short int mode_=CK_BUFFER_REG) {
    init(ptr_, cnt_, cb_, mode_);
  }

  void init(const void *ptr_, size_t cnt_, CkCallback &cb_, unsigned short int mode_=CK_BUFFER_REG) {
    ptr  = ptr_;
    cnt  = cnt_;
    cb   = cb_;
    pe   = CkMyPe();
    mode = mode_;

    isRegistered = false;
    // Register memory everytime new values are initialized
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

    // Set machine layer information when mode is not CK_BUFFER_NOREG
    if(mode != CK_BUFFER_NOREG) {

      CmiSetRdmaCommonInfo(&layerInfo[0], ptr, cnt);

      /* Set the pointer layerInfo unconditionally for layers that don't require pinning (MPI, PAMI)
       * or if mode is REG, PREREG on layers that require pinning (GNI, Verbs, OFI) */
#if CMK_REG_REQUIRED
      if(mode == CK_BUFFER_REG || mode == CK_BUFFER_PREREG)
#endif
      {
        CmiSetRdmaBufferInfo(layerInfo + CmiGetRdmaCommonInfoSize(), ptr, cnt, mode);
        isRegistered = true;
      }
    }
  }

  void setMode(unsigned short int mode_) { mode = mode_; }

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
    if(mode != CK_BUFFER_PREREG && mode != CK_BUFFER_NOREG) {
      CmiDeregisterMem(ptr, layerInfo + CmiGetRdmaCommonInfoSize(), pe, mode);
      isRegistered = false;
    }
#endif
  }

  void pup(PUP::er &p) {
    p((char *)&ptr, sizeof(ptr));
    p((char *)&ref, sizeof(ref));
    p|cnt;
    p|cb;
    p|pe;
    p|mode;
    p|isRegistered;
    PUParray(p, layerInfo, CMK_COMMON_NOCOPY_DIRECT_BYTES + CMK_NOCOPY_DIRECT_BYTES);
  }

  friend void CkRdmaDirectAckHandler(void *ack);

  friend void constructSourceBufferObject(NcpyOperationInfo *info, CkNcpyBuffer &src);
  friend void constructDestinationBufferObject(NcpyOperationInfo *info, CkNcpyBuffer &dest);

  friend envelope* CkRdmaIssueRgets(envelope *env);
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




/*********************************** Zerocopy Entry Method API ****************************/
#define CkSendBuffer(...) CkNcpyBuffer(__VA_ARGS__)

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
  void *msg; // pointer to the Charm++ message which will be enqueued after completion of all Rgets
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
envelope* CkRdmaIssueRgets(envelope *env);

void handleEntryMethodApiCompletion(NcpyOperationInfo *info);

void handleReverseEntryMethodApiCompletion(NcpyOperationInfo *info);

// Method called to pack rdma pointers
void CkPackRdmaPtrs(char *msgBuf);

// Method called to pack rdma pointers
void CkUnpackRdmaPtrs(char *msgBuf);

// Get the number of ncpy ops using the metadata message
int getRdmaNumOps(envelope *env);

// Get the sum of ncpy buffer sizes using the metadata message
int getRdmaBufSize(envelope *env);

// Ack handler function for the nocopy EM API
void CkRdmaEMAckHandler(int destPe, void *ack);

#endif /* End of CMK_ONESIDED_IMPL */

#endif

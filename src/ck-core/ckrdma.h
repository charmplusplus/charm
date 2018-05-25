/*
 * Charm Onesided API Utility Functions
 */

#ifndef _CKRDMA_H_
#define _CKRDMA_H_

#include "envelope.h"

#define CK_BUFFER_REG     CMK_BUFFER_REG
#define CK_BUFFER_UNREG   CMK_BUFFER_UNREG
#define CK_BUFFER_PREREG  CMK_BUFFER_PREREG
#define CK_BUFFER_NOREG   CMK_BUFFER_NOREG

/* CK_MSG_RDMA is passed in as entry method opts in the generated code for an entry
 * method containing RDMA parameters. In the SMP mode with IMMEDIATE message support,
 * it is used to mark the entry method invocation as IMMEDIATE to have the comm thread
 * handle the metadata message. In all other cases (Non-SMP mode, No comm thread support),
 * its value is used as 0.
 */
#if CMK_ONESIDED_IMPL && CMK_SMP && CK_MSG_IMMEDIATE
#define CK_MSG_RDMA CK_MSG_IMMEDIATE
#else
#define CK_MSG_RDMA 0
#endif

#if CMK_ONESIDED_IMPL

/* Sender Functions */

//Prepare metadata message with the relevant machine specific info
void CkRdmaPrepareMsg(envelope **env, int pe);

//Create a new message with machine specific information
envelope* CkRdmaCreateMetadataMsg(envelope *env, int pe);

//Handle ack received on the sender by invoking callback
void CkHandleRdmaCookie(void *cookie);



/* Receiver Functions */

//Copy the message using pointers when it's on the same PE/node
envelope* CkRdmaCopyMsg(envelope *env);

/*
 * Extract rdma based information from the metadata message,
 * allocate buffers and issue RDMA get call
 */
void CkRdmaIssueRgets(envelope *env);

/*
 * Method called to update machine specific information and pointers
 * inside Ckrdmawrappers
 */
void CkUpdateRdmaPtrs(envelope *msg, int msgsize, char *recv_md, char *src_md);

/*
 * Method called to pack rdma pointers
 * inside Ckrdmawrappers
 */
void CkPackRdmaPtrs(char *msgBuf);

/*
 * Method called to unpack rdma pointers
 * inside Ckrdmawrappers
 */
void CkUnpackRdmaPtrs(char *msgBuf);

//Get the number of rdma ops using the metadata message
int getRdmaNumOps(envelope *env);

//Get the sum of rdma buffer sizes using the metadata message
int getRdmaBufSize(envelope *env);


#endif /* End of CMK_ONESIDED_IMPL */


/* Support for Nocopy Direct API */

/* Use 0 sized headers for generic Direct API implementation */
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

// Ack handler function which invokes the callbacks on the source and destination PEs
void CkRdmaAckHandler(void *cookie);
void CkRdmaDirectAckHandler(void *ack);

// Class to represent an acknowledgement structure
class CkNcpyAck{
  public:
  // pointer to the buffer
  const void *ptr;

  // reference pointer
  // This is an optional arbitrary pointer set by the user before performing the get/put
  // operation. It is returned back in the CkNcpyAck object.
  const void *ref;

  CkNcpyAck(const void *ptr_, const void *ref_) : ptr(ptr_), ref(ref_) {}
};

PUPbytes(CkNcpyAck);

// Class to represent an RDMA buffer
class CkNcpyBuffer{
  private:
  // bool to indicate registration for current values of ptr and cnt on pe
  bool isRegistered;

  public:
  // pointer to the buffer
  const void *ptr;

  // number of bytes
  size_t cnt;

  // callback to be invoked on the sender/receiver
  CkCallback cb;

  // home pe
  int pe;

  #ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wpedantic"
  #endif
  // machine specific information about the buffer
  char layerInfo[CMK_COMMON_NOCOPY_DIRECT_BYTES + CMK_NOCOPY_DIRECT_BYTES];
  #ifdef __GNUC__
  #pragma GCC diagnostic pop
  #endif

  // mode
  unsigned short int mode;

  // reference pointer
  const void *ref;

  CkNcpyBuffer() : ptr(NULL), pe(-1), ref(NULL), mode(CK_BUFFER_UNREG) {}

  CkNcpyBuffer(const void *ptr_, size_t cnt_, CkCallback &cb_, unsigned short int mode_=CK_BUFFER_UNREG) {
    init(ptr_, cnt_, cb_, mode_);
  }

  void init(const void *ptr_, size_t cnt_, CkCallback &cb_, unsigned short int mode_=CK_BUFFER_UNREG) {
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

    if(isRegistered == true)
      return;

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
      }
      isRegistered = true;
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

  void get(CkNcpyBuffer &source);
  void put(CkNcpyBuffer &destination);

  // Deregister(Unpin) the memory that is registered for the buffer
  void deregisterMem() {
    // Check that this object is local when deregisterMem is called
    CkAssert(CkNodeOf(pe) == CkMyNode());

    if(isRegistered == false)
      return;

#if CMK_REG_REQUIRED
    CmiDeregisterMem(ptr, layerInfo + CmiGetRdmaCommonInfoSize(), pe, mode);
#endif

    isRegistered = false;
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

};

enum class ncpyTransferMode : char { MEMCPY, CMA, RDMA };

#endif

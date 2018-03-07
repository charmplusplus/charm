/*
 * Charm Onesided API Utility Functions
 */

#ifndef _CKRDMA_H_
#define _CKRDMA_H_

#include "envelope.h"

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

// Ack handler function which invokes the callbacks on the source and destination PEs
void CkRdmaAckHandler(void *cookie);
void CkRdmaAckHandler(void *cbPtr, int pe, const void *ptr);

class CkNcpyDestination;

// Class to represent an RDMA source
class CkNcpySource{
  public:
  // pointer to the source buffer
  const void *ptr;

  // number of bytes
  size_t cnt;

  // callback to be invoked on the sender
  CkCallback cb;

  // home pe
  int pe;

  // machine specific information about the source pointer
  char layerInfo[CMK_NOCOPY_DIRECT_BYTES];

  CkNcpySource() : ptr(NULL), pe(-1) {}

  CkNcpySource(const void *ptr_, size_t cnt_, CkCallback cb_) : ptr(ptr_), cnt(cnt_), cb(cb_){
    pe = CkMyPe();
    // set the source pointer layerInfo
    CmiSetRdmaSrcInfo(&layerInfo, ptr, cnt);
  }

  void rput(CkNcpyDestination destination);

  void releaseResource();
};
PUPbytes(CkNcpySource);

// Class to represent an RDMA destination
class CkNcpyDestination{
  public:
  // pointer to the destination buffer
  const void *ptr;

  // number of bytes
  size_t cnt;

  // callback to be invoked on the receiver
  CkCallback cb;

  // home pe
  int pe;

  // machine specific information about the destination pointer
  char layerInfo[CMK_NOCOPY_DIRECT_BYTES];

  CkNcpyDestination() : ptr(NULL), pe(-1) {}

  CkNcpyDestination(const void *ptr_, size_t cnt_, CkCallback cb_) : ptr(ptr_), cnt(cnt_), cb(cb_) {
    pe = CkMyPe();

    // set the destination pointer layerInfo
    CmiSetRdmaDestInfo(&layerInfo, ptr, cnt);
  }

  void rget(CkNcpySource source);

  void releaseResource();
};
PUPbytes(CkNcpyDestination);



#endif

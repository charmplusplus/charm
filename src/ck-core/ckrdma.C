/*
 * Charm Onesided API Utility Functions
 */

#include "charm++.h"
#include "converse.h"

#if CMK_ONESIDED_IMPL

#if CMK_SMP && CMK_IMMEDIATE_MSG
/*readonly*/ extern CProxy_ckcallback_group _ckcallbackgroup;
#endif

/* Sender Functions */

/*
 * Add the machine specific information if metadata message
 * sent across network. Do nothing if message is sent to same PE/Node
 */
void CkRdmaPrepareMsg(envelope **env, int pe){
  if(CmiNodeOf(pe)!=CmiMyNode()){
    envelope *prevEnv = *env;
    *env = CkRdmaCreateMetadataMsg(prevEnv, pe);
    CkFreeMsg(EnvToUsr(prevEnv));
  }
#if CMK_SMP && CMK_IMMEDIATE_MSG
  //Reset the immediate bit if it is a within node/pe message
  else
    CmiResetImmediate(*env);
#endif
}

/*
 * Add machine specific information to the msg. This includes information
 * information both common and specific to each rdma parameter
 * Metdata message format: <-env-><-msg-><-migen-><-mispec1-><-mispec2->..<-mispecn->
 * migen: machine info generic to rdma operation
 * mispec: machine info specific to rdma operation for n rdma operations
 */
envelope* CkRdmaCreateMetadataMsg(envelope *env, int pe){

  int numops = getRdmaNumOps(env);
  int msgsize = env->getTotalsize();

  //CmiGetRdmaInfoSize returns the size of machine specific information
  // for numops RDMA operations
  int mdsize = msgsize + CmiGetRdmaInfoSize(numops);

  //pack before copying
  CkPackMessage(&env);

  //Allocate a new metadata message, set machine specific info
  envelope *copyenv = (envelope *)CmiAlloc(mdsize);
  memcpy(copyenv, env, msgsize);
  copyenv->setTotalsize(mdsize);
#if CMK_SMP && CMK_IMMEDIATE_MSG
  CmiBecomeImmediate(copyenv);
#endif

  //Set the generic information
  char *md = (char *)copyenv + msgsize;
  CmiSetRdmaInfo(md, pe, numops);
  md += CmiGetRdmaGenInfoSize();

  //Set the rdma op specific information
  CkUnpackMessage(&copyenv);
  PUP::fromMem up((void *)(((CkMarshallMsg *)EnvToUsr(copyenv))->msgBuf));
  PUP::toMem p((void *)(((CkMarshallMsg *)EnvToUsr(copyenv))->msgBuf));
  up|numops;
  p|numops;
  for(int i=0; i<numops; i++){
    CkRdmaWrapper w; up|w;
    CkRdmaWrapper *wack = new CkRdmaWrapper(w);
    void *ack = CmiSetRdmaAck(CkHandleRdmaCookie, wack);
    w.callback = (CkCallback *)ack;
    p|w;
    CmiSetRdmaOpInfo(md, w.ptr, w.cnt, w.callback, pe);
    md += CmiGetRdmaOpInfoSize();
  }
  CkPackMessage(&copyenv);

  //return the env with the machine specific information
  return copyenv;
}

/*
 * Method called on sender when ack is received
 * Access the CkRdmaWrapper using the cookie passed and invoke callback
 */
void CkHandleRdmaCookie(void *cookie){
  CkRdmaWrapper *w = (CkRdmaWrapper *)cookie;
  CkCallback *cb= w->callback;

#if CMK_SMP && CMK_IMMEDIATE_MSG
  //call to callbackgroup to call the callback when calling from comm thread
  //this add one more trip through the scheduler
  _ckcallbackgroup[w->srcPe].call(*cb, sizeof(void *), (char*)&w->ptr);
#else
  //direct call to callback when calling from worker thread
  cb->send(sizeof(void *), &w->ptr);
#endif

  delete cb;
  delete (CkRdmaWrapper *)cookie;
}


/* Receiver Functions */

/*
 * Method called when received message is on the same PE/Node
 * This involves a direct copy from the sender's buffer into the receiver's buffer
 * A new message is allocated with size = existing message size + sum of all rdma buffers.
 * The newly allocated message contains the existing marshalled message along with
 * space for the rdma buffers which are copied from the source buffers.
 * The buffer and the message are allocated contiguously to free the buffer
 * when the message gets free.
 */
envelope* CkRdmaCopyMsg(envelope *env){
  int numops, bufsize, msgsize;
  bufsize = getRdmaBufSize(env);
  msgsize = env->getTotalsize();

  CkPackMessage(&env);
  envelope *copyenv = (envelope *)CmiAlloc(CK_ALIGN(msgsize, 16) + bufsize);
  memcpy(copyenv, env, msgsize);
  copyenv->setTotalsize(CK_ALIGN(msgsize, 16) + bufsize);
  copyenv->setRdma(false);

  char* buf = (char *)copyenv + CK_ALIGN(msgsize, 16);
  CkUnpackMessage(&copyenv);
  PUP::toMem p((void *)(((CkMarshallMsg *)EnvToUsr(copyenv))->msgBuf));
  PUP::fromMem up((void *)((CkMarshallMsg *)EnvToUsr(copyenv))->msgBuf);
  up|numops;
  p|numops;
  for(int i=0; i<numops; i++){
    CkRdmaWrapper w;
    up|w;
    //Copy the buffer from the source pointer to the message
    memcpy(buf, w.ptr, w.cnt);

    //Invoke callback as it is safe to rewrite into the source buffer
    (w.callback)->send(sizeof(void *), &w.ptr);
    delete w.callback;

    //Update the CkRdmaWrapper pointer of the new message
    w.ptr = buf;

    //Update the pointer
    buf += CK_ALIGN(w.cnt, 16);
    p|w;
  }
  CkPackMessage(&copyenv);
  return copyenv;
}

/*
 * Extract rdma based information from the metadata message,
 * allocate buffers and issue RDMA get call
 */
void CkRdmaIssueRgets(envelope *env){
  /*
   * Determine the buffer size and the message size from the
   * metadata message. Message size is the metadata message size without
   * the machine specific information
   */
  int numops, bufsize, msgsize;
  bufsize = getRdmaBufSize(env);
  numops = getRdmaNumOps(env);
  msgsize = env->getTotalsize() - CmiGetRdmaInfoSize(numops);

  //Allocate the buffer on the receiver
  envelope *copyenv = (envelope *)CmiAlloc(CK_ALIGN(msgsize, 16) + bufsize);

  //Copy the metadata message(without the machine specific info) into the buffer
  memcpy(copyenv, env, msgsize);

#if CMK_SMP && CMK_IMMEDIATE_MSG
  CmiResetImmediate(copyenv);
#endif

  char *recv_md = (char *) malloc(CmiGetRdmaRecvInfoSize(numops));
  CkUpdateRdmaPtrs(copyenv, msgsize, recv_md, ((char *)env) + msgsize);
  //Set the total size of the message
  copyenv->setTotalsize(CK_ALIGN(msgsize, 16) + bufsize);

  // Set rdma to be false to prevent message handler on the receiver
  // from intercepting it
  copyenv->setRdma(false);

  // Free the existing message
  CkFreeMsg(EnvToUsr(env));

  //Call the lower layer API for performing RDMA gets
  CmiIssueRgets(recv_md, copyenv->getSrcPe());
}


/*
 * Method called to update machine specific information for receiver
 * using the metadata message given by the sender along with updating
 * pointers of the CkRdmawrappers
 */
void CkUpdateRdmaPtrs(envelope *env, int msgsize, char *recv_md, char *src_md){
  CkUnpackMessage(&env);
  PUP::toMem p((void *)(((CkMarshallMsg *)EnvToUsr(env))->msgBuf));
  PUP::fromMem up((void *)(((CkMarshallMsg *)EnvToUsr(env))->msgBuf));
  int numops;
  up|numops;
  p|numops;

  //Use the metadata info to set the machine info for receiver
  //generic info for all RDMA operations
  CmiSetRdmaRecvInfo(recv_md, numops, env, src_md, env->getTotalsize());
  recv_md += CmiGetRdmaGenRecvInfoSize();
  char *buf = ((char *)env) + CK_ALIGN(msgsize, 16);
  for(int i=0; i<numops; i++){
    CkRdmaWrapper w;
    up|w;
    //Set RDMA operation specific info
    CmiSetRdmaRecvOpInfo(recv_md, buf, w.callback, w.cnt, i, src_md);
    recv_md += CmiGetRdmaOpRecvInfoSize();
    w.ptr = buf;
    buf += CK_ALIGN(w.cnt, 16);
    p|w;
  }
  CkPackMessage(&env);
}

//Determine the number of rdma ops from the message
int getRdmaNumOps(envelope *env){
  int numops;
  CkUnpackMessage(&env);
  PUP::fromMem up((void *)((CkMarshallMsg *)EnvToUsr(env))->msgBuf);
  up|numops;
  CkPackMessage(&env);
  return numops;
}

/*
 * Determine the total size of the buffers to be copied
 * This is to be allocated at the end of the message
 */
int getRdmaBufSize(envelope *env){
  /*
   * Determine the number of rdma operations and the sum of all
   * rdma buffer sizes by iterating over the ckrdmawrappers in the message
   */
  int numops, bufsize=0;
  CkUnpackMessage(&env);
  PUP::fromMem up((void *)((CkMarshallMsg *)EnvToUsr(env))->msgBuf);
  up|numops;
  for(int i=0; i<numops; i++){
    CkRdmaWrapper w; up|w;
    bufsize += CK_ALIGN(w.cnt, 16);
  }
  CkPackMessage(&env);
  return bufsize;
}

#endif

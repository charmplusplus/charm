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
  CkPackRdmaPtrs(((CkMarshallMsg *)EnvToUsr(copyenv))->msgBuf);
  CkPackMessage(&copyenv);
  return copyenv;
}

/*
 * Extract rdma based information from the metadata message,
 * allocate buffers and issue RDMA get call
 */
void CkRdmaIssueRgets(envelope *env, bool free){

  if(CmiNodeOf(env->getSrcPe()) == CmiMyNode()){
    envelope *newEnv = CkRdmaCopyMsg(env);
    CsdEnqueue(newEnv);
    return;
  }
  /*
   * Determine the buffer size('bufsize') and the message size('msgsize')
   * from the metadata message. 'msgSize' is the metadata message size
   * without the sender's machine specific information.
   */
  int numops, bufsize, msgsize;
  bufsize = getRdmaBufSize(env);
  numops = getRdmaNumOps(env);
  msgsize = env->getTotalsize() - CmiGetRdmaInfoSize(numops);

  /* Allocate the receiver's message, which contains the metadata message sent by the sender
   * (without its machine specific info) of size 'msgSize', the entire receiver's buffer of
   * size 'bufsize', and the receiver's machine specific info of size 'CmiGetRdmaRecvInfoSize(numops)'.
   * The receiver's machine specific info is added to this message to avoid separately freeing it
   * in the machine layer.
   */
  envelope *copyenv = (envelope *)CmiAlloc(CK_ALIGN(msgsize, 16) + bufsize + CmiGetRdmaRecvInfoSize(numops));

  //Copy the metadata message(without the machine specific info) into the buffer
  memcpy(copyenv, env, msgsize);

#if CMK_SMP && CMK_IMMEDIATE_MSG
  CmiResetImmediate(copyenv);
#endif

  //Receiver's machine specific info is at an offset, after the sender md and the receiver's buffer
  char *recv_md = ((char *)copyenv) + CK_ALIGN(msgsize, 16) + bufsize;

  CkUnpackMessage(&copyenv);
  CkUpdateRdmaPtrs(copyenv, msgsize, recv_md, ((char *)env) + msgsize);
  CkPackRdmaPtrs(((CkMarshallMsg *)EnvToUsr(copyenv))->msgBuf);
  CkPackMessage(&copyenv);

  /* Set the total size of the message excluding the receiver's machine specific info
   * which is not required when the receiver's entry method executes
   */
  copyenv->setTotalsize(CK_ALIGN(msgsize, 16) + bufsize);

  // Set rdma to be false to prevent message handler on the receiver
  // from intercepting it
  copyenv->setRdma(false);

  // Existing message freed by the runtime
  if(free)
    CkFreeMsg(EnvToUsr(env));

  //Call the lower layer API for performing RDMA gets
  CmiIssueRgets(recv_md, copyenv->getSrcPe());
}


/*
 * Method called to update machine specific information for receiver
 * using the metadata message given by the sender along with updating
 * pointers of the CkRdmawrappers
 * - assumes that the msg is unpacked
 */
void CkUpdateRdmaPtrs(envelope *env, int msgsize, char *recv_md, char *src_md){
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
}


/*
 * Method called to pack rdma pointers inside CkRdmaWrappers in the message
 * Assumes that msg is unpacked
 */
void CkPackRdmaPtrs(char *msgBuf){
  envelope *env = UsrToEnv(env);
  PUP::toMem p((void *)msgBuf);
  PUP::fromMem up((void *)msgBuf);
  int numops;
  up|numops;
  p|numops;

  //Pack Rdma pointers in CkRdmaWrappers
  for(int i=0; i<numops; i++){
    CkRdmaWrapper w;
    up|w;
    w.ptr = (void *)((char *)w.ptr - (char *)msgBuf);
    p|w;
  }
}


/*
 * Method called to unpack rdma pointers inside CkRdmaWrappers in the message
 * Assumes that msg is unpacked
 */
void CkUnpackRdmaPtrs(char *msgBuf){
  PUP::toMem p((void *)msgBuf);
  PUP::fromMem up((void *)msgBuf);
  int numops;
  up|numops;
  p|numops;

  //Unpack Rdma pointers in CkRdmaWrappers
  for(int i=0; i<numops; i++){
    CkRdmaWrapper w;
    up|w;
    w.ptr = (void *)((char *)msgBuf + (size_t)w.ptr);
    p|w;
  }
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

CkRdmaPostHandle* createRdmaPostHandle(int numops){
  CkRdmaPostHandle* ret = (CkRdmaPostHandle*)malloc(sizeof(CkRdmaPostHandle)
                                        + numops * sizeof(CkRdmaPostStruct));
  ret->nstructs = numops;
  return ret;
}


CkRdmaPostHandle* CkGetRdmaPostHandle(envelope *env){
  CkMarshallMsg *msg = (CkMarshallMsg *)EnvToUsr(env);
  CkMarshallMsg *copymsg = (CkMarshallMsg *)CkCopyMsg((void **)&msg);
  CkUnpackMessage(&env);
  envelope *copyenv = UsrToEnv(copymsg);
  CkUnpackMessage(&copyenv);
  PUP::fromMem up((void *)(copymsg->msgBuf));
  int numops; up|numops;
  //CkPrintf("numops: %d \n, numops");
  CkRdmaPostHandle* handle = createRdmaPostHandle(numops);
  handle->msg = copymsg;
  for(int i=0; i<numops; i++){
    CkRdmaWrapper w;
    up|w;
    handle->structs[i].cnt = w.cnt;
  }
  return handle;
}



/* Receive Side Utility Functions */


/*
 * Use pointers provided in the handle to issue Rgets.
 */
void CkRdmaPost(CkRdmaPostHandle *handle){

  /*
   * Determine the buffer size and the message size from the
   * metadata message. Message size is the metadata message size without
   * the machine specific information. Use the pointers provided in the handle
   */
  envelope *env = UsrToEnv(handle->msg);
  if(env->getSrcPe()==CkMyPe()||CkNodeOf(env->getSrcPe())==CkMyNode()){
    CkUpdateRdmaPtrsPost(env, handle);
    return;
  }


  PUP::fromMem p((void *)(((CkMarshallMsg *)handle->msg)->msgBuf));
  int numops; p|numops;
  int msgsize = env->getTotalsize() - CmiGetRdmaInfoSize(numops);

  char *recv_md = (char *) malloc(CmiGetRdmaRecvInfoSize(numops));
  CkUpdateRdmaPtrsPost(env, msgsize, recv_md, ((char *)env) + msgsize, handle);


  //ckout<<"Issuing Rgets "<<numops<<endl;
  //Call the lower layer API for performing RDMA gets
  CmiIssueRgets(recv_md, env->getSrcPe());
}

  void CkUpdateRdmaPtrsPost(envelope *env, CkRdmaPostHandle* handle){
    CkUnpackMessage(&env);
    PUP::toMem p((void *)(((CkMarshallMsg *)EnvToUsr(env))->msgBuf));
    PUP::fromMem up((void *)((CkMarshallMsg *)EnvToUsr(env))->msgBuf);
    int numops;
    up|numops; p|numops;
    for(int i=0; i<numops; i++){
      CkRdmaWrapper w;
      up|w;
      //Copy the buffer from the source pointer to the message
      memcpy(handle->structs[i].ptr, w.ptr, w.cnt);

      //Invoke callback as it is safe to rewrite into the source buffer
      (w.callback)->send(sizeof(void *), &w.ptr);
      free(w.callback);

      //Update the CkRdmaWrapper pointer of the new message
      w.ptr = handle->structs[i].ptr;
      p|w;
    }
    CkPackMessage(&env);
    CsdEnqueue(env);
  }

/*
 * Method called to update machine specific information for receiver
 * using the metadata message given by the sender along with updating
 * pointers of the CkRdmawrappers using the Post Handle.
 * Assumes message to be unpacked
 */
void CkUpdateRdmaPtrsPost(envelope *env, int msgsize, char *recv_md, char *src_md, CkRdmaPostHandle* handle){
  CkAssert(!env->isPacked());
  /* pack buffer */
  PUP::toMem p((void *)(((CkMarshallMsg *)EnvToUsr(env))->msgBuf));
  /* unpack buffer */
  PUP::fromMem up((void *)(((CkMarshallMsg *)EnvToUsr(env))->msgBuf));
  int numops;
  up|numops;
  p|numops;

  //Use the metadata info to set the machine info for receiver
  //generic info for all RDMA operations
  CmiSetRdmaRecvInfo(recv_md, numops, env, src_md, env->getTotalsize());
  recv_md += CmiGetRdmaGenRecvInfoSize();
  for(int i=0; i<numops; i++){
    CkRdmaWrapper w;
    up|w;
    //Set RDMA operation specific info
    //CkPrintf("CmiSetRdmaRecvOpInfo: w.cnt: %d, i: %d, handle->struct[i].ptr: %p \n", w.cnt, i, handle->structs[i].ptr);
    CmiSetRdmaRecvOpInfo(recv_md, handle->structs[i].ptr, w.callback, w.cnt, i, src_md);
    recv_md += CmiGetRdmaOpRecvInfoSize();
    w.ptr = handle->structs[i].ptr;
    p|w;
  }
  CkPackMessage(&env);
}



#endif

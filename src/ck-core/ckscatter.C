#include "charm++.h"
#include "converse.h"

void getScatterInfo(void *msg, CkScatterWrapper *w){
  envelope *env = UsrToEnv(msg);
  //!Assumption: message not packed yet
  PUP::fromMem up((void *)(((CkMarshallMsg *)msg)->msgBuf));
  CkScatterWrapper _w;
  up|_w;
  *w = _w;
}


void* createScatterMsg(void *msg, CkScatterWrapper &w, int ind){
  envelope *env = UsrToEnv(msg);
  int msgsize = env->getTotalsize();
  //CkPrintf("createScatterMsg msgsize: %d, bufsize: %d\n", msgsize, w.cnt[ind]);
  
  envelope *copyenv = (envelope *)CmiAlloc(CK_ALIGN(msgsize, 16) + w.cnt[ind]);
  CkPackMessage(&env);
  memcpy(copyenv, env, msgsize); 
  CkUnpackMessage(&env);
  CkUnpackMessage(&copyenv);

  copyenv->setTotalsize(CK_ALIGN(msgsize, 16) + w.cnt[ind]);

  char* buf = (char *)copyenv + CK_ALIGN(msgsize, 16);
  memcpy(buf, ((char *)w.buf) + w.disp[ind], w.cnt[ind]);
  //ckout<<"createScattermsg: "<<*((int *)((char *)w.buf + w.disp[ind]))<<endl;

  PUP::toMem p((void *)(((CkMarshallMsg *)EnvToUsr(copyenv))->msgBuf));
  PUP::fromMem up((void *)(((CkMarshallMsg *)EnvToUsr(copyenv))->msgBuf));
  CkScatterWrapper _w;
  up|_w;
  _w.setsize(w.cnt[ind]);
  size_t offset = buf - (((CkMarshallMsg *)EnvToUsr(copyenv))->msgBuf);
  _w.setoffset(offset);
  p|_w;

  return EnvToUsr(copyenv); 
}

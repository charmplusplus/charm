#include "charm++.h"
#include "converse.h"

void getScattervInfo(void *msg, CkScattervWrapper *w){
  envelope *env = UsrToEnv(msg);
  //!Assumption: message not packed yet
  PUP::fromMem up((void *)(((CkMarshallMsg *)msg)->msgBuf));
  up|*w;
}


void* createScattervMsg(void *msg, CkScattervWrapper &w, int ind){
  envelope *env = UsrToEnv(msg);
  int msgsize = env->getTotalsize();
  //CkPrintf("createScattervMsg msgsize: %d, bufsize: %d\n", msgsize, w.cnt[ind]);
 
  envelope *copyenv = (envelope *)CmiAlloc(CK_ALIGN(msgsize, 16) + w.cnt[ind]);
  CkPackMessage(&env);
  memcpy(copyenv, env, msgsize);
  CkUnpackMessage(&env);
  CkUnpackMessage(&copyenv);

  copyenv->setTotalsize(CK_ALIGN(msgsize, 16) + w.cnt[ind]);

  char* buf = (char *)copyenv + CK_ALIGN(msgsize, 16);
  memcpy(buf, ((char *)w.buf) + w.disp[ind], w.cnt[ind]);
  //ckout<<"createScattervmsg: "<<*((int *)((char *)w.buf + w.disp[ind]))<<endl;

  void *copybuf = (void *)(((CkMarshallMsg *)EnvToUsr(copyenv))->msgBuf);
  PUP::toMem p(copybuf);
  PUP::fromMem up(copybuf);
  CkScattervWrapper _w;
  up|_w;
  _w.setSize(w.cnt[ind]);
  size_t offset = buf - ((char *)copybuf);
  _w.setOffset(offset);
  p|_w;
  return EnvToUsr(copyenv);
}

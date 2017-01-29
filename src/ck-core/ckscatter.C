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

/**
 * Create Scatter Msg for one chare array element
 * Creates a terminal msg
**/
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


/**
 * Create Scatter msg for a list of chare array elements
 * Buffers for every element in the list comes at the end of the msg
**/
void* createScatterMsg(void *msg, CkScatterWrapper &w, std::vector<int> &indices){

       for(int i=0; i<indices.size(); i++){
            CkPrintf("In createScatterMsg, i:%d, index: %d \n", i, indices[i]);
       }

       envelope *env = UsrToEnv(msg);
       int msgsize = env->getTotalsize();
       int *disp = (int *) w.disp;
       CkArrayIndex *dest = (CkArrayIndex *) w.dest;
       int *cnt = (int *) w.cnt;
       int bufsize = 0;
       int ndest = indices.size();
       for(int k=0; k<ndest; k++){
            int ind = indices[k];
            bufsize += cnt[ind];
            bufsize = CK_ALIGN(bufsize, 16);
       }
       int infosize = ndest * (sizeof(int)          //displacement
                            +  sizeof(CkArrayIndex) //dest
                            +  sizeof(int)          //count
                              );
       envelope *copyenv = (envelope *)CmiAlloc(CK_ALIGN(msgsize, 16) + bufsize + infosize);
       CkPackMessage(&env);
       memcpy(copyenv, env, msgsize); 
       CkUnpackMessage(&env);
       CkUnpackMessage(&copyenv);
       copyenv->setTotalsize(CK_ALIGN(msgsize, 16) + bufsize + infosize);

       CkScatterWrapper si;
       si.aid = w.aid;
       si.ndest = ndest;
       //copy scatterv buffer
       char* buf = (char *)copyenv + CK_ALIGN(msgsize, 16);
       size_t offset = buf - (((CkMarshallMsg *)EnvToUsr(copyenv))->msgBuf);
       si.setoffset(offset);
       int tbufsize = 0;
       for(int k=0; k<ndest; k++){
            int ind = indices[k];
            memcpy(buf + tbufsize, ((char *)w.buf) + w.disp[ind], w.cnt[ind]);
            tbufsize += cnt[ind];
            tbufsize = CK_ALIGN(tbufsize, 16);
       }
       //update displacement array
       int *mdisp = (int *)(buf + tbufsize);
       offset = ((char*) mdisp) - (((CkMarshallMsg *)EnvToUsr(copyenv))->msgBuf);
       si.setoffset_disp(offset);
       tbufsize = 0;
       for(int k=0; k<ndest; k++){
            int ind = indices[k];
            mdisp[k] = tbufsize;
            tbufsize += cnt[ind];
            tbufsize = CK_ALIGN(tbufsize, 16);
       }
       //update destination array
       CkArrayIndex *mdest = (CkArrayIndex *)(&mdisp[ndest]);
       offset = ((char*) mdest) - (((CkMarshallMsg *)EnvToUsr(copyenv))->msgBuf);
       si.setoffset_dest(offset);
       for(int k=0; k<ndest; k++){
            int ind = indices[k];
            mdest[k] = dest[ind];
       }
       //update count array
       int *mcnt = (int *)(&mdest[ndest]);
       offset = ((char*) mcnt) - (((CkMarshallMsg *)EnvToUsr(copyenv))->msgBuf);
       si.setoffset_cnt(offset);
       for(int k=0; k<ndest; k++){
            int ind = indices[k];
            mcnt[k] = cnt[ind];
       }

       PUP::toMem p((void *)(((CkMarshallMsg *)EnvToUsr(copyenv))->msgBuf));
       p|si;
       return EnvToUsr(copyenv);
}





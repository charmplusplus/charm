#include "ampiimpl.h"

CkChareID ampimain::handle;
ampi_comm_struct ampimain::ampi_comms[AMPI_MAX_COMM];
int ampimain::ncomms = 0;

static void 
allReduceHandler(void *arg, int dataSize, void *data)
{
  ampi_comm_struct *commspec = (ampi_comm_struct *) arg;
  int type = commspec->rspec.type;
  if(type==0) 
  { // allreduce
    ampi::bcastraw(data, dataSize, commspec->aid);
  } else 
  { // reduce
    ampi::sendraw(0, AMPI_REDUCE_TAG, data, dataSize, commspec->aid, 
                  commspec->rspec.root);
  }
}

extern void CreateMetisLB(void);

ampimain::ampimain(CkArgMsg *m)
{
  int i;
  qwait = 0;
  for(i=0;i<AMPI_MAX_COMM;i++)
    ampi_comms[i].nobj = CkNumPes();
  i = 0;
  while(i<AMPI_MAX_COMM && CmiGetArgInt(m->argv, "+vp", &ampi_comms[i].nobj))
    i++;
  CreateMetisLB();
  numDone = 0;
  ampi_setup();
  nobjs = 0;
  char *dname;
  int isRestart;
  isRestart = CmiGetArgString(m->argv, "+restart", &dname);
  for(i=0;i<ncomms;i++)
  {
    nobjs += ampi_comms[i].nobj;
    ampi_comms[i].aid = CProxy_ampi::ckNew(new AmpiStartMsg(i), 
                                           ampi_comms[i].nobj);
    CProxy_ampi jarray(ampi_comms[i].aid);
    jarray.setReductionClient(allReduceHandler,(void*)&ampi_comms[i]);
    if(isRestart) {
      int j;
      for(j=0; j<ampi_comms[i].nobj; j++) {
        DirMsg *dmsg = new DirMsg(dname);
        jarray[j].restart(dmsg);
      }
    } else {
      int j;
      for(j=0; j<ampi_comms[i].nobj; j++) {
        ArgsInfo *argsinfo = new ArgsInfo(CmiGetArgc(m->argv), 
                                          CmiCopyArgs(m->argv));
        jarray[j].run(argsinfo);
      }
    }
  }
  delete m;
  handle = thishandle;
}

void
ampimain::done(void)
{
  numDone++;
  if(numDone==nobjs) {
    CkExit();
  }
}

void
ampimain::checkpoint(void)
{
  qwait++;
  if(qwait == nobjs)
    CkStartQD(CkIndex_ampimain::checkpointOnQd(), &thishandle);
}

void
ampimain::checkpointOnQd(void)
{
  for(int i=0;i<ncomms;i++)
  {
    CProxy_ampi jarray(ampi_comms[i].aid);
    for(int j=0; j<ampi_comms[i].nobj; j++)
      jarray[j].saveState();
  }
  qwait = 0;
}

void
ampimain::register_main(void (*mainfunc)(int, char **), char *name, int namelen)
{
  if(ncomms == AMPI_MAX_COMM)
  {
    CkAbort("AMPI> Number of registered comm_worlds exceeded limit.\n");
  }
  ampi_comms[ncomms].mainfunc = mainfunc;
  ampi_comms[ncomms].name = new char[namelen+1];
  memcpy(ampi_comms[ncomms].name, name, namelen);
  ampi_comms[ncomms].name[namelen] = '\0';
  ncomms++;
}

extern "C" void 
ampi_register_main(void (*mainfunc)(int, char **), char *name, int namelen)
{
  ampimain::register_main(mainfunc, name, namelen);
}

#include "ampimain.def.h"

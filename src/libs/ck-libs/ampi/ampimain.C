#include "ampiimpl.h"

CkChareID ampimain::handle;
CkArrayID ampimain::ampiAid;

static void allReduceHandler(void *redtype,int dataSize,void *data)
{
  int type = *((int*)redtype);
  if(type==0) { // allreduce
    ampi::bcastraw(data, dataSize, ampimain::ampiAid);
  } else { // reduce
    ampi::sendraw(0, AMPI_REDUCE_TAG, data, dataSize, ampimain::ampiAid, _rednroot);
  }
}

ampimain::ampimain(CkArgMsg *m)
{
  int i;
  nblocks = CkNumPes();
  for(i=1;i<m->argc;i++) {
    if(strncmp(m->argv[i], "+vp", 3) == 0) {
      if (strlen(m->argv[i]) > 3) {
        sscanf(m->argv[i], "+vp%d", &nblocks);
      } else {
        if (m->argv[i+1]) {
          sscanf(m->argv[i+1], "%d", &nblocks);
        }
      }
      break;
    }
  }
  numDone = 0;
  ampiAid = CProxy_ampi::ckNew(nblocks);
  CProxy_ampi jarray(ampiAid);
  jarray.setReductionClient(allReduceHandler,(void*)&_redntype);
  for(i=0; i<nblocks; i++) {
    ArgsInfo *argsinfo = new ArgsInfo(m->argc, m->argv);
    jarray[i].run(argsinfo);
  }
  delete m;
  handle = thishandle;
}

void
ampimain::done(void)
{
  numDone++;
  if(numDone==nblocks) {
    CkExit();
  }
}


#include "ampimain.def.h"

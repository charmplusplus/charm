#include "ampimain.decl.h"
#include "ampiimpl.h"

class ampimain : public Chare
{
  int nblocks;
  int numDone;
  CkArrayID arr;
  public:
    ampimain(CkArgMsg *);
    void done(void);
    void qd(void);
};

static inline void itersDone(void) { CProxy_ampimain pm(mainhandle); pm.done(); }

extern void _initCharm(int argc, char **argv);

extern "C" void conversemain_(int *argc,char _argv[][80],int length[])
{
  int i;
  char **argv = new char*[*argc+2];

  for(i=0;i <= *argc;i++) {
    if (length[i] < 100) {
      _argv[i][length[i]]='\0';
      argv[i] = &(_argv[i][0]);
    } else {
      argv[i][0] = '\0';
    }
  }
  argv[*argc+1]=0;
  
  ConverseInit(*argc, argv, _initCharm, 0, 0);
}

CkChareID mainhandle;
CkArrayID _ampiAid;

static void allReduceHandler(void *,int dataSize,void *data)
{
  TempoArray::ckTempoBcast(0, data, dataSize, _ampiAid);
}

ampimain::ampimain(CkArgMsg *m)
{
  int i;
  nblocks = CkNumPes();
  for(i=1;i<m->argc;i++) {
    if(strncmp(m->argv[i], "+vp", 3) == 0) {
      if (strlen(m->argv[i]) > 2) {
        sscanf(m->argv[i], "+vp%d", &nblocks);
      } else {
        if (m->argv[i+1]) {
          sscanf(m->argv[i+1], "%d", &nblocks);
        }
      }
      break;
    }
  }
  CProxy_migrator::ckNew();
  numDone = 0;
//  delete m;
  // CkGroupID mapID = CProxy_BlockMap::ckNew();
  // CProxy_ampi jarray(nblocks, mapID);
  _ampiAid = CProxy_ampi::ckNew(nblocks);
  // CkRegisterArrayReductionHandler(_ampiAid,allReduceHandler,0);
  CProxy_ampi jarray(_ampiAid);
  jarray.setReductionClient(allReduceHandler,0);
  for(i=0; i<nblocks; i++) {
    ArgsInfo *argsinfo = new ArgsInfo(m->argc, m->argv);
    jarray[i].run(argsinfo);
  }
  delete m;
  mainhandle = thishandle;
}

void
ampimain::qd(void)
{
  // CkWaitQD();
  // CkPrintf("Created Elements\n");
  CProxy_ampi jarray(arr);
  for(int i=0; i<nblocks; i++) {
    ArgsInfo *argsinfo = new ArgsInfo(0, NULL);
    jarray[i].run(argsinfo);
  }
  return;
}

//CpvExtern(int, _numSwitches);

void
ampimain::done(void)
{
  numDone++;
  if(numDone==nblocks) {
    // ckout << "Exiting" << endl;
    CkExit();
  }
}


#include "ampimain.def.h"

#include "f90main.h"
#include "charm-api.h"

extern "C" void FTN_NAME(F90CHARMMAIN,f90charmmain)(int, char **);

extern void _initCharm(int argc, char **argv);

CkChareID mainhandle;

// called from futil.f90 program main
extern "C" void FTN_NAME(CONVERSEMAIN, conversemain)(int *argc,char _argv[][80],int length[])
{
  int i;
  char **argv = new char*[*argc+2];

  for(i=0;i<=*argc;i++) {
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


f90main::f90main(CkArgMsg *msg)
{
  int argc = msg->argc;
  char **argv = msg->argv;
  delete msg;

  count = 0;
  mainhandle = thishandle;

  FTN_NAME(F90CHARMMAIN,f90charmmain)(argc, argv);

  /*
  int executor_grp = CProxy_executor::ckNew(); 
  CProxy_executor grp(executor_grp);
  grp.run();
  */
}

void f90main::done()
{
  count++;
  if (count == CkNumPes()) CkExit();
}

/*
void executor::run()
{
  CkPrintf("[%d] running.\n", CkMyPe());
  int i;
  main_(&i);
  CkPrintf("%d\n", i);
  CProxy_main mp(mainhandle);
  mp.done();
}
*/


#include "f90main.def.h"

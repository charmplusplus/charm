/** \file: bigsim_init.C -- Converse BlueGene Emulator Code
 *  Emulator written by Gengbin Zheng, gzheng@uiuc.edu on 5/16/2003
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <fstream>
#include <iostream>
using namespace std;

#include "bigsim_debug.h"
#undef DEBUGLEVEL
#define DEBUGLEVEL 10 
//#define  DEBUGF(x)      //CmiPrintf x;

#include "queueing.h"
#include "blue.h"
#include "blue_impl.h"    	// implementation header file
//#include "blue_timing.h" 	// timing module

#include "bigsim_ooc.h"

extern CmiStartFn bgMain(int argc, char **argv);

/* called by a AMPI thread of certan rank to attatch itself */
extern "C" void BgAttach(CthThread t)
{
//  CthShadow(t, cta(threadinfo)->getThread());
  CtvAccessOther(t, threadinfo)= cta(threadinfo);
    // special thread scheduling
  BgSetStrategyBigSimDefault(t);
    // set serial number
  CthSetSerialNo(t, CtvAccessOther(t, threadinfo)->cth_serialNo++);
}

extern "C" void BgSetStartOutOfCore(){
    DEBUGM(4, ("Set startOutOfCore!(node: %d, gId: %d, id: %d)\n", tMYNODEID, tMYGLOBALID, tMYID));
    if(cta(threadinfo)->startOutOfCore==0)
	cta(threadinfo)->startOOCChanged=1;
    cta(threadinfo)->startOutOfCore = 1;
}

extern "C" void BgUnsetStartOutOfCore(){
    DEBUGM(4, ("UnSet startOutOfCore!(node: %d, gId: %d, id: %d)\n", tMYNODEID, tMYGLOBALID, tMYID));
    if(cta(threadinfo)->startOutOfCore==1)
	cta(threadinfo)->startOOCChanged=1;
    cta(threadinfo)->startOutOfCore = 0;
}

// quiescence detection callback
// only used when doing timing correction to wait for 
static void BroadcastShutdown(void *null, double t)
{
  /* broadcast to shutdown */
  CmiPrintf("BG> In BroadcastShutdown after quiescence. \n");

  int msgSize = CmiBlueGeneMsgHeaderSizeBytes;
  void *sendmsg = CmiAlloc(msgSize);
  CmiSetHandler(sendmsg, cva(simState).exitHandler);
  CmiSyncBroadcastAllAndFree(msgSize, sendmsg);

  CmiDeliverMsgs(-1);
  if(CmiMyPe() == 0) {
    CmiPrintf("\nBG> BigSim emulator shutdown gracefully!\n");
    CmiPrintf("BG> Emulation took %f seconds!\n", CmiWallTimer()-cva(simState).simStartTime);
  }
  CsdExitScheduler();
/*
  ConverseExit();
  exit(0);
*/
}

void BgShutdown()
{
  /* when doing timing correction, do a converse quiescence detection
     to wait for all timing correction messages
  */

  if (!correctTimeLog) {
    /* broadcast to shutdown */
    int msgSize = CmiBlueGeneMsgHeaderSizeBytes;
    void *sendmsg = CmiAlloc(msgSize);
    
    CmiSetHandler(sendmsg, cva(simState).exitHandler);
    CmiSyncBroadcastAllAndFree(msgSize, sendmsg);
    
    //CmiAbort("\nBG> BlueGene emulator shutdown gracefully!\n");
    // CmiPrintf("\nBG> BlueGene emulator shutdown gracefully!\n");
    /* don't return */
    // ConverseExit();
    CmiDeliverMsgs(-1);

    if(bgUseOutOfCore)
        deInitTblThreadInMem();
    
    if(CmiMyPe() == 0) {
      CmiPrintf("\nBG> BigSim emulator shutdown gracefully!\n");
      CmiPrintf("BG> Emulation took %f seconds!\n", CmiWallTimer()-cva(simState).simStartTime);
    }
    ConverseExit();
    exit(0);
  }
  else {
  
    int msgSize = CmiBlueGeneMsgHeaderSizeBytes;
    void *sendmsg = CmiAlloc(msgSize); 
CmiPrintf("\n\n\nBroadcast begin EXIT\n");
    CmiSetHandler(sendmsg, cva(simState).beginExitHandler);
    CmiSyncBroadcastAllAndFree(msgSize, sendmsg);

    CmiStartQD(BroadcastShutdown, NULL);

#if 0
    // trapped here, so close the log
    BG_ENTRYEND();
    stopVTimer();
    // hack to remove the pending message for this work thread
    tAFFINITYQ.deq();

    CmiDeliverMsgs(-1);
    ConverseExit();
#endif
  }
}

int BGMach::traceProjections(int pe)
{
  if (procList.isEmpty()) return 1;
  return procList.includes(pe);
}

void BGMach::setNetworkModel(char *model)
{
  if (!strcmp(model, "dummy"))
    network = new DummyNetwork;
  else if (!strcmp(model, "lemieux"))
    network = new LemieuxNetwork;
  else if (!strcmp(model, "bluegenep")) {
    network = new BlueGenePNetwork;
    network->setDimensions(x, y, z, numWth);
  } else if (!strcmp(model, "bluegene"))
        network = new BlueGeneNetwork;
  else if (!strcmp(model, "redstorm"))
        network = new RedStormNetwork;
  else if (!strcmp(model, "ibmpower"))
        network = new IBMPowerNetwork;
  else
        CmiAbort("BG> unknown network setup");
}

int BGMach::read(char *file)
{
  ifstream configFile(file);
  if (configFile.fail()) {
    cout << "Bad config file, trouble opening\n";
    exit(1);
  }

  char parameterName  [1024];
  char parameterValue [1024];
                                                                                
  if (CmiMyPe() == 0)
  CmiPrintf("Reading Bluegene Config file %s ...\n", file);
                                                                                
  while (true) {
    configFile >> parameterName >> parameterValue;
    if (configFile.eof())
      break;
                                                                                
    // CmiPrintf("%s %s\n", parameterName, parameterValue);

    if (!strcmp(parameterName, "x")) {
      x = atoi(parameterValue);
      continue;
    }
    if (!strcmp(parameterName, "y")) {
      y = atoi(parameterValue);
      continue;
    }
    if (!strcmp(parameterName, "z")) {
      z = atoi(parameterValue);
      continue;
    }
    if (!strcmp(parameterName, "cth")) {
      numCth = atoi(parameterValue);
      continue;
    }
    if (!strcmp(parameterName, "wth")) {
      numWth = atoi(parameterValue);
      continue;
    }
    if (!strcmp(parameterName, "stacksize")) {
      stacksize = atoi(parameterValue);
      continue;
    }
    if (!strcmp(parameterName, "timing")) {
      if (!strcmp(parameterValue, "elapse"))
        timingMethod = BG_ELAPSE;
      else if (!strcmp(parameterValue, "walltime"))
        timingMethod = BG_WALLTIME;
      else if (!strcmp(parameterValue, "counter"))
        timingMethod = BG_COUNTER;
      else CmiAbort("BG> unknown timing method");
      continue;
    }
    if (!strcmp(parameterName, "cpufactor")) {
      cpufactor = atof(parameterValue);
      continue;
    }
    if (!strcmp(parameterName, "fpfactor")) {
      fpfactor = atof(parameterValue);
      continue;
    }
    if (!strcmp(parameterName, "log")) {
      if (!strcmp(parameterValue, "yes"))
        genTimeLog = 1;
      continue;
    }
    if (!strcmp(parameterName, "correct")) {
      if (!strcmp(parameterValue, "yes"))
        correctTimeLog = 1;
      continue;
    }
    if (!strcmp(parameterName, "traceroot")) {
      traceroot = (char *)malloc(strlen(parameterValue)+4);
      sprintf(traceroot, "%s/", parameterValue);
      continue;
    }
    if (!strcmp(parameterName, "network")) {
      setNetworkModel(parameterValue);
      continue;
    }
    if (!strcmp(parameterName, "projections")) {
      procList.set(strdup(parameterValue));
      continue;
    }
    if (!strcmp(parameterName, "record")) {
      if (!strcmp(parameterValue, "yes"))
        record = 1;
      continue;
    }
    if (!strcmp(parameterName, "recordprocessors")) {
      recordprocs.set(strdup(parameterValue));
      continue;
    }
    /* Parameters related with out-of-core execution */
//    if (!strcmp(parameterName, "bgooc")) {      
//        bgUseOutOfCore = 1;
//        bgOOCMaxMemSize = atof(parameterValue);
//        continue;
//    }           
    if (!strcmp(parameterName, "timercost")) {
      timercost = atof(parameterValue);
      continue;
    }

    if (CmiMyPe() == 0)
      CmiPrintf("skip %s '%s'\n", parameterName, parameterValue);
  }

  configFile.close();
  return 0;
}

int main(int argc,char *argv[])
{
  ConverseInit(argc,argv,(CmiStartFn)bgMain,0,0);
  return 0;
}




/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/


/*
  File: Blue_init.C -- Converse BlueGene Emulator Code
  Emulator written by Gengbin Zheng, gzheng@uiuc.edu on 5/16/2003
*/ 

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "cklists.h"

#define  DEBUGF(x)      //CmiPrintf x;

#include "queueing.h"
#include "blue.h"
#include "blue_impl.h"    	// implementation header file
#include "blue_timing.h" 	// timing module

extern CmiStartFn bgMain(int argc, char **argv);

/* called by a AMPI thread of certan rank to attatch itself */
extern "C" void BgAttach(CthThread t)
{
//  CthShadow(t, cta(threadinfo)->getThread());
  CtvAccessOther(t, threadinfo)= cta(threadinfo);
}

// quiescence detection callback
// only used when doing timing correction to wait for 
static void BroadcastShutdown(void *null)
{
  /* broadcast to shutdown */
  CmiPrintf("BG> In BroadcastShutdown after quiescence. \n");

  int msgSize = CmiBlueGeneMsgHeaderSizeBytes;
  void *sendmsg = CmiAlloc(msgSize);
  CmiSetHandler(sendmsg, cva(simState).exitHandler);
  CmiSyncBroadcastAllAndFree(msgSize, sendmsg);

  CmiDeliverMsgs(-1);
  CmiPrintf("\nBG> BlueGene emulator shutdown gracefully!\n");
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
    CmiPrintf("\nBG> BlueGene emulator shutdown gracefully!\n");
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

int main(int argc,char *argv[])
{
  ConverseInit(argc,argv,(CmiStartFn)bgMain,0,0);
  return 0;
}




#include <stdio.h>
#include "converse.h"
#include "conv-ccs.h"

void handler(char *msg)
{
  if(CcsIsRemoteRequest()) {
    char answer[1024];
    char *name=msg+CmiMsgHeaderSizeBytes;
    sprintf(answer, "hello %s from processor %d\n", name, CmiMyPe());
    CmiPrintf("CCS Ping handler called on %d with '%s'.\n",CmiMyPe(),name);
    CcsSendReply(strlen(answer)+1, answer);
  }
}

void user_main(int argc, char **argv)
{
int i;
  CcsRegisterHandler("ping2", handler);
  CcsRegisterHandler("ping", handler);
  CmiPrintf("CCS Handlers registered.  Waiting for net requests...\n");
  
}

main(int argc, char **argv)
{
  ConverseInit(argc, argv, user_main, 0, 0);
}

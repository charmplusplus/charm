#include <stdlib.h>
#include <converse.h>

CpvDeclare(int,msgSize);
CpvDeclare(int,userData);
CpvDeclare(int,recvHandler);
CpvDeclare(int,exitHandler);

void sendData()
{
  //Allocate message
  char *msg = (char *)CmiAlloc(CpvAccess(msgSize)+CmiMsgHeaderSizeBytes );
  //set allocated space to contain user data
  *((int *)(msg+CmiMsgHeaderSizeBytes)) =  CpvAccess(userData) ;
  //set Handler
  CmiSetHandler(msg,CpvAccess(recvHandler));
  //Send Message
  CmiSyncSendAndFree(0, CpvAccess(msgSize)+CmiMsgHeaderSizeBytes, msg);
}
//We finished for all message sizes. Exit now
CmiHandler recvHandlerFunc(char *msg)
{
	int myData = *((int *)(msg+CmiMsgHeaderSizeBytes));
	if (myData == CpvAccess(userData))
			CmiPrintf ("Received Expected Value\n");	
    CmiFree(msg);
 
	// Broadcast message 
    void *sendmsg = CmiAlloc(CmiMsgHeaderSizeBytes);
    CmiSetHandler(sendmsg,CpvAccess(exitHandler));
    CmiSyncBroadcastAllAndFree(CmiMsgHeaderSizeBytes,sendmsg);
}
// Exit now
CmiHandler exitHandlerFunc(char *msg)
{
    CmiFree(msg);
    CsdExitScheduler();
    return 0;
}

//Converse main. Initialize variables and register handlers
CmiStartFn mymain()
{
    CpvInitialize(int,msgSize);
    CpvInitialize(int,userData);
    CpvInitialize(int,recvHandler);
    CpvInitialize(int,exitHandler);
    CpvAccess(recvHandler) = CmiRegisterHandler((CmiHandler) recvHandlerFunc);
    CpvAccess(exitHandler) = CmiRegisterHandler((CmiHandler) exitHandlerFunc);
    CpvAccess(msgSize) = 4;
    CpvAccess(userData) = 1454;
    if (CmiMyPe() == 0)
        sendData();
    return 0;
}

int main(int argc,char *argv[])
{
    ConverseInit(argc,argv,(CmiStartFn)mymain,0,0);
    return 0;
}

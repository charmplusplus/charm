#include <converse.h>
#include <charm.h>
#ifdef CMK_OPTIMIZE
#define NDEBUG
#endif
#include <assert.h>
#include "envelope.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "fifo.h"
#include "queueing.h"

#if CMK_DEBUG_MODE

#define NUM_MESSAGES 100

extern "C" void  CpdInitializeHandlerArray(void);
extern void handlerArrayRegister(int, hndlrIDFunction, hndlrIDFunction);
extern char* genericViewMsgFunction(char *msg, int type);
extern char* getMsgListSched(void);
extern char* getMsgListPCQueue(void);
extern char* getMsgListFIFO(void);
extern char* getMsgListDebug(void);
extern char* getMsgContentsSched(int index);
extern char* getMsgContentsPCQueue(int index);
extern char* getMsgContentsFIFO(int index);
extern char* getMsgContentsDebug(int index);
extern void  msgListCache(void);
extern void  msgListCleanup(void);
extern int   getCharmMsgHandlers(int *handleArray);
extern char* getEnvInfo(envelope *env);
extern char* getSymbolTableInfo(void);

extern "C" void  CqsEnumerateQueue(Queue, void ***);
extern "C" void  FIFO_Enumerate(FIFO_QUEUE*, void***);

CpvDeclare(handlerType, handlerArray);

void **schedQueue=0;
void **FIFOQueue=0;
void **DQueue=0;

int schedIndex;
int debugIndex;
int FIFOIndex;

void msgListCleanup(void)
{
  if(schedQueue != 0) CmiFree(schedQueue);
  if(FIFOQueue != 0) free(FIFOQueue);
  if(DQueue != 0) free(DQueue);
  schedIndex = 0;
  FIFOIndex = 0;
  debugIndex = 0;

  schedQueue = 0;
  FIFOQueue = 0;
  DQueue = 0;
}

void msgListCache(void)
{
  CqsEnumerateQueue((Queue)CpvAccess(CsdSchedQueue), &schedQueue);
  FIFO_Enumerate((FIFO_QUEUE *)CpvAccess(CmiLocalQueue), &FIFOQueue);
  schedIndex = 0;
  FIFOIndex = 0;
  debugIndex = 0;
}

extern "C"
void CpdInitializeHandlerArray(void){
  int i;

  CpvInitialize(handlerType, handlerArray);
  for(i = 0; i < MAX_NUM_HANDLERS; i++){
    CpvAccess(handlerArray)[i][0] = 0;
    CpvAccess(handlerArray)[i][1] = 0;
  }
}

void handlerArrayRegister(int hndlrID, hndlrIDFunction fHeader, 
                                       hndlrIDFunction fContent){
    CpvAccess(handlerArray)[hndlrID][0] = fHeader;
    CpvAccess(handlerArray)[hndlrID][1] = fContent;
}

static const char *HeaderUnknownFormat =
"<HEADER>:Unknown Format #"
;

// type = 0 header required
//      = 1 contents required
char* genericViewMsgFunction(char *msg, int type){
  int hndlrID;
  char *temp;
  hndlrIDFunction f;

  hndlrID = CmiGetHandler(msg);
  f = CpvAccess(handlerArray)[hndlrID][type];
  if(f == 0){
    // Undefined Content/Header function
    temp = (char *)malloc(strlen(HeaderUnknownFormat)+1);
    strcpy(temp, HeaderUnknownFormat);
    return(temp);
  } else{
    return((*f)(msg));
  }
}

char* getMsgListSched(void)
{
  int ending;
  int count = 0;
  char *list;
  char t[10];
  int maxLength;

  ending = NUM_MESSAGES;
  if ( (ending + schedIndex) > ((Queue)(CpvAccess(CsdSchedQueue)))->length) {
    ending = (((Queue)(CpvAccess(CsdSchedQueue)))->length) - schedIndex;
  }
  maxLength = ending * sizeof(char) * 20 + 1;
  list = (char *)malloc(maxLength);
  strcpy(list, "");

  for(int i = schedIndex; i < ending + schedIndex; i++){
    char *temp = genericViewMsgFunction((char *)schedQueue[i], 0);
    if(strlen(list) + strlen(temp) + 10 > maxLength){ 
      free(temp);
      break;
    }
    strcat(list, temp);
    strcat(list, "#");
    sprintf(t, "%d", i);
    strcat(list, t);
    strcat(list, "#");
    count++;
    free(temp);
  }
  schedIndex += count;
  return(list);
}

static const char *NotImpl = "Not Implemented";

char* getMsgListPCQueue(void)
{
  char *list;

  list = (char *)malloc(strlen(NotImpl)+1);
  strcpy(list, NotImpl);
  return(list);
}

char* getMsgListFIFO(void)
{
  int ending;
  char *temp;
  int count = 0;
  char *list;
  char t[10];
  int maxLength;

  ending = NUM_MESSAGES;
  if ( (ending+FIFOIndex) > ((FIFO_QUEUE *)(CpvAccess(CmiLocalQueue)))->fill) {
    ending = (((FIFO_QUEUE *)(CpvAccess(CmiLocalQueue)))->fill) - FIFOIndex;
  }
  maxLength = ending * sizeof(char) * 20 + 1;
  list = (char *)malloc(maxLength);
  strcpy(list, "");

  for(int i=FIFOIndex; i < FIFOIndex+ending; i++){
    temp = genericViewMsgFunction((char *)FIFOQueue[i], 0);
    if(strlen(list) + strlen(temp) + 10 > maxLength){
      free(temp); 
      break;
    }
    strcat(list, temp);
    strcat(list, "#");
    sprintf(t, "%d", i);
    strcat(list, t);
    strcat(list, "#");
    count++;
    free(temp);
  }
  FIFOIndex += count;
  return(list);
}

char* getMsgListDebug(void)
{
  int ending;
  int count = 0;
  char *list;
  char t[10];
  int maxLength;
  char *temp;

  ending = NUM_MESSAGES;
  if ( (ending+debugIndex) > ((FIFO_QUEUE *)(CpvAccess(debugQueue)))->fill) {
      ending = (((FIFO_QUEUE *)(CpvAccess(debugQueue)))->fill) - debugIndex;
  }
  maxLength = ending * sizeof(char) * 20 + 1;
  list = (char *)malloc(maxLength);
  strcpy(list, "");

  if(DQueue != 0) free(DQueue);
  FIFO_Enumerate((FIFO_QUEUE *)CpvAccess(debugQueue), &DQueue);

  for(int i=debugIndex; i < ending+debugIndex; i++){
    temp = genericViewMsgFunction((char *)DQueue[i], 0);
    if(strlen(list) + strlen(temp) + 10 > maxLength){ 
      free(temp);
      break;
    }
    strcat(list, temp);
    strcat(list, "#");
    sprintf(t, "%d", i);
    strcat(list, t);
    strcat(list, "#");
    count++;
    free(temp);
  }
  debugIndex += count;
  return(list);
}

char* getMsgContentsSched(int index)
{
  return genericViewMsgFunction((char *)schedQueue[index], 1);
}

char* getMsgContentsPCQueue(int index)
{
  char *temp;
  temp = (char *)malloc(strlen(NotImpl)+1);
  strcpy(temp, NotImpl);
  return(temp);
}

char* getMsgContentsFIFO(int index)
{
  return genericViewMsgFunction((char *)FIFOQueue[index], 1);
}

char* getMsgContentsDebug(int index)
{
  return genericViewMsgFunction((char *)DQueue[index], 1);
}

#endif

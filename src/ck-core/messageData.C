
#include <charm++.h>
#include <iostream.h>
#include <strstream.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <fifo.h>
#include <queueing.h>

#if CMK_DEBUG_MODE

#define NUM_MESSAGES 100

//function declarations
void CpdInitializeHandlerArray();
void handlerArrayRegister(int);
char* genericViewMsgFunction(char *msg, int type);
char* getMsgListSched();
char* getMsgListPCQueue();
char* getMsgListFIFO();
char* getMsgListDebug();
char* getMsgContentsSched(int index);
char* getMsgContentsPCQueue(int index);
char* getMsgContentsFIFO(int index);
char* getMsgContentsDebug(int index);
void msgListCache();
void msgListCleanup();

extern "C" void CqsEnumerateQueue(Queue, void ***);
extern "C" void FIFO_Enumerate(FIFO_QUEUE*, void***);
extern "C" int getCharmMsgHandlers(int *handleArray);
extern "C" char* getEnvInfo(ENVELOPE *env);
extern "C" char* getSymbolTableInfo();

CpvDeclare(int *, handlerArray);
CpvDeclare(int, noOfHandlers);

void **schedQueue=0;
void **FIFOQueue=0;
void **DQueue=0;

int schedIndex;
int debugIndex;
int FIFOIndex;

void msgListCleanup(){
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

void msgListCache(){
  CqsEnumerateQueue((Queue)CpvAccess(CsdSchedQueue), &schedQueue);
  FIFO_Enumerate((FIFO_QUEUE *)CpvAccess(CmiLocalQueue), &FIFOQueue);
  schedIndex = 0;
  FIFOIndex = 0;
  debugIndex = 0;
}

void CpdInitializeHandlerArray(){
    CpvInitialize(int *, handlerArray);
    CpvInitialize(int, noOfHandlers);
    
    // Allocate memory to store the array of handlers
    CpvAccess(handlerArray) = (int *)malloc(10 * sizeof(int));
    CpvAccess(noOfHandlers) = 0;
}

void handlerArrayRegister(int hndlrID){
    CpvAccess(handlerArray)[CpvAccess(noOfHandlers)] = hndlrID;
    CpvAccess(noOfHandlers)++;
}

// type = 0 header required
//      = 1 contents required
char* genericViewMsgFunction(char *msg, int type){
    int hndlrID;
    char *unknownContentsMsg;
    char *unknownFormatMsg;
    char *temp;
    strstream str;
    
    hndlrID = CmiGetHandler(msg);

    //Look though the handlers in the handlerArray
    //and perform the appropriate action
    if((hndlrID == CpvAccess(handlerArray)[0]) || (hndlrID == CpvAccess(handlerArray)[1])){// Charm handlers
	
	//For now call the getEnvInfo function,
	//Later, incorporate Milind's changes
	if(type == 0){
	    return(getEnvInfo((ENVELOPE *)msg));
	}
	else{
	    //str << "Contents not known in this implementation" << '\0';
	    temp = (char *)malloc((strlen("Contents not known in this implementation") + 1) * sizeof(char));
	    strcpy(temp, "Contents not known in this implementation");
	    //return(str.str());
	    return(temp);
	}
    }
    else {
        //str << "<HEADER>:Unknown # Format #" << '\0';
        temp = (char *)malloc((strlen("<HEADER>:Unknown # Format #") + 1) * sizeof(char));
	strcpy(temp, "<HEADER>:Unknown # Format #");
	return(temp);
        //return(str.str());
    }
}

char* getMsgListSched(){
    strstream str;
    int ending;
    int count = 0;
    char *list;
    char t[10];
    int maxLength;
    /** debugging **/
    char *temp, *p;
    /*** ***/

    ending = NUM_MESSAGES;
    if ( (ending + schedIndex) >
         ((Queue)(CpvAccess(CsdSchedQueue)))->length) {
      ending = (((Queue)(CpvAccess(CsdSchedQueue)))->length) 
               - schedIndex;
    }
    maxLength = ending * sizeof(char) * 20 + 1;
    list = (char *)malloc(maxLength);
    strcpy(list, "");

    //CqsEnumerateQueue((Queue)CpvAccess(CsdSchedQueue), &schedQueue);
    for(int i = schedIndex; i < ending + schedIndex; i++){
        temp = genericViewMsgFunction((char *)schedQueue[i], 0);
	//str << temp << "#" << i << "#";
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
    //str << '\0';
    schedIndex += count;
  
    //p = str.str();
    
    /*** debugging ****/
    if(list == NULL) CmiPrintf("list is NULL\n");
    else CmiPrintf("list length = %d\n", strlen(list));
    /**** *****/
    
    //return(p);
    
    //Debugging
    CmiPrintf("schedIndex = %d\n", schedIndex);


    return(list);
}

char* getMsgListPCQueue(){
    strstream str;
    char *list;

    list = (char *)malloc((strlen("Not implemented") + 1) * sizeof(char));
    strcpy(list, "Not implemented");
    //str << "Not implemented" << '\0';
    //return(str.str());
    return(list);
}

char* getMsgListFIFO(){
    strstream str;
    int ending;
    char *temp;
    int count = 0;
    char *list;
    char t[10];
    int maxLength;

    ending = NUM_MESSAGES;
    if ( (ending + FIFOIndex) >
         ((FIFO_QUEUE *)(CpvAccess(CmiLocalQueue)))->fill) {
      ending = (((FIFO_QUEUE *)(CpvAccess(CmiLocalQueue)))->fill) 
               - FIFOIndex;
    }
    maxLength = ending * sizeof(char) * 20 + 1;
    list = (char *)malloc(maxLength);
    strcpy(list, "");

    //FIFO_Enumerate((FIFO_QUEUE *)CpvAccess(CmiLocalQueue), &FIFOQueue);
    for(int i = FIFOIndex; i < FIFOIndex + ending; i++){
        temp = genericViewMsgFunction((char *)FIFOQueue[i], 0);
        //str << temp << "#" << i << "#";
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
    //str << '\0';
    FIFOIndex += count;
    
    //return(str.str());
    return(list);
}

char* getMsgListDebug(){
    strstream str;
    int ending;
    int count = 0;
    char *list;
    char t[10];
    int maxLength;
    char *temp;

    ending = NUM_MESSAGES;
    if ( (ending + debugIndex) >
         ((FIFO_QUEUE *)(CpvAccess(debugQueue)))->fill) {
      ending = (((FIFO_QUEUE *)(CpvAccess(debugQueue)))->fill) 
               - debugIndex;
    }
    maxLength = ending * sizeof(char) * 20 + 1;
    list = (char *)malloc(maxLength);
    strcpy(list, "");

    //Debugging
    CmiPrintf("ending in msgListDebug = %d %d\n", ending, debugIndex);

    FIFO_Enumerate((FIFO_QUEUE *)CpvAccess(debugQueue), &DQueue);

    for(int i = debugIndex; i < ending + debugIndex; i++){
        temp = genericViewMsgFunction((char *)DQueue[i], 0);
        //str << genericViewMsgFunction((char *)DQueue[i], 0) << "#" << i << "#";
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
    //str << '\0';
    debugIndex += count;
    
    //return (str.str());
    return(list);
}

char* getMsgContentsSched(int index){
    strstream str;
    char *temp;
    //CqsEnumerateQueue((Queue)CpvAccess(CsdSchedQueue), &schedQueue);
    //str << genericViewMsgFunction((char *)schedQueue[index], 1);
    //str << '\0';
    temp = genericViewMsgFunction((char *)schedQueue[index], 1);
    //return (str.str());
    return(temp);
}

char* getMsgContentsPCQueue(int index){
    strstream str;
    char *temp;
    //str << "Not implemented";
    //str << '\0';
    //return(str.str());
    temp = (char *)malloc((strlen("Not implemented") + 1) * sizeof(char));
    strcpy(temp, "Not implemented");
    return(temp);
}

char* getMsgContentsFIFO(int index){
    strstream str;
    char *temp;
    //FIFO_Enumerate((FIFO_QUEUE *)CpvAccess(CmiLocalQueue), &FIFOQueue);
    //str << genericViewMsgFunction((char *)FIFOQueue[index], 1);
    //str << '\0';	
    temp = genericViewMsgFunction((char *)FIFOQueue[index], 1);
    //return (str.str());
    return(temp);
}

char* getMsgContentsDebug(int index){
    strstream str;
    char *temp;
    //FIFO_Enumerate((FIFO_QUEUE *)CpvAccess(debugQueue), &DQueue);
    //str << genericViewMsgFunction((char *)DQueue[index], 1);
    //str << '\0';	
    temp = genericViewMsgFunction((char *)DQueue[index], 1);
    //return (str.str());
    return(temp);
}

#endif

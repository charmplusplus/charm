#include <stdio.h>
#include <stdlib.h>
#include "converse.h"
#include "ampiProjections.h"
#include "traceCoreCommon.h"
#include "ampiEvents.h"
#include "ck.h"

static int current_tag  = -1;
static int current_src = -1;
static int current_count = -1;

extern "C" void initAmpiProjections(){
	ampi_beginProcessing(current_tag,current_src,current_count);
}

extern "C" void closeAmpiProjections(){	
	ampi_endProcessing();
}

extern "C" void ampi_beginProcessing(int tag,int src,int count){
	int *iData = (int *)malloc(sizeof(int)*3);
	iData[0] = tag;
	iData[1] = src;
	iData[2] = count;
	current_tag = tag;
	current_src = src;
	current_count = count;
	LogEvent1(_AMPI_LANG_ID,_E_BEGIN_AMPI_PROCESSING,3,iData);
}

extern "C" void ampi_endProcessing(){
	int *iData = (int *)malloc(sizeof(int)*3);
	iData[0] = current_tag;
	iData[1] = current_src;
	iData[2] = current_count;
	LogEvent1(_AMPI_LANG_ID,_E_END_AMPI_PROCESSING,3,iData);
}

extern "C" void ampi_msgSend(int tag,int dest,int count,int size){
	int *iData = (int *)malloc(sizeof(int)*4);
	iData[0] = tag;
	iData[1] = dest;
	iData[2] = count;
	iData[3] = size;
	//CmiPrintf("Size = %d\n",size);
	LogEvent1(_AMPI_LANG_ID,_E_AMPI_MSG_SEND,4,iData);
}

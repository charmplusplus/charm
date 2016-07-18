#include <stdio.h>
#include <stdlib.h>
#include "converse.h"
#include "../../../ck.h"
#include "ampiProjections.h"
#include "../../../traceCoreCommon.h"
#include "ampiEvents.h"
#include "cklists.h"
#include "ampi.h"
#include "ampiimpl.h"
static int current_rank  = -1;
static int current_src = -1;
static int current_count = -1;
extern  ampi *getAmpiInstance(MPI_Comm );


/* this dataType stores the information for each function
	 function.
*/
typedef struct {
  int funcNo;
  char *funcName;
} funcData;

typedef struct {
  int index;
  CkVec<int> funcList;
} vprocData;

CkVec<funcData *> _funcTable; /*stores the name and index for the different functions (might add somethings later on)*/
CkVec<vprocData *> vprocTable; /*stores the activation stack (only those functions that are being traced) for each virtual processor*/

extern "C" void initAmpiProjections(void){
  //ampi_beginProcessing(current_rank,current_src,current_count);
}

extern "C" void closeAmpiProjections(void){
  ampi_endProcessing(-1);
}

extern "C" void ampi_beginProcessing(int rank,int src,int count){
  int iData[3];
  iData[0] = rank; //rank of the chunk that begins executing
  iData[1] = src;
  if(vprocTable.size() <= rank || rank < 0){
    iData[2] = -1;
  }else{
    if(vprocTable[rank]->funcList.size() <= 0){
      iData[2] = -1;
    }else{
      iData[2] = vprocTable[rank]->funcList[vprocTable[rank]->funcList.size()-1];
    }
  }
  current_rank = rank;
  current_src = src;
  current_count = count;
  LogEvent1(_AMPI_LANG_ID,_E_BEGIN_AMPI_PROCESSING,3,iData);
}

extern "C" void ampi_endProcessing(int rank){
  int iData[3];
  iData[0] = current_rank;
  iData[1] = current_src;
  if(vprocTable.size() <= rank || rank < 0){
    iData[2] = -1;
  }else{
    if(vprocTable[rank]->funcList.size() <= 0){
      iData[2] = -1;
    }else{
      iData[2] = vprocTable[rank]->funcList[vprocTable[rank]->funcList.size()-1];
    }
  }

  LogEvent1(_AMPI_LANG_ID,_E_END_AMPI_PROCESSING,3,iData);
}

extern "C" void ampi_msgSend(int tag,int dest,int count,int size){
  int iData[4];
  iData[0] = tag;
  iData[1] = dest;
  iData[2] = count;
  iData[3] = size;
  //CkPrintf("Size = %d\n",size);
  LogEvent1(_AMPI_LANG_ID,_E_AMPI_MSG_SEND,4,iData);
}

extern "C" int ampi_registerFunc(char *funcName){
  for(int i=0;i<_funcTable.size();i++){
    if(strcmp(_funcTable[i]->funcName,funcName)==0){
      return _funcTable[i]->funcNo;
    }
  }
  funcData *funcElem = new funcData;
  funcElem->funcNo = _funcTable.size();
  funcElem->funcName = funcName;
  _funcTable.push_back(funcElem);
  return funcElem->funcNo;
}

extern "C" void ampi_beginFunc(int funcNo,MPI_Comm comm){
  ampi *ptr = getAmpiInstance(comm);
  int myindex = ptr->thisIndex;
  vprocData *procElem;
  if(vprocTable.size() <= myindex){
    procElem = new vprocData;
    procElem->index = myindex;
    procElem->funcList.push_back(funcNo);
    vprocTable.insert(myindex,procElem);
  }else{
    vprocTable[myindex]->funcList.push_back(funcNo);
  }
  int iData[2];
  iData[0] = myindex;
  iData[1] = funcNo;
  LogEvent3(_AMPI_LANG_ID,_E_AMPI_BEGIN_FUNC,2,iData,strlen(_funcTable[funcNo]->funcName)+1,
            _funcTable[funcNo]->funcName);
}

extern "C" void ampi_endFunc(int funcNo,MPI_Comm comm){
  ampi *ptr = getAmpiInstance(comm);
  int myindex = ptr->thisIndex;
  if(vprocTable.size() <= myindex){
  }else{
    int size = vprocTable[myindex]->funcList.size();
    if(size > 0){
      vprocTable[myindex]->funcList.remove(size-1);
    }
    int iData[2];
    iData[0] = myindex;
    iData[1] = funcNo;
    LogEvent1(_AMPI_LANG_ID,_E_AMPI_END_FUNC,2,iData);
  }
}


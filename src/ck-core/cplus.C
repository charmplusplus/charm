#include "charm++.h"
#include "trace.h"
#include <stdio.h>
#include <stdlib.h>

/* this is the handle of the main chare, used in place of MainChareID */
/* If you make these Cpv or Csv, you have to change the charm++ xlator too */
ChareIDType mainhandle;
ChareIDType NULL_HANDLE;

int _CK_NumTables = 0 ;
/* distributed table ssv ids are assigned at run time with _CK_NumTables */


/* These functions are exported from this file, and called from the core */
extern "C" void StartQuiescence(int , ChareIDType *);
extern "C" void CPlus_CallCharmInit(int, char **) ;
extern "C" void CPlus_SetMainChareID() ;

extern "C" void * CPlus_GetAccMsgPtr(_CK_Accumulator *) ;
extern "C" void CPlus_CallCombineFn(_CK_Accumulator *acc, void *msg) ;
extern "C" void CPlus_SetAccId(_CK_Accumulator *acc, int bocnum) ;
extern "C" void *CPlus_CallAccInit(int id, void *msg) ;

extern "C" void * CPlus_GetMonoMsgPtr(_CK_Monotonic *) ;
extern "C" void CPlus_CallUpdateFn(_CK_Monotonic *mono, void *msg) ;
extern "C" void CPlus_SetMonoId(_CK_Monotonic *mono, int bocnum) ;
extern "C" void *CPlus_CallMonoInit(int id, void *msg) ;



void *
comm_object::operator new(CMK_SIZE_T size) 
{
	CmiPrintf("[%d] ERROR: wrong new operator for message allocation\n",CmiMyPe()) ;
	return (void *)0;
}

void *
_CK_Object::operator new(CMK_SIZE_T size) 
{
	CmiPrintf("[%d] ERROR: wrong new operator for chare object allocation\n",CmiMyPe()) ;
	return (void *)0;
}

void * 
_CK_Object::operator new(CMK_SIZE_T size, void *buf) 
{
        return buf ;
}


_CK_Object::_CK_Object() {
        CHARE_BLOCK *chareblock = CpvAccess(currentChareBlock) ;
        SetID_onPE(thishandle, CmiMyPe());
        SetID_chare_magic_number(thishandle,GetID_chare_magic_number(chareblock->selfID)) ;
        SetID_chareBlockPtr(thishandle, chareblock);
#if CMK_DEBUG_MODE
	putObject(this);
#endif
}

_CK_Object::~_CK_Object() {
#if CMK_DEBUG_MODE
        removeObject(this);
#endif
}

char *
_CK_Object::showHeader(){
  return("Default Header");  
}

char *
_CK_Object::showContents(){
  char *ret;

  ret = (char *)malloc(sizeof(char) *15);
  sprintf(ret, "Contents : %d", (int)this);
  return(ret);
}

groupmember::groupmember()
{
        CHARE_BLOCK *cblock = CpvAccess(currentChareBlock) ;
        thisgroup = cblock->x.boc_num ;
}








void CPlus_StartQuiescence(int epnum, ChareIDType cid)
{
        StartQuiescence(epnum, &cid) ;
}
 
void CPlus_SetMainChareID()
{
  /* sets mainhandle, NULL_HANDLE etc */

  SetID_onPE(mainhandle, 0);
  if (CmiMyPe() == 0) 
    SetID_chare_magic_number(mainhandle, 
		GetID_chare_magic_number(CpvAccess(mainChareBlock)->selfID)) ;
  else
    SetID_chare_magic_number(mainhandle, 
				CpvAccess(mainChare_magic_number));
  
  SetID_chareBlockPtr(mainhandle, CpvAccess(mainChareBlock));

  /* set the NULL_HANDLE field */
  SetID_onPE(NULL_HANDLE, CK_PE_INVALID);
}



/**************************************************************************
    FOLLOWING FUNCTIONS ARE FOR ACCS AND MONOS 
**************************************************************************/

void * CPlus_GetAccMsgPtr(_CK_Accumulator *acc)
{
	return(acc->_CK_GetMsgPtr()) ;
}

void CPlus_CallCombineFn(_CK_Accumulator *acc, void *msg)
{
	acc->_CK_Combine(msg) ;
}

void CPlus_SetAccId(_CK_Accumulator *acc, int bocnum)
{
	acc->_CK_MyId = bocnum ;
}

void *CPlus_CallAccInit(int id, void *msg)
{
/*	return( (CPlus_AccMonoFnTable[id-BASE_EP_NUM])(msg) ) ;	*/
	return( ((ACCFNTYPE)(CsvAccess(PseudoTable)[id].initfn))(msg) ) ;
}


void * CPlus_GetMonoMsgPtr(_CK_Monotonic *mono)
{
	return(mono->_CK_GetMsgPtr()) ;
}

void CPlus_CallUpdateFn(_CK_Monotonic *mono, void *msg)
{
	mono->_CK_SysUpdate(msg) ;
}

void CPlus_SetMonoId(_CK_Monotonic *mono, int bocnum)
{
	mono->_CK_MyId = bocnum ;
}

void *CPlus_CallMonoInit(int id, void *msg)
{
	return( ((ACCFNTYPE)(CsvAccess(PseudoTable)[id].initfn))(msg) ) ;
}

FUNCTION_PTR CFunctionRefToName(int index)
{
	return (CsvAccess(_CK_9_GlobalFunctionTable)[index]) ;
}

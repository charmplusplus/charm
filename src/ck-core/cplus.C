/***************************************************************************
 * RCS INFORMATION:
 *
 *      $RCSfile$
 *      $Author$        $Locker$                $State$
 *      $Revision$      $Date$
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 ***************************************************************************
 * REVISION HISTORY:
 *
 * $Log$
 * Revision 2.0  1995-09-06 17:41:24  sanjeev
 * *** empty log message ***
 *
 ***************************************************************************/
static char ident[] = "@(#)$Header$";


#include <stdio.h>
#include <varargs.h>
#include "chare.h"
#include "msg_macros.h"
#include "globals.h"
#include "performance.h"

#include "c++interface.h"


CpvDeclare(ChareIDType, NULL_HANDLE) ;
/* this is the value used in Insert, etc when no handle is specified */
CpvDeclare(ChareIDType, mainhandle) ;
/* this is the handle of the main chare, used in place of MainChareID */
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




void CPlus_ChareExit()
{
	_CK_Object *temp = (_CK_Object *)CpvAccess(currentChareBlock->chareptr) ;
	delete temp ;

        SetID_chare_magic_number(CpvAccess(currentChareBlock)->selfID,-1) ;
	CmiFree(CpvAccess(currentChareBlock));
}

void CPlus_StartQuiescence(int epnum, ChareIDType cid)
{
        StartQuiescence(epnum, &cid) ;
}
 
void CPlus_SetMainChareID()
{
  /* sets mainhandle, NULL_HANDLE etc */

  SetID_onPE(CpvAccess(mainhandle), 0);
  if (CmiMyPe() == 0) 
    SetID_chare_magic_number(CpvAccess(mainhandle), 
		GetID_chare_magic_number(CpvAccess(mainChareBlock)->selfID)) ;
  else
    SetID_chare_magic_number(CpvAccess(mainhandle), 
				CpvAccess(mainChare_magic_number));
  
  SetID_chareBlockPtr(CpvAccess(mainhandle), CpvAccess(mainChareBlock));

  /* set the NULL_HANDLE field */
  SetID_onPE(CpvAccess(NULL_HANDLE), CK_PE_INVALID);
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

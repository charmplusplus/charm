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
 * Revision 2.13  1997-07-18 21:39:46  milind
 * Fixed a minor bug caused by recent projections changes.
 *
 * Revision 2.12  1995/11/02 22:50:47  knauff
 * Switched #include "c++interface.h" and #include <stdio.h> to fix
 * problems on SP.
 *
 * Revision 2.11  1995/11/02  20:23:20  sanjeev
 * added CFunctionRefToName
 *
 * Revision 2.10  1995/10/31  23:15:22  knauff
 * Undid my previous unnecessary change.
 *
 * Revision 2.9  1995/10/31  23:06:08  knauff
 * Changed all size_T's to CMK_SIZE_T
 *
 * Revision 2.8  1995/10/12  20:14:15  sanjeev
 * fixed problems while compiling with CC
 *
 * Revision 2.7  1995/10/11  19:30:33  sanjeev
 * removed CPlus_ChareExit
 *
 * Revision 2.6  1995/10/11  17:54:40  sanjeev
 * fixed Charm++ chare creation
 *
 * Revision 2.5  1995/10/02  20:43:11  knauff
 * Added return value to new operator.
 *
 * Revision 2.4  1995/09/26  19:46:35  sanjeev
 * moved new operator here
 *
 * Revision 2.3  1995/09/14  18:43:47  gursoy
 * fixed the paranthesis error which showed up after the previous fix
 *
 * Revision 2.2  1995/09/14  18:41:43  gursoy
 * fixed a cpv wrong usage
 *
 * Revision 2.1  1995/09/07  21:26:02  jyelon
 * Added prefixes to Cpv and Csv macros, fixed bugs thereby revealed.
 *
 * Revision 2.0  1995/09/06  17:41:24  sanjeev
 * *** empty log message ***
 *
 ***************************************************************************/
static char ident[] = "@(#)$Header$";

#include <varargs.h>
#include "chare.h"
#include "msg_macros.h"
#include "globals.h"
#include "trace.h"

#include "c++interface.h"

#include <stdio.h>

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
comm_object::operator new(size_t size) 
{
	CmiPrintf("[%d] ERROR: wrong new operator for message allocation\n",CmiMyPe()) ;
	return (void *)0;
}

void *
_CK_Object::operator new(size_t size) 
{
	CmiPrintf("[%d] ERROR: wrong new operator for chare object allocation\n",CmiMyPe()) ;
	return (void *)0;
}

void * 
_CK_Object::operator new(size_t size, void *buf) 
{
        return buf ;
}


_CK_Object::_CK_Object() {
        CHARE_BLOCK *chareblock = CpvAccess(currentChareBlock) ;
        SetID_onPE(thishandle, CmiMyPe());
        SetID_chare_magic_number(thishandle,GetID_chare_magic_number(chareblock->selfID)) ;
        SetID_chareBlockPtr(thishandle, chareblock);
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

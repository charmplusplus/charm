/***************************************************************************
 * RCS INFORMATION:
 *
 *	$RCSfile$
 *	$Author$	$Locker$		$State$
 *	$Revision$	$Date$
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 ***************************************************************************
 * REVISION HISTORY:
 *
 * $Log$
 * Revision 2.1  1995-06-08 17:07:12  gursoy
 * Cpv macro changes done
 *
 * Revision 1.3  1995/04/13  20:55:22  sanjeev
 * Changed Mc to Cmi
 *
 * Revision 1.2  1994/12/01  23:58:00  sanjeev
 * interop stuff
 *
 * Revision 1.1  1994/11/03  17:38:56  brunner
 * Initial revision
 *
 ***************************************************************************/
static char ident[] = "@(#)$Header$";
#include "chare.h"
#include "globals.h"
#include "mono.h"

void * GetMonoMsgPtr() ;

extern void * CPlus_CallMonoInit() ;
extern void * CPlus_GetMonoMsgPtr() ;
extern void CPlus_CallUpdateFn() ;
extern void CPlus_SetMonoId() ;


void * MonoValue(bocnum)
int bocnum;
{
        return( GetMonoMsgPtr((MONO_DATA *) GetBocDataPtr(bocnum)) );
}

MonoIDType CreateMono(id, initmsg, ReturnEP, ReturnID)
int id;
char *initmsg;
EntryPointType ReturnEP;
ChareIDType *ReturnID;
{
	int boc;
	ENVELOPE *env = (ENVELOPE  *) ENVELOPE_UPTR(initmsg);

	SetEnv_other_id(env, id);
    	boc = GeneralCreateBoc(sizeof(MONO_DATA), MONO_BranchInit_EP,
			 initmsg,  ReturnEP, ReturnID);
TRACE(CmiPrintf("[%d] CreateMono: boc = %d\n", CmiMyPe(), boc));
	return(boc);
}


MONO_BranchInit_Fn(msg, mydata)
void *msg;
MONO_DATA *mydata;
{
	ENVELOPE * env = (ENVELOPE *) ENVELOPE_UPTR(msg);

	mydata->id = GetEnv_other_id(env);
	mydata->time = 0;

        if ( IsCharmPlusPseudo(mydata->id) ) {
            mydata->dataptr = CPlus_CallMonoInit(mydata->id, msg) ;
            CPlus_SetMonoId(mydata->dataptr,MyBocNum(mydata)) ;
        }
        else
            mydata->dataptr = (void *) (*(CsvAccess(PseudoTable)[mydata->id].initfn))
					(NULL, msg);
}



_CK_9MONO_BranchNewValue(mydata, x)
MONO_DATA *mydata; 
char *x;
{
	if ((*(CsvAccess(PseudoTable)[mydata->id].pseudo_type.mono.updatefn))
		(mydata->dataptr, x))
	{
		_CK_BroadcastMono(GetMonoMsgPtr(mydata), ((BOC_BLOCK *)mydata-1)->boc_num) ;
	}
}


MONO_BranchUpdate_Fn(msg, mydata)
char *msg;
MONO_DATA *mydata;
{
        if ( IsCharmPlusPseudo(mydata->id) )
                CPlus_CallUpdateFn(mydata->dataptr,msg) ;
        else
		(*(CsvAccess(PseudoTable)[mydata->id].pseudo_type.mono.updatefn))
		(mydata->dataptr, msg);
}



MonoAddSysBocEps()
{
	EpTable[MONO_BranchInit_EP] = MONO_BranchInit_Fn;
	EpTable[MONO_BranchUpdate_EP] = MONO_BranchUpdate_Fn;
}


void * _CK_9GetMonoDataPtr(monodata)
MONO_DATA *monodata;
{
	return(monodata->dataptr);
}

FUNCTION_PTR _CK_9GetMonoCompareFn(monodata)
MONO_DATA *monodata;
{
	return(CsvAccess(PseudoTable)[monodata->id].pseudo_type.mono.updatefn);
}

void * GetMonoMsgPtr(mydata)
MONO_DATA *mydata ;
{
	if ( IsCharmPlusPseudo(mydata->id) ) 
		return(CPlus_GetMonoMsgPtr(mydata->dataptr)) ;
	else
		return(mydata->dataptr) ;
}

_CK_BroadcastMono(msg, bocnum)
void *msg ; 
int bocnum ;
{
	char *tmsg;

	tmsg = (char *) CkCopyMsg(msg);
	GeneralBroadcastMsgBranch(MONO_BranchUpdate_EP,
			tmsg,	IMMEDIATEcat, BroadcastBocMsg, bocnum) ;
}

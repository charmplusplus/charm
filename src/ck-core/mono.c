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
 * Revision 2.5  1997-10-29 23:52:49  milind
 * Fixed CthInitialize bug on uth machines.
 *
 * Revision 2.4  1995/09/01 02:13:17  jyelon
 * VID_BLOCK, CHARE_BLOCK, BOC_BLOCK consolidated.
 *
 * Revision 2.3  1995/07/27  20:29:34  jyelon
 * Improvements to runtime system, general cleanup.
 *
 * Revision 2.2  1995/07/24  01:54:40  jyelon
 * *** empty log message ***
 *
 * Revision 2.1  1995/06/08  17:07:12  gursoy
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

void _CK_BroadcastMono();


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
  boc=CreateBoc(CsvAccess(CkChare_MONO), CsvAccess(CkEp_MONO_BranchInit),
		initmsg,  ReturnEP, ReturnID);
  TRACE(CmiPrintf("[%d] CreateMono: boc = %d\n", CmiMyPe(), boc));
  return(boc);
}


void MONO_BranchInit_Fn(msg, mydata)
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



void _CK_9MONO_BranchNewValue(mydata, x)
MONO_DATA *mydata; 
char *x;
{
	if ((*(CsvAccess(PseudoTable)[mydata->id].pseudo_type.mono.updatefn))
		(mydata->dataptr, x))
	{
		_CK_BroadcastMono(GetMonoMsgPtr(mydata), MyBocNum(mydata));
	}
}


void MONO_BranchUpdate_Fn(msg, mydata)
char *msg;
MONO_DATA *mydata;
{
        if ( IsCharmPlusPseudo(mydata->id) )
                CPlus_CallUpdateFn(mydata->dataptr,msg) ;
        else
		(*(CsvAccess(PseudoTable)[mydata->id].pseudo_type.mono.updatefn))
		(mydata->dataptr, msg);
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

void _CK_BroadcastMono(msg, bocnum)
void *msg ; 
int bocnum ;
{
	char *tmsg;

	tmsg = (char *) CkCopyMsg(msg);
	GeneralBroadcastMsgBranch(CsvAccess(CkEp_MONO_BranchUpdate),
			tmsg,	ImmBroadcastBocMsg, bocnum) ;
}


void MonoAddSysBocEps(void)
{
  CsvAccess(CkChare_MONO) =
    registerChare("CkChare_MONO",sizeof(MONO_DATA),NULL);

  CsvAccess(CkEp_MONO_BranchInit) =
    registerBocEp("CkEp_MONO_BranchInit",
		  MONO_BranchInit_Fn,
		  CHARM, 0, CsvAccess(CkChare_MONO));
  CsvAccess(CkEp_MONO_BranchUpdate) =
    registerBocEp("CkEp_MONO_BranchUpdate",
		  MONO_BranchUpdate_Fn,
		  CHARM, 0, CsvAccess(CkChare_MONO));
}



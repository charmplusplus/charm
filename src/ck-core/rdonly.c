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
 * Revision 2.9  1997-10-29 23:52:51  milind
 * Fixed CthInitialize bug on uth machines.
 *
 * Revision 2.8  1997/07/18 21:21:10  milind
 * all files of the form perf-*.c have been changed to trace-*.c, with
 * name expansions. For example, perf-proj.c has been changed to
 * trace-projections.c.
 * performance.h has been renamed as trace.h, and perfio.c has been
 * renamed as traceio.c.
 * Corresponding changes have been made in the Makefile too.
 * Earlier, there used to be three libck-core-*.a where * was projections,
 * summary or none. Now, there will be a single libck-core.a and
 * three libck-trace-*.a where *=projections, summary and none.
 * The execmode parameter to charmc script has been renamed as
 * tracemode.
 * Also, the perfModuleInit function has been renamed as traceModuleInit,
 * RecdPerfMsg => RecdTraceMsg
 * CollectPerfFromNodes => CollectTraceFromNodes
 *
 * Revision 2.7  1997/03/24 23:14:04  milind
 * Made Charm-runtime 64-bit safe by removing conversions of pointers to
 * integers. Also, removed charm runtime's dependence of unused argv[]
 * elements being 0. Also, added sim-irix-64 version. It works.
 *
 * Revision 2.6  1995/09/07 21:21:38  jyelon
 * Added prefixes to Cpv and Csv macros, fixed bugs thereby revealed.
 *
 * Revision 2.5  1995/09/07  05:26:49  gursoy
 * made the necessary changes related to CharmInitLoop--> handler fuction
 *
 * Revision 2.4  1995/09/05  22:03:34  sanjeev
 * removed call to CPlus_GetMagicNumber
 *
 * Revision 2.3  1995/07/24  01:54:40  jyelon
 * *** empty log message ***
 *
 * Revision 2.2  1995/07/22  23:45:15  jyelon
 * *** empty log message ***
 *
 * Revision 2.1  1995/06/08  17:07:12  gursoy
 * Cpv macro changes done
 *
 * Revision 1.3  1995/04/21  22:43:18  sanjeev
 * fixed mainchareid bug
 *
 * Revision 1.2  1994/12/01  23:58:02  sanjeev
 * interop stuff
 *
 * Revision 1.1  1994/11/03  17:39:01  brunner
 * Initial revision
 *
 ***************************************************************************/
static char ident[] = "@(#)$Header$";
#include "chare.h"
#include "globals.h"
#include "trace.h"


CpvExtern(char *, ReadBufIndex);
CpvExtern(char *, ReadFromBuffer);

/************************************************************************/
/* The following functions are used to copy the read-buffer out of and 	*/
/* into the read only variables. 					*/
/************************************************************************/
void _CK_13CopyToBuffer(srcptr, var_size) 
char *srcptr;
int var_size;
{
  int i;

  for (i=0; i<var_size; i++) 
    *CpvAccess(ReadBufIndex)++ = *srcptr++;
}

void _CK_13CopyFromBuffer(destptr, var_size) 
char *destptr;
int var_size;
{
  int i;

  for (i=0; i<var_size; i++) 
    *destptr++ = *CpvAccess(ReadFromBuffer)++;
}


void BroadcastReadBuffer(ReadBuffer, size, mainChareBlock)
char *ReadBuffer;
int size;
struct chare_block * mainChareBlock;
{
	ENVELOPE * env;

	env = ENVELOPE_UPTR(ReadBuffer);
	SetEnv_msgType(env, ReadVarMsg);
	
	/* this is where we add the information for the main chare
	block */
	SetEnv_chareBlockPtr(env, mainChareBlock);
	SetEnv_chare_magic_number(env, 
			GetID_chare_magic_number(mainChareBlock->selfID));

        CkCheck_and_BcastInitNL(env);
}


void ReadMsgInit(msg, id)
char *msg;
int id;
{
	int packed;
	ENVELOPE *env ;

	env = ENVELOPE_UPTR(msg);
	CpvAccess(NumReadMsg)++;
	SetEnv_msgType(env, ReadMsgMsg);
	SetEnv_other_id(env, id);
	if (GetEnv_isPACKED(env) == UNPACKED)
		packed = 1;
	else
		 packed = 0;

        CkCheck_and_BcastInitNFNL(env); 
}


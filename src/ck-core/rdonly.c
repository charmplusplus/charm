#include "charm.h"

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
	CmiSetHandler(env,CsvAccess(HANDLE_INIT_MSG_Index));
	CmiSyncBroadcastAndFree(GetEnv_TotalSize(env), env);
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
        PACK(env);
        CmiSetHandler(env,CsvAccess(HANDLE_INIT_MSG_Index));
	CmiSyncBroadcast(GetEnv_TotalSize(env), env);
	UNPACK(env);
}


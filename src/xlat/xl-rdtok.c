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
 * Revision 2.0  1995-06-05 18:52:05  brunner
 * Reorganized file structure
 *
 * Revision 1.2  1995/04/23  18:30:18  milind
 * Changed list of keywords to include Cmi functions.
 *
 * Revision 1.1  1994/11/03  17:41:53  brunner
 * Initial revision
 *
 ***************************************************************************/
static char ident[] = "@(#)$Header$";
#include "xl-lex.h"
#include "xl-sym.h"

struct token TokenData[]={
{ "chare", 257 },
{ "main", 258 },
{ "message", 259 },
{ "entry", 260 },
{ "private", 261 },
{ "DataInit", 262 },
{ "CharmInit", 263 },
{ "QUIESCENCE", 264 },
{ "module", 265 },
{ "BranchOffice", 266 },
{ "readonly", 267 },
{ "monotonic", 268 },
{ "table", 269 },
{ "accumulator", 270 },
{ "static", 271 },
{ "pack", 272 },
{ "unpack", 273 },
{ "varSize", 274 },
{ "function", 275 },
{ "branch", 276 },
{ "sizeof", 277 },
{ "auto", 278 },
{ "break", 279 },
{ "case", 280 },
{ "char", 281 },
{ "const", 282 },
{ "continue", 283 },
{ "default", 284 },
{ "do", 285 },
{ "double", 286 },
{ "else", 287 },
{ "enum", 288 },
{ "extern", 289 },
{ "float", 290 },
{ "for", 291 },
{ "goto", 292 },
{ "if", 293 },
{ "int", 294 },
{ "long", 295 },
{ "register", 296 },
{ "return", 297 },
{ "short", 298 },
{ "signed", 299 },
{ "struct", 300 },
{ "switch", 301 },
{ "typedef", 302 },
{ "union", 303 },
{ "unsigned", 304 },
{ "void", 305 },
{ "while", 306 },
{ "interface", 307 },
{ "ChareIDType", 308 },
{ "EntryPointType", 309 },
{ "PeNumType", 310 },
{ "PackIDType", 311 },
{ "WriteOnceID", 312 },
{ "PVECTOR", 313 },
{ "ChareNumType", 314 },
{ "EntryNumType", 315 },
{ "BOOLEAN", 316 },
{ "PrivateCall", 359 },
{ "BranchCall", 360 },
{ "public", 362 },
{ "FunctionNameToRef", 363 },
{ "CkAllocMsg", 364 },
{ "CkAllocPrioMsg", 365 },
{ "ReadMsgInit", 366 },
{ "Accumulate", 367 },
{ "NewValue", 368 },
{ "NULL_VID", 0 },
{ "NULL_PE", 0 },
{ "AccIDType", 369 },
{ "MonoIDType", 370 },
{ "DummyMsg", 371 },
{ "FunctionRefType", 372 },
{ "FUNCTION_PTR", 373 },
{ "FunctionRefToName", 374 },
{ "implicit", 375 },
{ "CkBlockedRecv", 376 },
{ "dag", 377 },
{ "MATCH", 378 },
{ "AUTOFREE", 379 },
{ "when", 380 },
{ "ChareCall", 381 },
{ "ChareNameType", 382 },
{ "export_to_C", 383 },
{ 0, 0 }
};

struct token KeyData[]={
{ "extern", 0 },
{ "auto", 0 },
{ "register", 0 },
{ "static", 0 },
{ "function", 0 },
{ "sizeof", 0 },
{ "break", 0 },
{ "case", 0 },
{ "const", 0 },
{ "continue", 0 },
{ "default", 0 },
{ "do", 0 },
{ "else", 0 },
{ "for", 0 },
{ "goto", 0 },
{ "if", 0 },
{ "return", 0 },
{ "switch", 0 },
{ "while", 0 },
{ "PrivateCall", 0 },
{ "BranchCall", 0 },
{ "ReadMsgInit", 0 },
{ 0,0 }
};

char *SyscallData[]=
{
"_CK_CreateChare",
"SendMsg",
"MyChareID",
"MyParentID",
"MainChareID",
"CkAlloc",
"CkAllocMsg",
"CkAllocPrioMsg",
"CkFree",
"CkFreeMsg",
"CkPrintf",
"CkScanf",
"ChareExit",
"CkTimer",
"CkUTimer",
"CkHTimer",
"CkExit",
"CkCopyMsg",
"CmiTimer",
"CmiNumPe",
"CmiMyPe",
"_CK_CreateBoc",
"_CK_SendMsgBranch",
"_CK_BroadcastMsgBranch",
"_CK_MyBocNum",
"ReadValue",
"ReadInit",
"_CK_MyBranchID",
"_CK_CreateAcc",
"_CK_CreateMono",
"CollectValue",
"MonoValue",
"_CK_Find",
"_CK_Delete",
"_CK_Insert",
"WriteOnce",
"DerefWriteOnce",
"CkPriorityPtr",
"atoi",
"itoa",
"isalpha",
"isdigit",
"islower",
"isspace",
"isupper",
"rand",
"random",
"srand",
"srandom",
"strcmp",
"strlen",
"strsave",
"power",
"lower",
"CkAllocPrioBuffer",
"CkAllocBuffer",
"McUTimer",
"McHTimer",
"CkAllocPackBuffer",
"CmiSpanTreeRoot",
"CmiSpanTreeParent",
"CmiSpanTreeChild",
"CmiNumSpanTreeChildren",
"fprintf",
"fscanf",
"fclose",
"SetRefNumber",
"GetRefNumber",
"StartQuiescence",
"IsChareLocal",
"GetChareDataPtr",
"McTotalNumPe",
0
};

#define TRUE 1
#define FALSE 0

#define SYSCALLS 100

struct token *TokenArray;
int           TotalTokens=0;

struct token *KeyArray;
int           TotalKeys=0;

void ReadTokens()
{
TokenArray = TokenData;
for (TotalTokens=0; TokenArray[TotalTokens].name; TotalTokens++);
}

void ReadKeys()
{
KeyArray = KeyData;
for (TotalKeys=0; KeyArray[TotalKeys].name; TotalKeys++);
}

void InsertSysCalls()
{
  int i;
  SYMTABPTR worksymtab;

  for (i=0; SyscallData[i]; i++)
    {
    worksymtab = Insert(SyscallData[i],CurrentTable);
    worksymtab->idtype = SYSCALLNAME;
    worksymtab->type = INTPTR;
    }
}

int SearchKey(key)
char *key;
{ int i;

  for (i=0;i<TotalTokens;i++)
	if (!strcmp(key,TokenArray[i].name)) return(TokenArray[i].tokenvalue);
  return(-1);
}

int IsKey(tokenstring)
char *tokenstring;
{ int i;

  for (i=0;i<TotalKeys;i++)
	if (!strcmp(tokenstring,KeyArray[i].name)) return(TRUE);
  return(FALSE);
}



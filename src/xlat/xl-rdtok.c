#include "xl-lex.h"
#include "xl-sym.h"
#include "xl-yacc.tab.h"

struct token TokenData[]={
{ "chare",   CHARE },
{ "main",    MAIN },
{ "message", MESSAGE },
{ "entry",   ENTRY },
{ "private", PRIVATE },
{ "DataInit", DATAINIT },
{ "CharmInit", CHAREINIT },
{ "QUIESCENCE", QUIESCENCE },
{ "module", MODULE },
{ "BranchOffice", BRANCHOFFICE },
{ "readonly", READONLY },
{ "monotonic", MONOTONIC },
{ "table", TABLE },
{ "accumulator", ACCUMULATOR },
{ "static", STATIC },
{ "pack", PACK },
{ "unpack", UNPACK },
{ "varSize", VARSIZE },
{ "function", FUNCTION },
{ "branch", BRANCH },
{ "sizeof", SIZEOF },
{ "auto", AUTO },
{ "break", BREAK },
{ "case", CASE },
{ "char", CHAR },
{ "const", CONST },
{ "continue", CONTINUE },
{ "default", DEFAULT },
{ "do", DO },
{ "double", DOUBLE },
{ "else", ELSE },
{ "enum", ENUM },
{ "extern", EXTERN },
{ "float", FLOAT },
{ "for", FOR },
{ "goto", GOTO },
{ "if", IF },
{ "int", INT },
{ "long", LONG },
{ "register", REGISTER },
{ "return", RETURN },
{ "short", SHORT },
{ "signed", SIGNED },
{ "struct", STRUCT },
{ "switch", SWITCH },
{ "typedef", TYPEDEF },
{ "union", UNION },
{ "unsigned", UNSIGNED },
{ "void", VOID },
{ "while", WHILE },
{ "interface", INTERFACE },
{ "ChareIDType", ChareIDType },
{ "EntryPointType", EntryPointType },
{ "PeNumType", PeNumType },
{ "PackIDType", PackIDType },
{ "WriteOnceID", WriteOnceID },
{ "PVECTOR", PVECTOR },
{ "ChareNumType", ChareNumType },
{ "EntryNumType", EntryNumType },
{ "BOOLEAN", BOOLEAN },
{ "PrivateCall", PRIVATECALL },
{ "BranchCall", BRANCHCALL },
{ "public", PUBLIC },
{ "CkAllocMsg", CKALLOCMSG },
{ "CkAllocPrioMsg", CKALLOCPRIOMSG },
{ "ReadMsgInit", READMSGINIT },
{ "Accumulate", ACCUMULATE },
{ "NewValue", NEWVALUE },
{ "AccIDType", ACCIDTYPE },
{ "MonoIDType", MONOIDTYPE },
{ "DummyMsg", DUMMYMSG },
{ "FunctionRefType", FunctionRefType },
{ "FUNCTION_PTR", FUNCTION_PTR },
{ "implicit", IMPLICIT },
{ "CkBlockedRecv", BLOCKEDRECV },
{ "dag", DAG },
{ "MATCH", MATCH },
{ "AUTOFREE", AUTOFREE },
{ "when", WHEN },
{ "ChareCall", CHARECALL },
{ "ChareNameType", ChareNameType },
{ "export_to_C", EXPORT_TO_C },
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
"_CK_ImmSendMsgBranch",
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



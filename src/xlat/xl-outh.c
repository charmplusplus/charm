#include "xl-lex.h"
#include "xl-sym.h"

SYMTABPTR       GlobalEntryFind();

extern char    *itoa();
extern char    *AppendMap();
extern char    *AppendMapIndex();
#define IMPORTINITCODE 2
#define MSGFILLCODE 11
#define EPTOPACKIDCODE 12
#define CHARECOUNT 14
#define BOCCOUNT 15
#define CHAREINITCODE 16
#define READBUFFERCODE 17
#define WRITEBUFFERCODE 18
#define BUFFERSIZECODE 19
#define PSEUDOCOUNT 22
#define PSEUDOINITCODE 23

/*
 * code 4 for localptr, code 9 for function tables, code 3 for "main" table
 * and structure init. calls, code 5 for "main" function and EP counts, code
 * 13 for required variables. code 20,21 is reserved for global vars.
 * 
 * _CKx, where x is in 0..9 is for internally generated functions, x an alphabet
 * signifies module name for structs. x=_ is ALWAYS followed by one of the
 * above codes.
 */

/***** IMPORTANT NOTE : _CK_4localptr is CkLocalPtr in yaccspec *****/

extern SYMTABPTR ImportModule, ModuleDefined;
extern char    *Map();

void            GenerateStruct();
void            InitializeStruct();

char           *CEPVAR = "CpvAccess(_CK_13ChareEPCount)";
char           *TOTMSG = "CpvAccess(_CK_13TotalMsgCount)";
char           *PACKMSG = "CpvAccess(_CK_13PackMsgCount)";
char           *PACKOFFSET = "CpvAccess(_CK_13PackOffset)";
char           *CkGenericAlloc = "GenericCkAlloc";
char           *CkVarSizeAlloc = "CsvAccess(MsgToStructTable)";
char           *REFSUFFIX = "_ref";
char           *CkReadMsgTable = "_CK_ReadMsgTable"; /* ??? */

char           *CkSizeArray = "_CK_13SizeArray"; /* ??? */
char           *CkSizeArrayIndex = "_CK_13SizeArrayIndex"; /* ??? */
char           *CkCopyFromBuffer = "_CK_13CopyFromBuffer";
char           *CkCopyToBuffer = "_CK_13CopyToBuffer";

char           *CkPrefix = "_CK_";
char           *CkPrefix_ = "_CK";
char           *CkMainTable = "void _CK_3mainTable(ft,fi,cet,cei,bet,bei,cetimp,betimp,cetname,betname,cetchare,betchare)\nint *fi,*bei,*cei,*cetimp,*betimp;\nFNPTRTYPE ft,cet,bet;\nchar *cetname[], *betname[];\nint cetchare[], betchare[];\n";
char           *CkMainMsgTableInit = "void _CK_3mainMessageTableInit(mt,emt,bocemt)\nMSG_STRUCT mt[];\nint emt[];\nint bocemt[];";
char           *CkMainMsgPUAInit = "void _CK_3mainmsgpuainit(mt)\nMSG_STRUCT mt[];";
char           *CkCallMsgFill = "_CK_3mainmsgpuainit(mt)";
char           *CkMainEPtoMsgNo = "void _CK_3mainEPtoMsgNo(ept, bocept)\nint ept[], bocept[];";
char           *CkCallEPFill = "_CK_3mainEPtoMsgNo(emt, bocemt)";
char           *CkMainDataInit = "void _CK_3maindatainit()";
char           *CkMainChareInit = "void _CK_3mainChareInit(ct,cname)\nint ct[];\nchar *cname[];\n";
char           *CkMainPseudoInit = "void _CK_3mainPseudoInit(amt)\nPSEUDO_STRUCT amt[];";

char           *CkMainCopyFromBuffer = "void _CK_3mainCopyFromBuffer()";
char           *CkMainCopyToBuffer = "void _CK_3mainCopyToBuffer()";
char           *CkMainBufferSize = "int _CK_5mainBufferSize(i)";
/*
 * char *CkMainDataInit="void _CK_3maindatainit(_CK_4localptr)\nvoid
 * *dataptr;";
 */
char           *CkMainFCount = "int _CK_5mainFunctionCount()";
char           *CkMainCECount = "int _CK_5mainChareEPCount()";
char           *CkMainMainCECount = "int _CK_5mainMainChareEPCount()";
char           *CkMainBECount = "int _CK_5mainBranchEPCount()";
char           *CkMainMsgCount = "int _CK_5mainMessageCount()";
char           *CkMainChareCount = "int _CK_5mainChareCount()";
char           *CkMainPseudoCount = "int _CK_5mainPseudoCount()";
char           *CkMainQEP = "int _CK_5mainQuiescenceEP()";
char           *CkMainDataSize = "int _CK_5mainDataSize()\n{ return sizeof(main_Data); }\n\n";
char           *CkMainCharm = "void _CK_3mainCharm(_CK_4NULL,_CK_4localptr,argc,argv)\nvoid *_CK_4NULL,*_CK_4localptr;\nint argc;\nchar *argv[];";
char           *CkMainQuiescence = "void _CK_3mainQuiescence(_CK_4NULL,_CK_4localptr)\nvoid *_CK_4localptr,*_CK_4NULL;";
char           *CKMAINDATAFUNCTION = "(_CK_4NULL,_CK_4localptr,argc,argv)";
char           *CKMAINCHAREFUNCTION = "(_CK_4NULL,_CK_4localptr,argc,argv)";
char           *CKMAINQUIESCENCEFUNCTION = "(_CK_4NULL,_CK_4localptr)";
char           *FunctionType = "FUNCTION_PTR";
char           *VoidFnPtr = "VOIDFNPTR";
char           *FunctionPtrType = "FNPTRTYPE";

FILE           *outh1, *outh2;
FILE           *outh, *outh0;

int             ALLMESSAGES;

static int EmitComma;

void 
Indent(level)
int             level;
{
	int             tabs, spaces, i;
	outh = outh1;
	tabs = level / 4;
	spaces = level % 4;
	for (i = 0; i < tabs; i++)
		fprintf(outh, "\t");
	for (i = 0; i < spaces; i++)
		fprintf(outh, "  ");
}

void 
RecursiveGenerateStruct(table, level)
SYMTABPTR       table;
int             level;
{
	if (table == NULL)
		return;
	RecursiveGenerateStruct(table->left, level);
	RecursiveGenerateStruct(table->right, level);
	if (!strcmp(table->name, " "))
		return;
	outh = outh1;
	switch (table->idtype)
	{
	case BOCNAME:
	case CHARENAME:
		GenerateStruct(table->name, table->type->table, TRUE,
			       CkPrefix, level);
		Indent(level);
		fprintf(outh, "int %s;\n", table->name);
		break;
	case PRIVATEFNNAME:
		break;
	case ACCNAME:
	case MONONAME:
	case MESSAGENAME:
	case PUBLICFNNAME:
	case ENTRYNAME:
		Indent(level);
		fprintf(outh, "int %s;\n", table->name);
		break;
	case FNNAME:
		Indent(level);
		fprintf(outh, "%s %s;\n", FunctionType, table->name);
		Indent(level);
		fprintf(outh, "int %s%s;\n", table->name, REFSUFFIX);
	default:
		break;
	}
}

void
RecursiveGenerateProtos(table)
SYMTABPTR       table;
{
	if (table == NULL)
		return;
	RecursiveGenerateProtos(table->left);
	RecursiveGenerateProtos(table->right);
	if (!strcmp(table->name, " ")) 
	    return;
	switch (table->idtype)
	{
	case PRIVATEFNNAME:
		break;
	case PUBLICFNNAME:
		break;
	case ENTRYNAME:
		break;
	case FNNAME:
                break;
	}
}



void 
RecursiveInitializeStruct(table, level)
SYMTABPTR       table;
int             level;
{
	if (table == NULL)
		return;
	RecursiveInitializeStruct(table->left, level);
	RecursiveInitializeStruct(table->right, level);
	if (!strcmp(table->name, " "))
		return;
	outh = outh1;
	switch (table->idtype)
	{
	case BOCNAME:
	case CHARENAME:
		InitializeStruct(table->name, table->type->table, TRUE,
			       CkPrefix, level);
		if(EmitComma)
			fprintf(outh,",");
		fprintf(outh, "0 ");
		EmitComma = 1;
		break;
	case PRIVATEFNNAME:
		break;
	case ACCNAME:
	case MONONAME:
	case MESSAGENAME:
	case PUBLICFNNAME:
	case ENTRYNAME:
		if(EmitComma)
			fprintf(outh,",");
		fprintf(outh, "0 ");
		EmitComma = 1;
		break;
	case FNNAME:
		if(EmitComma)
			fprintf(outh,",");
		fprintf(outh, "_CkNullFunc, ");
		fprintf(outh, "0 ");
		EmitComma = 1;
	default:
		break;
	}
}

void 
GenerateStruct(name, table, ownflag, prefix, level)
SYMTABPTR       table;
int             ownflag;
char           *prefix;
int             level;
{
	if ((InPass1) || (table == NULL))
		return;
	outh = outh1;
	if (!ownflag)
		fprintf(outh, "extern ");
	Indent(level);
	if(level==0 && ownflag)
		fprintf(outh, "struct %s%s_type {\n", prefix,name);
	else
		fprintf(outh, "struct {\n");

	RecursiveGenerateStruct(table, level + 1);
	Indent(level);
	if(level==0 && ownflag)
		fprintf(outh, "};\n", prefix, name);
	else
		fprintf(outh, "} %s%s;\n", prefix, name);
	if (level == 0)
		fprintf(outh, "\n");
}

/* added by milind 5/2/95 */
void 
InitializeStruct(name, table, ownflag, prefix, level)
SYMTABPTR       table;
int             ownflag;
char           *prefix;
int             level;
{
	if ((InPass1) || (table == NULL))
		return;
	outh = outh1;
	if (!ownflag)
		return;
	if(level==0)
	{
		fprintf(outh,"extern _CkNullFunc();\n");
		fprintf(outh,"struct %s%s_type %s%s  = {",prefix,name,
				prefix,name);
		EmitComma = 0;
	}
	else
	{
		if(EmitComma)
			fprintf(outh,",");
		fprintf(outh,"{");
		EmitComma = 0;
	}
	RecursiveInitializeStruct(table, level + 1);
	fprintf(outh, "}");
	EmitComma = 1;
	if (level == 0)
		fprintf(outh, ";\n");
}

void 
CreateStructures(table)
SYMTABPTR       table;
{
	GenerateStruct(ModuleDefined->name, ModuleDefined->type->table, 1, CkPrefix_, 0);
        RecursiveGenerateProtos(ModuleDefined->type->table);
	InitializeStruct(ModuleDefined->name, ModuleDefined->type->table, 1, CkPrefix_, 0);
}

void 
ImportStructInit(table, module)
SYMTABPTR       table;
char           *module;
{
	char           *temp, *dummy;
	if (table == NULL)
		return;

	ImportStructInit(table->left, module);
	ImportStructInit(table->right, module);
	outh = outh2;
	switch (table->idtype)
	{
	case BOCNAME:
	case CHARENAME:
		ImportStructInit(table->type->table, module);
		break;
	case FNNAME:
		temp = ModulePrefix(module, table->name);
		fprintf(outh, "  %s%s = ", temp, REFSUFFIX);
		dummy = MyModulePrefix(module, table->name);
		fprintf(outh, "%s%s;\n", dummy, REFSUFFIX);
		fprintf(outh, "%s = (%s) %s;\n", temp, FunctionType, dummy);
		dontfree(temp);
		dontfree(dummy);
		break;
	case PRIVATEFNNAME:
		break;
	case PUBLICFNNAME:
	case ENTRYNAME:
		temp = ModuleCharePrefix(module, table->charename->name,
					 table->name);
		fprintf(outh, "  %s = ", temp);
		dontfree(temp);
		temp = MyModuleCharePrefix(module, table->charename->name,
					   table->name);
		fprintf(outh, "%s;\n", temp);
		dontfree(temp);
		break;
	default:
		break;
	}
}

void 
CreateImportInitFunction()
{
	if (ModuleDefined == NULL)
		error("Module Undefined? Something Wrong!", EXIT);
	outh = outh2;
	fprintf(outh, "void %s%d%s()\n{\n", CkPrefix, IMPORTINITCODE,
		ModuleDefined->name);
	ImportStructInit(ImportModule, ModuleDefined->name);
	fprintf(outh, "}\n\n");
}

/* added by milind */

void CreateImportModuleComponentFill();

void 
CreateOwnImportInitFunction()
{
	if (ModuleDefined == NULL)
		error("Module Undefined? Something Wrong!", EXIT);
	outh = outh2;
	fprintf(outh,
		"char *_CK_%s_struct_id=\"\\0charmc autoinit %s_struct\";\n",
		ModuleDefined->name, ModuleDefined->name);
	fprintf(outh, "void %s%s_struct_init()\n{\n", CkPrefix,
		ModuleDefined->name);
	fprintf(outh, "/* ImportStructInit */\n");
	ImportStructInit(ImportModule, ModuleDefined->name);
	fprintf(outh, "/* CreateImportModuleComponentFill */\n");
	CreateImportModuleComponentFill(ImportModule, ModuleDefined->name, 
					BOCNAME);
	CreateImportModuleComponentFill(ImportModule, ModuleDefined->name, 
					CHARENAME);
	CreateImportModuleComponentFill(ImportModule, ModuleDefined->name, 
					ENTRYNAME);
	CreateImportModuleComponentFill(ImportModule, ModuleDefined->name, 
					MESSAGENAME);
	CreateImportModuleComponentFill(ImportModule, ModuleDefined->name, 
					ACCNAME);
	CreateImportModuleComponentFill(ImportModule, ModuleDefined->name, 
					MONONAME);
	CreateImportModuleComponentFill(ImportModule, ModuleDefined->name, 
					TABLENAME);
	CreateImportModuleComponentFill(ImportModule, ModuleDefined->name,
					PRIVATEFNNAME);
	CreateImportModuleComponentFill(ImportModule, ModuleDefined->name,
					PUBLICFNNAME);
	CreateImportModuleComponentFill(ImportModule, ModuleDefined->name,
					FNNAME);
	fprintf(outh, "}\n\n");
}

void 
CallOtherModuleInits(table)
SYMTABPTR       table;
{
	if (table == NULL)
		return;
	CallOtherModuleInits(table->left);
	CallOtherModuleInits(table->right);
	outh = outh2;
	if ((table->idtype == MODULENAME) && (table->declflag == IMPORTED))
	  {
	    fprintf(outh, "\n  %s%s_init();", CkPrefix, table->name);
	  }
}

void
CreateOwnMsgInitFunction(table, name)
SYMTABPTR       table;
char           *name;
{
	char           *temp;
	if (table == NULL)
		return;
	CreateOwnMsgInitFunction(table->left, name);
	CreateOwnMsgInitFunction(table->right, name);

	if (table->idtype != MESSAGENAME)
		return;
	temp = MyModulePrefix(name, table->name);
	fprintf(outh, "  %s = registerMsg(", temp);
	fprintf(outh, "\"%s\",",table->name);
	/* AllocFunction */
	if (table->localid <= 0)
	{
		fprintf(outh, "(%s) %s,", FunctionType, CkGenericAlloc);
	}
	else
	{
		fprintf(outh, "(%s) %s,", FunctionType, 
			Map(name, table->name, "ALLOC"));
	}
	/* Pack and Unpack Functions */
	if (table->localid == 0)
	{
		fprintf(outh, "NULL,NULL,");
	}
	else
	{
		fprintf(outh, "(%s) %s,", FunctionType,
			Map(name, itoa(table->msgno), "PACK"));
		fprintf(outh, "(%s) %s,",
			FunctionType, Map(name, itoa(table->msgno), "UNPACK"));
	}
	fprintf(outh, "sizeof(%s)); \n", table->name);
	dontfree(temp);
}

void
CreateOwnChareInitFunction(table)
SYMTABPTR       table;
{
	char           *temp;

	if (table == NULL)
		return;

	CreateOwnChareInitFunction(table->left);
	CreateOwnChareInitFunction(table->right);

	if ((table->idtype != CHARENAME) && (table->idtype != BOCNAME))
		return;

	temp = MyModulePrefix(table->modname->name, table->name);
	fprintf(outh, "  %s = ", temp);
	fprintf(outh, "registerChare(");
	fprintf(outh, "\"%s\",", table->name);
	fprintf(outh, "sizeof(%s_Data),",table->name);
	fprintf(outh, "(%s) NULL);\n",FunctionType);
	dontfree(temp);
}

void
CreateOwnEpInitFunction(table, module)
SYMTABPTR       table;
char           *module;
{
	char           *temp, *temp2;
	SYMTABPTR	dummy;
	char *eptype;

	if (table == NULL)
		return;

	CreateOwnEpInitFunction(table->left, module);
	CreateOwnEpInitFunction(table->right, module);

	outh = outh2;

	if((table->idtype == BOCNAME) || (table->idtype==CHARENAME))
	{
		CreateOwnEpInitFunction(table->type->table, module);
		return;
	}

	if(table->idtype != ENTRYNAME)
		return;

	if (table->charename->idtype == BOCNAME)
		eptype = "registerBocEp";
	else
		eptype = "registerEp";

	temp = MyModuleCharePrefix(module,
		       table->charename->name, table->name);
	fprintf(outh, "%s = ", temp);
	fprintf(outh, "%s(", eptype);
	temp = Map(module, table->charename->name, table->name);
	fprintf(outh, "\"%s\",", temp);
	fprintf(outh, "(%s) %s,", FunctionType, temp);
	fprintf(outh, "CHARM,");
	/* put msg index here */
	dummy = (SYMTABPTR) table->type;
	if (dummy)
	{
		if(!strcmp(module, dummy->modname->name))
			temp2 = MyModulePrefix(module,dummy->name);
		else
			temp2 = ModulePrefix(dummy->modname->name, dummy->name);
	}
	else
		temp2 = "0";
	fprintf(outh, "%s,", temp2);
	/* put chare index here */
	temp = MyModulePrefix(table->modname->name, table->charename->name);
	fprintf(outh, "%s);\n",temp);
	if(!dummy)
	{
		fprintf(outh, "\n /* Register Main Chare */\n");
		fprintf(outh, "  registerMainChare(");
		fprintf(outh, "%s,",temp);
		temp = MyModuleCharePrefix(module,
		       		table->charename->name, table->name);
		fprintf(outh, "%s,CHARM);\n",temp);
	}
	dontfree(temp);
}

void
CreateOwnFuncInitFunction(table, module)
SYMTABPTR       table;
char           *module;
{
	char           *temp, *temp2;

	if (table == NULL)
		return;

	CreateOwnFuncInitFunction(table->left, module);
	CreateOwnFuncInitFunction(table->right, module);

	outh = outh2;

	if((table->idtype == BOCNAME) || (table->idtype==CHARENAME))
	{
		CreateOwnFuncInitFunction(table->type->table, module);
		return;
	}
	
	if((table->idtype != FNNAME) && (table->idtype!=PUBLICFNNAME))
		return;
	if(table->idtype == FNNAME)
	{
		temp = MyModulePrefix(module, table->name);
		fprintf(outh, "  %s%s = registerFunction(", temp, REFSUFFIX);
		fprintf(outh, "(%s) %s);\n", FunctionType, table->name);
		fprintf(outh, "  %s = (%s) %s;\n", temp, FunctionType,
			table->name);
		dontfree(temp);
	}
	else
	{
		temp = Map(module, table->charename->name, table->name);
		temp2 = MyModuleCharePrefix(module, table->charename->name
					   ,table->name);
		fprintf(outh, "  %s = registerFunction((%s) %s);\n", temp2,
			FunctionType, temp);
		dontfree(temp2);
	}
}

FillOwnCopyFromBuffer(table, name)
SYMTABPTR       table;
char           *name;
{
	char           *temp;
	if (table == NULL)
		return;
	FillOwnCopyFromBuffer(table->left, name);
	FillOwnCopyFromBuffer(table->right, name);

	if (table->idtype == READONLYVAR)
		fprintf(outh, "  %s(&%s,sizeof(%s));\n",
			CkCopyFromBuffer, AppendMap(name, table->name), 
			AppendMap(name, table->name));
	if (table->idtype == READONLYARRAY)
		fprintf(outh, "  %s(%s,sizeof(%s));\n",
			CkCopyFromBuffer, AppendMap(name, table->name), 
			AppendMap(name, table->name));
	if (table->idtype == READONLYMSG)
	{
		temp = MakeString(AppendMapIndex(name, table->name));
		fprintf(outh, "  temp = (void **)&(%s);\n", 
			AppendMap(name, table->name));
		fprintf(outh, "  *temp = _CK_ReadMsgTable[%s];\n", temp);
		dontfree(temp);
	}
}

FillOwnCopyToBuffer(table, name)
SYMTABPTR       table;
char           *name;
{
	if (table == NULL)
		return;
	FillOwnCopyToBuffer(table->left, name);
	FillOwnCopyToBuffer(table->right, name);

	if (table->idtype == READONLYVAR)
		fprintf(outh, "  %s(&%s,sizeof(%s));\n",
			CkCopyToBuffer, AppendMap(name, table->name), 
			AppendMap(name, table->name));
	if (table->idtype == READONLYARRAY)
		fprintf(outh, "  %s(%s,sizeof(%s));\n",
			CkCopyToBuffer, AppendMap(name, table->name), 
			AppendMap(name, table->name));
}

CreateOwnCopyFromBuffer()
{
	outh = outh2;

	fprintf(outh, "%s%s_CopyFromBuffer(_CK_ReadMsgTable)\n",
			CkPrefix,
			ModuleDefined->name);
	fprintf(outh, "void **_CK_ReadMsgTable;\n{\n");
	fprintf(outh, "  void **temp;\n\n");
	FillOwnCopyFromBuffer(ModuleDefined->type->table, ModuleDefined->name);
	fprintf(outh, "}\n\n");
}

CreateOwnCopyToBuffer()
{
	outh = outh2;

	fprintf(outh, "%s%s_CopyToBuffer()\n{\n",CkPrefix, ModuleDefined->name);
	FillOwnCopyToBuffer(ModuleDefined->type->table, ModuleDefined->name);
	fprintf(outh, "}\n\n");
}

FillOwnBufferSize(table, name)
SYMTABPTR       table;
char           *name;
{
	if (table == NULL)
		return;
	FillOwnBufferSize(table->left, name);
	FillOwnBufferSize(table->right, name);

	if ((table->idtype == READONLYVAR) || (table->idtype == READONLYARRAY))
		fprintf(outh, "  count += sizeof(%s);\n", 
			AppendMap(name, table->name));

	if (table->idtype == READONLYMSG)
	{
		fprintf(outh, "  %s = registerReadOnlyMsg();\n", 
			AppendMapIndex(name, table->name));
		fprintf(outh1, "static int %s;\n", AppendMapIndex(name, table->name));
	}
}

OwnPseudoTableFill(table)
SYMTABPTR       table;
{
	char           *temp;
	if (table == NULL)
		return;
	OwnPseudoTableFill(table->left);
	OwnPseudoTableFill(table->right);

	switch(table->idtype)
	{
	case ACCNAME:
		temp = MyModulePrefix(table->modname->name, table->name);
		fprintf(outh, "  %s = registerAccumulator(\"%s\",",temp,
				table->name);
		fprintf(outh, "(%s)%s,",FunctionType,
		    Map(table->modname->name, itoa(table->msgno), "INIT"));
		fprintf(outh, "(%s)%s,",FunctionType, 
		    Map(table->modname->name, itoa(table->msgno), "INCREMENT"));
		fprintf(outh, "(%s)%s,",FunctionType, 
		    Map(table->modname->name, itoa(table->msgno), "COMBINE"));
		fprintf(outh, "CHARM);\n");
		dontfree(temp);
		break;
	case MONONAME:
		temp = MyModulePrefix(table->modname->name, table->name);
		fprintf(outh, "  %s = registerMonotonic(\"%s\",",temp,
				table->name);
		fprintf(outh, "(%s)%s,", FunctionType,
			Map(table->modname->name, itoa(table->msgno), "INIT"));
		fprintf(outh, "(%s)%s,",FunctionType, 
			Map(table->modname->name,itoa(table->msgno), "UPDATE"));
		fprintf(outh, "CHARM);\n");
		dontfree(temp);
		break;
	case TABLENAME:
		temp = AppendMap(CurrentModule->name, table->name);
		fprintf(outh, "  %s = registerTable(", temp);
		fprintf(outh, "\"%s\",(%s)NULL, (%s)", table->name, 
			FunctionType, FunctionType);
		if(table->msgno)
			fprintf(outh, "%s);\n",
				Map(table->modname->name, itoa(table->msgno), 
					"HASH"));
		else
			fprintf(outh, "NULL);\n");
		dontfree(temp);
		break;
	default:
		break;
	}
}

void
CreateOwnModuleInitFunction()
{
	outh = outh2;
	fprintf(outh,
		"char *_CK_%s_id=\"\\0charmc autoinit %s\";\n",
		ModuleDefined->name, ModuleDefined->name);
	fprintf(outh, "%s%s_init()\n",CkPrefix,ModuleDefined->name);
	fprintf(outh, "{\n  static int visited=0;\n  int count;\n\n");
	fprintf(outh, "  if (visited) return; else visited=1;\n");

	fprintf(outh, "\n  /*Register Messages*/\n");
	CreateOwnMsgInitFunction(ModuleDefined->type->table, 
			ModuleDefined->name);

	fprintf(outh, "\n  /*Register Chares*/\n");
	CreateOwnChareInitFunction(ModuleDefined->type->table);

	fprintf(outh, "\n  /*Register EntryPoints*/\n");
	CreateOwnEpInitFunction(ModuleDefined->type->table, 
				ModuleDefined->name);

	fprintf(outh, "\n  /*Register Functions*/\n");
	CreateOwnFuncInitFunction(ModuleDefined->type->table, 
				ModuleDefined->name);

	fprintf(outh, "\n  /*Register Monotonics, Tables and Accumulators*/\n");
	OwnPseudoTableFill(ModuleDefined->type->table);

	fprintf(outh, "\n  /*Register Read Only Var & Msg*/\n");
	fprintf(outh, "  count = 0;\n");
	FillOwnBufferSize(ModuleDefined->type->table, ModuleDefined->name);
	fprintf(outh, "  registerReadOnly(count,");
	fprintf(outh, "(%s)%s%s_CopyFromBuffer,", FunctionType,CkPrefix,
			ModuleDefined->name);
	fprintf(outh, "(%s)%s%s_CopyToBuffer);\n", FunctionType,CkPrefix,
			ModuleDefined->name);

	fprintf(outh, "\n  /*Call own import struct Init*/\n");
	fprintf(outh, "  %s%s_struct_init();\n", CkPrefix,
		ModuleDefined->name);

	fprintf(outh, "\n  /*Call Other module Inits*/\n");
	CallOtherModuleInits(StackBase->tableptr);

	fprintf(outh, "\n}\n\n");
}

/* end addition --milind */

void 
CreateImportModuleComponentFill(table, name, modcomponent)
char           *name;
SYMTABPTR       table;
int             modcomponent;
{
	char           *temp1, *temp2;
	if (table == NULL)
		return;
	CreateImportModuleComponentFill(table->left, name, modcomponent);
	CreateImportModuleComponentFill(table->right, name, modcomponent);

	if (table->idtype == modcomponent)
	{
                temp1 = ModulePrefix(name, table->name);
		temp2 = MyModulePrefix(name, table->name);
		fprintf(outh, " memcpy((void*)&%s,(void*)&%s,sizeof(%s));\n",
			temp1, temp2, temp2);
		dontfree(temp1);
		dontfree(temp2);
	}
}

void 
GenerateOuth()
{
	CreateStructures(StackBase->tableptr);
	CreatePUAFunctions(ModuleDefined->type->table);
	CreateImportInitFunction();
	/* addition by milind */
	CreateOwnImportInitFunction();
	CreateOwnCopyFromBuffer();
	CreateOwnCopyToBuffer();
	CreateOwnModuleInitFunction();
	/* end addition -- milind */
}

/*****************************************************************************/
/* Generating Alloc, Pack and Unpack Functions 		     */
/*****************************************************************************/

CreatePUAFunctions(table)
SYMTABPTR       table;
{
	if (table == NULL)
		return;
	CreatePUAFunctions(table->left);
	CreatePUAFunctions(table->right);

	if (!((table->idtype == MESSAGENAME) && (table->localid > 0)))
		return;

	CreateAllocFunction(table);
	if (table->userpack == 0)
	{
		CreatePackFunction(table);
		CreateUnPackFunction(table);
	}
}

CreatePackFunction(table)
SYMTABPTR       table;
{
	SYMTABPTR       sym, node;
	char           *temp;
	outh = outh2;
	fprintf(outh, "static %s(in,out,len)\n", Map(table->modname->name, itoa(table->msgno), "PACK"));
	fprintf(outh, "%s *in,**out;\nint *len;\n{ *len = 0;\n", table->name);
	fprintf(outh, "  *out = in;\n");
	sym = table->type->table->next;
	while (sym != table->type->table)
	{
		if (sym->idtype == VARSIZENAME)
		{
			node = (SYMTABPTR) sym->type;
			if ((node->idtype != SYSTEMTYPENAME) && (strcmp(node->modname->name,
						      ModuleDefined->name)))
				temp = Map(node->modname->name, "0", node->name);
			else
				temp = node->name;
			fprintf(outh, "  in->%s = (%s *) ((char *)in->%s - ((char *)&(in->%s)));\n",
				sym->name, temp, sym->name, sym->name);
		}
		sym = sym->next;
	}
	fprintf(outh, "}\n\n");
}

CreateUnPackFunction(table)
SYMTABPTR       table;
{
	SYMTABPTR       sym, node;
	char           *temp;
	outh = outh2;
	fprintf(outh, "static %s(in,out)\n", Map(table->modname->name, itoa(table->msgno), "UNPACK"));
	fprintf(outh, "%s *in,**out;\n{ *out = in;\n", table->name);
	sym = table->type->table->next;
	while (sym != table->type->table)
	{
		if (sym->idtype == VARSIZENAME)
		{
			node = (SYMTABPTR) sym->type;
			if ((node->idtype != SYSTEMTYPENAME) && (strcmp(node->modname->name,
						      ModuleDefined->name)))
				temp = Map(node->modname->name, "0", node->name);
			else
				temp = node->name;
			fprintf(outh, "  in->%s = (%s *) ((char *)(&(in->%s)) + (int)in->%s);\n",
				sym->name, temp, sym->name, sym->name);
		}
		sym = sym->next;
	}
	fprintf(outh, "}\n\n");
}

CreateAllocFunction(table)
SYMTABPTR       table;
{
	SYMTABPTR       sym, node;
	int             count = 0;
	char           *name;
	outh = outh2;
	fprintf(outh, "static void *%s(msgno,size,prio,array)\nint msgno,size,prio,array[];\n",
		Map(table->modname->name, table->name, "ALLOC"));
	fprintf(outh, "{ int totsize=0;\n  int temp,dummy,sarray[%d];\n", table->localid);
	fprintf(outh, "  %s *ptr;\n", table->name);
	fprintf(outh, "\n  totsize = temp = (size%%_CK_VARSIZE_UNIT)?_CK_VARSIZE_UNIT*((size+_CK_VARSIZE_UNIT)/_CK_VARSIZE_UNIT):size;\n");
	sym = table->type->table->next;
	while (sym != table->type->table)
	{
		if (sym->idtype == VARSIZENAME)
		{
			node = (SYMTABPTR) sym->type;
			if ((node->idtype != SYSTEMTYPENAME) && (strcmp(node->modname->name,
						      ModuleDefined->name)))
				name = Map(node->modname->name, "0", node->name);
			else
				name = node->name;
			fprintf(outh, "  size = sizeof(%s)*array[%d];\n", name, count);
			fprintf(outh, "\n  dummy = (size%%_CK_VARSIZE_UNIT)?_CK_VARSIZE_UNIT*((size+_CK_VARSIZE_UNIT)/_CK_VARSIZE_UNIT):size;\n");
			fprintf(outh, "  sarray[%d]=dummy;\n", count++);
			fprintf(outh, "  totsize += dummy;\n");
		}
		sym = sym->next;
	}
	fprintf(outh, "  ptr = (%s *)%s(msgno,totsize,prio);\n", table->name, CkGenericAlloc);
	sym = table->type->table->next;
	count = 0;
	fprintf(outh, "\n  dummy=temp;\n");
	while (sym != table->type->table)
	{
		if (sym->idtype == VARSIZENAME)
		{
			node = (SYMTABPTR) sym->type;
			if ((node->idtype != SYSTEMTYPENAME) && (strcmp(node->modname->name,
						      ModuleDefined->name)))
				name = Map(node->modname->name, "0", node->name);
			else
				name = node->name;
			fprintf(outh, "  ptr->%s = (%s *)((char *)ptr + dummy);\n",
				sym->name, name);
			fprintf(outh, "  dummy += sarray[%d];\n", count++);
		}
		sym = sym->next;
	}

	fprintf(outh, "  return((void *)ptr);\n");
	fprintf(outh, "}\n\n");
}

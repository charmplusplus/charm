extern StackStruct *StackTop ;
extern StackStruct *GlobalStack ;

extern AggState *PermanentAggTable[] ;
extern int PermanentAggTableSize ;

extern ChareInfo * ChareTable[] ;
extern ChareInfo * BOCTable[] ;
extern MsgStruct MessageTable[] ;
extern AccStruct * AccTable[] ;
extern AccStruct * MonoTable[] ;
extern char * DTableTable[] ;
extern char * ReadTable[] ;
extern char * ReadMsgTable[] ;
extern FunctionStruct FunctionTable[] ;
extern SymEntry SymTable[] ;

extern int charecount ;
extern int boccount ;
extern int TotalEntries ;
extern int TotalMsgs ;
extern int TotalAccs ;
extern int TotalMonos ;
extern int TotalDTables ;
extern int TotalReads ;
extern int TotalReadMsgs ;
extern int TotalFns ;
extern int TotalSyms ;

extern HandleEntry ChareHandleTable[] ;
extern HandleEntry BOCHandleTable[] ;
extern HandleEntry AccHandleTable[] ;
extern HandleEntry MonoHandleTable[] ;
extern HandleEntry WrOnHandleTable[] ;

extern int ChareHandleTableSize ;
extern int BOCHandleTableSize ;
extern int AccHandleTableSize ;
extern int MonoHandleTableSize ;
extern int WrOnHandleTableSize ;

extern int CurrentLine ;
extern int CurrentScope ;  /* 1 means file scope, >1 means inside a block*/
extern char CurrentFileName[] ;
extern int CurrentAccess, CurrentAggType, CurrentStorage;
extern int CurrentCharmType ;
extern int CurrentCharmNameIndex ;
extern char CurrentTypedef[] ;
extern char CurrentDeclType[] ;
extern char CurrentAggName[] ;
extern char CurrentChare[] ;
extern char CurrentEP[] ;
extern char CurrentFn[] ;
extern char CurrentMsgParm[] ;
extern char CurrentSharedHandle[] ;
extern AccStruct *CurrentAcc ;
extern ChareInfo *CurrentCharePtr ;
extern char *EpMsg;
extern char CurrentAsterisk[] ;
extern char SendEP[] ;
extern char SendChare[] ;
extern char SendPe[] ;
extern char *ParentArray[] ;
extern int SendType ;
extern int main_argc_argv ;
extern int foundargs ;
extern int numparents ;
extern int SendMsgBranchPoss ;
extern int FoundHandle ;
extern int FilledAccMsg ;
extern int FoundConstructorBody ;
extern int IsMonoCall ;
extern int FoundParms ;
extern int FoundLocalBranch ;
extern int FoundTable ;
extern int FoundVarSize ;
extern int FoundReadOnly ;
extern int StructScope ;

extern FILE *yyin ;
extern FILE *outfile ;
extern FILE *headerfile ;
extern FILE *graphfile ;

extern int MakeGraph ;

extern char prevtoken[] ;
/* extern char modname[] ;  */
extern char OutBuf[] ;
extern char CoreName[] ;

extern char * EpMsg ;

extern int main_argc_argv ;
extern int shouldprint ;
extern int foundargs ;

extern int ErrVal ;
extern int AddedScope ;
extern int FoundGlobalScope ;
extern int wchar_is_predefined;
extern int ptrdiff_is_predefined;
extern int InsideChareCode ;

extern int NewOpType ;
extern int FoundDeclarator ; 


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
 * Revision 2.1  1995-09-06 04:21:31  sanjeev
 * new Charm++ syntax, CHARE_BLOCK changes
 *
 * Revision 2.0  1995/06/05  19:01:24  brunner
 * Reorganized directory structure
 *
 * Revision 1.3  1995/03/23  05:11:53  sanjeev
 * changes for printing call graph
 *
 * Revision 1.2  1994/11/11  05:32:43  brunner
 * Removed ident added by accident with RCS header
 *
 * Revision 1.1  1994/11/07  15:41:10  brunner
 * Initial revision
 *
 ***************************************************************************/

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

extern int InsideChareCode ;

extern int NewOpType ;
extern int FoundDeclarator ; 


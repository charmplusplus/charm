#define YYSTYPE char *
#include "xp-t.tab.h"
#include "xp-lexer.h"
#include "xp-extn.h"


char *CheckSendError() ;
char *Mystrstr() ;
char *Strstr() ;
EP *SearchEPList() ;

ProcessEP(epname,defined)
char *epname ;
int defined ;
{
/* called after the header of an EP has been parsed */	

	int val ;
	EP *ep, *e, *eprev ;
	ChareInfo *chare ;

	strcpy(CurrentEP,epname) ;
	TotalEntries++ ;


	if ( strcmp(CurrentChare,"main")==0 && strcmp(epname,"main")==0 ) {
		if ( foundargs )
			main_argc_argv = TRUE ;
		else
			main_argc_argv = FALSE ;
	}

/* Enter EP into Chare / BOC table */
        if ( CurrentAggType == CHARE )
                chare = ChareTable[FoundInChareTable(ChareTable,charecount+1,CurrentChare)] ;
        else if ( CurrentAggType == BRANCHED )
                chare = BOCTable[FoundInChareTable(BOCTable,boccount+1,CurrentChare)] ;

	/* check if the EP is already there */
	for ( e=chare->eps; e!=NULL; e=e->next ) {
		if ( strcmp(epname,e->epname)==0 ) {
			e->defined = defined ;
			return ;
		}
	}


        ep = (EP *)malloc(sizeof(EP)) ;
        strcpy(ep->epname,epname) ;
	ep->chare = chare ;
        ep->inherited = 0 ;
        ep->defined = defined ;
	if ( CurrentStorage == VIRTUAL )
		ep->isvirtual = TRUE ;
	else
		ep->isvirtual = FALSE ;
	if ( chare->eps == NULL ) {
		chare->eps = ep ;
		ep->next = NULL ;
	}
	else { /* insert in lexicographic order */
		InsertAlpha(chare,ep) ;
	}
		
	
/* Find message name in this ep */
	if ( strcmp(CurrentChare,"main")==0 && strcmp(CurrentEP,"main")==0 )
	{/*	fprintf(stderr,"Got main::%s\n",ep->epname) ;  */
		;
	}
	else {
		strcpy(ep->msgname,EpMsg) ;
	}
}


ProcessFn(fnname)
char *fnname ;
{
/* called after the header of an EP has been parsed */	

	FN *fn, *f ;
	ChareInfo *chare ;

	strcpy(CurrentEP,fnname) ;

/* Enter EP into Chare / BOC table */
        if ( CurrentAggType == CHARE )
                chare = ChareTable[FoundInChareTable(ChareTable,charecount+1,CurrentChare)] ;
        else if ( CurrentAggType == BRANCHED )
                chare = BOCTable[FoundInChareTable(BOCTable,boccount+1,CurrentChare)] ;

	/* check if the FN is already there */
	for ( f=chare->fns; f!=NULL; f=f->next ) {
		if ( strcmp(fnname,f->fnname)==0 ) 
			return ;
	}

        fn = (FN *)malloc(sizeof(FN)) ;
        strcpy(fn->fnname,fnname) ;
	fn->next = chare->fns ;
	chare->fns = fn ;
}






InsertSymTable(string)
char *string ;
{
	if ( isaTYPE(string) ) {
	/* Not necessarily a redefinition : the first could have been
	   just a declaration. A real redef will be reported by C++ compiler */
  	/* Also possiblility : typedefs with same name (class X and enum X) */
		return ;
	}

	if ( TotalSyms >= MAXSYMBOLS ) {
		printf("INTERNAL ERROR : Symbol Table overflow : Cannot handle more than %d typenames\n",MAXSYMBOLS ) ;
		printf("TO FIX ERROR : Increase MAXSYMBOLS variable in translator/t.h\n") ;
	}

	SymTable[TotalSyms].name = (char *)malloc((strlen(string)+1)*sizeof(char)) ;
	strcpy(SymTable[TotalSyms].name,string) ;
	SymTable[TotalSyms].permanentindex = -1 ;

	TotalSyms++ ;
}



int FoundInChareTable(table,tablelen,name)
ChareInfo *table[] ;
int tablelen ;
char *name ;
{
	int i ;

	for ( i=tablelen-1; i>=0; i-- ) 
		if ( strcmp(table[i]->name,name) == 0 )
			return(i) ;

	return(-1) ;
}	


int FoundInAccTable(table,tablelen,name)
AccStruct *table[] ;
int tablelen ;
char *name ;
{
	int i ;
	char *nptr = strstr(name,"::") ;

	if ( nptr == NULL )
		nptr = name ;
	else
		nptr += 2 ;

	for ( i=tablelen-1; i>=0; i-- ) 
		if ( strcmp(table[i]->name,nptr) == 0 )
			return(i) ;

	return(-1) ;
}	

int FoundInMsgTable(name)
char *name ;
{
	int i ;

	for ( i=TotalMsgs-1; i>=0; i-- ) 
		if ( strcmp(MessageTable[i].name,name) == 0 )
			return(i) ;

	return(-1) ;
}	


int CheckCharmName()
{
/* Check if CurrentTypedef is CHARM type, for use by the 
   possibly upcoming handle decl */
	int i, ind ;
	int ScopedType=FALSE ;
	char *lastagg ;
	char *type ;
	char *sptr ;

	if ( strcmp(CurrentTypedef,"")==0 )
		return 0 ;

/*	FLUSHBUF() ;cant flush here because we want to remove 
	the CurrentTypedef */
	
	sptr = Mystrstr(OutBuf,CurrentTypedef) ;
	if ( sptr != NULL ) {	/* This will happen for 1st var in the list */
		*sptr = '\0' ;
		type = CurrentTypedef ;
	}
	else {
		/* fprintf(stderr,"TRANSLATOR ERROR in handle decl: %s, line %d: \n",CurrentFileName,CurrentLine) ;   */
		/* It is possible that CurrentTypedef is not found in OutBuf
		   because CurrentTypedef is not set properly always */
		return 0 ;
	}



	FLUSHBUF() ;

	if ( Strstr(type,"::") != NULL ) {
		AddScope(type) ;
		ScopedType = TRUE ;
		lastagg = Mystrstr(type,"::") + 2 ;
	}	
	else
		lastagg = type ;


	if ( (ind=FoundInChareTable(ChareTable,charecount+1,lastagg)) != -1 ) {
		CurrentCharmType = CHARE ;
		fprintf(outfile,"ChareIDType") ;
	}
	else if ( (ind=FoundInChareTable(BOCTable,boccount+1,lastagg)) != -1 ){
		CurrentCharmType = BRANCHED ;
		fprintf(outfile,"int") ;
	}
	else if ( (ind=FoundInAccTable(AccTable,TotalAccs,lastagg)) != -1 ) {
		CurrentCharmType = ACCUMULATOR ;
		fprintf(outfile,"AccIDType") ;
	}
	else if ( (ind=FoundInAccTable(MonoTable,TotalMonos,lastagg)) != -1 ) {
		CurrentCharmType = MONOTONIC ;
		fprintf(outfile,"MonoIDType") ;
	}
	else if ( strcmp(lastagg,"writeonce")==0 ) {
		CurrentCharmType = WRITEONCE ;
		fprintf(outfile,"WriteOnceID") ;
		ind = -1 ;
	}
	else {
		CurrentCharmType = -1 ;
		fprintf(outfile,"%s ",type) ; /* put type back into output */
		ind = -1 ;
	}

	if ( ind != -1 ) {
		if ( !ScopedType )
			CurrentCharmNameIndex = ind ;
		else { /* find the PermanentAggTable index of lastagg */
			for ( i=TotalSyms-1; i>=0; i-- )
				if ( strcmp(SymTable[i].name,lastagg)==0 )
					break ;
			if ( i==-1 )
				fprintf(stderr,"TRANSLATOR ERROR: %s, line %d : class/struct not found in symbol table\n",CurrentFileName,CurrentLine) ;
			CurrentCharmNameIndex = BASE_PERM_INDEX + SymTable[i].permanentindex ;
			/* > BASE_PERM_INDEX means it is a permanentindex*/
		}
	}
	else
		CurrentCharmNameIndex = -1 ;

	strcpy(prevtoken,"") ;


	if ( ScopedType )
		RemoveScope(type) ;

	if ( CurrentCharmType == -1 )
		return 0 ;
	else
		return 1 ;
}

			
SyntaxError(string)
char *string ;
{
	if ( strcmp(string,"") != 0 )
		fprintf(stderr," in %s.\n",string);
	else 
		fprintf(stderr,".\n") ;
	ErrVal = TRUE ;
}

CharmError(string)
{
	fprintf(stderr,"ERROR : %s, line %d : %s.\n",CurrentFileName,CurrentLine,string) ;
	ErrVal = TRUE ;
}


InsertHandleTable(table,size,id)
HandleEntry table[] ;
int *size ;
char *id ;
{
	table[*size].name = (char *)malloc(sizeof(char)*strlen(id)+1) ;
	strcpy(table[*size].name,id) ;
	table[*size].typestr = (char *)malloc((strlen(CurrentAsterisk)+1)*sizeof(char)+1) ;
	strcpy(table[*size].typestr, CurrentAsterisk) ;
	(*size)++ ;
}


SearchHandleTable(table,size,name)
HandleEntry table[] ;
int size ;
char *name ;
{
	int i ;

	for ( i=size-1; i>=0; i-- ) {
		if ( strcmp(table[i].name,name) == 0 ) 
			return(i) ;
	}
	return(-1) ;
}

EP *SearchEPList(eplist,ep)
EP *eplist ;
char *ep ;
{
	EP *anep ;

	for ( anep=eplist; anep!=NULL; anep=anep->next ) 
		if ( strcmp(anep->epname,ep) == 0 )
			return(anep) ;
	return(NULL) ;
}


char *CheckSendError(SendChare,SendEP,Msg,SendType,charename)
char *SendChare, *SendEP, *Msg ;
int SendType ;
char **charename ;
{
/* This fn checks (not very accurately right now) whether SendEP is a valid
   entry point of the handle SendChare, and fills the type of SendChare
   in charename */

	int ind, len ;
	char errstr[256] ;
	EP *ep, *e ;
	ChareInfo *chare ;
	char *ident, *lastcoln, *scopestr ;

	if ( strcmp(SendChare,"thishandle")==0 || strcmp(SendChare,"thisgroup")==0 ) {
		*charename = (char *)malloc(sizeof(char)*(strlen(CurrentChare)+1));
		strcpy(*charename,CurrentChare) ;
		scopestr = (char *)malloc(sizeof(char)*2) ;
		strcpy(scopestr," ") ;
		return scopestr ;
	}
	else if ( strcmp(SendChare,"mainhandle")==0 ) {
		*charename = (char *)malloc(sizeof(char)*5);
		strcpy(*charename,"main") ;
		scopestr = (char *)malloc(sizeof(char)*2) ;
		strcpy(scopestr," ") ;
		return scopestr ;
	}

	if ( (ident=Mystrstr(SendChare,"::")) != NULL ){ 
		/* handle itself is elsewhere */
		AddScope(SendChare) ;
		ident += 2 ;  /* so it points to the handle identifier */
	}
	else
		ident = SendChare ;
		

	if ( SendType == SIMPLE ) {
		ind = SearchHandleTable(ChareHandleTable,ChareHandleTableSize,ident) ;
		if ( ind == -1 ) {
			sprintf(errstr,"%s is not a charm.handle or is a complex expression",ident) ;
			CharmError(errstr) ;
			scopestr = NULL ;
			*charename = NULL ;
			goto endfn ;
		}

		len = strlen(ChareHandleTable[ind].typestr) ;
		scopestr = (char *)malloc(sizeof(char)*(len+1)) ;
		strcpy(scopestr,ChareHandleTable[ind].typestr) ;

		lastcoln = Mystrstr(scopestr,"::") ;
		if ( lastcoln != NULL ) {
			*lastcoln = '\0' ;
			*charename = lastcoln + 2 ;
			goto endfn ;
		}

		*charename = scopestr ;	
		scopestr = (char *)malloc(sizeof(char)*2) ;
		strcpy(scopestr," ") ;

		/* Search for charename in ChareTable or BOCTable, since
		   the handle could be a normal chare handle or a branch-id
		   of a BOC branch */

		ind = FoundInChareTable(ChareTable,charecount+1,*charename) ;
		if ( ind != -1 ) 
			chare = ChareTable[ind] ;
		else {
			ind = FoundInChareTable(BOCTable,boccount+1,*charename) ;
			if ( ind != -1 ) 
				chare = BOCTable[ind] ;
			else
				goto endfn ;
		}
		for ( e=chare->eps; e!=NULL; e=e->next ) {
			if ( strcmp(e->epname,SendEP) == 0 ) {
				break ;
			}
		}
		if ( e == NULL ) /* didnt find EP, so it is EntryPointType */
			*charename = NULL ;

	}
	else if ( SendType == BRANCH || SendType == BROADCAST ) {
		ind = SearchHandleTable(BOCHandleTable,BOCHandleTableSize,ident) ;
		if ( ind == -1 ) {
			sprintf(errstr,"%s is not a Branched Chare handle or is a complex expression",ident) ;
			CharmError(errstr) ;
			scopestr = NULL ;
			goto endfn ;
		}

		len = strlen(BOCHandleTable[ind].typestr) ;
		scopestr = (char *)malloc(sizeof(char)*(len+1)) ;
		strcpy(scopestr,BOCHandleTable[ind].typestr) ;

		lastcoln = Mystrstr(scopestr,"::") ;
		if ( lastcoln != NULL ) {
			*lastcoln = '\0' ;
			*charename = lastcoln + 2 ;
			goto endfn ;
		}

		*charename = scopestr ;	
		scopestr = (char *)malloc(sizeof(char)*2) ;
		strcpy(scopestr," ") ;

		/* Search for charename in BOCTable */
		ind = FoundInChareTable(BOCTable,boccount+1,*charename) ;
		if ( ind == -1 ) 
			goto endfn ;
		for ( e=BOCTable[ind]->eps; e!=NULL; e=e->next ) {
			if ( strcmp(e->epname,SendEP) == 0 ) {
				break ;
			}
		}
		if ( e == NULL ) /* didnt find EP, so it is EntryPointType */
			*charename = NULL ;
	}

endfn:  if ( Strstr(SendChare,"::") != NULL ) 
		RemoveScope(SendChare) ;

	return(scopestr) ;
}




OutputSend(SendChare, SendEP, msg, SendType, charename, scopestr, SendPe)
char *SendChare; 
char *SendEP; 
char *msg; 
int SendType;
char *charename; 
char *scopestr; 
char *SendPe;
{
    if ( charename != NULL ) {
	if ( SendType == SIMPLE ) {
		  /* the cid is a handle=ChareIDType, so put &cid */
		fprintf(outfile,"SendMsg(%s_CK_ep_%s_%s,(void *)%s,&(%s)",
				scopestr,charename,SendEP,msg,SendChare) ;
		if ( MakeGraph )
			fprintf(graphfile,"SENDCHARE %s %s : %s %s\n", 
				CurrentChare, CurrentEP, charename, SendEP) ;
	}
	else if ( SendType == BRANCH ) {
		fprintf(outfile,"GeneralSendMsgBranch(%s_CK_ep_%s_%s,(void *)%s,%s,-1,(int)(%s)",scopestr,charename,SendEP,msg,SendPe,SendChare) ;
		if ( MakeGraph )
			fprintf(graphfile,"SENDBOC %s %s : %s %s %s\n", 
			CurrentChare, CurrentEP, charename, SendEP, SendPe) ;
	}
	else if ( SendType == BROADCAST ) {
		fprintf(outfile,"GeneralBroadcastMsgBranch(%s_CK_ep_%s_%s,(void *)%s,-1,(int)(%s)",scopestr,charename,SendEP,msg,SendChare) ;
		if ( MakeGraph )
			fprintf(graphfile,"BROADCASTBOC %s %s : %s %s\n", 
				CurrentChare, CurrentEP, charename, SendEP) ;
	}
	else 
		fprintf(stderr,"TRANSLATOR ERROR: SendType unknown\n");
    }
    else {
	/* We have an EntryPointType as the SendEP */
	if ( SendType == SIMPLE ) {
		  /* the cid is a handle=ChareIDType, so put &cid */
		fprintf(outfile,"SendMsg(%s,(void *)%s,&(%s)",SendEP,msg,
								SendChare) ;
	}
	else if ( SendType == BRANCH ){
		fprintf(outfile,"GeneralSendMsgBranch(%s,(void *)%s,%s,-1,(int)(%s)",
						SendEP,msg,SendPe,SendChare) ;
	}
	else if ( SendType == BROADCAST ){
		fprintf(outfile,"GeneralBroadcastMsgBranch(%s,(void *)%s,-1,(int)(%s)",SendEP,msg,SendChare) ;
	}
	else 
		fprintf(stderr,"TRANSLATOR ERROR: SendType unknown\n");
    }
}


InsertObjTable(name)
char *name ;
{
	int num, i ;	
	ChareInfo *chare ;
	char *mymsg ;
	char *myacc ;

	CurrentCharePtr = NULL ;
	strcpy(CurrentChare,"_CK_NOTACHARE") ; 

	if ( CurrentAggType == CHARE || CurrentAggType == BRANCHED )
	{	strcpy(CurrentChare,name) ;	
		if ( CurrentAggType == CHARE ) {
			if ((num=FoundInChareTable(ChareTable,charecount+1,
							CurrentChare))!=-1) {
				CurrentCharePtr = ChareTable[num] ;
				return ;
			}
			else
				num = ++charecount ;	
		}
		else {
			if ((num=FoundInChareTable(BOCTable,boccount+1,
							CurrentChare))!=-1) {
				CurrentCharePtr = BOCTable[num] ;
				return ;
			}
			else
				num = ++boccount ;
		}
		chare = (ChareInfo *) malloc(sizeof(ChareInfo)) ;
		strcpy(chare->name,CurrentChare) ;
		chare->eps = NULL ;
		chare->fns = NULL ;
		chare->parents = NULL ;
		if ( CurrentAggType == CHARE )
			ChareTable[num] = chare ;
		else /* CurrentAggType == BRANCHED */
			BOCTable[num] = chare ;
		CurrentCharePtr = chare ;
	}	
	else if ( CurrentAggType == MESSAGE ) {
		mymsg = (char *)malloc((strlen(name)+1)*sizeof(char)) ;
		strcpy(mymsg,name) ;
		MessageTable[TotalMsgs].name = mymsg ;
		MessageTable[TotalMsgs].pack = FALSE ;
		MessageTable[TotalMsgs].numvarsize = 0 ;
		MessageTable[TotalMsgs].varsizearray = NULL ;
		TotalMsgs++ ;
	}
	else if ( CurrentAggType == ACCUMULATOR ) { 
		/* this is an acc defn */
		CurrentAcc = AccTable[TotalAccs] = (AccStruct *)malloc(sizeof(AccStruct)) ;

		CurrentAcc->name = (char *)malloc((strlen(name)+1)*sizeof(char)) ;
		strcpy(CurrentAcc->name,name) ;
		TotalAccs++ ;
	}
	else if ( CurrentAggType == MONOTONIC ) { 
		/* this is a mono defn */
		CurrentAcc = MonoTable[TotalMonos] = (AccStruct *)malloc(sizeof(AccStruct)) ;

		CurrentAcc->name = (char *)malloc((strlen(name)+1)*sizeof(char)) ;
		strcpy(CurrentAcc->name,name) ;
		TotalMonos++ ;
	}
	else if ( CurrentAggType == READONLY ) { 
		/* this is a readonly defn */
		ReadTable[TotalReads] = (char *)malloc((strlen(name)+1)*sizeof(char)) ;
		strcpy(ReadTable[TotalReads],name) ;
		TotalReads++ ;
	}
	else if ( CurrentAggType == READMSG ) { 
		/* this is a readonly msg defn */
		ReadMsgTable[TotalReadMsgs] = (char *)malloc((strlen(name)+1)*sizeof(char)) ;
		strcpy(ReadMsgTable[TotalReadMsgs],name) ;
		TotalReadMsgs++ ;
	}
	else if ( CurrentAggType == DTABLE ) { 
		/* this is a table defn */
		DTableTable[TotalDTables] = (char *)malloc((strlen(name)+1)*sizeof(char)) ;
		strcpy(DTableTable[TotalDTables],name) ;
		TotalDTables++ ;
	}
} 



CheckSharedHandle(name)
char *name ;
{
	int ind ;
	char *sptr ;

	ind = SearchHandleTable(AccHandleTable,AccHandleTableSize,name) ;
        if ( ind == -1 ) {
		ind = SearchHandleTable(MonoHandleTable,MonoHandleTableSize,name) ;
		if ( ind == -1 ) {
			ind = SearchHandleTable(WrOnHandleTable,WrOnHandleTableSize,name) ;
			if ( ind == -1 ) 
				return ;
		}
	}

	strcpy(CurrentSharedHandle,name) ;
}


SetDefinedIfEp(str)
char *str ;
{
	char *col ;
	char chare[128], *ep ;
	ChareInfo *chptr = NULL ;
	int ch, bo ;
	EP *e ;
	FN *f ;
	int i ;

	col = Mystrstr(str,"::") ;
	if ( col == NULL )
		return ;

	/* sscanf(str,"%s::%s",chare,ep) ; */

	for ( i=0; str!=col; i++,str++ )
		chare[i] = *str ;
	chare[i] = '\0' ;
	ep = col + 2 ;

	ch = FoundInChareTable(ChareTable,charecount+1,chare) ;
	bo = FoundInChareTable(BOCTable,boccount+1,chare) ;

	if ( ch != -1 )
		chptr = ChareTable[ch] ;
	else if ( bo != -1 )
		chptr = BOCTable[bo] ;
	if ( chptr != NULL ) {
		for ( e=chptr->eps; e!=NULL; e=e->next ) {
			if ( strcmp(e->epname,ep) == 0 ) {
				e->defined = TRUE ;
				strcpy(CurrentChare,chptr->name) ;
				strcpy(CurrentEP,e->epname) ;
				CurrentCharePtr = chptr ;
				InsideChareCode = 1 ;
				return 1 ;
			}
		}
		for ( f=chptr->fns; f!=NULL; f=f->next ) {
			if ( strcmp(f->fnname,ep) == 0 ) {
				strcpy(CurrentChare,chptr->name) ;
				strcpy(CurrentEP,f->fnname) ;
				CurrentCharePtr = chptr ;
				InsideChareCode = 1 ;
				return 1 ;
			}
		}
	}
	return 0 ;
}



char *Mystrstr(big,small)
char *big, *small ;
{
/* The idea is to find the LAST position in big where small occurs ;
   the usual strstr gives the FIRST occurrence of small in big */
	
	char *last ;
	char * first = Strstr(big,small) ;

	while ( first != NULL ) {
		last = Strstr(first+1,small) ;
		if ( last == NULL )
			return(first) ;
		first = last ;
	}
	return(first) ;
}


MonoFN()
{
	return 0;
}



PushStack()
{
/* called when the prevtoken is a '{', from t.l  */
/* Also from AddScope */

	StackStruct *newtop = (StackStruct *)malloc(sizeof(StackStruct)) ;

	if ( StackTop == NULL )
		GlobalStack = newtop ;

	newtop->next = StackTop ;

	newtop->charecount = charecount ;
	newtop->boccount = boccount ;
	newtop->TotalMsgs = TotalMsgs ;
	newtop->TotalAccs = TotalAccs ;
	newtop->TotalMonos = TotalMonos ;
	newtop->TotalReads = TotalReads ;
	newtop->TotalReadMsgs = TotalReadMsgs ;
	newtop->TotalSyms = TotalSyms ;
	newtop->ChareHandleTableSize = ChareHandleTableSize ;
	newtop->BOCHandleTableSize = BOCHandleTableSize ;
	newtop->AccHandleTableSize = AccHandleTableSize ;
	newtop->MonoHandleTableSize = MonoHandleTableSize ;

	StackTop = newtop ;
}

PopStack()
{
/* called on when the prevtoken is a '}', from t.l  */
/* Also from RemoveScope */

	StackStruct *prevtop ;

	if ( StackTop == NULL ) {
		fprintf(stderr,"ERROR : %s, line %d : unmatched { and } braces.\n", CurrentFileName, CurrentLine) ;
		return ;
	}
		

	charecount = StackTop->charecount ;
	boccount = StackTop->boccount ;
	TotalMsgs = StackTop->TotalMsgs ;
	TotalAccs = StackTop->TotalAccs ;
	TotalMonos = StackTop->TotalMonos ;
	TotalReads = StackTop->TotalReads ;
	TotalReadMsgs = StackTop->TotalReadMsgs ;
	TotalSyms = StackTop->TotalSyms ;
	ChareHandleTableSize = StackTop->ChareHandleTableSize ;
	BOCHandleTableSize = StackTop->BOCHandleTableSize ;
	AccHandleTableSize = StackTop->AccHandleTableSize ;
	MonoHandleTableSize = StackTop->MonoHandleTableSize ;

	prevtop = StackTop ;
	StackTop = StackTop->next ;

	if ( StackTop == NULL )
		GlobalStack = NULL ;

	free(prevtop) ;
}




FillPermanentAggTable(name)
char *name ;
{
	int i ;
	AggState *n = PermanentAggTable[PermanentAggTableSize++] = (AggState *)
						     malloc(sizeof(AggState)) ;
	strcpy(n->name,name) ;
/* put the index of this agg into symbol table */
	for ( i=0; i<TotalSyms; i++ ) {
		if ( strcmp(SymTable[i].name,name) == 0 ) {
			SymTable[i].permanentindex = PermanentAggTableSize-1;
			break ;
		}
	}


/* First fill in all the Handle tables */

	if ( (n->ChareHandleTableSize = ChareHandleTableSize - StackTop->ChareHandleTableSize) != 0 ) {
		n->ChareHandleTable = (HandleEntry *)malloc(n->ChareHandleTableSize*sizeof(HandleEntry)) ;
		for ( i=0; i<n->ChareHandleTableSize; i++ )
			n->ChareHandleTable[i] = ChareHandleTable[i+StackTop->ChareHandleTableSize] ;
	}
	if ( (n->BOCHandleTableSize = BOCHandleTableSize - StackTop->BOCHandleTableSize) != 0 ) {
		n->BOCHandleTable = (HandleEntry *)malloc(n->BOCHandleTableSize*sizeof(HandleEntry)) ;
		for ( i=0; i<n->BOCHandleTableSize; i++ )
			n->BOCHandleTable[i] = BOCHandleTable[i+StackTop->BOCHandleTableSize] ;
	}
	if ( (n->AccHandleTableSize = AccHandleTableSize - StackTop->AccHandleTableSize) != 0 ) {
		n->AccHandleTable = (HandleEntry *)malloc(n->AccHandleTableSize*sizeof(HandleEntry)) ;
		for ( i=0; i<n->AccHandleTableSize; i++ )
			n->AccHandleTable[i] = AccHandleTable[i+StackTop->AccHandleTableSize] ;
	}
	if ( (n->MonoHandleTableSize = MonoHandleTableSize - StackTop->MonoHandleTableSize) != 0 ) {
		n->MonoHandleTable = (HandleEntry *)malloc(n->MonoHandleTableSize*sizeof(HandleEntry)) ;
		for ( i=0; i<n->MonoHandleTableSize; i++ )
			n->MonoHandleTable[i] = MonoHandleTable[i+StackTop->MonoHandleTableSize] ;
	}


/* Now fill in all the typeDEF tables */

	if ( TotalSyms == StackTop->TotalSyms ) {
		n->TotalSyms = 0 ;  /* No new typeDEFS introduced in this agg*/
		return ;
	}

	if ( (n->TotalSyms = TotalSyms - StackTop->TotalSyms) != 0 ) {
		n->SymTable = (SymEntry *)malloc(n->TotalSyms*sizeof(SymEntry)) ;
		for ( i=0; i<n->TotalSyms; i++ )
			n->SymTable[i] = SymTable[i+StackTop->TotalSyms] ;
	}
        if ( (n->charecount = charecount - StackTop->charecount) != 0){
		n->ChareTable = (ChareInfo **)malloc(n->charecount*sizeof(ChareInfo *)) ;
		for ( i=1; i<=n->charecount; i++ ) 
			n->ChareTable[i] = ChareTable[i+StackTop->charecount] ;
	}
       	if ( (n->boccount = boccount - StackTop->boccount) != 0 ){
		n->BOCTable = (ChareInfo **)malloc(n->boccount*sizeof(ChareInfo *)) ;
		for ( i=1; i<=n->boccount; i++ ) 
			n->BOCTable[i] = BOCTable[i+StackTop->boccount] ;
	}
       	if ( (n->TotalMsgs = TotalMsgs - StackTop->TotalMsgs) != 0 ){
		n->MessageTable = (MsgStruct *)malloc(n->TotalMsgs*sizeof(MsgStruct));
		for ( i=0; i<n->TotalMsgs; i++ ) {
			n->MessageTable[i].name = MessageTable[i+StackTop->TotalMsgs].name;
			n->MessageTable[i].pack = MessageTable[i+StackTop->TotalMsgs].pack;
		}
	}
       	if ( (n->TotalAccs = TotalAccs - StackTop->TotalAccs) != 0 ){
		n->AccTable = (AccStruct **)malloc(n->TotalAccs*sizeof(AccStruct *)) ;
		for ( i=0; i<n->TotalAccs; i++ ) 
			n->AccTable[i] = AccTable[i+StackTop->TotalAccs] ;
	}
       	if ( (n->TotalMonos = TotalMonos - StackTop->TotalMonos) != 0){
		n->MonoTable = (AccStruct **)malloc(n->TotalMonos*sizeof(AccStruct *)) ;
		for ( i=0; i<n->TotalMonos; i++ ) 
			n->MonoTable[i] = MonoTable[i+StackTop->TotalMonos] ;
	}
}

	
int InsideAddScope=0 ;
		
AddScope(name)
char *name;
{
	int i ;
	char *firstcoln, *ptr, *rest ;
	char classname[MAX_NAME_LENGTH] ;


	firstcoln=Strstr(name,"::") ;

	if ( firstcoln == NULL ) 
		return ;
	if ( firstcoln==name ) {
		FoundGlobalScope = 1 ;
		firstcoln += 2 ;
	}

	InsideAddScope = 1 ;	
	PushStack() ;  	
	/* record state before the function is entered. This matches
	   with the call in RemoveScope() */

	rest = name ;
	
	while ( firstcoln != NULL ) {
		for ( i=0,ptr=rest; *ptr!=':'; ptr++ )
			classname[i++] = *ptr ;
		classname[i] = '\0' ;
		rest = firstcoln+2 ;

		AddOneScope(classname) ;

		firstcoln=Strstr(rest,"::") ;
	}
	InsideAddScope = 0 ;	
}
	

AddOneScope(name)
char *name ;
{
	int i, ind = -1 ;
	AggState *n ;

	if ( strcmp(CurrentAggName,name) == 0 )
		return ;

	/* The AddedScope variable is to be used only for indicating that
	   a TEMPORARY scope has been added, eg in :: stuff.
	   It is used when we dont know (from context) whether a scope has
	   been added or not, eg when the adding and removing occur in 
 	   different parts of the yacc file */
	/* For functions, etc, the AddScope() and RemoveScope() pair is always
	   called, so AddedScope should not be set (it causes confusion) */

	if ( AddedScope==0 && !InsideAddScope ) {
	/* second condition is to prevent two PushStacks */
		PushStack() ;
		AddedScope = 1  ;
	}

	/* find name's entry in PermanentAggTable */
	if ( !FoundGlobalScope || GlobalStack==NULL ) {
 		for ( i=TotalSyms-1; i>=0; i-- ) { 
		/* name HAS to be there in the SymTable because it
	   	   gets updated every time by AddOneScope */
			if ( strcmp(name,SymTable[i].name)==0 ) {	
				ind = SymTable[i].permanentindex ; 
				break ; 
			}
		}
	}
	else {
 		for ( i=0; i<GlobalStack->TotalSyms; i++ ) { 
			if ( strcmp(name,SymTable[i].name)==0 ) {	
				ind = SymTable[i].permanentindex ; 
				break ; 
			}
		}
		FoundGlobalScope = 0 ;
	}
	if ( i == TotalSyms || ind == -1 ) {
		fprintf(stderr,"ERROR: %s, line %d : %s is not a class/struct type.\n",CurrentFileName,CurrentLine,name) ;
		return ;
	}

	n = PermanentAggTable[ind] ;

	for ( i=0; i<n->ChareHandleTableSize; i++ )
		ChareHandleTable[ChareHandleTableSize++] = n->ChareHandleTable[i] ;
	for ( i=0; i<n->BOCHandleTableSize; i++ )
		BOCHandleTable[BOCHandleTableSize++] = n->BOCHandleTable[i] ;
	for ( i=0; i<n->AccHandleTableSize; i++ )
		AccHandleTable[AccHandleTableSize++] = n->AccHandleTable[i] ;
	for ( i=0; i<n->MonoHandleTableSize; i++ )
		MonoHandleTable[MonoHandleTableSize++] = n->MonoHandleTable[i];

	if ( n->TotalSyms == 0 ) {
		/* No new typeDEFS introduced in this agg*/
		return ;
	}

	for ( i=0; i<n->TotalSyms; i++ )
		SymTable[TotalSyms++] = n->SymTable[i] ;
	for ( i=0; i<n->charecount; i++ ) 
		ChareTable[++charecount] = n->ChareTable[i] ;
	for ( i=0; i<n->boccount; i++ ) 
		BOCTable[++boccount] = n->BOCTable[i] ;
	for ( i=0; i<n->TotalMsgs; i++ ) {
		MessageTable[TotalMsgs++].name =n->MessageTable[i].name ;
		MessageTable[TotalMsgs++].pack =n->MessageTable[i].pack ;
	}
	for ( i=0; i<n->TotalAccs; i++ ) 
		AccTable[TotalAccs++] = n->AccTable[i] ;
	for ( i=0; i<n->TotalMonos; i++ ) 
		MonoTable[TotalMonos++] = n->MonoTable[i] ;
}


RemoveScope(name)
char *name ;
{
        char *firstcoln ;

        firstcoln=Strstr(name,"::") ;

        if ( firstcoln == NULL )
                return ;

	PopStack() ;	
	/* restore state to before entering function. This matches
           with the call in AddScope() */

}


InsertVarSize(type, name)
char *type, *name ;
{	MsgStruct *thismsg ;

        thismsg = &(MessageTable[TotalMsgs-1]) ;
        if ( thismsg->numvarsize == 0 ) {
                thismsg->varsizearray = (VarSizeStruct *)
                          malloc(sizeof(VarSizeStruct)*MAX_VARSIZE) ;
        }
        thismsg->varsizearray[thismsg->numvarsize].type = 
                        (char *)malloc((sizeof(char)+1)*strlen(type)) ;
        thismsg->varsizearray[thismsg->numvarsize].name =
                        (char *)malloc((sizeof(char)+1)*strlen(name)) ;
	strcpy(thismsg->varsizearray[thismsg->numvarsize].type,type) ;
	strcpy(thismsg->varsizearray[thismsg->numvarsize].name,name) ;

	thismsg->numvarsize++ ;
	thismsg->pack = TRUE ;
}



InsertFunctionTable(name,defined)
char *name ;
int defined ;
{
	int i, j ;

	if ( Strstr(name,"::") != NULL )
		return ;	/* this is a aggregate member function defn */

	if ( Strstr(name,"operator") != NULL )
		return ;  /* for now, dont put operators in function table */

	if ( defined ) { /* search for overloading */
	    for ( i=0; i<TotalFns; i++ ) {
		if ( strcmp(FunctionTable[i].name,name) == 0 &&
		     FunctionTable[i].defined ) {
			/* HACK : cant handle overloading yet : 
			   so remove from table */
			for ( j=i; j<TotalFns-1; j++ ) {
				FunctionTable[j].name = FunctionTable[j+1].name ;
				FunctionTable[j].defined = FunctionTable[j+1].defined ;
			}
			TotalFns-- ;
			return ;
		}
	    }
	}

	FunctionTable[TotalFns].name = (char *)malloc((strlen(name)+1)*sizeof(char)) ;
	strcpy(FunctionTable[TotalFns].name,name) ;
	FunctionTable[TotalFns].defined = defined ;
	TotalFns++ ;
}
	


EP * AlreadyInList(thisep,eplist)
EP *thisep, *eplist ;
{
	EP *epl ;

	for ( epl=eplist; epl!=NULL; epl=epl->next )
		if ( strcmp(epl->epname, thisep->epname) == 0 )
			return epl ;
	return NULL ;
}


AddInheritedEps(chare)
ChareInfo *chare ;
{
	int i, j, count ;
	ChareInfo *p ;
	ChareList *pl ;
	EP *newep, *parep, *thisep ;

	for ( pl=chare->parents; pl!=NULL; pl=pl->next ) {
		p = pl->chare ;
		/* Add eplist of parent to eplist of this chare 
		   for inheritance */
        	for ( parep=p->eps; parep!=NULL; parep=parep->next ) {
               		if ((thisep=AlreadyInList(parep,chare->eps))) {
				thisep->inherited = 1 ;
				thisep->parentep = parep ;
				if ( parep->isvirtual )
					thisep->isvirtual = TRUE ;
			}
               		else {
               			newep = (EP *)malloc(sizeof(EP)) ;
               			strcpy(newep->epname,parep->epname) ;
               			strcpy(newep->msgname,parep->msgname) ;
				newep->parentep = parep ;
               			newep->inherited = 1 ;
				newep->defined = 0 ;
				if ( parep->isvirtual )
					newep->isvirtual = TRUE ;
				newep->chare = chare ;
				InsertAlpha(chare,newep) ;
			}
        	}
	}
}

	
	

InsertAlpha(chare,ep)
ChareInfo *chare ;
EP *ep ;
{
	int val ;
	EP *e, *eprev ;

	if ( chare->eps == NULL ) {
		chare->eps = ep ;
		return ;
	}
	for ( eprev=NULL,e=chare->eps; e!=NULL; eprev=e,e=e->next ) {
		val = strcmp(ep->epname,e->epname) ;
		if ( val < 0 ) {	
			/* insert ep before e */	
			if ( eprev == NULL )
				chare->eps = ep ;
			else
				eprev->next = ep ;
			ep->next = e ;
			break ;
		}
		else if ( val == 0 ) /* already in list OR overloaded*/
			break ;
	}
	if ( e == NULL ) { /* insert at end of list */
		eprev->next = ep ;
		ep->next = NULL ;
	}
}


CheckConstructorEP(name,defined)
char *name ;
int defined ;
{
	char *begin ;

	if ( CurrentAggType==CHARE || CurrentAggType==BRANCHED || InsideChareCode ) {
                if (CurrentAccess == ENTRY  || 
                    (strcmp(CurrentChare,"main")==0 && strcmp(name,"main")==0))
		{
                        ProcessEP(name,defined);
                }
        }
}




InsertParent(chare, parent, table, tablecount)
ChareInfo *chare ;
char *parent ;
ChareInfo **table ;
int tablecount ;
{
	int i, found=0 ;
	ChareList *save ;
	ChareInfo *p ;
	EP *ep, *newep ;

	for ( i=0; i<tablecount; i++ ) {
		if ( strcmp(table[i]->name,parent) == 0 ) {
			found = 1 ;
			break ;
		}
	}
	if ( !found )
		return ;
	save = chare->parents ;
	chare->parents = (ChareList *)malloc(sizeof(ChareList)) ;
	p = chare->parents->chare = table[i] ;
	chare->parents->next = save ;
}

		


char *Strstr(b,s)
char *b, *s ;
{
	int i=0, j=0, nextbeg=1 ;
	int ls, lb ;

	ls = strlen(s) ;
	lb = strlen(b) ;

	if ( ls > lb )
		return(NULL) ;

	while ( i < ls && j < lb )	{
		if ( s[i] == b[j] ) {
			i++ ;
			j++ ;
		}
		else {
			i = 0 ;
			j = nextbeg ;
			nextbeg++ ;
		}
	}
	
	if ( j == lb && i != ls ) 
		return(NULL) ;
	else
		return(b+j-ls) ;
}



OutputNewChareMsg(name, arg, placement)
    char *name; char *arg; char *placement;
{
	char *sptr ;
	int type ;

	if ( FoundDeclarator ) { /* we have "new Msg *" or "new Msg [...]" */
		FoundDeclarator = FALSE ;
		return ;
	}

	/* First find whether name is a Chare, BOC or Message */	
	if ( NewOpType == NEWCHARE )
		type = CHARE ;
	else if ( NewOpType == NEWGROUP )
		type = BRANCHED ;
	else if ( FoundInMsgTable(name) != -1 ) 
		type = MESSAGE ;
	else if ( FoundInAccTable(AccTable,TotalAccs,name) != -1 ) 
		type = ACCUMULATOR ;
	else if ( FoundInAccTable(MonoTable,TotalMonos,name) != -1 ) 
		type = MONOTONIC ;
	else
		return ;

	sptr = strstr(OutBuf, "new") ;

	if ( sptr != NULL ) 
		*sptr = '\0' ;
	else 
		fprintf(stderr,"TRANSLATOR ERROR : %s, line %d : couldnt discard new.\n",CurrentFileName,CurrentLine) ;
	FLUSHBUF() ;


	if ( type == CHARE || type == BRANCHED ) {
		if ( strchr(arg,',') != NULL )
			fprintf(stderr,"TRANSLATOR ERROR : %s, line %d : new chare has more than one arg.\n",CurrentFileName,CurrentLine) ;
		
		if ( type == CHARE ) {
			if ( placement == NULL || *placement=='\0' ) 
				/* (0xFFF2) is CK_PE_ANY */
				fprintf(outfile,"CreateChare(_CK_chare_%s, _CK_ep_%s_%s, %s, 0, (0xFFF2)",name, name, name, arg) ; 
			else 
				fprintf(outfile,"CreateChare(_CK_chare_%s, _CK_ep_%s_%s, %s, %s",name, name, name, arg, placement) ;
		}
		else {
			if ( placement == NULL || *placement=='\0' ) 
				fprintf(outfile,"CreateBoc(_CK_chare_%s, _CK_ep_%s_%s, %s, -1, 0",name, name, name, arg) ;
			else 
				fprintf(outfile,"CreateBoc(_CK_chare_%s, _CK_ep_%s_%s, %s, %s",name, name, name, arg, placement) ;
		}
	}
	else if ( type == ACCUMULATOR ) {
		if ( strchr(arg,',') != NULL )
			fprintf(stderr,"TRANSLATOR ERROR : %s, line %d : new accumulator has more than one arg.\n",CurrentFileName,CurrentLine) ;
		
		if ( placement == NULL || *placement=='\0' ) 
			fprintf(outfile,"CreateAcc(_CK_acc_%s, %s, -1, 0",name, arg) ;
		else 
			fprintf(outfile,"CreateAcc(_CK_acc_%s, %s, %s",name, arg, placement) ;
	}
	else if ( type == MONOTONIC ) {
		if ( strchr(arg,',') != NULL )
			fprintf(stderr,"TRANSLATOR ERROR : %s, line %d : new monotonic has more than one arg.\n",CurrentFileName,CurrentLine) ;
		
		if ( placement == NULL || *placement=='\0' ) 
			fprintf(outfile,"CreateMono(_CK_mono_%s, %s, -1, 0",name, arg) ;
		else 
			fprintf(outfile,"CreateMono(_CK_mono_%s, %s, %s",name, arg, placement) ;
	}
	else { /* type == MESSAGE */
		if ( placement == NULL || *placement=='\0' ) 
			fprintf(outfile,"new (_CK_%s._CK_msg_%s) %s", CoreName, name, name) ;
		else  /* take care of placement = "sizes, prio" */
			fprintf(outfile,"new (_CK_%s._CK_msg_%s, %s) %s", CoreName, name, placement, name) ;
		if ( arg != NULL && *arg!='\0' ) 
			fprintf(outfile,"(%s",arg) ;
	}
}



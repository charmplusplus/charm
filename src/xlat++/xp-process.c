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
 * Revision 2.0  1995-06-05 19:01:24  brunner
 * Reorganized directory structure
 *
 * Revision 1.7  1995/05/05  22:55:38  sanjeev
 * Fixed ReadOnlyMsg bug
 *
 * Revision 1.6  1995/05/03  20:58:33  sanjeev
 * initialize module struct etc for detecting uninitialized modules
 *
 * Revision 1.5  1995/04/23  17:50:07  sanjeev
 * stuff to output PPG
 *
 * Revision 1.4  1995/03/23  05:11:33  sanjeev
 * changes for printing call graph
 *
 * Revision 1.3  1995/02/14  00:08:35  sanjeev
 * removed module.list stuff, no longer reqd in new runtime
 *
 * Revision 1.2  1994/12/10  19:00:20  sanjeev
 * interoperability stuff
 *
 * Revision 1.1  1994/11/03  17:42:10  brunner
 * Initial revision
 *
 ***************************************************************************/
static char ident[] = "@(#)$Header$";


/*------- VIEW THIS FILE IN A 130 COLUMN WINDOW ---------*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#include "xp-t.h"
#include "xp-extn.h"

int foundMain = 0 ;

static int TotalEps=0 ;

void GenerateStructsFns() ;
void GenerateRegisterCalls() ;


main(argc,argv)
int argc ;
char *argv[] ;
{
	int retval ;

	InitFiles(argc,argv) ;

	InsertSymTable("writeonce") ;
	InsertSymTable("EntryPointType") ;
	InsertSymTable("FunctionRefType") ;
	InsertSymTable("ChareIDType") ;	/* for the table class */
	retval = yyparse() ;

	if ( shouldprint )
		strcat(OutBuf,prevtoken) ;
	fprintf(outfile,"%s",OutBuf) ;

	if ( ErrVal || retval>0 ) {
		fprintf(stderr,"\nThere are errors in the input.\n");
		CloseFiles() ;
		exit(1) ;
	}

	/* Generate _CK_chare_ChareType,  _CK_ep_ChareType_EP, 
	   _CK_acc_AccType, _CK_mono_MonoType, _CK_func_FnName  variables.
	   Also generate the _CK_ModuleName struct which holds all the 
	   _CK_msg_MessageType variables.
	   Also generate _CK_create_ChareType() and _CK_call_ChareType_EP()
	   functions */

	GenerateStructsFns() ;


	/* Generate calls to all the "register" routines */

	GenerateRegisterCalls() ;

	if ( MakeGraph )
		OutputNames() ;

	CloseFiles() ;

	return(0) ;
}



InitFiles(argc,argv)
int argc ;
char *argv[] ;
{
	char pgm[MAX_NAME_LENGTH], headername[MAX_NAME_LENGTH] ;
        char *bgn, *end;
	FILE *modfp ;
	char buf[256], thismod[256] ;

	char *envstring ;
	char graphname[MAX_NAME_LENGTH] ;


	if (argc!=3) { 
		fprintf(stderr,"Usage: translate <InFile> <OutFile>. Stop.\n");
		exit(1); 
	}

	strcpy(pgm,argv[1]);

        bgn = strrchr(argv[1], '/');
        if (bgn==0) bgn=argv[1];
        end = bgn;
        while (1)
            {
            char c = *end;
            if (!(((c>='a')&&(c<='z'))||
                  ((c>='A')&&(c<='Z'))||
                  ((c>='0')&&(c<='9'))||
                  (c=='_'))) break;
            end++;
            }
	strncpy(CoreName, bgn, end-bgn);
	CoreName[(end-bgn)] = '\0' ;

	sprintf(headername,"%s.headers",CoreName) ;

	yyin = fopen(argv[1],"r") ;
	outfile = fopen(argv[2],"w") ;
	headerfile = fopen(headername,"w") ;

	if ( yyin==NULL || outfile==NULL || headerfile==NULL ) {
		fprintf(stderr,"ERROR : Cannot open input or output file(s). Stop.\n") ;
		exit(1) ;
	}
	
	fprintf(outfile,"#define CPLUS_FILE\n") ;
	fprintf(outfile,"#include \"ckdefs.h\" \n") ;
/*	fprintf(outfile,"#include \"env_macros.h\" \n") ;  Amitabh put this */
	fprintf(outfile,"#include \"chare.h\" \n") ;
	fprintf(outfile,"#include \"c++interface.h\" \n") ;
	fprintf(outfile,"#include \"%s\"\n",headername) ;
/*	fprintf(outfile,"#include \"stdio.h\"\n") ; */
/*	fprintf(outfile,"#line 1 \"%s.P\"\n",CoreName) ;   */

/*	fprintf(outfile,"extern \"C\" char * sprintf(char *, const char* ...);") ;	*/

	strcpy(prevtoken,"") ;
	strcpy(OutBuf,"") ;



	/* Create pgm.graph if required */
	
	envstring = getenv("GRAPH") ;	
	if ( envstring == NULL || *envstring == '0' )
		return ;

	MakeGraph = 1 ;

	sprintf(graphname,"%s.graph",CoreName) ;
	graphfile = fopen(graphname,"w") ;
	if ( graphfile == NULL ) {
		fprintf(stderr,"ERROR : Cannot open graph file. Stop.\n") ;
		exit(1) ;
	}
}


CloseFiles()
{

	fclose(yyin) ;
	fclose(outfile) ;
	fclose(headerfile) ;
	if ( MakeGraph )
		fclose(graphfile) ;
}




#define EXT(x) ( x ? " " : "extern " )
#define INIT(x) ( x ? "=0" : " " )


void GenerateStructsFns()
{
        int i, j, nv ;
	FILE *epfile ;
	ChareInfo *chare ;
	EP *ep ;
	int thisismain ;
	char extstr[16] ;
	char InitStr[128] ;

	TotalEps = 0 ;

	epfile = headerfile ;
	fprintf(epfile,"\n\n\n/******************************************************************/\n\n") ;
	fprintf(outfile,"\n\n\n/******************************************************************/\n\n") ;

/* Output all chare and boc names */
	for ( i=0; i<=charecount; i++ ) {
		if ( ChareTable[i]->eps != NULL )
			fprintf(epfile,"%sint _CK_chare_%s %s ;\n",EXT(ChareTable[i]->eps->defined),ChareTable[i]->name,INIT(ChareTable[i]->eps->defined)) ;
	}

	for ( i=0; i<=boccount; i++ ) {
		if ( BOCTable[i]->eps != NULL )
			fprintf(epfile,"%sint _CK_chare_%s %s ;\n",EXT(BOCTable[i]->eps->defined),BOCTable[i]->name,INIT(BOCTable[i]->eps->defined)) ;
	}


/* Output Chare variables and functions */
	thisismain = 0 ;
	for ( i=0; i<=charecount; i++ ) {
		chare = ChareTable[i] ;
		if ( chare->eps == NULL )
			continue ;
		if ( strcmp(chare->name,"main") == 0 )
			thisismain = 1 ;
		if ( chare->eps->defined ) {
		    for  ( ep=chare->eps; ep!=NULL; ep=ep->next,TotalEps++ ) {
			fprintf(epfile,"\tint _CK_ep_%s_%s =0 ;\n",chare->name,ep->epname) ;
			if (thisismain && strcmp(ep->epname,"main")==0) {
        			foundMain = 1 ;
                		fprintf(outfile,"\nvoid _CK_call_main_main(_CK_Object *obj, int argc, char *argv[])\n{\n");
                		if ( main_argc_argv )
                       			fprintf(outfile,"\t((main *)obj)->_CKmain(argc,argv) ;\n") ;
                		else {
                       			fprintf(outfile,"\targc = 0 ;\n") ;
                       			fprintf(outfile,"\targv = NULL ;\n") ;
                       			fprintf(outfile,"\t((main *)obj)->_CKmain() ;\n") ;
                		}
				fprintf(outfile,"}\n") ;
			}
			else {
				fprintf(outfile,"void _CK_call_%s_%s(void *m, _CK_Object *obj)\n{\n",chare->name,ep->epname) ;
				fprintf(outfile,"\t((%s *)obj)->%s((%s *)m) ;\n}\n",chare->name,ep->epname,ep->msgname) ;
			}
		    }
		}
		else {
			for  ( ep=chare->eps; ep!=NULL; ep=ep->next,TotalEps++)
				fprintf(epfile,"\textern int _CK_ep_%s_%s ;\n",chare->name,ep->epname) ;
		}

		if ( chare->eps->defined ) {
	    	    fprintf(outfile,"_CK_Object *_CK_create_%s(MAGIC_NUMBER_TYPE m) {\n",chare->name) ;
	    	    fprintf(outfile,"\treturn(new %s(m)) ;\n}\n\n",chare->name);
		}
	}


/* output BOC variables and fns */
	for ( i=0; i<=boccount; i++ ) {
		chare = BOCTable[i] ;
		if ( chare->eps == NULL )
			continue ;
		if ( chare->eps->defined ) {
			for  ( ep=chare->eps; ep!=NULL; ep=ep->next,TotalEps++)
			{
				fprintf(epfile,"\tint _CK_ep_%s_%s =0 ;\n",chare->name,ep->epname) ;
				fprintf(outfile,"void _CK_call_%s_%s(void *m, _CK_Object *obj)\n{\n", chare->name,ep->epname) ;
				fprintf(outfile,"\t((%s *)obj)->%s((%s *)m) ;\n}\n", chare->name,ep->epname,ep->msgname) ;
			}
		}
		else {
			for  ( ep=chare->eps; ep!=NULL; ep=ep->next,TotalEps++)
				fprintf(epfile,"\textern int _CK_ep_%s_%s ;\n",chare->name,ep->epname) ;
		}

		if ( chare->eps->defined ) {
	    	    fprintf(outfile,"_CK_BOC *_CK_create_%s() {\n",chare->name) ;
	    	    fprintf(outfile,"\treturn(new %s()) ;\n}\n\n",chare->name) ;
		}
	}

/* Output all acc and mono names */
	for ( i=0; i<TotalAccs; i++ ) {
		fprintf(epfile,"%sint _CK_acc_%s %s ;\n",EXT(AccTable[i]->defined),AccTable[i]->name,INIT(AccTable[i]->defined)) ;
		if ( AccTable[i]->defined ) {
	    	    fprintf(outfile,"void *_CK_create_%s(void *msg) {\n",AccTable[i]->name) ;
	    	    fprintf(outfile,"\treturn(new %s((%s *)msg)) ;\n}\n\n",AccTable[i]->name,AccTable[i]->initmsgtype) ;
		}
	}
	for ( i=0; i<TotalMonos; i++ ) {
		fprintf(epfile,"%sint _CK_mono_%s %s ;\n",EXT(MonoTable[i]->defined),MonoTable[i]->name,INIT(MonoTable[i]->defined)) ;
		if ( MonoTable[i]->defined ) {
	    	    fprintf(outfile,"void *_CK_create_%s(void *msg) {\n",MonoTable[i]->name) ;
	    	    fprintf(outfile,"\treturn(new %s((%s *)msg)) ;\n}\n\n",MonoTable[i]->name,MonoTable[i]->initmsgtype) ;
		}
	}


        for ( i=0; i<TotalReadMsgs; i++ )
                fprintf(epfile,"int _CK_index_%s ;\n",ReadMsgTable[i]) ;

/* Output all global Function Names */
	for ( i=0; i<TotalFns; i++ ) {
	    if ( !(FunctionTable[i].defined) ) 
		fprintf(epfile,"extern int _CK_func_%s ;\n",FunctionTable[i].name) ;
	    else
		fprintf(epfile,"int _CK_func_%s =0;\n",FunctionTable[i].name) ;
	}

	fprintf(epfile,"\n\n") ;




/* generate the _CK_ModuleName struct which holds all the _CK_msg_MessageType 
   variables.	*/

	fprintf(headerfile,"\n\n\n/******************************************************************/\n\n") ;

	strcpy(InitStr,"0") ;
	for ( i=1; i<TotalMsgs; i++ ) 
		strcat(InitStr,",0") ;

	fprintf(headerfile,"struct {\n",CoreName) ;
	for ( i=0; i<TotalMsgs; i++ ) {
		fprintf(headerfile,"\tint _CK_msg_%s ;\n",MessageTable[i].name) ;
	}
	fprintf(headerfile,"} _CK_%s ={%s} ;\n\n",CoreName,InitStr) ;


/* for all MsgTypes output the pack - unpack call functions and the alloc fns  */
	for ( i=0; i<TotalMsgs; i++ ) {
		char *nam = MessageTable[i].name ;
		if ( MessageTable[i].pack ) {
			fprintf(outfile,"static void _CK_pack_%s(void *in, void **out,int *length)\n{\n",nam) ; 
			fprintf(outfile,"\t(*out) = ((%s *)in)->pack(length) ;\n}\n",nam) ;
			if ( MessageTable[i].numvarsize > 0 ) {
				fprintf(outfile,"static void _CK_unpack_%s(void *in, void **out)\n{\n",nam) ; 
				fprintf(outfile,"\t(*out) = in ;\n") ;
				fprintf(outfile,"\t((%s *)in)->unpack() ;\n}\n",nam) ;
			}
			else {
				fprintf(outfile,"static void _CK_unpack_%s(void *in, void **out)\n{\n",nam) ; 
				fprintf(outfile,"\t%s * m = (%s *)GenericCkAlloc(_CK_%s._CK_msg_%s,sizeof(%s),0) ;\n", nam, nam, 
													CoreName, nam, nam) ;
				fprintf(outfile,"\t(*out) = (void *) m ;\n") ;
				fprintf(outfile,"\tm->unpack(in) ;\n}\n") ;
			}
		}
		/* Output the alloc function if this is a varsize msg */
		if ( (nv=MessageTable[i].numvarsize) > 0 ) {
		    VarSizeStruct *vs = MessageTable[i].varsizearray ;
		    fprintf(outfile,"static void *_CK_alloc_%s(int msgno, int size, int *array, int prio)\n{\n",nam) ; 
		    fprintf(outfile,"\tint totsize=0, temp, dummy, sarray[%d] ;\n",nv) ;
		    fprintf(outfile,"\t%s * ptr ;\n",nam) ;
		    fprintf(outfile,"\ttotsize = temp = (size%%_CK_VARSIZE_UNIT)?_CK_VARSIZE_UNIT*((size+_CK_VARSIZE_UNIT)/_CK_VARSIZE_UNIT):size;\n\n") ;
		    for ( j=0; j<nv; j++ ) {
		    	fprintf(outfile,"\tsize = sizeof(%s)*array[%d];\n",vs[j].type,j) ;
		    	fprintf(outfile,"\tdummy = (size%%_CK_VARSIZE_UNIT)?_CK_VARSIZE_UNIT*((size+_CK_VARSIZE_UNIT)/_CK_VARSIZE_UNIT):size;\n") ;
		    	fprintf(outfile,"\tsarray[%d]=dummy;\n",j) ;
		    	fprintf(outfile,"\ttotsize += dummy;\n") ;
		    }

		    fprintf(outfile,"\n\tptr = (%s *)GenericCkAlloc(msgno,totsize,prio);\n",nam) ;
		    fprintf(outfile,"\tdummy=temp;\n\n") ;
		    for ( j=0; j<nv; j++ ) {
		    	fprintf(outfile,"\tptr->%s = (%s *)((char *)ptr + dummy);\n", vs[j].name, vs[j].type) ;
		    	fprintf(outfile,"\tdummy += sarray[%d];\n",j) ;
		    }
		    fprintf(outfile,"\nreturn((void *)ptr);\n}\n\n") ;
		}
	}


/* Output _CK_mod_CopyFromBuffer, _CK_mod_CopyToBuffer for readonly vars */
	fprintf(outfile,"extern \"C\" void _CK_13CopyFromBuffer(void *,int) ;\n") ;
	fprintf(outfile,"void _CK_%s_CopyFromBuffer(void **_CK_ReadMsgTable)\n",CoreName) ;
	fprintf(outfile,"{\n") ;
	for ( i=0; i<TotalReads; i++ ) 
		fprintf(outfile,"\t_CK_13CopyFromBuffer(&%s,sizeof(%s)) ;\n",ReadTable[i],ReadTable[i]) ;
	for ( i=0; i<TotalReadMsgs; i++ ) {
		fprintf(outfile,"    {   void **temp = (void **)(&%s);\n",ReadMsgTable[i]) ;
		fprintf(outfile,"        *temp = _CK_ReadMsgTable[_CK_index_%s];\n",ReadMsgTable[i]) ;
		fprintf(outfile,"    }\n") ;
	}
	fprintf(outfile,"}\n\n") ;

	fprintf(outfile,"extern \"C\" void _CK_13CopyToBuffer(void *,int) ;\n") ;
	fprintf(outfile,"extern \"C\" void ReadMsgInit(void *,int) ;\n") ;

	fprintf(outfile,"void _CK_%s_CopyToBuffer()\n",CoreName) ;
	fprintf(outfile,"{\n") ;
	for ( i=0; i<TotalReads; i++ ) 
		fprintf(outfile,"\t_CK_13CopyToBuffer(&%s,sizeof(%s)) ;\n",ReadTable[i],ReadTable[i]) ;
	for ( i=0; i<TotalReadMsgs; i++ ) 
		fprintf(outfile,"\tReadMsgInit(%s,_CK_index_%s) ;\n",ReadMsgTable[i],ReadMsgTable[i]) ;
	fprintf(outfile,"}\n\n") ;

}



void GenerateRegisterCalls()
{
	char *nam ;
	int i, j, count ;
	ChareInfo **Table, *chare ;
	EP *ep ;

/* generate the beginning of this Module's Init function */
	fprintf(outfile,"extern \"C\" void _CK_%s_init() ;\n",CoreName) ;
	fprintf(outfile,"void _CK_%s_init()\n{\n",CoreName) ;

/* first register all messages */
	for ( i=0; i<TotalMsgs; i++ ) {
		nam = MessageTable[i].name ;
		fprintf(outfile,"_CK_%s._CK_msg_%s = registerMsg(\"%s\", ", CoreName, nam, nam) ;

		if ( MessageTable[i].numvarsize == 0 )
			fprintf(outfile,"(FUNCTION_PTR)&GenericCkAlloc, ") ; 
		else
			fprintf(outfile,"(FUNCTION_PTR)&_CK_alloc_%s, ",nam) ;

		if ( MessageTable[i].pack == FALSE ) 
			fprintf(outfile,"NULL, NULL, ") ; 
		else 
			fprintf(outfile,"(FUNCTION_PTR)&_CK_pack_%s, (FUNCTION_PTR)&_CK_unpack_%s, ", nam, nam) ;

		fprintf(outfile,"sizeof(%s)) ;\n\n",nam) ;
	}
	fprintf(outfile,"\n\n") ;


/* now register all chares and BOCs and their EPs */
	for ( j=0; j<=1; j++ ) {
	    if ( j == 0 ) {
		count = charecount ;
		Table = ChareTable ;
	    }
	    else if ( j == 1 ) {
		count = boccount ;
		Table = BOCTable ;
	    }
	    for ( i=0; i<=count; i++ ) {
		chare = Table[i] ;
		if ( chare->eps == NULL )
			continue ;
		if ( !chare->eps->defined ) 
			continue ;
		fprintf(outfile,"_CK_chare_%s = registerChare(\"%s\", 0, (FUNCTION_PTR)&_CK_create_%s) ;\n\n",chare->name,chare->name,chare->name) ;

                for  ( ep=chare->eps; ep!=NULL; ep=ep->next ) {
			if ( j == 0 ) 
				fprintf(outfile,"_CK_ep_%s_%s = registerEp(\"%s\", (FUNCTION_PTR)&_CK_call_%s_%s, CHARMPLUSPLUS,", chare->name,
										ep->epname, ep->epname, chare->name,ep->epname) ;
			else
				fprintf(outfile,"_CK_ep_%s_%s = registerBocEp(\"%s\", (FUNCTION_PTR)&_CK_call_%s_%s, CHARMPLUSPLUS,",chare->name,
										ep->epname, ep->epname, chare->name,ep->epname) ;

			if ( j==0 && strcmp(chare->name,"main")==0 && 
			     strcmp(ep->epname,"main")==0 ) 
				fprintf(outfile,"0, _CK_chare_%s) ;\n\n",chare->name) ;
			else
				fprintf(outfile,"_CK_%s._CK_msg_%s, _CK_chare_%s) ;\n\n",CoreName, ep->msgname, chare->name) ;
		}			
 
	    }
	}
	fprintf(outfile,"\n\n") ;


/* register the main chare */
	if ( foundMain ) 
		fprintf(outfile,"registerMainChare(_CK_chare_main, _CK_ep_main_main, CHARMPLUSPLUS) ;\n\n\n") ;
	fprintf(outfile,"\n\n") ;


/* now register all global fns */
        for ( i=0; i<TotalFns; i++ ) {
        	if ( FunctionTable[i].defined )
            		fprintf(outfile,"_CK_func_%s = registerFunction((FUNCTION_PTR)&%s) ;\n",FunctionTable[i].name, FunctionTable[i].name) ;
        }
	fprintf(outfile,"\n\n") ;


/* now register all accs, monotonics, tables */
        for ( i=0; i<TotalAccs; i++ ) {
                if ( AccTable[i]->defined )
                	fprintf(outfile,"_CK_acc_%s = registerAccumulator(\"%s\", (FUNCTION_PTR)&_CK_create_%s, NULL, NULL, CHARMPLUSPLUS) ;\n",
									AccTable[i]->name,AccTable[i]->name,AccTable[i]->name) ;
        }
        for ( i=0; i<TotalMonos; i++ ) {
                if ( MonoTable[i]->defined )
                	fprintf(outfile,"_CK_mono_%s = registerMonotonic(\"%s\", (FUNCTION_PTR)&_CK_create_%s, NULL, CHARMPLUSPLUS) ;\n",
									MonoTable[i]->name,MonoTable[i]->name,MonoTable[i]->name) ;
        }
        for ( i=0; i<TotalDTables; i++ ) {
                fprintf(outfile,"%s.SetId(registerTable(\"%s\", NULL, NULL)) ;\n",DTableTable[i],DTableTable[i]) ;
        }
	fprintf(outfile,"\n\n") ;


/* now register readonlies and readonli messages */
        fprintf(outfile,"int readonlysize=0 ;\n") ;
        for ( i=0; i<TotalReads; i++ )
                fprintf(outfile,"readonlysize += sizeof(%s) ;\n",ReadTable[i]) ;

        fprintf(outfile,"\nregisterReadOnly(readonlysize, (FUNCTION_PTR)&_CK_%s_CopyFromBuffer, (FUNCTION_PTR)&_CK_%s_CopyToBuffer) ;\n",CoreName,CoreName) ;

	/* this is only needed to give a unique index to all all readonly msgs */
        for ( i=0; i<TotalReadMsgs; i++ )
                fprintf(outfile,"_CK_index_%s = registerReadOnlyMsg() ;\n",ReadMsgTable[i]) ;


/* This is the closing brace of the Module-init function */
        fprintf(outfile,"\n}\n") ;
}




OutputNames()
{
/* Output all chare and EP names into the pgm.graph file */
/* Try to make format same as Projections' pgm.sts file */

	int numchares=-1, numeps=-1 ;
	int i, j, count ;
	ChareInfo **Table, *chare ;
	EP *ep ;
	FN *fn ;
	int toteps = 0 ;

/* first output counts of chares, eps, msgs */
	fprintf(graphfile,"TOTAL_CHARES %d\n",charecount+1+boccount+1) ;
	fprintf(graphfile,"TOTAL_EPS %d\n",TotalEps) ;
	fprintf(graphfile,"TOTAL_MSGS %d\n",TotalMsgs) ;

/* first output all messages */
	for ( i=0; i<TotalMsgs; i++ ) { 
	  /* 0 below is because I dont know the message size at compile time */
		fprintf(graphfile,"MESSAGE %d 0 %s\n",i,MessageTable[i].name);
	}

/* now register all chares and BOCs and their EPs */
	for ( j=0; j<=1; j++ ) {
	    if ( j == 0 ) {
		count = charecount ;
		Table = ChareTable ;
	    }
	    else if ( j == 1 ) {
		count = boccount ;
		Table = BOCTable ;
	    }
	    for ( i=0; i<=count; i++ ) {
		chare = Table[i] ;

		/* count number of EPs */
                for  ( toteps=0,ep=chare->eps; ep!=NULL; ep=ep->next ) 
			toteps++ ;
                for  ( fn=chare->fns; fn!=NULL; fn=fn->next ) 
			toteps++ ;

		fprintf(graphfile,"CHARE %d %s %d\n",++numchares,chare->name,
							toteps) ;

		if ( chare->eps == NULL )
			continue ;
                for  ( ep=chare->eps; ep!=NULL; ep=ep->next ) {
		    if ( j == 0 ) {
			fprintf(graphfile,"ENTRY CHARE %d %s %d %d\n",++numeps,
						ep->epname, numchares, 
						FoundInMsgTable(ep->msgname)) ;
		    }
		    else {
			fprintf(graphfile,"ENTRY BOC %d %s %d %d\n",++numeps,
						ep->epname, numchares, 
						FoundInMsgTable(ep->msgname)) ;
		    }
		}	
                for  ( fn=chare->fns; fn!=NULL; fn=fn->next ) {
		    if ( j == 0 ) {
			fprintf(graphfile,"FUNCTION CHARE %d %s %d\n",++numeps,
						fn->fnname, numchares ) ;
		    }
		    else {
			fprintf(graphfile,"FUNCTION BOC %d %s %d\n",++numeps,
						fn->fnname, numchares ) ;
		    }
		}	
	    }
	}
	fprintf(graphfile,"END\n") ;
}


Graph_OutputCreate(chareboc, LastArg, LastChare, LastEP)
char *chareboc;
char *LastArg;
char *LastChare;
char *LastEP;
{
	if ( strcmp(chareboc,"_CK_CreateBoc") == 0 ) {
		fprintf(graphfile,"CREATEBOC %s %s : %s %s\n", CurrentChare,
					CurrentEP, LastChare, LastEP) ;
	}
	else if ( strcmp(chareboc,"_CK_CreateChare") == 0 ) {
                fprintf(graphfile,"CREATECHARE %s %s : %s %s %s\n",
                        CurrentChare, CurrentEP, LastChare, LastEP, LastArg) ;
	}
}

Graph_OutputPrivateCall(fnname)
char *fnname ;
{
	/* First find if this is indeed a public/privatecall */
	FN *f ;

	for ( f=CurrentCharePtr->fns; f!=NULL; f=f->next )
		if ( strcmp(fnname,f->fnname) == 0 ) {
			if (FoundInChareTable(ChareTable,charecount+1,
                                                        CurrentChare)!=-1) 
				fprintf(graphfile,"CALLCHARE %s %s : %s %s\n", 
				CurrentChare, CurrentEP, CurrentChare, fnname);
			else if (FoundInChareTable(BOCTable,boccount+1,
                                                        CurrentChare)!=-1) 
				fprintf(graphfile,"CALLBOC %s %s : %s %s\n", 
				CurrentChare, CurrentEP, CurrentChare, fnname);
			return ;
		}
}







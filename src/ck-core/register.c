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
 * Revision 2.0  1995-06-02 17:27:40  brunner
 * Reorganized directory structure
 *
 * Revision 1.5  1995/04/02  00:49:08  sanjeev
 * changes for separating Converse
 *
 * Revision 1.4  1995/03/21  20:44:43  sanjeev
 * Changed registerHandler to CsdRegisterHandler
 *
 * Revision 1.3  1995/03/17  23:38:33  sanjeev
 * changes for better message format
 *
 * Revision 1.2  1994/12/10  19:01:24  sanjeev
 * bug fixes for working with Charm++ translator
 *
 * Revision 1.1  1994/12/02  00:08:12  sanjeev
 * Initial revision
 *
 ***************************************************************************/
static char ident[] = "@(#)$Header$";


#include "chare.h"
#include "globals.h"


/* indexes */
extern int _CK_MainChareIndex ;
extern int _CK_MainEpIndex ;
extern int ReadBuffSize ;

int fnCount=0;

int chareEpsCount=0;

int msgCount=0;

int chareCount=0;

int pseudoCount=0;

int readCount=0 ;

int readMsgCount=0 ;




int registerMsg(name,allocf,packf,unpackf,size)
char *name;
FUNCTION_PTR allocf, packf, unpackf;
int size ;
{
/* fills in MsgToStructTable */
	MsgToStructTable[msgCount].alloc = allocf ;
	MsgToStructTable[msgCount].pack = packf ;
	MsgToStructTable[msgCount].unpack = unpackf ;
	MsgToStructTable[msgCount].size = size ;
	msgCount ++ ;
	return(msgCount-1) ;
}


int registerBocEp(name,epFunc,epType,msgIndx,chareIndx)
char *name;
FUNCTION_PTR epFunc ;
int epType ;
int msgIndx, chareIndx;
{
/* fills in EpTable, EpIsImplicitTable, EpNameTable, EpChareTable
   EpToMsgTable */
	EpTable[chareEpsCount] = epFunc ;
	EpIsImplicitTable[chareEpsCount] = 0 ;
	EpNameTable[chareEpsCount] = (char *)malloc(strlen(name)*sizeof(char)+1);
	strcpy(EpNameTable[chareEpsCount], name) ;
	EpChareTable[chareEpsCount] = chareIndx ;
	EpToMsgTable[chareEpsCount] = msgIndx ;
	EpLanguageTable[chareEpsCount] = epType ;

	EpChareTypeTable[chareEpsCount] = BOC ;

	chareEpsCount++ ;
	return(chareEpsCount-1) ;
}



int registerEp(name,epFunc,epType,msgIndx,chareIndx)
char *name;
FUNCTION_PTR epFunc ;
int epType ;
int msgIndx, chareIndx;
{
/* fills in EpTable, EpIsImplicitTable, EpNameTable, EpChareTable
   EpToMsgTable */

	EpTable[chareEpsCount] = epFunc ;
	EpIsImplicitTable[chareEpsCount] = 0 ;
	EpNameTable[chareEpsCount] = (char *)malloc(strlen(name)*sizeof(char)+1);
	strcpy(EpNameTable[chareEpsCount], name) ;
	EpChareTable[chareEpsCount] = chareIndx ;
	EpToMsgTable[chareEpsCount] = msgIndx ;
	EpLanguageTable[chareEpsCount] = epType ;

	EpChareTypeTable[chareEpsCount] = CHARE ;

	chareEpsCount++ ;
	return(chareEpsCount-1) ;
}

int registerChare(name,dataSz,createfn)
char *name;
int dataSz;
FUNCTION_PTR createfn ;
{
/* fills in ChareSizesTable, ChareNamesTable */
	ChareSizesTable[chareCount] = dataSz ;
	ChareNamesTable[chareCount] = (char *)malloc(strlen(name)*sizeof(char)+1);
	ChareFnTable[chareCount] = createfn ;

        strcpy(ChareNamesTable[chareCount], name) ;
	chareCount++ ;
	return(chareCount-1) ;
}


int registerFunction(fn)
FUNCTION_PTR fn ;
{
/* fills in _CK_9_GlobalFunctionTable */
	_CK_9_GlobalFunctionTable[fnCount] = fn ;
	fnCount++ ;	
	return(fnCount-1) ;
}


int registerMonotonic(name, initfn, updatefn,language)
char *name ;
FUNCTION_PTR initfn, updatefn ;
int language ;
{
	PseudoTable[pseudoCount].name = (char *)malloc(strlen(name)*sizeof(char)+1);
	strcpy(PseudoTable[pseudoCount].name,name) ;
	PseudoTable[pseudoCount].type = MONOTONIC ;
	PseudoTable[pseudoCount].initfn = initfn ;
	PseudoTable[pseudoCount].language = language ;
	PseudoTable[pseudoCount].pseudo_type.mono.updatefn = updatefn ;
	pseudoCount++ ;

	return(pseudoCount-1) ;
}

int registerTable(name, initfn, hashfn)
char *name ;
FUNCTION_PTR initfn, hashfn ;
{
	PseudoTable[pseudoCount].name = (char *)malloc(strlen(name)*sizeof(char)+1);
	strcpy(PseudoTable[pseudoCount].name,name) ;
	PseudoTable[pseudoCount].type = TABLE ;
	PseudoTable[pseudoCount].initfn = initfn ;
	PseudoTable[pseudoCount].pseudo_type.table.hashfn = hashfn ;
	pseudoCount++ ;

	return(pseudoCount-1) ;
}

int registerAccumulator(name, initfn, addfn, combinefn,language)
char *name ;
FUNCTION_PTR initfn, addfn, combinefn ;
int language ;
{
	PseudoTable[pseudoCount].name = (char *)malloc(strlen(name)*sizeof(char)+1);
	strcpy(PseudoTable[pseudoCount].name,name) ;
	PseudoTable[pseudoCount].type = ACCUMULATOR ;
	PseudoTable[pseudoCount].initfn = initfn ;
	PseudoTable[pseudoCount].language = language ;
	PseudoTable[pseudoCount].pseudo_type.acc.addfn = addfn ;
	PseudoTable[pseudoCount].pseudo_type.acc.combinefn = combinefn ;
	pseudoCount++ ;

	return(pseudoCount-1) ;
}


int registerReadOnlyMsg()
{
/* this is only needed to give a unique index to all all readonly msgs */
	readMsgCount++ ;
	return(readMsgCount-1) ;
}


void registerReadOnly(size, fnCopyFromBuffer, fnCopyToBuffer)
int size ;
FUNCTION_PTR fnCopyFromBuffer, fnCopyToBuffer ;
{
/* this is called only once per module */
	ROCopyFromBufferTable[readCount] = fnCopyFromBuffer ;
	ROCopyToBufferTable[readCount] = fnCopyToBuffer ;
	ReadBuffSize += size ;
	readCount++ ;
}



void registerMainChare(m, ep, type)
int m, ep ;
int type ;
{
	_CK_MainChareIndex = m ;
	_CK_MainEpIndex = ep ;
	MainChareLanguage = type ;
}

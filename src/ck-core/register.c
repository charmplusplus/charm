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
 * Revision 2.1  1995-06-08 17:07:12  gursoy
 * Cpv macro changes done
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
CsvExtern(int, _CK_MainChareIndex);
CsvExtern(int, _CK_MainEpIndex);
CsvExtern(int, ReadBuffSize);

CpvDeclare(int, fnCount); 
CpvDeclare(int, chareEpsCount);
CpvDeclare(int, msgCount);
CpvDeclare(int, chareCount);
CpvDeclare(int, pseudoCount);
CpvDeclare(int, readCount);
CpvDeclare(int, readMsgCount);


void registerModuleInit()
{
     CpvInitialize(int, fnCount);
     CpvInitialize(int, chareEpsCount);
     CpvInitialize(int, msgCount);
     CpvInitialize(int, chareCount);
     CpvInitialize(int, pseudoCount);
     CpvInitialize(int, readCount);
     CpvInitialize(int, readMsgCount);

     CpvAccess(fnCount) = 0;
     CpvAccess(chareEpsCount) = 0;
     CpvAccess(msgCount) = 0;
     CpvAccess(chareCount) = 0;
     CpvAccess(pseudoCount) = 0;
     CpvAccess(readCount) = 0;
     CpvAccess(readMsgCount) = 0;
}


int registerMsg(name,allocf,packf,unpackf,size)
char *name;
FUNCTION_PTR allocf, packf, unpackf;
int size ;
{
/* fills in MsgToStructTable */
	CsvAccess(MsgToStructTable)[CpvAccess(msgCount)].alloc = allocf ;
	CsvAccess(MsgToStructTable)[CpvAccess(msgCount)].pack = packf ;
	CsvAccess(MsgToStructTable)[CpvAccess(msgCount)].unpack = unpackf ;
	CsvAccess(MsgToStructTable)[CpvAccess(msgCount)].size = size ;
	CpvAccess(msgCount) ++ ;
	return(CpvAccess(msgCount)-1) ;
}


int registerBocEp(name,epFunc,epType,msgIndx,chareIndx)
char *name;
FUNCTION_PTR epFunc ;
int epType ;
int msgIndx, chareIndx;
{
/* fills in EpTable, EpIsImplicitTable, EpNameTable, EpChareTable
   EpToMsgTable */
	CsvAccess(EpTable)[CpvAccess(chareEpsCount)] = epFunc ;
	CsvAccess(EpIsImplicitTable)[CpvAccess(chareEpsCount)] = 0 ;
	CsvAccess(EpNameTable)[CpvAccess(chareEpsCount)] = 
                               (char *)CmiSvAlloc(strlen(name)*sizeof(char)+1);
	strcpy(CsvAccess(EpNameTable)[CpvAccess(chareEpsCount)], name) ;
	CsvAccess(EpChareTable)[CpvAccess(chareEpsCount)] = chareIndx ;
	CsvAccess(EpToMsgTable)[CpvAccess(chareEpsCount)] = msgIndx ;
	CsvAccess(EpLanguageTable)[CpvAccess(chareEpsCount)] = epType ;

	CsvAccess(EpChareTypeTable)[CpvAccess(chareEpsCount)] = BOC ;

	CpvAccess(chareEpsCount) ++ ;
	return(CpvAccess(chareEpsCount)-1) ;
}



int registerEp(name,epFunc,epType,msgIndx,chareIndx)
char *name;
FUNCTION_PTR epFunc ;
int epType ;
int msgIndx, chareIndx;
{
/* fills in EpTable, EpIsImplicitTable, EpNameTable, EpChareTable
   EpToMsgTable */

	CsvAccess(EpTable)[CpvAccess(chareEpsCount)] = epFunc ;
	CsvAccess(EpIsImplicitTable)[CpvAccess(chareEpsCount)] = 0 ;
	CsvAccess(EpNameTable)[CpvAccess(chareEpsCount)] = 
                              (char *)CmiSvAlloc(strlen(name)*sizeof(char)+1);
	strcpy(CsvAccess(EpNameTable)[CpvAccess(chareEpsCount)], name) ;
	CsvAccess(EpChareTable)[CpvAccess(chareEpsCount)] = chareIndx ;
	CsvAccess(EpToMsgTable)[CpvAccess(chareEpsCount)] = msgIndx ;
	CsvAccess(EpLanguageTable)[CpvAccess(chareEpsCount)] = epType ;

	CsvAccess(EpChareTypeTable)[CpvAccess(chareEpsCount)] = CHARE ;

	CpvAccess(chareEpsCount)++ ;
	return(CpvAccess(chareEpsCount-1)) ;
}

int registerChare(name,dataSz,createfn)
char *name;
int dataSz;
FUNCTION_PTR createfn ;
{
/* fills in ChareSizesTable, ChareNamesTable */
	CsvAccess(ChareSizesTable)[CpvAccess(chareCount)] = dataSz ;
	CsvAccess(ChareNamesTable)[CpvAccess(chareCount)] = 
                             (char *)CmiSvAlloc(strlen(name)*sizeof(char)+1);
	CsvAccess(ChareFnTable)[CpvAccess(chareCount)] = createfn ;

        strcpy(CsvAccess(ChareNamesTable)[CpvAccess(chareCount)], name) ;
	CpvAccess(chareCount)++ ;
	return(CpvAccess(chareCount-1)) ;
}


int registerFunction(fn)
FUNCTION_PTR fn ;
{
/* fills in _CK_9_GlobalFunctionTable */
	_CK_9_GlobalFunctionTable[CpvAccess(fnCount)] = fn ;
	CpvAccess(fnCount)++;	
	return(CpvAccess(fnCount-1)) ;
}


int registerMonotonic(name, initfn, updatefn,language)
char *name ;
FUNCTION_PTR initfn, updatefn ;
int language ;
{
	CsvAccess(PseudoTable)[CpvAccess(pseudoCount)].name = 
                         (char *)CmiSvAlloc(strlen(name)*sizeof(char)+1);
	strcpy(CsvAccess(PseudoTable)[CpvAccess(pseudoCount)].name,name) ;
	CsvAccess(PseudoTable)[CpvAccess(pseudoCount)].type = MONOTONIC ;
	CsvAccess(PseudoTable)[CpvAccess(pseudoCount)].initfn = initfn ;
	CsvAccess(PseudoTable)[CpvAccess(pseudoCount)].language = language ;
	CsvAccess(PseudoTable)[CpvAccess(pseudoCount)].pseudo_type.mono.updatefn=updatefn;
	CpvAccess(pseudoCount)++ ;

	return(CpvAccess(pseudoCount-1)) ;
}

int registerTable(name, initfn, hashfn)
char *name ;
FUNCTION_PTR initfn, hashfn ;
{
	CsvAccess(PseudoTable)[CpvAccess(pseudoCount)].name = 
                            (char *)CmiSvAlloc(strlen(name)*sizeof(char)+1);
	strcpy(CsvAccess(PseudoTable)[CpvAccess(pseudoCount)].name,name) ;
	CsvAccess(PseudoTable)[CpvAccess(pseudoCount)].type = TABLE ;
	CsvAccess(PseudoTable)[CpvAccess(pseudoCount)].initfn = initfn ;
	CsvAccess(PseudoTable)[CpvAccess(pseudoCount)].pseudo_type.table.hashfn = hashfn ;
	CpvAccess(pseudoCount)++ ;

	return(CpvAccess(pseudoCount-1)) ;
}

int registerAccumulator(name, initfn, addfn, combinefn,language)
char *name ;
FUNCTION_PTR initfn, addfn, combinefn ;
int language ;
{
	CsvAccess(PseudoTable)[CpvAccess(pseudoCount)].name = 
                             (char *)CmiSvAlloc(strlen(name)*sizeof(char)+1);
	strcpy(CsvAccess(PseudoTable)[CpvAccess(pseudoCount)].name,name) ;
	CsvAccess(PseudoTable)[CpvAccess(pseudoCount)].type = ACCUMULATOR ;
	CsvAccess(PseudoTable)[CpvAccess(pseudoCount)].initfn = initfn ;
	CsvAccess(PseudoTable)[CpvAccess(pseudoCount)].language = language ;
	CsvAccess(PseudoTable)[CpvAccess(pseudoCount)].pseudo_type.acc.addfn = addfn ;
	CsvAccess(PseudoTable)[CpvAccess(pseudoCount)].pseudo_type.acc.combinefn=combinefn;
	CpvAccess(pseudoCount)++ ;

	return(CpvAccess(pseudoCount-1)) ;
}


int registerReadOnlyMsg()
{
/* this is only needed to give a unique index to all all readonly msgs */
	CpvAccess(readMsgCount)++ ;
	return(CpvAccess(readMsgCount-1)) ;
}


void registerReadOnly(size, fnCopyFromBuffer, fnCopyToBuffer)
int size ;
FUNCTION_PTR fnCopyFromBuffer, fnCopyToBuffer ;
{
/* this is called only once per module */
	CsvAccess(ROCopyFromBufferTable)[CpvAccess(readCount)] = fnCopyFromBuffer ;
	CsvAccess(ROCopyToBufferTable)[CpvAccess(readCount)] = fnCopyToBuffer ;
	CsvAccess(ReadBuffSize) += size ;
	CpvAccess(readCount)++ ;
}



void registerMainChare(m, ep, type)
int m, ep ;
int type ;
{
	CsvAccess(_CK_MainChareIndex) = m ;
	CsvAccess(_CK_MainEpIndex) = ep ;
	CsvAccess(MainChareLanguage) = type ;
}

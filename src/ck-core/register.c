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
 * Revision 2.11  1998-02-27 11:52:15  jyelon
 * Cleaned up header files, replaced load-balancer.
 *
 * Revision 2.10  1997/10/29 23:52:52  milind
 * Fixed CthInitialize bug on uth machines.
 *
 * Revision 2.9  1997/08/22 19:29:07  milind
 * Added user-event tracing.
 *
 * Revision 2.8  1996/03/28 14:45:11  kale
 * added registration of threaded  entry points.
 *
 * Revision 2.7  1995/10/11 17:52:51  sanjeev
 * fixed Charm++ chare creation
 *
 * Revision 2.6  1995/09/07  21:21:38  jyelon
 * Added prefixes to Cpv and Csv macros, fixed bugs thereby revealed.
 *
 * Revision 2.5  1995/07/27  20:29:34  jyelon
 * Improvements to runtime system, general cleanup.
 *
 * Revision 2.4  1995/07/22  23:45:15  jyelon
 * *** empty log message ***
 *
 * Revision 2.3  1995/06/29  21:51:53  narain
 * Changed members of MSG_STRUCT and PSEUDO_STRUCT : packfn, unpackfn and tbl
 *
 * Revision 2.2  1995/06/27  22:10:49  gursoy
 * fixed some CpvAccess'es (not CpvAccess(i-1) bu CpvAccess(i)-1)
 *
 * Revision 2.1  1995/06/08  17:07:12  gursoy
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


#include "charm.h"



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
CpvDeclare(int, eventCount);


void registerModuleInit()
{
     CpvInitialize(int, fnCount);
     CpvInitialize(int, chareEpsCount);
     CpvInitialize(int, msgCount);
     CpvInitialize(int, chareCount);
     CpvInitialize(int, pseudoCount);
     CpvInitialize(int, readCount);
     CpvInitialize(int, readMsgCount);
     CpvInitialize(int, eventCount);

     CpvAccess(fnCount) = 0;
     CpvAccess(chareEpsCount) = 0;
     CpvAccess(msgCount) = 0;
     CpvAccess(chareCount) = 0;
     CpvAccess(pseudoCount) = 0;
     CpvAccess(readCount) = 0;
     CpvAccess(readMsgCount) = 0;
     CpvAccess(eventCount) = 0;
}


int registerEvent(name)
char *name;
{
  CsvAccess(EventTable)[CpvAccess(eventCount)] = name;
  CpvAccess(eventCount) ++ ;
  return(CpvAccess(eventCount)-1) ;
}

int registerMsg(name,allocf,packf,unpackf,size)
char *name;
FUNCTION_PTR allocf, packf, unpackf;
int size ;
{
/* fills in MsgToStructTable */
	CsvAccess(MsgToStructTable)[CpvAccess(msgCount)].alloc = allocf ;
	CsvAccess(MsgToStructTable)[CpvAccess(msgCount)].packfn = packf ;
	CsvAccess(MsgToStructTable)[CpvAccess(msgCount)].unpackfn = unpackf ;
	CsvAccess(MsgToStructTable)[CpvAccess(msgCount)].size = size ;
	CpvAccess(msgCount) ++ ;
	return(CpvAccess(msgCount)-1) ;
}

void setThreadedEp( int entry) {
  (CsvAccess(EpInfoTable)+ entry)->threaded = 1;
}

void SetEp(ep,name,function,language,messageindex,chareindex,chare_or_boc)
char *name;
FUNCTION_PTR function ;
int ep, language, messageindex, chareindex, chare_or_boc;
{
  EP_STRUCT *epinfo = CsvAccess(EpInfoTable)+ep;
  char *nname = (char *)CmiSvAlloc(strlen(name)+1);
  strcpy(nname, name);

  epinfo->name        = nname;
  epinfo->function    = function;
  epinfo->language    = language;
  epinfo->messageindex= messageindex;
  epinfo->chareindex  = chareindex;
  epinfo->chare_or_boc= chare_or_boc;
  epinfo->threaded = 0; 
}

int registerEp(name,function,language,messageindex,chareindex)
char *name;
FUNCTION_PTR function ;
int language ;
int messageindex, chareindex;
{
  int index=CpvAccess(chareEpsCount)++;
  SetEp(index, name, function, language, messageindex, chareindex, CHARE);
  return index;
}

int registerBocEp(name,function,language,messageindex,chareindex)
char *name;
FUNCTION_PTR function ;
int language ;
int messageindex, chareindex;
{
  int index=CpvAccess(chareEpsCount)++;
  SetEp(index, name, function, language, messageindex, chareindex, BOC);
  return index;
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

        strcpy(CsvAccess(ChareNamesTable)[CpvAccess(chareCount)], name) ;
	CpvAccess(chareCount)++ ;
	return(CpvAccess(chareCount)-1) ;
}


int registerFunction(fn)
FUNCTION_PTR fn ;
{
/* fills in _CK_9_GlobalFunctionTable */
	CsvAccess(_CK_9_GlobalFunctionTable)[CpvAccess(fnCount)] = fn ;
	CpvAccess(fnCount)++;	
	return(CpvAccess(fnCount)-1) ;
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

	return(CpvAccess(pseudoCount)-1) ;
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
	CsvAccess(PseudoTable)[CpvAccess(pseudoCount)].pseudo_type.tbl.hashfn = hashfn ;
	CpvAccess(pseudoCount)++ ;

	return(CpvAccess(pseudoCount)-1) ;
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

	return(CpvAccess(pseudoCount)-1) ;
}


int registerReadOnlyMsg()
{
/* this is only needed to give a unique index to all all readonly msgs */
	CpvAccess(readMsgCount)++ ;
	return(CpvAccess(readMsgCount)-1) ;
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

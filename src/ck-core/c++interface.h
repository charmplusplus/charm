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
 * Revision 2.18  1995-10-12 20:13:59  sanjeev
 * fixed problems while compiling with CC
 *
 * Revision 2.17  1995/10/11  19:30:33  sanjeev
 * removed CPlus_ChareExit
 *
 * Revision 2.16  1995/10/11  17:54:40  sanjeev
 * fixed Charm++ chare creation
 *
 * Revision 2.15  1995/10/03  19:54:21  sanjeev
 * new BOC syntax
 *
 * Revision 2.14  1995/09/26  19:46:46  sanjeev
 * moved new operator to cplus.C
 *
 * Revision 2.13  1995/09/21  16:39:01  sanjeev
 * *** empty log message ***
 *
 * Revision 2.12  1995/09/20  23:09:47  sanjeev
 * added comm_object
 *
 * Revision 2.11  1995/09/20  15:10:18  sanjeev
 * removed externs for Cmi stuff
 *
 * Revision 2.10  1995/09/19  21:44:54  brunner
 * Moved declaration of CmiTimer to converse.h from here.
 *
 * Revision 2.9  1995/09/07  21:21:38  jyelon
 * Added prefixes to Cpv and Csv macros, fixed bugs thereby revealed.
 *
 * Revision 2.8  1995/09/06  21:48:50  jyelon
 * Eliminated 'CkProcess_BocMsg', using 'CkProcess_ForChareMsg' instead.
 *
 * Revision 2.7  1995/09/05  22:35:32  sanjeev
 * removed _CK_MyBocNum
 *
 * Revision 2.6  1995/09/05  22:02:09  sanjeev
 * modified _CK_Object, _CK_BOC for new ChareBlock format.
 *
 * Revision 2.5  1995/09/01  02:13:17  jyelon
 * VID_BLOCK, CHARE_BLOCK, BOC_BLOCK consolidated.
 *
 * Revision 2.4  1995/07/27  20:29:34  jyelon
 * Improvements to runtime system, general cleanup.
 *
 * Revision 2.3  1995/07/22  23:44:13  jyelon
 * *** empty log message ***
 *
 * Revision 2.2  1995/06/14  19:39:26  gursoy
 * *** empty log message ***
 *
 * Revision 2.1  1995/06/08  17:09:41  gursoy
 * Cpv macro changes done
 *
 * Revision 1.6  1995/05/04  22:11:15  jyelon
 * *** empty log message ***
 *
 * Revision 1.5  1995/04/13  20:53:46  sanjeev
 * Changed Mc to Cmi
 *
 * Revision 1.4  1994/12/10  19:00:55  sanjeev
 * added extern decls for register fns
 *
 * Revision 1.3  1994/12/02  00:01:57  sanjeev
 * interop stuff
 *
 * Revision 1.2  1994/11/11  05:31:26  brunner
 * Removed ident added by accident with RCS header
 *
 * Revision 1.1  1994/11/07  15:39:49  brunner
 * Initial revision
 *
 ***************************************************************************/

#ifndef C_PLUS_INTERFACE_H
#define C_PLUS_INTERFACE_H

#define NULL_EP -1

/* This is to get size_t */
typedef unsigned int size_t ;

class _CK_Object ;
class groupmember ;

/* GroupIdType is the generic type usable for all BOC ids */
typedef ChareNumType GroupIdType ;

/* EPFnType is a pointer to a _CK_call_Chare_EP() function */
typedef void (*EPFnType)(void *, _CK_Object *) ;

/* CHAREFNTYPE is a pointer to a _CK_create_ChareName() function */
typedef _CK_Object * (*CHAREFNTYPE)(int) ;

/* BOCFNTYPE is a pointer to a _CK_create_BocName() function */
typedef groupmember * (*BOCFNTYPE)(void) ;

/* ACCFNTYPE is a pointer to a _CK_create_AccName() function */
typedef void * (*ACCFNTYPE)(void *) ;

/* ALLOCFNTYPE is a pointer to a _CK_alloc_MsgName() function */
typedef void * (*ALLOCFNPTR)(int, int, int*, int) ;




/* this is the handle of the main chare, used in place of MainChareID */
extern ChareIDType mainhandle;
extern ChareIDType NULL_HANDLE;



/* These are C functions in the Charm runtime system */

extern "C" int registerMsg(char *name, FUNCTION_PTR allocf, FUNCTION_PTR packf, FUNCTION_PTR unpackf, int size) ;

extern "C" int registerBocEp(char *name, FUNCTION_PTR epFunc , int epType , int msgIndx, int chareIndx) ;

extern "C" int registerEp(char *name, FUNCTION_PTR epFunc , int epType , int msgIndx, int chareIndx) ;

extern "C" int registerChare(char *name, int dataSz, FUNCTION_PTR createfn) ;

extern "C" int registerFunction(FUNCTION_PTR fn) ;

extern "C" int registerMonotonic(char *name , FUNCTION_PTR initfn, FUNCTION_PTR updatefn , int language) ;

extern "C" int registerTable(char *name , FUNCTION_PTR initfn, FUNCTION_PTR hashfn) ;

extern "C" int registerAccumulator(char *name , FUNCTION_PTR initfn, FUNCTION_PTR addfn, FUNCTION_PTR combinefn , int language) ;

extern "C" int registerReadOnlyMsg() ;

extern "C" void registerReadOnly(int size , FUNCTION_PTR fnCopyFromBuffer, FUNCTION_PTR fnCopyToBuffer) ;

extern "C" void registerMainChare(int m, int ep , int type) ;

extern "C" void * GenericCkAlloc(int, int, int) ;
extern "C" void * VarSizeCkAlloc(int, int, int, int[]) ;
extern "C" void * CkAllocPackBuffer(void *, int) ;
extern "C" int CreateBoc(int, int, void *, int, ChareIDType *) ;
extern "C" void CreateChare(int, int, void *, ChareIDType *, int) ;
extern "C" int CreateAcc(int, void *, int, ChareIDType *) ;
extern "C" int CreateMono(int, void *, int, ChareIDType *) ;
extern "C" void CkExit() ;
extern "C" void ChareExit() ;
extern "C" void CkFreeMsg(void *) ;
extern "C" void GeneralSendMsgBranch(int, void *, int, int, int) ;
extern "C" void GeneralBroadcastMsgBranch(int, void *, int, int) ;
extern "C" void SendMsg(int, void *, ChareIDType *) ;
extern "C" void *GetBocDataPtr(int) ;
extern "C" void SetBocBlockPtr(int, CHARE_BLOCK *);

extern "C" void VidRetrieveMessages(CHARE_BLOCK *, PeNumType, CHARE_BLOCK *) ;
extern "C" void SendNodeStatistics() ;
extern "C" void close_log() ;
extern "C" void PrintStsFile(char *) ;
extern "C" void trace_creation(int, int, ENVELOPE *) ;
extern "C" void trace_begin_execute(ENVELOPE *) ;
extern "C" void trace_end_execute(int, int, int) ;

extern "C" int CPlus_GetMagicNumber(_CK_Object *) ;
extern "C" void CPlus_StartQuiescence(int, ChareIDType) ;

extern "C" void * _CK_9GetAccDataPtr(void *) ;
extern "C" void * _CK_9GetMonoDataPtr(void *) ;
extern "C" void _CK_BroadcastMono(void *, int) ;
extern "C" void CollectValue(int, int, ChareIDType *) ;
extern "C" void * MonoValue(int) ;

extern "C" void * CkPriorityPtr(void *) ;
extern "C" ENVELOPE *CkCopyEnv(ENVELOPE *) ;






/* These are messages which are created in C code in the runtime 
   and received by a Charm++ chare. So they cant inherit from 
   comm_object, else the layout of the message changes. The translator
   puts them in the symbol table at initialization time (in process.c),
   so these definitions dont need to be seen by the translator (they are 
   seen by the C++ compiler because this file is #included in the program. */

class GroupIdMessage {	// sizeof(GroupIdMessage) MUST be 4
public:	GroupIdType group ;

	void *operator new(size_t size) {	// should never be called
		size = 0 ;	// to prevent CC from generating "size unused"
		return NULL ;
	}

        void operator delete(void *msg) {
                CkFreeMsg(msg) ;
        }
} ;

class QuiescenceMessage {// used in quiescence module
public:	int emptyfield ;

	void *operator new(size_t size) {	// should never be called
		size = 0 ;	// to prevent CC from generating "size unused"
		return NULL ;
	}

        void operator delete(void *msg) {
                CkFreeMsg(msg) ;
        }
} ;

class TableMessage {	
// used by distributed tables, must have exactly the
// same size and format as TBL_MSG in tbl.h
public: int key ;
        char *data ;
 
	void *operator new(size_t size) {	// should never be called
		size = 0 ;	// to prevent CC from generating "size unused"
		return NULL ;
	}

        void operator delete(void *msg) {
                CkFreeMsg(msg) ;
        }
} ;




/****** This is the top level class from which all message types inherit *****/

class comm_object {
public:	void operator delete(void *msg) {
		CkFreeMsg(msg) ;
	}

	void *operator new(size_t size) ;

	void *operator new(size_t size, int id) {
		return (void *)GenericCkAlloc(id, size, 0) ;
	}

	void *operator new(size_t size, int id, int prio) {
		return (void *)GenericCkAlloc(id, size, prio) ;
	}

	void *operator new(size_t size, int id, int* sizes) {
		return (void *)((ALLOCFNPTR)(CsvAccess(MsgToStructTable)[id].alloc))(id, size, sizes, 0) ;
	}

	void *operator new(size_t size, int id, int prio, int* sizes) {
		return (void *)((ALLOCFNPTR)(CsvAccess(MsgToStructTable)[id].alloc))(id, size, sizes, prio) ;
	}
} ;






/******* Top level chare class at root of chare hierarchy ********/

class _CK_Object {  
public:
	ChareIDType thishandle ;   

	_CK_Object() ;

        void * operator new(size_t size) ;
 
        void * operator new(size_t size, void *buf) ;

 	void operator delete(void *obj) {
		obj = 0 ;	// to prevent CC from generating "obj unused"
		ChareExit() ;
	}
} ;



class groupmember : public _CK_Object {  /* top level BOC object */
public:
	int thisgroup ;  /* stores BocNum */

	groupmember() ;
} ;


class _CK_Accumulator { /* top level Accumulator object */

public:
	int _CK_MyId ;

	virtual void * _CK_GetMsgPtr() = 0 ;

	virtual void _CK_Combine(void *) = 0 ;

	void CollectValue(int EpNum, ChareIDType cid)
	{
		::CollectValue(_CK_MyId, EpNum, &cid) ; 
		/* in node_acc.c */
	}
} ;

class _CK_Monotonic { /* top level Monotonic object */

public:
	int _CK_MyId ;

	virtual void * _CK_GetMsgPtr() = 0 ;

	virtual void _CK_SysUpdate(void *) = 0 ;  /* called by system */
} ;


#endif

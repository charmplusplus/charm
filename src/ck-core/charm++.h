#ifndef C_PLUS_INTERFACE_H
#define C_PLUS_INTERFACE_H

extern "C" {
#include "charm.h"
}

class _CK_Object ;
class groupmember ;

#define NULL_EP -1

/* GroupIdType is the generic type usable for all BOC ids */
typedef int GroupIdType ;

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


/* These are C++ functions in the runtime system */
FUNCTION_PTR CFunctionRefToName(int index) ;


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

extern "C" void * VarSizeCkAlloc(int, int, int, int[]) ;
extern "C" void * CkAllocPackBuffer(void *, int) ;
extern "C" int CreateBoc(int, int, void *, int, ChareIDType *) ;
extern "C" void CreateChare(int, int, void *, ChareIDType *, int) ;
extern "C" int CreateAcc(int, void *, int, ChareIDType *) ;
extern "C" int CreateMono(int, void *, int, ChareIDType *) ;
extern "C" void CkExit() ;
extern "C" void ChareExit() ;
extern "C" void CkFreeMsg(void *) ;
extern "C" void GeneralMulticastMsgBranch(int, void *, int, int, CmiGroup) ;
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

extern "C" unsigned int *CkPrioPtrFn(void *);
extern "C" int           CkPrioSizeBitsFn(void *);          
extern "C" int           CkPrioSizeBytesFn(void *);          
extern "C" int           CkPrioSizeWordsFn(void *);
extern "C" void          CkPrioConcatFn(void *, void *, unsigned int);

extern "C" void          CkSetQueueing(void *, int);

extern "C" ENVELOPE *CkCopyEnv(ENVELOPE *) ;


extern "C" void  SetRefNumber(void *m, int n);
extern "C" int   GetRefNumber(void *m);

extern "C" void      futuresModuleInit();
extern "C" void      futuresCreateBOC();
extern "C" void*     CRemoteCallBranchFn(int ep, void * m, int g, int p);
extern "C" void*     CRemoteCallFn(int ep, void *m, ChareIDType *id);
extern "C" void      CSendToFuture(void *m, int processor);



/* These are messages which are created in C code in the runtime 
   and received by a Charm++ chare. So they cant inherit from 
   comm_object, else the layout of the message changes. The translator
   puts them in the symbol table at initialization time (in process.c),
   so these definitions dont need to be seen by the translator (they are 
   seen by the C++ compiler because this file is #included in the program. */

class GroupIdMessage {	// sizeof(GroupIdMessage) MUST be 4
public:	GroupIdType groupid ;

	void *operator new(CMK_SIZE_T size) {	// should never be called
		size += 0 ;	// to prevent CC from generating "size unused"
		return NULL ;
	}

        void operator delete(void *msg) {
                CkFreeMsg(msg) ;
        }
} ;

class QuiescenceMessage {// used in quiescence module
public:	int emptyfield ;

	void *operator new(CMK_SIZE_T size) {	// should never be called
		size += 0 ;	// to prevent CC from generating "size unused"
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
 
	void *operator new(CMK_SIZE_T size) {	// should never be called
		size += 0 ;	// to prevent CC from generating "size unused"
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

	void *operator new(CMK_SIZE_T size) ;

	void *operator new(CMK_SIZE_T size, int id) {
		return (void *)GenericCkAlloc(id, size, 0) ;
	}

	void *operator new(CMK_SIZE_T size, int id, int prio) {
		return (void *)GenericCkAlloc(id, size, prio) ;
	}

	void *operator new(CMK_SIZE_T size, int id, int* sizes) {
		return (void *)((ALLOCFNPTR)(CsvAccess(MsgToStructTable)[id].alloc))(id, size, sizes, 0) ;
	}

	void *operator new(CMK_SIZE_T size, int id, int prio, int* sizes) {
		return (void *)((ALLOCFNPTR)(CsvAccess(MsgToStructTable)[id].alloc))(id, size, sizes, prio) ;
	}
} ;






/******* Top level chare class at root of chare hierarchy ********/

class _CK_Object {  
public:
	ChareIDType thishandle ;   

	_CK_Object() ;

        void * operator new(CMK_SIZE_T size) ;
 
        void * operator new(CMK_SIZE_T size, void *buf) ;

 	void operator delete(void *obj) {
		obj = obj ;	// to prevent CC from generating "obj unused"
		ChareExit() ;
	}
} ;


/* for use in the non-translator version */
class chare_object : public _CK_Object {} ;


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

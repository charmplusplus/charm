#include "charm.h"
#include "ck.h"
#include "ckobjid.h"

/**
	The method for returning the actual object pointed to by an id 
	If the object doesnot exist on the processor it returns NULL
**/

void* CkObjID::getObject(){
	
		switch(type){
			case TypeChare:	
				return CkLocalChare(&data.chare.id);
			case TypeMainChare:
				return CkLocalChare(&data.chare.id);
			case TypeGroup:
	
				CkAssert(data.group.onPE == CkMyPe());
				return CkLocalBranch(data.group.id);
			case TypeNodeGroup:
				CkAssert(data.group.onPE == CkMyNode());
				//CkLocalNodeBranch(data.group.id);
				{
					CmiImmediateLock(CksvAccess(_nodeGroupTableImmLock));
				  void *retval = CksvAccess(_nodeGroupTable)->find(data.group.id).getObj();
				  CmiImmediateUnlock(CksvAccess(_nodeGroupTableImmLock));					
	
					return retval;
				}	
			case TypeArray:
				{
	
	
					CkArrayID aid(data.array.id);
	
					if(aid.ckLocalBranch() == NULL){ return NULL;}
	
					CProxyElement_ArrayBase aProxy(aid,data.array.idx.asChild());
	
					return aProxy.ckLocal();
				}
			default:
				CkAssert(0);
		}
}


int CkObjID::guessPE(){
		switch(type){
			case TypeChare:
			case TypeMainChare:
				return data.chare.id.onPE;
			case TypeGroup:
			case TypeNodeGroup:
				return data.group.onPE;
			case TypeArray:
				{
					CkArrayID aid(data.array.id);
					if(aid.ckLocalBranch() == NULL){
						return -1;
					}
					return aid.ckLocalBranch()->lastKnown(data.array.idx.asChild());
				}
			default:
				CkAssert(0);
		}
};

char *CkObjID::toString(char *buf) const {
	
	switch(type){
		case TypeChare:
			sprintf(buf,"Chare %p PE %d \0",data.chare.id.objPtr,data.chare.id.onPE);
			break;
		case TypeMainChare:
			sprintf(buf,"Chare %p PE %d \0",data.chare.id.objPtr,data.chare.id.onPE);	
			break;
		case TypeGroup:
			sprintf(buf,"Group %d	PE %d \0",data.group.id.idx,data.group.onPE);
			break;
		case TypeNodeGroup:
			sprintf(buf,"NodeGroup %d	Node %d \0",data.group.id.idx,data.group.onPE);
			break;
		case TypeArray:
			{
				const CkArrayIndexMax &idx = data.array.idx.asChild();
				const int *indexData = idx.data();
				sprintf(buf,"Array |%d %d %d| id %d \0",indexData[0],indexData[1],indexData[2],data.array.id.idx);
				break;
			}
		default:
			CkAssert(0);
	}
	
	return buf;
};

void CkObjID::updatePosition(int PE){
	if(guessPE() == PE){
		return;
	}
	switch(type){
		case TypeChare:
		case TypeMainChare:
			CkAssert(data.chare.id.onPE == PE);
			break;
		case TypeGroup:
		case TypeNodeGroup:
			CkAssert(data.group.onPE == PE);
			break;
		default:
			CkAssert(0);
	}
}

// #endif    //CMK_MESSAGE_LOGGING

#include "Array1D.h"
#include "ArrayMap.h"
#include "ArrayElement.h"

void *
ArrayMigrateMessage::alloc(int msgnum,int size,int *array,int priobits)
{
  int totalsize;
  totalsize = size + array[0]*sizeof(char) + 8;
#if 0
  CPrintf("Allocating %d %d %d\n",msgnum,totalsize,priobits);
#endif
  ArrayMigrateMessage *newMsg = (ArrayMigrateMessage *)
    GenericCkAlloc(msgnum,totalsize,priobits);
#if 0
  CPrintf("Allocated %d\n",newMsg);
#endif
  newMsg->elementData = (char *)newMsg + ALIGN8(size);

  return (void *) newMsg;
}
  
void *
ArrayMigrateMessage::pack(int *length)
{
#if 0
  CPrintf("PE %d Packing %d %d %d\n",CMyPe(),from,index,elementSize);
#endif
  elementData = (void *)((char *)elementData - (char *)&elementData);
  return this;
}

void 
ArrayMigrateMessage::unpack(void *in)
{
  ArrayMigrateMessage *me = (ArrayMigrateMessage *)in;
#if 0
  CPrintf("PE %d Unpacking this=%d from=%d index=%d elementSize=%d\n",
	  CMyPe(),this,from,index,elementSize);
  CPrintf("PE %d Unpacking me=%d from=%d index=%d elementSize=%d\n",
	  CMyPe(),me,me->from,me->index,me->elementSize);
#endif
  me->elementData = (char *)&(me->elementData) + (int)me->elementData;
}

GroupIDType Array1D::CreateArray(int numElements,
				 ChareIndexType mapChare,
				 EntryIndexType mapConstructor,
				 ChareIndexType elementChare,
				 EntryIndexType elementConstructor,
				 EntryIndexType elementMigrator)
{
  int group;

  ArrayCreateMessage *msg = 
    new(MessageIndex(ArrayCreateMessage)) ArrayCreateMessage;

  msg->numElements = numElements;
  msg->mapChareType = mapChare;
  msg->mapConstType = mapConstructor;
  msg->elementChareType = elementChare;
  msg->elementConstType = elementConstructor;
  msg->elementMigrateType = elementMigrator;
  group = new_group(Array1D,ArrayCreateMessage,msg);
#if 0
  CPrintf("Created group %d\n",group);
#endif
  return group;
}

Array1D::Array1D(ArrayCreateMessage *msg)
{
  numElements = msg->numElements;
  elementChareType = msg->elementChareType;
  elementConstType = msg->elementConstType;
  elementMigrateType = msg->elementConstType;

  if (CMyPe()==0)
  {
    ArrayMapCreateMessage *mapMsg = 
      new(MsgIndex(ArrayMapCreateMessage))  ArrayMapCreateMessage;
    mapMsg->numElements = numElements;
    mapMsg->arrayID = thishandle;
    mapMsg->groupID = thisgroup;
    CreateBoc(msg->mapChareType,msg->mapConstType,mapMsg,-1,0);
  }
#if 0
  CPrintf("Array1D constructed\n");
#endif
  delete msg;
}

void Array1D::RecvMapID(ArrayMap *mPtr, ChareIDType mHandle,
			GroupIDType mGroup)
{
  map = mPtr;
  mapHandle = mHandle;
  mapGroup = mGroup;

  elementIDs = new ElementIDs[numElements];
  elementIDsReported = 0;
  numLocalElements=0;
  int i;
  for(i=0; i < numElements; i++)
  {
    elementIDs[i].state = creating;
    elementIDs[i].originalPE = elementIDs[i].pe = map->procNum(i);
    elementIDs[i].curHop = 0;
    if (elementIDs[i].pe != CMyPe())
    {
      elementIDs[i].element = NULL;
    }
    else
    {
      numLocalElements++;

      ChareIDType vid;
      ArrayElementCreateMessage *msg = 
	new (MessageIndex(ArrayElementCreateMessage)) 
	  ArrayElementCreateMessage;
      
      msg->numElements = numElements;
      msg->arrayID = thishandle;
      msg->groupID = thisgroup;
      msg->arrayPtr = this;
      msg->index = i;
      CreateChare(elementChareType, elementConstType, msg, 
		  &vid, CMyPe());
    }
  }
}

void Array1D::RecvElementID(int index, ArrayElement *elem, ChareIDType handle)
{
  elementIDs[index].state = here;
  elementIDs[index].element = elem;
  elementIDs[index].elementHandle = handle;
  elementIDsReported++;

#if 0
  if (elementIDsReported == numLocalElements)
    CPrintf("PE %d all elements reported in\n",CMyPe());
#endif
}

void Array1D::send(ArrayMessage *msg, int index, EntryIndexType ei)
{
  msg->destIndex = index;
  msg->entryIndex = ei;
  if (elementIDs[index].state == here)
  {
#if 0
    CPrintf("PE %d sending local message to index %d\n",CMyPe(),index);
#endif
    SendMsg(ei,msg,&elementIDs[index].elementHandle);
  }
  else if (elementIDs[index].state == moving_to)
  {
#if 0
    CPrintf("PE %d sending message to migrating index %d on PE %d\n",
	    CMyPe(),index,elementIDs[index].pe);
#endif
    CSendMsgBranch(Array1D,RecvForElement,ArrayMessage,msg,
		   thisgroup,elementIDs[index].pe);
  } else if (elementIDs[index].state == arriving)
  {
#if 0
    CPrintf("PE %d sending message for index %d to myself\n",
	    CMyPe(),index);
#endif
    CSendMsgBranch(Array1D,RecvForElement,ArrayMessage,msg,
		   thisgroup,CMyPe());
  } else
  {
#if 0
    CPrintf("PE %d sending message to index %d on original PE %d\n",
	    CMyPe(),index,elementIDs[index].originalPE);
#endif
    CSendMsgBranch(Array1D,RecvForElement,ArrayMessage,msg,
		   thisgroup,elementIDs[index].originalPE);
  }
}

void Array1D::broadcast(ArrayMessage *msg, EntryIndexType ei)
{
  CPrintf("Broadcast not implemented\n");
}

void Array1D::RecvForElement(ArrayMessage *msg)
{
#if 0
  CPrintf("PE %d RecvForElement sending to index %d\n",CMyPe(),msg->destIndex);
#endif
  if (elementIDs[msg->destIndex].state == here) 
  {
#if 0
    CPrintf("PE %d DELIVERING index %d RecvForElement state %d\n",
	    CMyPe(),msg->destIndex,elementIDs[msg->destIndex].state);
#endif
    SendMsg(msg->entryIndex,msg,&elementIDs[msg->destIndex].elementHandle);
  }
  else if (elementIDs[msg->destIndex].state == at)
  {
#if 0
    CPrintf("PE %d Sending to SELF index %d RecvForElement state %d\n",
	    CMyPe(),msg->destIndex,elementIDs[msg->destIndex].state);
#endif
    CSendMsgBranch(Array1D,RecvForElement,ArrayMessage,msg,
		   thisgroup,elementIDs[msg->destIndex].pe);
  }
  else
  {
#if 0
    CPrintf("PE %d Sending to SELF index %d RecvForElement state %d\n",
	    CMyPe(),msg->destIndex,elementIDs[msg->destIndex].state);
#endif
    CSendMsgBranch(Array1D,RecvForElement,ArrayMessage,msg,
		   thisgroup,elementIDs[msg->destIndex].originalPE);
  }
}

void Array1D::migrateMe(int index, int where)
{
  int bufSize = elementIDs[index].element->packsize();

  ArrayMigrateMessage *msg = 
    new (MessageIndex(ArrayMigrateMessage),1,&bufSize)
    ArrayMigrateMessage;

  msg->index = index;
  msg->from = CMyPe();
  msg->elementSize = bufSize;
  msg->hopCount = elementIDs[index].curHop + 1;
  elementIDs[index].element->pack(msg->elementData);
#if 0
  CPrintf("Sending to %d\n",where);
#endif
  numLocalElements--;
  CSendMsgBranch(Array1D,RecvMigratedElement,ArrayMigrateMessage,msg,
		 thisgroup,where);
  
}

void Array1D::RecvMigratedElement(ArrayMigrateMessage *msg)
{
  ChareIDType vid;
  
#if 0
  CPrintf("PE %d received migrated element from %d\n",CMyPe(),msg->from);
#endif
  int index =msg->index;

  elementIDs[index].state = arriving;
  elementIDs[index].pe = CMyPe();
  elementIDs[index].curHop = msg->hopCount;
  elementIDs[index].cameFrom = msg->from;

  ArrayElementMigrateMessage *new_msg = 
    new (MessageIndex(ArrayElementMigrateMessage)) ArrayElementMigrateMessage;

  new_msg->index = index;
  new_msg->numElements = numElements;
  new_msg->arrayID = thishandle;
  new_msg->groupID = thisgroup;
  new_msg->arrayPtr = this;
  
  CreateChare(elementChareType, elementMigrateType, new_msg, &vid, CMyPe());

  delete msg;
}

void Array1D::RecvMigratedElementID(int index, ArrayElement *elem,
				    ChareIDType handle)
{
#if 0
  CPrintf("PE %d index %d receiving migrated element handle %d\n",
	  CMyPe(),index,handle);
#endif
  elementIDs[index].state = here;
  elementIDs[index].element = elem;
  elementIDs[index].elementHandle = handle;

  ArrayElementAckMessage *ack_msg = 
    new (MessageIndex(ArrayElementAckMessage)) ArrayElementAckMessage;

  ack_msg->hopCount = elementIDs[index].curHop;
  ack_msg->index = index;
  ack_msg->arrivedAt = elementIDs[index].pe;
  ack_msg->handle = elementIDs[index].elementHandle;
  ack_msg->deleteElement = true;

  CSendMsgBranch(Array1D,AckMigratedElement,ArrayElementAckMessage,ack_msg,
		 thisgroup,elementIDs[index].cameFrom);
  
  if (elementIDs[index].cameFrom != elementIDs[index].originalPE)
  {
    ack_msg = new 
      (MessageIndex(ArrayElementAckMessage)) ArrayElementAckMessage;

    ack_msg->hopCount = elementIDs[index].curHop;
    ack_msg->index = index;
    ack_msg->arrivedAt = elementIDs[index].pe;
    ack_msg->handle = elementIDs[index].elementHandle;
    ack_msg->deleteElement = false;

    CSendMsgBranch(Array1D,AckMigratedElement,ArrayElementAckMessage,ack_msg,
		   thisgroup,elementIDs[index].originalPE);
  }
  numLocalElements++;
}

void Array1D::AckMigratedElement(ArrayElementAckMessage *msg)
{
  int index = msg->index;

#if 0
  CPrintf("PE %d Message acknowledged hop=%d curHop=%d\n",
	  CMyPe(),msg->hopCount,elementIDs[index].curHop);
#endif

  if (msg->hopCount > elementIDs[index].curHop)
  {
    if (msg->deleteElement)
    {
      ArrayElementExitMessage *exitmsg = new 
	(MessageIndex(ArrayElementExitMessage)) ArrayElementExitMessage;
#if 0
      CPrintf("I want to delete the element %d\n",index);
#endif
      SendMsg(EntryIndex(ArrayElement,exit,ArrayElementExitMessage),exitmsg,
	      &elementIDs[index].elementHandle);
    }
    elementIDs[index].pe = msg->arrivedAt;
    elementIDs[index].state = at;
    elementIDs[index].elementHandle = msg->handle;
  }
  else if (msg->hopCount <= elementIDs[index].curHop)
  {
    CPrintf("PE %d STALE Message acknowledged hop=%d curHop=%d\n",
	    CMyPe(),msg->hopCount,elementIDs[index].curHop);
    
  }
  delete msg;
}

#include "Array1D.bot.h"

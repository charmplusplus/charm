#include "ckarray.h"
#include "CkArray.def.h"

void *
ArrayMigrateMessage::alloc(int msgnum,int size,int *array,int priobits)
{
  int totalsize;
  totalsize = size + array[0]*sizeof(char) + 8;
#if 0
  CkPrintf("Allocating %d %d %d\n",msgnum,totalsize,priobits);
#endif
  ArrayMigrateMessage *newMsg = (ArrayMigrateMessage *)
    CkAllocMsg(msgnum,totalsize,priobits);
#if 0
  CkPrintf("Allocated %d\n",newMsg);
#endif
  newMsg->elementData = (char *)newMsg + ALIGN8(size);
  return (void *) newMsg;
}
  
void *
ArrayMigrateMessage::pack(ArrayMigrateMessage* in)
{
#if 0
  CkPrintf("PE %d Packing %d %d %d\n",CkMyPe(),from,index,elementSize);
#endif
  in->elementData = (void*)((char*)in->elementData-(char *)&(in->elementData));
  return (void*) in;
}

ArrayMigrateMessage* 
ArrayMigrateMessage::unpack(void *in)
{
  ArrayMigrateMessage *me = new (in) ArrayMigrateMessage;
#if 0
  CkPrintf("PE %d Unpacking this=%d from=%d index=%d elementSize=%d\n",
    CkMyPe(),this,from,index,elementSize);
  CkPrintf("PE %d Unpacking me=%d from=%d index=%d elementSize=%d\n",
    CkMyPe(),me,me->from,me->index,me->elementSize);
#endif
  me->elementData = (char *)&(me->elementData) + (int)me->elementData;
  return me;
}

CkGroupID Array1D::CreateArray(int numElements,
                               ChareIndexType mapChare,
                               EntryIndexType mapConstructor,
                               ChareIndexType elementChare,
                               EntryIndexType elementConstructor,
                               EntryIndexType elementMigrator)
{
  int group;

  ArrayCreateMessage *msg = new ArrayCreateMessage;

  msg->numElements = numElements;
  msg->mapChareType = mapChare;
  msg->mapConstType = mapConstructor;
  msg->elementChareType = elementChare;
  msg->elementConstType = elementConstructor;
  msg->elementMigrateType = elementMigrator;
  group = CProxy_Array1D::ckNew(msg);
#if 0
  CkPrintf("Created group %d\n",group);
#endif
  return group;
}

Array1D::Array1D(ArrayCreateMessage *msg)
{
  numElements = msg->numElements;
  elementChareType = msg->elementChareType;
  elementConstType = msg->elementConstType;
  elementMigrateType = msg->elementConstType;

  if (CkMyPe()==0) {
    ArrayMapCreateMessage *mapMsg = new ArrayMapCreateMessage;
    mapMsg->numElements = numElements;
    mapMsg->arrayID = thishandle;
    mapMsg->groupID = thisgroup;
    CkCreateGroup(msg->mapChareType,msg->mapConstType,mapMsg,-1,0);
  }
#if 0
  CkPrintf("Array1D constructed\n");
#endif
  delete msg;
}

void Array1D::RecvMapID(ArrayMap *mPtr, CkChareID mHandle,
                        CkGroupID mGroup)
{
  map = mPtr;
  mapHandle = mHandle;
  mapGroup = mGroup;

  elementIDs = new ElementIDs[numElements];
  elementIDsReported = 0;
  numLocalElements=0;
  int i;
  for(i=0; i < numElements; i++) {
    elementIDs[i].state = creating;
    elementIDs[i].originalPE = elementIDs[i].pe = map->procNum(i);
    elementIDs[i].curHop = 0;
    if (elementIDs[i].pe != CkMyPe()) {
      elementIDs[i].element = NULL;
    } else {
      numLocalElements++;

      CkChareID vid;
      ArrayElementCreateMessage *msg = new ArrayElementCreateMessage;
      
      msg->numElements = numElements;
      msg->arrayID = thishandle;
      msg->groupID = thisgroup;
      msg->arrayPtr = this;
      msg->index = i;
      CkCreateChare(elementChareType, elementConstType, msg, &vid, CkMyPe());
    }
  }
}

void Array1D::RecvElementID(int index, ArrayElement *elem, CkChareID handle)
{
  elementIDs[index].state = here;
  elementIDs[index].element = elem;
  elementIDs[index].elementHandle = handle;
  elementIDsReported++;

#if 0
  if (elementIDsReported == numLocalElements)
    CkPrintf("PE %d all elements reported in\n",CkMyPe());
#endif
}

void Array1D::send(ArrayMessage *msg, int index, EntryIndexType ei)
{
  msg->destIndex = index;
  msg->entryIndex = ei;
  if (elementIDs[index].state == here) {
#if 0
    CkPrintf("PE %d sending local message to index %d\n",CkMyPe(),index);
#endif
    CkSendMsg(ei,msg,&elementIDs[index].elementHandle);
  } else if (elementIDs[index].state == moving_to) {
#if 0
    CkPrintf("PE %d sending message to migrating index %d on PE %d\n",
      CkMyPe(),index,elementIDs[index].pe);
#endif
    CProxy_Array1D arr(thisgroup);
    arr.RecvForElement(msg, elementIDs[index].pe);
  } else if (elementIDs[index].state == arriving) {
#if 0
    CkPrintf("PE %d sending message for index %d to myself\n",
      CkMyPe(),index);
#endif
    CProxy_Array1D arr(thisgroup);
    arr.RecvForElement(msg, CkMyPe());
  } else {
#if 0
    CkPrintf("PE %d sending message to index %d on original PE %d\n",
      CkMyPe(),index,elementIDs[index].originalPE);
#endif
    CProxy_Array1D arr(thisgroup);
    arr.RecvForElement(msg, elementIDs[index].originalPE);
  }
}

void Array1D::broadcast(ArrayMessage *msg, EntryIndexType ei)
{
  CkPrintf("Broadcast not implemented\n");
}

void Array1D::RecvForElement(ArrayMessage *msg)
{
#if 0
  CkPrintf("PE %d RecvForElement sending to index %d\n",CkMyPe(),msg->destIndex);
#endif
  if (elementIDs[msg->destIndex].state == here) {
#if 0
    CkPrintf("PE %d DELIVERING index %d RecvForElement state %d\n",
      CkMyPe(),msg->destIndex,elementIDs[msg->destIndex].state);
#endif
    CkSendMsg(msg->entryIndex,msg,&elementIDs[msg->destIndex].elementHandle);
  } else if (elementIDs[msg->destIndex].state == at) {
#if 0
    CkPrintf("PE %d Sending to SELF index %d RecvForElement state %d\n",
      CkMyPe(),msg->destIndex,elementIDs[msg->destIndex].state);
#endif
    CProxy_Array1D arr(thisgroup);
    arr.RecvForElement(msg, elementIDs[msg->destIndex].pe);
  } else {
#if 0
    CkPrintf("PE %d Sending to SELF index %d RecvForElement state %d\n",
      CkMyPe(),msg->destIndex,elementIDs[msg->destIndex].state);
#endif
    CProxy_Array1D arr(thisgroup);
    arr.RecvForElement(msg, elementIDs[msg->destIndex].originalPE);
  }
}

void Array1D::migrateMe(int index, int where)
{
  int bufSize = elementIDs[index].element->packsize();

  ArrayMigrateMessage *msg = new (&bufSize) ArrayMigrateMessage;

  msg->index = index;
  msg->from = CkMyPe();
  msg->elementSize = bufSize;
  msg->hopCount = elementIDs[index].curHop + 1;
  elementIDs[index].element->pack(msg->elementData);
#if 0
  CkPrintf("Sending to %d\n",where);
#endif
  numLocalElements--;
  CProxy_Array1D arr(thisgroup);
  arr.RecvMigratedElement(msg, where);
}

void Array1D::RecvMigratedElement(ArrayMigrateMessage *msg)
{
  CkChareID vid;
  
#if 0
  CkPrintf("PE %d received migrated element from %d\n",CkMyPe(),msg->from);
#endif
  int index =msg->index;

  elementIDs[index].state = arriving;
  elementIDs[index].pe = CkMyPe();
  elementIDs[index].curHop = msg->hopCount;
  elementIDs[index].cameFrom = msg->from;

  ArrayElementMigrateMessage *new_msg = new ArrayElementMigrateMessage;

  new_msg->index = index;
  new_msg->numElements = numElements;
  new_msg->arrayID = thishandle;
  new_msg->groupID = thisgroup;
  new_msg->arrayPtr = this;
  
  CkCreateChare(elementChareType, elementMigrateType, new_msg, &vid, CkMyPe());

  delete msg;
}

void Array1D::RecvMigratedElementID(int index, ArrayElement *elem,
                                    CkChareID handle)
{
#if 0
  CkPrintf("PE %d index %d receiving migrated element handle %d\n",
    CkMyPe(),index,handle);
#endif
  elementIDs[index].state = here;
  elementIDs[index].element = elem;
  elementIDs[index].elementHandle = handle;

  ArrayElementAckMessage *ack_msg = new ArrayElementAckMessage;

  ack_msg->hopCount = elementIDs[index].curHop;
  ack_msg->index = index;
  ack_msg->arrivedAt = elementIDs[index].pe;
  ack_msg->handle = elementIDs[index].elementHandle;
  ack_msg->deleteElement = true;

  CProxy_Array1D arr(thisgroup);
  arr.AckMigratedElement(ack_msg, elementIDs[index].cameFrom);
  
  if (elementIDs[index].cameFrom != elementIDs[index].originalPE) {
    ack_msg = new ArrayElementAckMessage;

    ack_msg->hopCount = elementIDs[index].curHop;
    ack_msg->index = index;
    ack_msg->arrivedAt = elementIDs[index].pe;
    ack_msg->handle = elementIDs[index].elementHandle;
    ack_msg->deleteElement = false;

    arr.AckMigratedElement(ack_msg, elementIDs[index].originalPE);
  }
  numLocalElements++;
}

void Array1D::AckMigratedElement(ArrayElementAckMessage *msg)
{
  int index = msg->index;

#if 0
  CkPrintf("PE %d Message acknowledged hop=%d curHop=%d\n",
    CkMyPe(),msg->hopCount,elementIDs[index].curHop);
#endif

  if (msg->hopCount > elementIDs[index].curHop) {
    if (msg->deleteElement) {
      ArrayElementExitMessage *exitmsg = new ArrayElementExitMessage;
#if 0
      CkPrintf("I want to delete the element %d\n",index);
#endif
      CProxy_ArrayElement elem(elementIDs[index].elementHandle);
      elem.exit(exitmsg);
    }
    elementIDs[index].pe = msg->arrivedAt;
    elementIDs[index].state = at;
    elementIDs[index].elementHandle = msg->handle;
  } else if (msg->hopCount <= elementIDs[index].curHop) {
    CkPrintf("PE %d STALE Message acknowledged hop=%d curHop=%d\n",
      CkMyPe(),msg->hopCount,elementIDs[index].curHop);
    
  }
  delete msg;
}


ArrayElement::ArrayElement(ArrayElementCreateMessage *msg)
{
  numElements = msg->numElements;
  arrayChareID = msg->arrayID;
  arrayGroupID = msg->groupID;
  thisArray = msg->arrayPtr;
  thisIndex = msg->index;
  delete msg;
}

ArrayElement::ArrayElement(ArrayElementMigrateMessage *msg)
{
  numElements = msg->numElements;
  arrayChareID = msg->arrayID;
  arrayGroupID = msg->groupID;
  thisArray = msg->arrayPtr;
  thisIndex = msg->index;
#if 0
  CkPrintf("ArrayElement:%d Receiving migrated element %d\n",
    CkMyPe(),thisIndex,numElements,
    thisArray,CLocalBranch(Array1D,arrayGroupID));
#endif
  delete msg;
}

void ArrayElement::finishConstruction(void)
{
  //  CkPrintf("Finish Constructor registering %d,%d\n",thisIndex,thishandle);
  thisArray->RecvElementID(thisIndex, this, thishandle);
}

void ArrayElement::finishMigration(void)
{
#if 0
  CkPrintf("Finish Migration registering %d,%d\n",thisIndex,thishandle);
#endif
  thisArray->RecvMigratedElementID(thisIndex, this, thishandle);
}

void ArrayElement::migrate(int where)
{
#if 0
  CkPrintf("Migrating element %d to %d\n",thisIndex,where);
#endif
  if (where != CkMyPe())
    thisArray->migrateMe(thisIndex,where);
#if 0
  else 
    CkPrintf("PE %d I won't migrating element %d to myself\n", where,thisIndex);
#endif

}

int ArrayElement::packsize(void)
{ 
  CkPrintf("ArrayElement::packsize not defined!\n");
  return 0;
}

void ArrayElement::pack(void *pack)
{ 
  CkPrintf("ArrayElement::pack not defined!\n");
}

void ArrayElement::exit(ArrayElementExitMessage *msg)
{
  delete msg;
#if 0
  CkPrintf("ArrayElement::exit exiting %d\n",thisIndex);
#endif
  delete this;
}

ArrayMap::ArrayMap(ArrayMapCreateMessage *msg)
{
  CkPrintf("PE %d creating ArrayMap\n",CkMyPe());
  arrayChareID = msg->arrayID;
  arrayGroupID = msg->groupID;
  array = CProxy_Array1D::ckLocalBranch(arrayGroupID);
  numElements = msg->numElements;

  delete msg;
}

void ArrayMap::finishConstruction(void)
{
  array->RecvMapID(this, thishandle, thisgroup);
}

RRMap::RRMap(ArrayMapCreateMessage *msg) : ArrayMap(msg)
{
  CkPrintf("PE %d creating RRMap for %d elements\n",CkMyPe(),numElements);

  finishConstruction();
}

RRMap::~RRMap()
{
  CkPrintf("Bye from RRMap\n");
}

int RRMap::procNum(int element)
{
  return ((element+1) % CkNumPes());
}

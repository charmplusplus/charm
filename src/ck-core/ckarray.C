#include "ckarray.h"
#include "CkArray.def.h"

void *
ArrayMigrateMessage::alloc(int msgnum,int size,int *array,int priobits)
{
  int totalsize;
  totalsize = size + array[0]*sizeof(char) + 8;
  // CkPrintf("Allocating %d %d %d\n",msgnum,totalsize,priobits);
  ArrayMigrateMessage *newMsg = (ArrayMigrateMessage *)
    CkAllocMsg(msgnum,totalsize,priobits);
  // CkPrintf("Allocated %d\n",newMsg);
  newMsg->elementData = (char *)newMsg + ALIGN8(size);
  return (void *) newMsg;
}
  
void *
ArrayMigrateMessage::pack(ArrayMigrateMessage* in)
{
  /*
  CkPrintf("%d:Packing %d %d %d\n",CkMyPe(),in->from,in->index,in->elementSize);
  */
  in->elementData = (void*)((char*)in->elementData-(char *)&(in->elementData));
  return (void*) in;
}

ArrayMigrateMessage* 
ArrayMigrateMessage::unpack(void *in)
{
  ArrayMigrateMessage *me = new (in) ArrayMigrateMessage;
  /*
  CkPrintf("PE %d Unpacking me=%d from=%d index=%d elementSize=%d\n",
    CkMyPe(),me,me->from,me->index,me->elementSize);
  */
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

  return group;
}

Array1D::Array1D(ArrayCreateMessage *msg)
{
  numElements = msg->numElements;
  elementChareType = msg->elementChareType;
  elementConstType = msg->elementConstType;
  elementMigrateType = msg->elementMigrateType;

  if (CkMyPe()==0) {
    ArrayMapCreateMessage *mapMsg = new ArrayMapCreateMessage;
    mapMsg->numElements = numElements;
    mapMsg->arrayID = thishandle;
    mapMsg->groupID = thisgroup;
    CkCreateGroup(msg->mapChareType,msg->mapConstType,mapMsg,-1,0);
  }
  /*
  CkPrintf("Array1D constructed\n");
  */
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
  for(i=0; i < numElements; i++)
  {
    elementIDs[i].originalPE = elementIDs[i].pe = map->procNum(i);
    elementIDs[i].curHop = 0;
    if (elementIDs[i].pe != CkMyPe())
    {
      elementIDs[i].state = at;
      elementIDs[i].element = NULL;
    }
    else
    {
      elementIDs[i].state = creating;
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

  /*
  if (elementIDsReported == numLocalElements)
    CkPrintf("PE %d all elements reported in\n",CkMyPe());
  */
}

void Array1D::send(ArrayMessage *msg, int index, EntryIndexType ei)
{
  msg->destIndex = index;
  msg->entryIndex = ei;
  if (elementIDs[index].state == here) {
    // CkPrintf("PE %d sending local message to index %d\n",CkMyPe(),index);
    CkSendMsg(ei,msg,&elementIDs[index].elementHandle);
  } else if (elementIDs[index].state == moving_to) {
    // CkPrintf("PE %d sending message to migrating index %d on PE %d\n",
      // CkMyPe(),index,elementIDs[index].pe);
    CProxy_Array1D arr(thisgroup);
    arr.RecvForElement(msg, elementIDs[index].pe);
  } else if (elementIDs[index].state == arriving) {
    // CkPrintf("PE %d sending message for index %d to myself\n",
      // CkMyPe(),index);
    CProxy_Array1D arr(thisgroup);
    arr.RecvForElement(msg, CkMyPe());
  } else {
    // CkPrintf("PE %d sending message to index %d on original PE %d\n",
      // CkMyPe(),index,elementIDs[index].originalPE);
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
  /*
  CkPrintf("PE %d RecvForElement sending to index %d\n",CkMyPe(),msg->destIndex);
  */
  if (elementIDs[msg->destIndex].state == here) {
    // CkPrintf("PE %d DELIVERING index %d RecvForElement state %d\n",
      // CkMyPe(),msg->destIndex,elementIDs[msg->destIndex].state);
    CkSendMsg(msg->entryIndex,msg,&elementIDs[msg->destIndex].elementHandle);
  } else if (elementIDs[msg->destIndex].state == at) {
    // CkPrintf("PE %d Sending to SELF index %d RecvForElement state %d\n",
      // CkMyPe(),msg->destIndex,elementIDs[msg->destIndex].state);
    CProxy_Array1D arr(thisgroup);
    arr.RecvForElement(msg, elementIDs[msg->destIndex].pe);
  } else {
    // CkPrintf("PE %d Sending to SELF index %d RecvForElement state %d\n",
      // CkMyPe(),msg->destIndex,elementIDs[msg->destIndex].state);
    CProxy_Array1D arr(thisgroup);
    arr.RecvForElement(msg, elementIDs[msg->destIndex].originalPE);
  }
}

void Array1D::migrateMe(int index, int where)
{
  int bufSize = elementIDs[index].element->packsize();

  ArrayMigrateMessage *msg = new (&bufSize, 0) ArrayMigrateMessage;

  msg->index = index;
  msg->from = CkMyPe();
  msg->elementSize = bufSize;
  msg->hopCount = elementIDs[index].curHop + 1;
  elementIDs[index].element->pack(msg->elementData);
  elementIDs[index].state = moving_to;
  elementIDs[index].pe = where;
  numLocalElements--;
  CProxy_Array1D arr(thisgroup);
  arr.RecvMigratedElement(msg, where);
}

void Array1D::RecvMigratedElement(ArrayMigrateMessage *msg)
{
  CkChareID vid;
  
  int index =msg->index;

  elementIDs[index].state = arriving;
  elementIDs[index].pe = CkMyPe();
  elementIDs[index].curHop = msg->hopCount;
  elementIDs[index].cameFrom = msg->from;
  elementIDs[index].migrateMsg = msg;

  ArrayElementMigrateMessage *new_msg = new ArrayElementMigrateMessage;

  new_msg->index = index;
  new_msg->numElements = numElements;
  new_msg->arrayID = thishandle;
  new_msg->groupID = thisgroup;
  new_msg->arrayPtr = this;
  new_msg->packData = msg->elementData;
  
  CkCreateChare(elementChareType, elementMigrateType, new_msg, &vid, CkMyPe());
}

void Array1D::RecvMigratedElementID(int index, ArrayElement *elem,
                                    CkChareID handle)
{
  // CkPrintf("PE %d index %d receiving migrated element handle %d\n",
    // CkMyPe(),index,handle);
  elementIDs[index].state = here;
  elementIDs[index].element = elem;
  elementIDs[index].elementHandle = handle;
  delete elementIDs[index].migrateMsg;

  ArrayElementAckMessage *ack_msg = new ArrayElementAckMessage;

  ack_msg->hopCount = elementIDs[index].curHop;
  ack_msg->index = index;
  ack_msg->arrivedAt = elementIDs[index].pe;
  ack_msg->handle = elementIDs[index].elementHandle;
  ack_msg->deleteElement = 1;

  CProxy_Array1D arr(thisgroup);
  arr.AckMigratedElement(ack_msg, elementIDs[index].cameFrom);
  
  if (elementIDs[index].cameFrom != elementIDs[index].originalPE) {
    ack_msg = new ArrayElementAckMessage;

    ack_msg->hopCount = elementIDs[index].curHop;
    ack_msg->index = index;
    ack_msg->arrivedAt = elementIDs[index].pe;
    ack_msg->handle = elementIDs[index].elementHandle;
    ack_msg->deleteElement = 0;

    arr.AckMigratedElement(ack_msg, elementIDs[index].originalPE);
  }
  numLocalElements++;
}

void Array1D::AckMigratedElement(ArrayElementAckMessage *msg)
{
  int index = msg->index;

  // CkPrintf("PE %d Message acknowledged hop=%d curHop=%d\n",
    // CkMyPe(),msg->hopCount,elementIDs[index].curHop);

  if (msg->hopCount > elementIDs[index].curHop) {
    if (msg->deleteElement) {
      ArrayElementExitMessage *exitmsg = new ArrayElementExitMessage;
      // CkPrintf("I want to delete the element %d\n",index);
      CProxy_ArrayElement elem(elementIDs[index].elementHandle);
      elem.exit(exitmsg);
    }
    elementIDs[index].pe = msg->arrivedAt;
    elementIDs[index].state = at;
    elementIDs[index].elementHandle = msg->handle;
  } else if (msg->hopCount <= elementIDs[index].curHop) {
    // CkPrintf("PE %d STALE Message acknowledged hop=%d curHop=%d\n",
      // CkMyPe(),msg->hopCount,elementIDs[index].curHop);
    
  }
  delete msg;
}


ArrayElement::ArrayElement(ArrayElementCreateMessage *msg)
{
  numElements = msg->numElements;
  arrayChareID = msg->arrayID;
  arrayGroupID = msg->groupID;
  thisArray = msg->arrayPtr;
  thisAID.setAid(thisArray->ckGetGroupId());
  thisAID._elem = (-1);
  thisIndex = msg->index;
}

ArrayElement::ArrayElement(ArrayElementMigrateMessage *msg)
{
  numElements = msg->numElements;
  arrayChareID = msg->arrayID;
  arrayGroupID = msg->groupID;
  thisArray = msg->arrayPtr;
  thisAID.setAid(thisArray->ckGetGroupId());
  thisAID._elem = (-1);
  thisIndex = msg->index;
}

void ArrayElement::finishConstruction(void)
{
  thisArray->RecvElementID(thisIndex, this, thishandle);
}

void ArrayElement::finishMigration(void)
{
  // CkPrintf("Finish Migration registering %d,%d\n",thisIndex,thishandle);
  thisArray->RecvMigratedElementID(thisIndex, this, thishandle);
}

void ArrayElement::migrate(int where)
{
  // CkPrintf("Migrating element %d to %d\n",thisIndex,where);
  if (where != CkMyPe())
    thisArray->migrateMe(thisIndex,where);
/*
  else 
    CkPrintf("PE %d I won't migrating element %d to myself\n", where,thisIndex);
*/
}

int ArrayElement::packsize(void)
{ 
  // CkPrintf("ArrayElement::packsize not defined!\n");
  return 0;
}

void ArrayElement::pack(void *pack)
{ 
  // CkPrintf("ArrayElement::pack not defined!\n");
}

void ArrayElement::exit(ArrayElementExitMessage *msg)
{
  delete msg;
  // CkPrintf("ArrayElement::exit exiting %d\n",thisIndex);
  delete this;
}

ArrayMap::ArrayMap(ArrayMapCreateMessage *msg)
{
  // CkPrintf("PE %d creating ArrayMap\n",CkMyPe());
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
  // CkPrintf("PE %d creating RRMap for %d elements\n",CkMyPe(),numElements);

  finishConstruction();
}

RRMap::~RRMap()
{
  // CkPrintf("Bye from RRMap\n");
}

int RRMap::procNum(int element)
{
  return ((element+1) % CkNumPes());
}

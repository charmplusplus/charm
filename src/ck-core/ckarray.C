#include "charm++.h"
#include "ck.h"
#include "CkArray.def.h"
#include "init.h"

CkGroupID _RRMapID;

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
  me->elementData = (char *)&(me->elementData) + (size_t)me->elementData;
  return me;
}

CkGroupID Array1D::CreateArray(int numElements,
                               CkGroupID mapID,
                               ChareIndexType elementChare,
                               EntryIndexType elementConstructor,
                               EntryIndexType elementMigrator)
{
  CkGroupID group;

  ArrayCreateMessage *msg = new ArrayCreateMessage;

  msg->numElements = numElements;
  msg->mapID = mapID;
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

  ArrayMapRegisterMessage *mapMsg = new ArrayMapRegisterMessage;
  mapMsg->numElements = numElements;
  mapMsg->arrayID = thishandle;
  mapMsg->groupID = thisgroup;

  bufferedForElement = new PtrQ();
  bufferedMigrated = new PtrQ();
  map = 0;

  ArrayMap *mapPtr = (ArrayMap *)CkLocalBranch(msg->mapID);

  if(mapPtr==0) {
    CProxy_ArrayMap pmap(msg->mapID);
    pmap.registerArray(mapMsg, CkMyPe());
  } else {
    mapPtr->registerArray(mapMsg);
  }

  delete msg;

  /*
  CkPrintf("Array1D constructed\n");
  */
}

void Array1D::RecvMapID(ArrayMap *mPtr, int mHandle)
{
  map = mPtr;
  mapHandle = mHandle;

  elementIDs = new ElementIDs[numElements];
  _MEMCHECK(elementIDs);
  elementIDsReported = 0;
  numLocalElements=0;
  int i;
  for(i=0; i < numElements; i++)
  {
    elementIDs[i].originalPE = elementIDs[i].pe = map->procNum(mapHandle, i);
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

      ArrayElementCreateMessage *msg = new ArrayElementCreateMessage;
      
      msg->numElements = numElements;
      msg->arrayID = thishandle;
      msg->groupID = thisgroup;
      msg->arrayPtr = this;
      msg->index = i;
      CkCreateChare(elementChareType, elementConstType, msg, 0, CkMyPe());
    }
  }
  CProxy_Array1D arr(thisgroup);
  ArrayMessage *amsg;
  while((amsg = (ArrayMessage *) bufferedForElement->deq())) {
    arr.RecvForElement(amsg, CkMyPe());
  }
  delete bufferedForElement;
  ArrayMigrateMessage *mmsg;
  while((mmsg = (ArrayMigrateMessage *) bufferedMigrated->deq())) {
    arr.RecvMigratedElement(mmsg, CkMyPe());
  }
  delete bufferedMigrated;
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

static int serial_num = 0;

void Array1D::send(ArrayMessage *msg, int index, EntryIndexType ei)
{
  msg->destIndex = index;
  msg->entryIndex = ei;
  msg->hopCount = 0;
  msg->serial_num = 1000*serial_num+CkMyPe();
  serial_num++;

  if (elementIDs[index].state == here) {
#if 0
    CPrintf("PE %d sending local message to index %d\n",CMyPe(),index);
#endif
    CProxy_Array1D arr(thisgroup);
    arr.RecvForElement(msg, CkMyPe());
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
 } else if (elementIDs[index].state == at) {
#if 0
    CPrintf("PE %d AT message to index %d on original PE %d\n",
            CMyPe(),elementIDs[index].state,index,
            elementIDs[index].pe);
#endif
    CProxy_Array1D arr(thisgroup);
    arr.RecvForElement(msg, elementIDs[index].pe);
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
  if(!map) {
    bufferedForElement->enq((void *)msg);
    return;
  }
  msg->hopCount++;
  if (elementIDs[msg->destIndex].state == here) {
    // CkPrintf("PE %d DELIVERING index %d RecvForElement state %d\n",
    // CkMyPe(),msg->destIndex,elementIDs[msg->destIndex].state);
    // CkSendMsg(msg->entryIndex,msg,&elementIDs[msg->destIndex].elementHandle);
    //    register int epIdx = env->getEpIdx();
    register int epIdx = msg->entryIndex;
    //    register void *obj = env->getObjPtr();
    CkChareID handle = elementIDs[msg->destIndex].elementHandle;
    register void *obj = handle.objPtr;
    _entryTable[epIdx]->call(msg, obj);

    //    EP_STRUCT *epinfo = CsvAccess(EpInfoTable)+msg->entryIndex;
    //    CHARE_BLOCK *chareblock = GetID_chareBlockPtr(handle);
    //    void *current_usr = msg;
    //USER_MSG_PTR(env);
    //    callep(epinfo->function,msg,chareblock->chareptr);
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
  if(!map) {
    bufferedMigrated->enq(msg);
    return;
  }
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
  
  CkCreateChare(elementChareType, elementMigrateType, new_msg, 0, CkMyPe());
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
  elementIDs[index].migrateMsg = NULL;

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
  thisAID._setAid(thisArray->ckGetGroupId());
  thisAID._elem = (-1);
  thisAID._setChare(0);
  thisIndex = msg->index;
}

ArrayElement::ArrayElement(ArrayElementMigrateMessage *msg)
{
  numElements = msg->numElements;
  arrayChareID = msg->arrayID;
  arrayGroupID = msg->groupID;
  thisArray = msg->arrayPtr;
  thisAID._setAid(thisArray->ckGetGroupId());
  thisAID._elem = (-1);
  thisAID._setChare(0);
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

void ArrayElement::exit(ArrayElementExitMessage *msg)
{
  delete msg;
  // CkPrintf("ArrayElement::exit exiting %d\n",thisIndex);
  delete this;
}

RRMap::RRMap(void)
{
  // CkPrintf("PE %d creating RRMap for %d elements\n",CkMyPe(),numElements);
  arrayVec = new PtrVec();
}

int RRMap::procNum(int /*arrayHdl*/, int element)
{
  return ((element+1) % CkNumPes());
}

void RRMap::registerArray(ArrayMapRegisterMessage *msg)
{
  int hdl = arrayVec->length();
  arrayVec->insert(hdl, (void *)(msg->numElements));
  Array1D* array = (Array1D *) CkLocalBranch(msg->groupID);
  delete msg;
  array->RecvMapID(this, hdl);
}


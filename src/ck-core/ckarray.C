#include "charm++.h"
#include "register.h"
#include "ckarray.h"
#include "ck.h"
#include "CkArray.def.h"
#include "init.h"

#if CMK_LBDB_ON
#include "LBDatabase.h"
#endif // CMK_LBDB_ON

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
#if CMK_LBDB_ON
  msg->loadbalancer = lbdb;
#endif
  group = CProxy_Array1D::ckNew(msg);

  return group;
}

Array1D::Array1D(ArrayCreateMessage *msg)
{
  numElements = msg->numElements;
  elementChareType = msg->elementChareType;
  elementConstType = msg->elementConstType;
  elementMigrateType = msg->elementMigrateType;

#if CMK_LBDB_ON
  the_lbdb = CProxy_LBDatabase(msg->loadbalancer).ckLocalBranch();
  if (the_lbdb == 0)
    CkPrintf("[%d] LBDatabase not created?\n",CkMyPe());

  //  int iii=0;
  //  CkPrintf("%d Hi from Array1D[%d]\n",iii++,CkMyPe());

  // Register myself as an array manager
  LDOMid myId;
  myId.id = (int)thisgroup;

  LDCallbacks myCallbacks;
  myCallbacks.migrate = staticMigrate;
  myCallbacks.setStats = staticSetStats;
  myCallbacks.queryEstLoad = staticQueryLoad;
  
  myHandle = the_lbdb->RegisterOM(myId,this,myCallbacks);
#endif

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

#if CMK_LBDB_ON
  // Tell the lbdb that I'm registering objects, until I'm done
  // registering them.
  the_lbdb->RegisteringObjects(myHandle);
#endif

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
#if CMK_LBDB_ON
  if (numLocalElements==0)
    the_lbdb->DoneRegisteringObjects(myHandle);
#endif
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

void Array1D::RecvElementID(int index, ArrayElement *elem,
			    CkChareID handle, CmiBool use_local_barrier)
{
  elementIDs[index].state = here;
  elementIDs[index].element = elem;
  elementIDs[index].elementHandle = handle;
  elementIDsReported++;

  //  if (elementIDsReported == numLocalElements)
  //    CkPrintf("PE %d all elements reported in\n",CkMyPe());

#if CMK_LBDB_ON
  // Register the object with the load balancer
  LDObjid elemID;
  elemID.id[0] = index;
  elemID.id[1] = elemID.id[2] = elemID.id[3] = 0;

  elementIDs[index].ldHandle = the_lbdb->RegisterObj(myHandle,elemID,0,1);

  if (use_local_barrier)
    RegisterElementForSync(index);
  else
    elementIDs[index].uses_barrier = CmiFalse;

  if (elementIDsReported == numLocalElements)
    the_lbdb->DoneRegisteringObjects(myHandle);
    
#endif

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
  delete msg;
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

    register int epIdx = msg->entryIndex;
    CkChareID handle = elementIDs[msg->destIndex].elementHandle;
    register void *obj = handle.objPtr;

#if CMK_LBDB_ON
    const int index = msg->destIndex;
    the_lbdb->ObjectStart(elementIDs[index].ldHandle);
    // Can't use msg after call(): The user may delete it!
    _entryTable[epIdx]->call(msg, obj);
    the_lbdb->ObjectStop(elementIDs[index].ldHandle);
#else
    _entryTable[epIdx]->call(msg, obj);
#endif

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
  //  CkPrintf("[%d] Element %d migrating to %d\n",CkMyPe(),index,where);

  ArrayMigrateMessage *msg = new (&bufSize, 0) ArrayMigrateMessage;

  msg->index = index;
  msg->from = CkMyPe();
  msg->elementSize = bufSize;
  msg->hopCount = elementIDs[index].curHop + 1;
#if CMK_LBDB_ON
  msg->uses_barrier = elementIDs[index].uses_barrier;
#endif

  elementIDs[index].element->pack(msg->elementData);
  elementIDs[index].state = moving_to;
  elementIDs[index].pe = where;

#if CMK_LBDB_ON
  the_lbdb->UnregisterObj(elementIDs[index].ldHandle);
  if (elementIDs[index].uses_barrier)
    the_lbdb->RemoveLocalBarrierClient(elementIDs[index].barrierHandle);
#endif
  numLocalElements--;
  elementIDsReported--;

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

  //  CkPrintf("[%d] Element %d migrated here from %d\n",CkMyPe(),index,msg->from);
#if CMK_LBDB_ON
  elementIDs[index].uses_barrier = msg->uses_barrier;
#endif
   
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

  if (CkMyPe() == elementIDs[index].cameFrom)
    CkPrintf("[%d] Error: Acknowledging element %d migrating to me!\n",
	     CkMyPe(),index);

  CProxy_Array1D arr(thisgroup);
  arr.AckMigratedElement(ack_msg, elementIDs[index].cameFrom);
  //  CkPrintf("[%d] Ack element %d to %d\n",
  //	   CkMyPe(),index, elementIDs[index].cameFrom);
  
  if ( (elementIDs[index].cameFrom != elementIDs[index].originalPE) 
       && (CkMyPe() != elementIDs[index].originalPE) ) {
    ack_msg = new ArrayElementAckMessage;

    ack_msg->hopCount = elementIDs[index].curHop;
    ack_msg->index = index;
    ack_msg->arrivedAt = elementIDs[index].pe;
    ack_msg->handle = elementIDs[index].elementHandle;
    ack_msg->deleteElement = 0;

    arr.AckMigratedElement(ack_msg, elementIDs[index].originalPE);
    //    CkPrintf("[%d] Ack element %d to source %d\n",
    //	     CkMyPe(),index, elementIDs[index].originalPE);
  }
  numLocalElements++;
  elementIDsReported++;

#if CMK_LBDB_ON
  // Register the object with the load balancer
  LDObjid elemID;
  elemID.id[0] = index;
  elemID.id[1] = elemID.id[2] = elemID.id[3] = 0;

  elementIDs[index].ldHandle = the_lbdb->RegisterObj(myHandle,elemID,0,1);

  if (elementIDs[index].uses_barrier)
    RegisterElementForSync(index);

  the_lbdb->Migrated(elementIDs[index].ldHandle);
#endif

}

void Array1D::AckMigratedElement(ArrayElementAckMessage *msg)
{
  int index = msg->index;

  //  CkPrintf("[%d] element %d acknowledged\n",CkMyPe(),index);

  if (msg->hopCount > elementIDs[index].curHop) {
    if (msg->deleteElement) {
      ArrayElementExitMessage *exitmsg = new ArrayElementExitMessage;
      //      CkPrintf("[%d] I want to delete the element %d\n",CkMyPe(),index);
      CProxy_ArrayElement elem(elementIDs[index].elementHandle);
      elem.exit(exitmsg);
      //      CkPrintf("[%d] Element %d deleted\n",CkMyPe(),index);
    }
    elementIDs[index].pe = msg->arrivedAt;
    elementIDs[index].state = at;
    elementIDs[index].elementHandle = msg->handle;
    elementIDs[index].curHop = msg->hopCount;
  } else if (msg->hopCount <= elementIDs[index].curHop) {
    //    CkPrintf("PE %d index %d STALE Message acknowledged hop=%d curHop=%d\n",
    //	     CkMyPe(),index,msg->hopCount,elementIDs[index].curHop);
    
  }
  delete msg;
}

#if CMK_LBDB_ON

void Array1D::staticMigrate(LDObjHandle _h, int _dest)
{
  (static_cast<Array1D*>(_h.omhandle.user_ptr))->Migrate(_h,_dest);
}

void Array1D::staticSetStats(LDOMHandle _h, int _state)
{
  (static_cast<Array1D*>(_h.user_ptr))->SetStats(_h,_state);   
}

void Array1D::staticQueryLoad(LDOMHandle _h)
{
  (static_cast<Array1D*>(_h.user_ptr))->QueryLoad(_h);
}

void Array1D::Migrate(LDObjHandle _h, int _dest)
{
  int id = _h.id.id[0];
  if (elementIDs[id].state != here)
    CkPrintf("%s(%d)[%d]: Migrate error, element not present\n",
	     __FILE__,__LINE__,CkMyPe());
  else
    elementIDs[id].element->migrate(_dest);
  
}

void Array1D::SetStats(LDOMHandle _h, int _state)
{
  CkPrintf("%s(%d)[%d]: SetStats request received\n",
	   __FILE__,__LINE__,CkMyPe());
}

void Array1D::QueryLoad(LDOMHandle _h)
{
  CkPrintf("%s(%d)[%d]: QueryLoad request received\n",
	   __FILE__,__LINE__,CkMyPe());
}

void Array1D::RegisterElementForSync(int index)
{
  //  CkPrintf("[%d] Registering element %d for barrier\n",CkMyPe(),index);
  if (elementIDsReported == 1) { // This is the first element reported
    // If this is a sync array, register a sync callback so I can
    // inform the db when I start registering objects 
    the_lbdb->AddLocalBarrierReceiver(staticRecvAtSync,
				      static_cast<void*>(this));
  }
    
  elementIDs[index].uses_barrier = CmiTrue;  
  elementIDs[index].barrierData.me = this;
  elementIDs[index].barrierData.index = index;

  elementIDs[index].barrierHandle = the_lbdb->
    AddLocalBarrierClient(staticResumeFromSync,
			  static_cast<void*>(&elementIDs[index].barrierData));

}

void Array1D::staticRecvAtSync(void* data)
{
  static_cast<Array1D*>(data)->RecvAtSync();
}

void Array1D::RecvAtSync()
{
  // If all of our elements leave, there won't be anything to
  // call DoneRegisteringObjects();
  the_lbdb->RegisteringObjects(myHandle);
}

void Array1D::staticResumeFromSync(void* data)
{
  ElementIDs::BarrierClientData* barrierData = 
    static_cast<ElementIDs::BarrierClientData*>(data);
  (barrierData->me)->ResumeFromSync(barrierData->index);
}

void Array1D::ResumeFromSync(int index)
{
  the_lbdb->DoneRegisteringObjects(myHandle);

  if (elementIDs[index].state == here)
    elementIDs[index].element->ResumeFromSync();
  else {
    CkPrintf("!!! I'm supposed to resume an element, but it has left !!!\n");
  }
}

void Array1D::AtSync(int index)
{
  the_lbdb->AtLocalBarrier(elementIDs[index].barrierHandle);
}
#endif // CMK_LBDB_ON

ArrayElement::ArrayElement(ArrayElementCreateMessage *msg)
{
  numElements = msg->numElements;
  arrayChareID = msg->arrayID;
  arrayGroupID = msg->groupID;
  thisArray = msg->arrayPtr;
  thisAID._setAid(thisArray->ckGetGroupId());
  thisAID._elem = (-1);
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
  thisIndex = msg->index;
}

void ArrayElement::finishConstruction(CmiBool use_local_barrier)
{
  thisArray->RecvElementID(thisIndex, this, thishandle, use_local_barrier);
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

void ArrayElement::AtSync(void)
{
  //  CkPrintf("Element %d at sync\n",thisIndex);
#if CMK_LBDB_ON
  thisArray->AtSync(thisIndex);
#endif
}

void ArrayElement::exit(ArrayElementExitMessage *msg)
{
  delete msg;
  // CkPrintf("ArrayElement::exit exiting %d\n",thisIndex);
  delete this;
}

#if 0
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
#endif

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


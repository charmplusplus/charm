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

#ifdef CK_ARRAY_REDUCTIONS
//Set reduction state variables
  reductionClient=NULL;
  reductionNo=0;//We'll claim we were just doing reduction number zero...
  reductionFinished=1;//...but it is done now.
  curMsgs=new PtrQ();
  futureBuffer=new PtrQ();
#endif

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
  myCallbacks.migrate = reinterpret_cast<LDMigrateFn>(staticMigrate);
  myCallbacks.setStats = reinterpret_cast<LDStatsFn>(staticSetStats);
  myCallbacks.queryEstLoad =
    reinterpret_cast<LDQueryEstLoadFn>(staticQueryLoad);
  
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

  // Add myself as a local barrier receiver, so I know when I might
  // be registering objects.
  the_lbdb->
    AddLocalBarrierReceiver(reinterpret_cast<LDBarrierFn>(staticRecvAtSync),
 			      static_cast<void*>(this));
  // Also, add a dummy local barrier client, so there will always be
  // something to call DoneRegisteringObjects()
  dummyBarrierHandle = the_lbdb->AddLocalBarrierClient(
    reinterpret_cast<LDResumeFn>(staticDummyResumeFromSync),
    static_cast<void*>(this));

  // Activate the AtSync for this one immediately.  Note, that since
  // we have not yet called DoneRegisteringObjects(), nothing
  // will happen yet.
  CProxy_Array1D(thisgroup).DummyAtSync(CkMyPe());
#endif

  for(i=0; i < numElements; i++) {
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
  msg->from_pe = CkMyPe();
  msg->destIndex = index;
  msg->entryIndex = ei;
  msg->hopCount = 0;
  msg->serial_num = 1000*serial_num+CkMyPe();
  serial_num++;

#if CMK_LBDB_ON
  LDObjid dest;
  dest.id[0] = index; dest.id[1] = dest.id[2] = dest.id[3] = 0;
  
  the_lbdb->Send(myHandle,dest,UsrToEnv(msg)->getTotalsize());

#endif
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
//Brain-dead broadcast algorithm:
// linear loop over array elements.
  for (int i=0;i<numElements;i++)
  {//Send a copy of the original message
    void *newMsg=CkCopyMsg((void **)&msg);
    send((ArrayMessage *)newMsg,i,ei);
  }
  //We've only sent off copies of the message, so
  // delete the original.
  CkFreeMsg((void *)msg);
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
  const int index = msg->destIndex;
  if (elementIDs[index].state == here) {
    // CkPrintf("PE %d DELIVERING index %d RecvForElement state %d\n",
    // CkMyPe(),msg->destIndex,elementIDs[msg->destIndex].state);

    register int epIdx = msg->entryIndex;
    CkChareID handle = elementIDs[index].elementHandle;
    register void *obj = handle.objPtr;

    if (msg->hopCount > 1) {
      //      CkPrintf("[%d] Sending update to %d for %d\n",
      //	       CkMyPe(),msg->from_pe,msg->destIndex);
      ArrayElementUpdateMessage* update = new ArrayElementUpdateMessage;
      update->index = index;
      update->hopCount = elementIDs[index].curHop;
      update->pe = CkMyPe();
      CProxy_Array1D arr(thisgroup);
      arr.UpdateLocation(update, msg->from_pe);
    }

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
  elementIDs[index].state = moving_to;
  elementIDs[index].pe = where;

  //  CkPrintf("[%d] Element %d migrating to %d\n",CkMyPe(),index,where);

  int bufSize = elementIDs[index].element->packsize();
  ArrayMigrateMessage *msg = new (&bufSize, 0) ArrayMigrateMessage;
  msg->index = index;
  msg->nContributions=elementIDs[index].element->nContributions;
  msg->from = CkMyPe();
  msg->elementSize = bufSize;
  msg->hopCount = elementIDs[index].curHop + 1;
#if CMK_LBDB_ON
  msg->uses_barrier = elementIDs[index].uses_barrier;
#endif

  elementIDs[index].element->pack(msg->elementData);

#if CMK_LBDB_ON
  the_lbdb->UnregisterObj(elementIDs[index].ldHandle);
  if (elementIDs[index].uses_barrier)
    the_lbdb->RemoveLocalBarrierClient(elementIDs[index].barrierHandle);
#endif
  numLocalElements--;
  elementIDsReported--;

  CProxy_Array1D arr(thisgroup);
  arr.RecvMigratedElement(msg, where);

#ifdef CK_ARRAY_REDUCTIONS
  tryEndReduction();//This may have been the guy we were waiting on...
#endif
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
  new_msg->nContributions=msg->nContributions;
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
      if(elementIDs[index].element == 0) {
        CkError("Already deleted element %d\n", index);
        abort();
      }
      ArrayElementExitMessage *exitmsg = new ArrayElementExitMessage;
      //      CkPrintf("[%d] I want to delete the element %d\n",CkMyPe(),index);
 //OSL 2/20/2000-- replaced CProxy_ArrayElement(CkChareId) constructor
      // with this horrible hack:
      CkSendMsg(CProxy_ArrayElement::__idx_exit_ArrayElementExitMessage,
		exitmsg,
		&elementIDs[index].elementHandle);
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

void Array1D::UpdateLocation(ArrayElementUpdateMessage* msg)
{
  const int index = msg->index;
  ElementIDs* elementID = &elementIDs[index];
  
  if (elementID->state == at) {
    const int msghop = msg->hopCount;
    if (msghop > elementID->curHop) {
      //CkPrintf("[%d] Receiving update %d from hop %d:%d to hop %d:%d\n",
      //       CkMyPe(),index,elementID->curHop,elementID->pe,
      //       msghop,msg->pe);
      elementID->pe = msg->pe;
    }
  }
  delete msg;
}

void Array1D::DummyAtSync()
{
  //  CkPrintf("[%d] DummyAtSync called\n",CkMyPe());
  // Since this is in the .ci file, and I can't use the #ifdef there,
  // I'll have to always declare the function, but only call the
  // load balancer if LBDB is on
#if CMK_LBDB_ON
  the_lbdb->AtLocalBarrier(dummyBarrierHandle);
#endif
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
//   if (elementIDsReported == 1) { // This is the first element reported
//     // If this is a sync array, register a sync callback so I can
//     // inform the db when I start registering objects 
//     the_lbdb->
//       AddLocalBarrierReceiver(reinterpret_cast<LDBarrierFn>(staticRecvAtSync),
// 			      static_cast<void*>(this));

//     // Also, add a dummy local barrier client, so there will always be
//     // something to call DoneRegisteringObjects()
//     the_lbdb->AddLocalBarrierClient(
//       reinterpret_cast<LDResumeFn>(staticDummyResumeFromSync),
//       static_cast<void*>(this));
//     CProxy_Array1D(thisgroup).dummyAtSync();
//   }
    
  elementIDs[index].uses_barrier = CmiTrue;  
  elementIDs[index].barrierData.me = this;
  elementIDs[index].barrierData.index = index;

  elementIDs[index].barrierHandle = the_lbdb->
    AddLocalBarrierClient(reinterpret_cast<LDResumeFn>(staticResumeFromSync),
			  static_cast<void*>(&elementIDs[index].barrierData));

}

void Array1D::staticDummyResumeFromSync(void* data)
{
  Array1D* me = static_cast<Array1D*>(data);
  me->DummyResumeFromSync();
}

void Array1D::DummyResumeFromSync()
{
  //  CkPrintf("[%d] DummyResume called\n",CkMyPe());
  the_lbdb->DoneRegisteringObjects(myHandle);
  CProxy_Array1D(thisgroup).DummyAtSync(CkMyPe());
  //  CkPrintf("[%d] DummyResume done\n",CkMyPe());
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
  thisArrayID = thisAID;
  thisIndex = msg->index;
  nContributions=0;
}

ArrayElement::ArrayElement(ArrayElementMigrateMessage *msg)
{
  numElements = msg->numElements;
  arrayChareID = msg->arrayID;
  arrayGroupID = msg->groupID;
  thisArray = msg->arrayPtr;
  thisAID._setAid(thisArray->ckGetGroupId());
  thisAID._elem = (-1);
  thisArrayID = thisAID;
  thisIndex = msg->index;
  nContributions=msg->nContributions;
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

#ifdef CK_ARRAY_REDUCTIONS
/////////////////////////////////////////////////////////////////////////////
/////////////////////// Array Reduction Implementation //////////////////////
// Orion Sky Lawlor, olawlor@acm.org, 11/15/1999
// Replaced fixed-sized arrays with PtrQs: 2/24/2000

//Debugging defines-- set this to 1 for tons of (useless?) debugging output.
#define GIVE_DEBUGGING_OUTPUT 0
#if GIVE_DEBUGGING_OUTPUT
#define RED_DEB(x) CkPrintf x
#else
#define RED_DEB(x) /*empty*/
#endif
#define RA "PE_%d/reduction %d%s: "
#define RB ,CkMyPe(),reductionNo,reductionFinished?"d":"s"

#include "ckarray_reductions.C" //Include reduction implementations


//Call contribute to add your contribution to a new global reduction.
// The array BOC will keep a copy the data. reducer must be the same on all PEs.
void ArrayElement::contribute(int dataSize,void *data,ArrayReductionFn reducer)
{
	RED_DEB(("index %d sends contribution %d to node %d\n",thisIndex,nContributions+1,CkMyPe()));
	ArrayReductionMessage *msg=ArrayReductionMessage::buildNew(dataSize,data);
	msg->source=thisIndex;
	msg->reductionNo=(++nContributions);
	msg->reducer=reducer;
	thisArray->addContribution(msg);
}

void Array1D::registerReductionHandler(ArrayReductionClientFn handler,void *param)
{
	reductionClient=handler;
	reductionClientParam=param;
}

void Array1D::RecvReductionMessage(ArrayReductionMessage *msg)
{
	RED_DEB((RA"recv'd remote contribution\n"RB));
	addContribution(msg);
	//No delete needed because addContribution keeps msg.
}

void Array1D::ReductionHeartbeat(ArrayReductionHeartbeatM *msg)
{
	RED_DEB((RA"recv'd heartbeat %d\n"RB,msg->currentReduction));
	tryBeginReduction(msg->currentReduction);
	tryEndReduction();//Try to finish immediately
	delete msg;
}


//This is called by ArrayElement::contribute() and RcvReductionMessage.
// reducer may be NULL. The given message is kept by Array1D.
void Array1D::addContribution(ArrayReductionMessage *m)
{
	int dubReduction=2*reductionNo+reductionFinished;
	RED_DEB((RA"recv'd contribution from %d for reduction %d\n"RB,m->source,m->reductionNo));
	
	if (m->reductionNo*2==dubReduction)
		addCurrentContribution(m);
	else if (m->reductionNo*2>dubReduction)
	{//We haven't dealt with this reduction number yet--
		if (reductionFinished && (m->reductionNo==reductionNo+1))
		{//We're ready to begin a new reduction
		        tryBeginReduction(m->reductionNo);
			addCurrentContribution(m);
		}
		else 
		{//A prior reduction is in progress-- buffer this message
			RED_DEB((RA"recv'd early contribution from %d\n"RB,m->source));
		     	futureBuffer->enq((void *)m);
      		}
	} else
	{//This message is for a reduction we already finished!
		if (CkMyPe()==0)
		{//This is an error in the reduction library
		  if (m->getSources()!=0)//Don't worry if it's just a heartbeat message
			CkAbort("ERROR! Root node recieved a message for a reduction which is already complete!\n");
		  delete m;
		} else //This message is late because of migration--
		{//forward it straight to the root
			RED_DEB((RA"recv'd late contribution from %d\n"RB,m->source));
			CProxy_Array1D arr(thisgroup);
			arr.RecvReductionMessage(m, 0);
			return;
		}
	}
}

//Add a contribution to the current reduction
void Array1D::addCurrentContribution(ArrayReductionMessage *m)
{
	RED_DEB((RA"recv'd %dth contribution\n"RB,curMsgs->length()));
	if (m->getSources()!=0)
		curMsgs->enq(m);
	if (!m->isSingleton())//This was a remote message
		nRemote++;
	nContributions+=m->getSources();
	
	tryEndReduction();
}



int i_min(int a,int b) {if (a<b) return a; else return b;}

//BeginReduction is called to start each reduction
//It computes number of remote messages and increments reductionNo.
// It will do nothing if reduction atLeast is already started.
typedef ArrayReductionMessage* ArrayReductionMessagePtr;
void Array1D::tryBeginReduction(int atLeast)
{
	if (reductionFinished && (reductionNo<atLeast))
	{
	  reductionFinished=0;
	  reductionNo++;//Start a new reduction
	
	  //This is the PE number of my first child
	  int firstKid=CkMyPe()*ARRAY_RED_TREE+1;

	  if (firstKid<CkNumPes()) //We have children-- expect 1 message from each
	  {//Let the kids know we're waiting for them (in case they don't know)
	    int lastKid=i_min(firstKid+ARRAY_RED_TREE,CkNumPes());
	    CProxy_Array1D arr(thisgroup);
	    for (int kid=firstKid;kid<lastKid;kid++)
	      arr.ReductionHeartbeat(new ArrayReductionHeartbeatM(reductionNo),kid);
	    expectedRemote=lastKid-firstKid;
	  }
	  else	
	    expectedRemote=0;//We are a leaf in the reduction tree.
	  
	  nContributions=nRemote=0;
	  RED_DEB((RA"starting reduction. expecting %d remote messages\n"RB,expectedRemote));
	}
}

//How many messages do we still need from local elements?
int Array1D::expectedLocalMessages(void)
{
	int i,nExpected=0;
	//We expect one message from each local who hasn't yet contributed
	for (i=0;i<numElements;i++)
	{
		if (elementIDs[i].state==here)
			if (elementIDs[i].element->nContributions<reductionNo)
				nExpected++;
		if (elementIDs[i].state==creating)
			nExpected++;
	}
	
	//We also expect one message from each occupant of the future file
	for (i=0;i<futureBuffer->length();i++)
	{
	  ArrayReductionMessage *f=(ArrayReductionMessage *)futureBuffer->deq();
	  if (f->reductionNo==reductionNo)/*<-is this reduction*/
	    if (f->isSingleton()) /*<-is local*/
	      nExpected++;
	  futureBuffer->enq((void *)f);
	}
	return nExpected;
}

void Array1D::tryEndReduction(void)//Check if all messages in, and if so, finish.
{
	if (!reductionFinished)
	{
	  if (CkMyPe()==0) //We are root-- must have *all* messages
	  {
	    if (nContributions==numElements) 
	      endReduction();//Every element has contributed
	  }
	  else 
	  //We aren't root-- just expect remote and local messages
	      if ((expectedRemote==nRemote)&&(expectedLocalMessages()==0))
		endReduction();//We have all the messages we're going to get
	}
}

//This is called at the end of each reduction
//It combines the msgs array, sends it off, and sets the finished flag
void Array1D::endReduction(void)
{
	int i;
       	RED_DEB((RA"about to finish reduction, %d contributions in\n"RB,curMsgs->length()));
	reductionFinished=1;	
	
	//Reduce messages into a single result
	ArrayReductionMessage *result;
	ArrayReductionFn reducer=NULL;
	if (curMsgs->length()==0)
	  //We have no messages to send-- compose an empty message
		result=ArrayReductionMessage::buildNew(0,NULL);
	else 
	{//Combine the current messages into a reduced message
		int arrLen=curMsgs->length();
		ArrayReductionMessage **arr=new ArrayReductionMessage*[arrLen];
		for (i=0;i<arrLen;i++)
		{
		  arr[i]=(ArrayReductionMessage *)curMsgs->deq();
		  if (arr[i]->reducer!=NULL)
		    reducer=arr[i]->reducer;//FInd a non-NULL reduction function
		}
		
		if (reducer==NULL) CkAbort("ERROR!  Reducer function is NULL!\n");
		
		result=(reducer)(arrLen,arr);
		
		//Delete old messages and array
		for (i=0;i<arrLen;i++)
		  delete arr[i];
		delete arr;
	}
	
	//Set the reduction number and number of message sources.
	result->reductionNo=reductionNo;
	result->source=-nContributions-1;//m->source is negative as a flag (composite)
	result->reducer=reducer;
	
	if (CkMyPe()==0)
	{//We are the root-- return result to user's handler
	  RED_DEB((RA"reduction finished.  Calling reduction client.\n"RB));
	  if (reductionClient!=NULL)
	    (*reductionClient)(reductionClientParam,result->dataSize,result->data);
	  delete result;
	} else {//We aren't root-- forward the message to our parent
	  int parentNode=(CkMyPe()-1)/ARRAY_RED_TREE;
	  RED_DEB((RA"forwarding reduced message to my parent, %d.\n"RB,parentNode));
	  //Send the new message to our parent node
	  CProxy_Array1D arr(thisgroup);
	  arr.RecvReductionMessage(result, parentNode);
	}
	
	
	RED_DEB((RA"finished with reduction.\n"RB,CkMyPe(),reductionNo));	
	//Check to see if we can handle any messages from the future now
	int orig_nFuture=futureBuffer->length();
	for (i=0;i<orig_nFuture;i++)
	{
		//Pop an element from the front of the future buffer
		ArrayReductionMessage *m=(ArrayReductionMessage *)futureBuffer->deq();
		RED_DEB((RA"handling future message %d of %d, from reduction %d\n"RB,i,orig_nFuture,m->reductionNo));
		//Try to handle this message-- if we still can't, it'll
		// go right back into the (end of the) futureBuffer.
		addContribution(m);
	}
}

////////////////////////////////
//ArrayReductionMessage support

//ReductionMessage default private constructor-- does nothing
ArrayReductionMessage::ArrayReductionMessage(){}

//Return the number of array elements from which this message's data came
int ArrayReductionMessage::isSingleton(void)
{
	if (source>=0) return 1;//This data came from a single element
	else return 0;//This data came from several elements
}

//Return the number of array elements from which this message's data came
int ArrayReductionMessage::getSources(void)
{
	if (source>=0) return 1;//This data came from a single element
	else return -source-1;//This data came from several elements
}

//This define gives the distance from the start of the ArrayReductionMessage
// object to the start of the user data area (just below last object field)
#define ARM_DATASTART ALIGN8(sizeof(ArrayReductionMessage))

//"Constructor"-- builds and returns a new ArrayReductionMessage.
//  the "data" array you specify will be copied into this object.
ArrayReductionMessage *ArrayReductionMessage::
	buildNew(int NdataSize,void *srcData)
{
	int totalsize=ARM_DATASTART+NdataSize;
	
	ArrayReductionMessage *ret = (ArrayReductionMessage *)
		CkAllocMsg(__idx,totalsize,0);
	ret->dataSize=NdataSize;
	ret->data=(void *)(ARM_DATASTART+(char *)ret);
	if (srcData!=NULL)
		memcpy(ret->data,srcData,NdataSize);
	return ret;
}

void *
ArrayReductionMessage::alloc(int msgnum,int size,int *sz,int priobits)
{
	int totalsize=ARM_DATASTART+(*sz);
	RED_DEB(("Allocating %d %d %d\n",msgnum,totalsize,priobits));
	ArrayReductionMessage *ret = (ArrayReductionMessage *)
		CkAllocMsg(msgnum,totalsize,priobits);
	ret->data=(void *)(ARM_DATASTART+(char *)ret);
	return (void *) ret;
}
  
void *
ArrayReductionMessage::pack(ArrayReductionMessage* in)
{
	RED_DEB(("PE %d Packing %d %d %d\n",CkMyPe(),in->source,in->reductionNo,in->dataSize));
	in->data = NULL;
	return (void*) in;
}

ArrayReductionMessage* 
ArrayReductionMessage::unpack(void *in)
{
	ArrayReductionMessage *ret = (ArrayReductionMessage *)in;
	RED_DEB(("PE %d Unpacking %d %d %d\n",
		CkMyPe(),ret->source,ret->reductionNo,ret->dataSize));
	ret->data=(void *)(ARM_DATASTART+(char *)ret);
	return ret;
}

#endif //CK_ARRAY_REDUCTIONS


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
  nFuture=0;reductionClient=NULL;
  reductionNo=0;//We'll claim we were just doing reduction number zero...
  reductionFinished=1;//...but it is done now.
  curReducer=NULL;
  curMsgs=NULL;
  curMax=nCur=nComposite=expectedComposite=0;
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

//Debugging defines-- set this to 1 for tons of (useless?) debugging output.
#define GIVE_DEBUGGING_OUTPUT 0
#if GIVE_DEBUGGING_OUTPUT
#define RED_DEB(x) CkPrintf x
#else
#define RED_DEB(x) /*empty*/
#endif
#define RA "PE_%d/reduction %d: "
#define RB ,CkMyPe(),reductionNo

#include "ckarray_reductions.C" //Include reduction implementations


//Call contribute to add your contribution to a new global reduction.
// The array BOC will keep a copy the data. reducer must be the same on all PEs.
void ArrayElement::contribute(int dataSize,void *data,ArrayReductionFn reducer)
{
	RED_DEB(("index %d sends contribution %d to node %d\n",thisIndex,nContributions+1,CkMyPe()));
	ArrayReductionMessage *msg=ArrayReductionMessage::buildNew(dataSize,data);
	msg->source=thisIndex;
	msg->reductionNo=(++nContributions);
	thisArray->addReductionContribution(msg,reducer);
}

void Array1D::registerReductionHandler(ArrayReductionClientFn handler,void *param)
{
	reductionClient=handler;
	reductionClientParam=param;
}

static int printARM(ArrayReductionMessage *m)
{
	CkPrintf("-----PE %d: Reduction message--------------------\n",CkMyPe());
	CkPrintf("\t  source=%d  (may be negative for composites)\n",m->source);
	CkPrintf("\t  reductionNo=%d\n",m->reductionNo);
	CkPrintf("\t  dataSize=%d\n",m->dataSize);
	CkPrintf("\t  data[0]=%08x\n",((int *)(m->data))[0]);
	//CkPrintf("\t  data[1]=%08x\n",((int *)(m->data))[1]);
	//CkPrintf("\t  data[2]=%08x\n",((int *)(m->data))[2]);
	//CkPrintf("\t  data[3]=%08x\n",((int *)(m->data))[3]);
	CkPrintf("------------------------------------\n");
	return 0;
}


void Array1D::RecvReductionMessage(ArrayReductionMessage *msg)
{
	RED_DEB((RA"recv'd remote contribution\n"RB,printARM(msg)));
	addReductionContribution(msg,NULL);
}

//This is called by ArrayElement::contribute() and RcvReductionMessage.
// reducer may be NULL. The given message is kept by Array1D.
void Array1D::addReductionContribution(
	ArrayReductionMessage *m,ArrayReductionFn reducer)
{
	if (m->reductionNo>reductionNo)
	{//We haven't dealt with this reduction number yet--
		if (reductionFinished)
		{//We're ready for a new reduction
			if (reducer==NULL)
				beginReduction(0);//We were invoked remotely
			else
				beginReduction(1);//We were invoked by a local element
		}
		else 
		{//A prior reduction is in progress-- buffer this message
			m->futureReducer=reducer;//Stash the reducer for later use
			futureBuffer[nFuture++]=m;
			RED_DEB((RA"recv'd %dth early contribution from %d\n"RB,nFuture,m->source));
			if (nFuture>=ARRAY_RED_FUTURE_MAX) CkAbort("Too many out-of-order reduction messages sent.\n");
			return;//We can't handle this message yet
		}
	} else if ((m->reductionNo<reductionNo)||
		((m->reductionNo==reductionNo)&&reductionFinished))
	{//This message is for a reduction we already finished!
		if (CkMyPe()==0)
		//This is an error in the reduction library
			CkAbort("ERROR! Root node recieved a message for a reduction which is already complete!\n");
		else //This message is late because of migration--
		{//forward it straight to the root
			RED_DEB((RA"recv'd late contribution from %d\n"RB,m->source));
			CProxy_Array1D arr(thisgroup);
			arr.RecvReductionMessage(m, 0);
			return;
		}
	}
	
	if (reducer!=NULL) curReducer=reducer;//Set the reduction function
	curMsgs[nCur++]=m;
	if (m->source<0)//This was a message from one of our kids
		nComposite++;
	RED_DEB((RA"recv'd %dth contribution\n"RB,nCur));
	
	tryEndReduction();
}

int i_min(int a,int b) {if (a<b) return a; else return b;}
//BeginReduction is called to start each reduction
//It allocates msgs array above, increments reductionNo
typedef ArrayReductionMessage* ArrayReductionMessagePtr;
void Array1D::beginReduction(int extraLocals)
{
	reductionFinished=0;
	reductionNo++;//Start a new reduction
	
	//This is the PE number of my first child
	int firstKid=(CkMyPe()<<ARRAY_RED_TREE_LOG)+1;
	if (firstKid<CkNumPes()) //We have children-- expect 1 message from each
		expectedComposite=i_min(ARRAY_RED_TREE,CkNumPes()-firstKid);
	else	expectedComposite=0;//We are a leaf in the reduction tree.
	
	//Allocate a buffer for the new expected messages
	curMax=expectedComposite+expectedLocalMessages()+extraLocals;
	if (CkMyPe()==0)
		//The root node may have to recieve a few extra messages
		// from migrating nodes.
		curMax+=250;
	curMsgs=new ArrayReductionMessagePtr[curMax];
	nComposite=nCur=0;
	RED_DEB((RA"starting reduction. expecting %d remote messages, %d local (+%d)\n"RB,expectedComposite,curMax-expectedComposite,extraLocals));
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
	for (i=0;i<nFuture;i++)
		if (futureBuffer[i]->reductionNo==reductionNo)/*<-is this reduction*/
			if (futureBuffer[i]->futureReducer!=NULL) /*<-is local*/
				nExpected++;
	return nExpected;
}

void Array1D::tryEndReduction(void)//Check if we're done, and if so, finish.
{
	if ((reductionFinished==0)&&(nCur!=0))
	{
		if (nCur==curMax)
			endReduction();//We have all the messages we can handle
		else if ((expectedComposite==nComposite)&&(expectedLocalMessages()==0))
			endReduction();//We have all the messages we're going to get
	}
}

//This is called at the end of each reduction
//It combines the msgs array, sends it off, and sets the finished flag
void Array1D::endReduction(void)
{
	int i;
	if (curReducer==NULL)
		CkAbort("Array reduction function is NULL!  Are there any array elements on this PE?");
	RED_DEB((RA"about to finish reduction, %d contributions in\n"RB,nCur));
	//Combine the current messages into a reduced message
	ArrayReductionMessage *m=(*curReducer)(nCur,curMsgs);
	//Set the reduction number and compute the number of message sources.
	m->reductionNo=reductionNo;
	int nSources=0;
	for (i=0;i<nCur;i++)
		nSources+=curMsgs[i]->getSources();
	m->source=-nSources;//m->source is negative as a flag (composite)
	
	if (CkMyPe()==0)
	{//We are the root-- check to see if we have all the messages yet
		if (nSources==numElements)
		{//We've collected the contributions from all elements--
		// call the user's reduction client function.
			reductionFinished=1;
			RED_DEB((RA"reduction finished.  Calling reduction client.\n"RB));
			if (reductionClient!=NULL)
				(*reductionClient)(reductionClientParam,m->dataSize,m->data);
			delete m;
		} else {//We don't have all the contributions yet--
			//Some stragglers must be migrating.  We'll get them next time.
			RED_DEB((RA"reduction NOT finished because some elements are migrating-- only have %d out of %d contributions.\n"RB,nSources,numElements));
			delete m;
			return;
		}
	} else {
	//We aren't root-- we can just forward the message to our parent
		reductionFinished=1;
		int parentNode=(CkMyPe()-1)>>ARRAY_RED_TREE_LOG;
		RED_DEB((RA"forwarding reduced message to my parent, %d.\n"RB,parentNode,printARM(m)));
		//Send the new message to our parent node
		CProxy_Array1D arr(thisgroup);
		arr.RecvReductionMessage(m, parentNode);
	}
	
	//We're finished-- clean up
	curReducer=NULL;//Flush the old reducer
	for (i=0;i<nCur;i++)
		delete curMsgs[i];
	delete curMsgs;//Delete buffered reduction messages
	RED_DEB((RA"finished with reduction.\n"RB,CkMyPe(),reductionNo));
	
	//Check to see if we can handle any messages from the future now
	int orig_nFuture=nFuture;
	for (i=0;i<orig_nFuture;i++)
	{
		//Pop an element from the front of the future buffer
		ArrayReductionMessage *m=futureBuffer[0];
		RED_DEB((RA"handling future message %d of %d, from reduction %d\n"RB,i,orig_nFuture,m->reductionNo));
		//Move everybody in the future buffer down a notch
		for (int j=0;j+1<nFuture;j++)
			futureBuffer[j]=futureBuffer[j+1];
		nFuture--;
		//Try to handle this message-- if we still can't, it'll
		// go right back into the (end of the) futureBuffer.
		addReductionContribution(m,m->futureReducer);
	}
}

////////////////////////////////
//ArrayReductionMessage support

//ReductionMessage default private constructor-- does nothing
ArrayReductionMessage::ArrayReductionMessage(){}

//Return the number of array elements from which this message's data came
int ArrayReductionMessage::getSources()
{
	if (source>=0) return 1;//This data came from a single element
	else return -source;//This data came from several elements
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


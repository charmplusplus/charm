///**
//   @addtogroup CharmComlib
//*/
///*@{*/
//
///** @file */
//
////#include "charm++.h"
////#include "envelope.h"
////#include "ckhashtable.h"
//#include "ComlibArrayListener.h"
//#include "ComlibManager.h"
//
///*
//ComlibArrayListener::ComlibArrayListener () 
//    : CkArrayListener(0){ //Carry 1 int for the sid, not used yet
//    nElements = 0;
//    ht = new CkHashtableT<CkArrayIndex, CkArrayIndex *>;
//    //    CkPrintf("Creating Array Listener\n");
//}
//*/
//
//ComlibArrayListener::ComlibArrayListener(CkArrayID &id) : CkArrayListener(0) {
//  setupFinished = 0;
//  thisArrayID = id;
//  ComlibPrintf("[%d] Creating ComlibArrayListener for array %d\n",CkMyPe(),((CkGroupID)id).idx);
//}
//
//ComlibArrayListener::ComlibArrayListener (CkMigrateMessage *m)
//    :CkArrayListener(m) {
//  /*
//    nElements = 0;
//    ht = new CkHashtableT<CkArrayIndex, CkArrayIndex *>;
//  */
//}
//
//void ComlibArrayListener::pup(PUP::er &p) {
//  ComlibPrintf("[%d] ComlibArrayListener pupping for %s, why?!?\n",CkMyPe(),
//               p.isPacking()?"packing":(p.isUnpacking()?"unpacking":"sizing"));
//}
//
//void ComlibArrayListener::ckEndInserting() {
//	CkAssert(0);
//	CkAssert(setupFinished==0);
//
//	ComlibPrintf("[%d] ComlibArrayListener::ckEndInserting\n",CkMyPe());
//	CProxy_ComlibManager cgproxy(CkpvAccess(cmgrID));
//	for (int i=0; i<userIDs.size(); ++i) {
//		cgproxy[0].bracketedFinishSetup(userIDs[i]);
//	}
//
//	setupFinished = 1;
//  
//}
//
//void ComlibArrayListener::ckElementCreating(ArrayElement *elt){
//  ComlibPrintf("[%d] ComlibArrayListener::ckElementCreating\n",CkMyPe());
//  for (int i=0; i<users.size(); ++i) {
//    users[i]->newElement(thisArrayID, elt->ckGetArrayIndex());
//  }
//  //addElement(elt, CmiFalse);
//    //CkPrintf("[%d] Element Created\n", CkMyPe());
//}
///*
//void ComlibArrayListener::ckElementDied(ArrayElement *elt){
//    deleteElement(elt, CmiFalse);
//}
//
//void ComlibArrayListener::ckElementLeaving(ArrayElement *elt){
//    deleteElement(elt, CmiTrue);
//}
//
//CmiBool ComlibArrayListener::ckElementArriving(ArrayElement *elt){
//    addElement(elt, CmiTrue);
//    return CmiTrue;
//}
//
//void ComlibArrayListener::addElement(ArrayElement *elt, 
//                                     CmiBool migration_flag){
//    if(nElements == 0)
//        thisArrayID = elt->ckGetArrayID();
//
//    ht->put(elt->thisIndexMax) = &(elt->thisIndexMax);
//    //elt->thisIndexMax.print();
//    nElements ++;
//
//    if(!migration_flag) {
//        for(int count = 0; count < strategyList.length(); count++){
//            CharmStrategy *strategy = (CharmStrategy *)
//                strategyList[count]->strategy;
//            if(isRegistered(elt, strategy)) {
//                strategyList[count]->numElements ++;
//            }
//        }   
//    }
//}
//
//void ComlibArrayListener::deleteElement(ArrayElement *elt, 
//                                        CmiBool migration_flag){
//    ht->remove(elt->thisIndexMax);
//    nElements --;
//    
//    if(!migration_flag) {
//        for(int count = 0; count < strategyList.length(); count++){
//            CharmStrategy *strategy = (CharmStrategy *)
//                strategyList[count]->strategy;
//            if(isRegistered(elt, strategy)) {
//                strategyList[count]->numElements --;
//            }
//        }   
//    }
//}
//
//int ComlibArrayListener::isRegistered(ArrayElement *elt, 
//                                      CharmStrategy *strat){
//    CkArrayIndex idx = elt->thisIndexMax;
//
//    CkArrayID st_aid;
//    int st_nelements;
//    CkArrayIndex *st_elem;
//    strat->ainfo.getSourceArray(st_aid, st_elem, st_nelements);
//
//    if(st_nelements < 0)
//        CkAbort("Not an Array Strategy\n");
//    
//    if(st_nelements == 0)
//        return 1;   
//
//    for(int count = 0; count < st_nelements; count ++)
//        if(st_elem[count].compare(idx))
//            return 1;
//
//    return 0;
//}
// 
////Assumes strategy is already present in the strategy table   
//void ComlibArrayListener::registerStrategy(StrategyTableEntry *stable_entry) {
//    strategyList.insertAtEnd(stable_entry);
//
//    CharmStrategy *strat = (CharmStrategy *) stable_entry->strategy;
//
//    CkArrayID st_aid;
//    int st_nelements;
//    CkArrayIndex *st_elem;
//    strat->ainfo.getSourceArray(st_aid, st_elem, st_nelements);
//
//    if(st_nelements == 0) {//All elements of array in strategy
//        stable_entry->numElements += nElements;
//
////         CkHashtableIterator *ht_iterator = ht->iterator();
////         ht_iterator->seekStart();
////         while(ht_iterator->hasNext()){
////             CkArrayIndex *idx;
////             ht_iterator->next((void **)&idx);
////             stable_entry->strategy->insertLocalIndex(*idx);       
////         }
//
//    }
//    else if (st_nelements > 0){ //Only some elements belong to strategy
//        for(int count = 0; count < st_nelements; count ++)
//            if(ht->get(st_elem[count]) != NULL) {
//                stable_entry->numElements ++;
//            }
//    }
//    else 
//        CkAbort("NOT an Array Strategy\n");
//
//}
//
//void ComlibArrayListener::getLocalIndices(CkVec<CkArrayIndex> &vec){
//    
//    CkHashtableIterator *ht_iterator = ht->iterator();
//    ht_iterator->seekStart();
//    while(ht_iterator->hasNext()){
//        CkArrayIndex *idx;
//        ht_iterator->next((void **)&idx);
//        vec.insertAtEnd(*idx);       
//    }
//}
//*/
//
///*@}*/

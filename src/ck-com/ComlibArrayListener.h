
///*@{*/
//
///** @file */
//
//#ifndef COMMLIBARRAYLISTENER_H
//#define COMMLIBARRAYLISTENER_H
//
//#include "ComlibStrategy.h"
//
///**
// * This class is used by the ComlibArrayInfo class to keep track of new objects
// * when the user specification is so. Namely, a ComlibArrayInfo can register
// * itself with this ArrayListener so that every element that for every element
// * that is created on this processor, the ComlibArrayInfo class is notified and
// * can take appropriate action.
// */
//class ComlibArrayListener : public CkArrayListener {
//  //int nElements;
//  int setupFinished;
//  CkArrayID thisArrayID;
//  /*
//  CkVec <StrategyTableEntry *> strategyList;
//  CkHashtableT<CkArrayIndex, CkArrayIndex*> *ht;
//    
//  int isRegistered(ArrayElement *elt, CharmStrategy *astrat);
//  void addElement(ArrayElement *elt, bool mogration_flag);
//  void deleteElement(ArrayElement *elt, bool migration_flag);
//  */
//  CkVec<ComlibArrayInfo*> users;
//  CkVec<int> userIDs;
//
// public:
//  ComlibArrayListener(CkArrayID &id);
//  ComlibArrayListener(CkMigrateMessage *);
//
//  inline bool operator==(CkArrayID &a) {
//    return (thisArrayID == a);
//  }
//
//  inline bool operator==(ComlibArrayListener &l) {
//    return operator==(l.thisArrayID);
//  }
//
//  inline void registerUser(ComlibArrayInfo *ai, int stratid=0) {
//    users.push_back(ai);
//    userIDs.push_back(stratid);
//  }
//
//  void ckEndInserting();
//
//  void ckElementCreating(ArrayElement *elt);
//  //void ckElementDied(ArrayElement *elt);
//    
//  //void ckElementLeaving(ArrayElement *elt);
//  //bool ckElementArriving(ArrayElement *elt);
//    
//  //Add strategy to listening list, strategy will get an the number
//  //of array elements lying on that processor
//  //void registerStrategy(StrategyTableEntry *);
//
//  //remove strategy from table, and now it will not get updates
//  //about this array
//  /*
//  void unregisterStrategy(StrategyTableEntry *entry) {
//    for(int count = 0; count < strategyList.size(); count++)
//      if(strategyList[count] == entry)
//	strategyList.remove(count);
//  }
//
//  void getLocalIndices(CkVec<CkArrayIndex> &vec);
//  */
//
//  void pup(PUP::er &p);
//  PUPable_decl(ComlibArrayListener);
//};
//
//#endif
//
///*@}*/

/**
   @addtogroup ComlibCharmStrategy
   @{
   
   @file 
 
*/

#ifndef ONE_TIME_MULTICAST_STRATEGY
#define ONE_TIME_MULTICAST_STRATEGY

#include "ComlibManager.h"
#include "ComlibSectionInfo.h"

/**
   The simplest multicast strategy. This strategy extracts the array section information, and packs the section information and the user message into a single message. The original message is delivered locally, and the new message is sent using CmiSyncListSendAndFree to all other processors containing destination objects. 

   This strategy is simpler than those which are derived from the MulticastStrategy class because those maintain a persistant record of previous array section information extracted from the messages, and those provide multiple implementations of the multicast tree (such as ring or multiring or all to all). Those strategies ought to be used when multiple multicasts are sent to the same array section. If an array section is not reused, then this strategy ought to be used.

   A class can be created which inherits from this class, but provides its own determineNextHopPEs method to specify any type of desired spanning tree. For example, OneTimeRingMulticastStrategy forwards the multicast messages in a ring while OneTimeTreeMulticastStrategy forwards messages along a tree of specified degree. In the future, topology aware (both network and core/cpu/node) strategies should be added.

   The local messages are delivered through the array manager using the CharmStrategy::deliverToIndices methods. If a destination chare is remote, the array manager will forward it on to the pe that contains the chare.
   
*/
class OneTimeMulticastStrategy: public Strategy, public CharmStrategy {
 private:
  
  ComlibSectionInfo sinfo; // This is used to create the multicast messages themselves

  void remoteMulticast(ComlibMulticastMsg * multMsg, bool rootPE);
  void localMulticast(CharmMessageHolder *cmsg);
  
 public:

  virtual void determineNextHopPEs(ComlibMulticastMsg * multMsg, int myIndex, int * &pelist, int &npes );
  
 OneTimeMulticastStrategy(CkMigrateMessage *m): Strategy(m), CharmStrategy(m){}
  
  OneTimeMulticastStrategy();
  ~OneTimeMulticastStrategy();
  
  void insertMessage(MessageHolder *msg) {insertMessage((CharmMessageHolder*)msg);}
  void insertMessage(CharmMessageHolder *msg);
  
  ///Called by the converse handler function
  void handleMessage(void *msg);    

  void pup(PUP::er &p);

  PUPable_decl(OneTimeMulticastStrategy);

};





/**
   A OneTimeMulticastStrategy that sends along a ring
*/
class OneTimeRingMulticastStrategy: public OneTimeMulticastStrategy {
  
 public:
  void determineNextHopPEs(ComlibMulticastMsg * multMsg, int myIndex, int * &pelist, int &npes );

 OneTimeRingMulticastStrategy(CkMigrateMessage *m): OneTimeMulticastStrategy(m) {}
 OneTimeRingMulticastStrategy(): OneTimeMulticastStrategy() {}
  ~OneTimeRingMulticastStrategy() {}
  
  void pup(PUP::er &p){ OneTimeMulticastStrategy::pup(p); }
  
  PUPable_decl(OneTimeRingMulticastStrategy);

};



/**
   A OneTimeMulticastStrategy that sends along a tree of arbitrary degree
*/
class OneTimeTreeMulticastStrategy: public OneTimeMulticastStrategy {
 private:
  int degree;
  
 public:
  
  void determineNextHopPEs(ComlibMulticastMsg * multMsg, int myIndex, int * &pelist, int &npes );
  
 OneTimeTreeMulticastStrategy(CkMigrateMessage *m): OneTimeMulticastStrategy(m) {}
 OneTimeTreeMulticastStrategy(int treeDegree=4): OneTimeMulticastStrategy(), degree(treeDegree) {}
  ~OneTimeTreeMulticastStrategy() {}
  
  void pup(PUP::er &p){ 
    OneTimeMulticastStrategy::pup(p); 
    p | degree;
  }
  
  PUPable_decl(OneTimeTreeMulticastStrategy);
  
};



#endif

/*@}*/

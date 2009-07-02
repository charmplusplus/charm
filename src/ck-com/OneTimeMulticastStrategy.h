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
   
   To create a new strategy:
   <ul>
   <li>Add a class declaration similar to the ones below, making sure they inherit from OneTimeMulticastStrategy. 
   <li>Add a PUPable entry in ComlibManager.ci 
   <li>Implement determineNextHopPEs in OneTimeMulticastStrategy.C. See information for OneTimeMulticastStrategy::determineNextHopPEs .
   </ul>

@todo  Buffer messages until strategy is fully enabled. The current version might have some startup issues if the multicast is used too early.

@todo  Implement topology aware subclasses. 

*/
class OneTimeMulticastStrategy: public Strategy, public CharmStrategy {
 private:
  
  ComlibSectionInfo sinfo; // This is used to create the multicast messages themselves

  void remoteMulticast(ComlibMulticastMsg * multMsg, bool rootPE);
  void localMulticast(CharmMessageHolder *cmsg);
  
 public:

  /** 
      Determine the set of PEs to which the message should be forwarded from this PE.
      Fill in pelist and npes to which the multicast message will be forwarded from this PE.

      @param [in] totalDestPEs The number of destination PEs to whom the message needs to be sent. This will always be > 0.
      @param [in] destPEs The list of PEs that eventually will be sent the message.
      @param [in] myIndex The index into destPEs for this PE.

      @param [out] pelist A list of PEs to which the message will be sent after this function returns. This function allocates the array with new. The caller will free it with delete[] if npes>0.
      @param [out] npes The size of pelist

  */
  virtual void determineNextHopPEs(const int totalDestPEs, const ComlibMulticastIndexCount* destPEs, const int myIndex, int * &pelist, int &npes );

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
   A strategy that sends along a ring through the destination processors.
*/
class OneTimeRingMulticastStrategy: public OneTimeMulticastStrategy {
  
 public:
  void determineNextHopPEs(const int totalDestPEs, const ComlibMulticastIndexCount* destPEs, const int myIndex, int * &pelist, int &npes );

 OneTimeRingMulticastStrategy(CkMigrateMessage *m): OneTimeMulticastStrategy(m) {}
 OneTimeRingMulticastStrategy(): OneTimeMulticastStrategy() {}
  ~OneTimeRingMulticastStrategy() {}
  
  void pup(PUP::er &p){ OneTimeMulticastStrategy::pup(p); }
  
  PUPable_decl(OneTimeRingMulticastStrategy);

};



/**
   A strategy that sends along a tree with user specified branching factor.
*/
class OneTimeTreeMulticastStrategy: public OneTimeMulticastStrategy {
 private:
  int degree;
  
 public:
  
  void determineNextHopPEs(const int totalDestPEs, const ComlibMulticastIndexCount* destPEs, const int myIndex, int * &pelist, int &npes );
  
 OneTimeTreeMulticastStrategy(CkMigrateMessage *m): OneTimeMulticastStrategy(m) {}

  /** Create a strategy with specified branching factor(which defaults to 4) */
 OneTimeTreeMulticastStrategy(int treeDegree=4): OneTimeMulticastStrategy(), degree(treeDegree) {}

  ~OneTimeTreeMulticastStrategy() {}
  
  void pup(PUP::er &p){ 
    OneTimeMulticastStrategy::pup(p); 
    p | degree;
  }
  
  PUPable_decl(OneTimeTreeMulticastStrategy);
};






/**
   A node-aware strategy that sends along a node-based tree with user specified branching factor. Once the message reaches the PE representative for each node, it is forwarded from the PE to all other destination PEs on the node. This strategy can result in imbalanced loads. The PEs along the tree have higher load than the other PEs.
*/
class OneTimeNodeTreeMulticastStrategy: public OneTimeMulticastStrategy {
 private:
  int degree;
  
 public:
  
  void determineNextHopPEs(const int totalDestPEs, const ComlibMulticastIndexCount* destPEs, const int myIndex, int * &pelist, int &npes );
  
 OneTimeNodeTreeMulticastStrategy(CkMigrateMessage *m): OneTimeMulticastStrategy(m) {}
  
  /** Create a strategy with specified branching factor(which defaults to 4) */
 OneTimeNodeTreeMulticastStrategy(int treeDegree=4): OneTimeMulticastStrategy(), degree(treeDegree) {}
  
  ~OneTimeNodeTreeMulticastStrategy() {}
  
  void pup(PUP::er &p){ 
    OneTimeMulticastStrategy::pup(p); 
    p | degree;
  }
  
  PUPable_decl(OneTimeNodeTreeMulticastStrategy);
};



/**
   A node-aware strategy that sends along a node-based tree with user specified branching factor. Once the message arrives at the first PE on the node, it is forwarded to the other PEs on the node through a ring.
*/
class OneTimeNodeTreeRingMulticastStrategy: public OneTimeMulticastStrategy {
 private:
  int degree;
  
 public:
  
  void determineNextHopPEs(const int totalDestPEs, const ComlibMulticastIndexCount* destPEs, const int myIndex, int * &pelist, int &npes );
  
 OneTimeNodeTreeRingMulticastStrategy(CkMigrateMessage *m): OneTimeMulticastStrategy(m) {}
  
  /** Create a strategy with specified branching factor(which defaults to 4) */
 OneTimeNodeTreeRingMulticastStrategy(int treeDegree=4): OneTimeMulticastStrategy(), degree(treeDegree) {}
  
  ~OneTimeNodeTreeRingMulticastStrategy() {}
  
  void pup(PUP::er &p){ 
    OneTimeMulticastStrategy::pup(p); 
    p | degree;
  }
  
  PUPable_decl(OneTimeNodeTreeRingMulticastStrategy);
};





#endif

/*@}*/

/**
   @addtogroup ComlibCharmStrategy
   @{
   
   @file 
 
*/

#ifndef CHUNK_MULTICAST_STRATEGY
#define CHUNK_MULTICAST_STRATEGY

#include "ComlibManager.h"
#include "ComlibSectionInfo.h"
#include <list>

/*class blkMsg1 {
public:
  double *data;
};*/

struct ChunkInfo {
  //int bcastPe;     ///< pe who is doing the broadcast, used for the hash key
  int chunkNumber;   ///< which chunk is this inside the whole message
  int chunkSize;     ///< it is the size of the data part of the message (without the converse header)
  int numChunks;   ///< total number of chunks to arrive
  //int messageSize;   ///< the entire message size, all included 
  int idx;	     ///< keeps track of the number of messages sent from this pe
  int srcPe;       ///< pe from whom I'm receiving the message
};



/**
   The simplest multicast strategy. This strategy extracts the array section information, and packs the section information and the user message into a single message. The original message is delivered locally, and the new message is sent using CmiSyncListSendAndFree to all other processors containing destination objects. If the destination entry method is [nokeep], then the multicast is delivered inline without extra copies to the local destination elements. If the destination is not [nokeep], then the message is delivered through the scheduler queue.  

   Projections can trace the messages for the [nokeep] destinations, but the sending entry method will end prematurely because inline calling of local entry methods is not fully supported by Projections. Messages multicast to non [nokeep] methods are displayed incorrectly, probably because the call to deliver overwrites the source pe & event.

@fixme Fix projections logging for the non [nokeep] version

   This strategy is simpler than those which are derived from the MulticastStrategy class because those maintain a persistant record of previous array section information extracted from the messages, and those provide multiple implementations of the multicast tree (such as ring or multiring or all to all). Those strategies ought to be used when multiple multicasts are sent to the same array section. If an array section is not reused, then this strategy ought to be used.

   A class can be created which inherits from this class, but provides its own determineNextHopPEs method to specify any type of desired spanning tree. For example, ChunkRingMulticastStrategy forwards the multicast messages in a ring while ChunkTreeMulticastStrategy forwards messages along a tree of specified degree. In the future, topology aware (both network and core/cpu/node) strategies should be added.

   The local messages are delivered through the array manager using the CharmStrategy::deliverToIndices methods. If a destination chare is remote, the array manager will forward it on to the pe that contains the chare.
   
   To create a new strategy:
   <ul>
   <li>Add a class declaration similar to the ones below, making sure they inherit from ChunkMulticastStrategy. 
   <li>Add a PUPable entry in ComlibManager.ci 
   <li>Implement determineNextHopPEs in ChunkMulticastStrategy.C. See information for ChunkMulticastStrategy::determineNextHopPEs .
   </ul>

@todo  Buffer messages until strategy is fully enabled. The current version might have some startup issues if the multicast is used too early.

@todo  Implement topology aware subclasses. 

*/

struct recvBuffer {
  int numChunks;
  envelope** recvChunks;
  int nrecv;
  int srcPe;
  int idx;
};

class ChunkMulticastStrategy: public Strategy, public CharmStrategy {
 private:
  
  ComlibSectionInfo sinfo; // This is used to create the multicast messages themselves

  void remoteMulticast(ComlibMulticastMsg * multMsg, bool rootPE, int chunkNumber, int numChunks);
  void localMulticast(CharmMessageHolder *cmsg);
  
 public:

  //int numChunks;
  //int nrecv;
  std::list< recvBuffer* > recvList;
  int sentCount;

  /** 
      Determine the set of PEs to which the message should be forwarded from this PE.
      Fill in pelist and npes to which the multicast message will be forwarded from this PE.

      @param [in] totalDestPEs The number of destination PEs to whom the message needs to be sent. This will always be > 0.
      @param [in] destPEs The list of PEs that eventually will be sent the message.
      @param [in] myIndex The index into destPEs for this PE.

      @param [out] pelist A list of PEs to which the message will be sent after this function returns. This function allocates the array with new. The caller will free it with delete[] if npes>0.
      @param [out] npes The size of pelist

  */
  virtual void determineNextHopPEs(const int totalDestPEs, const ComlibMulticastIndexCount* destPEs, const int myIndex, int * &pelist, int &npes, int chunkNumber, int numChunks);

    ChunkMulticastStrategy(CkMigrateMessage *m): Strategy(m), CharmStrategy(m){}
  
  ChunkMulticastStrategy();
  ~ChunkMulticastStrategy();
  
  void insertMessage(MessageHolder *msg) {insertMessage((CharmMessageHolder*)msg);}
  void insertMessage(CharmMessageHolder *msg);
  
  ///Called by the converse handler function
  void handleMessage(void *msg);    

  void pup(PUP::er &p);

  PUPable_decl(ChunkMulticastStrategy);

};





/**
   A strategy that sends along a ring through the destination processors.
*/
class ChunkRingMulticastStrategy: public ChunkMulticastStrategy {
  
 public:
  void determineNextHopPEs(const int totalDestPEs, const ComlibMulticastIndexCount* destPEs, const int myIndex, int * &pelist, int &npes, int chunkNumber, int numChunks );

 ChunkRingMulticastStrategy(CkMigrateMessage *m): ChunkMulticastStrategy(m) {}
 ChunkRingMulticastStrategy(): ChunkMulticastStrategy() {}
  ~ChunkRingMulticastStrategy() {}
  
  void pup(PUP::er &p){ ChunkMulticastStrategy::pup(p); }
  
  PUPable_decl(ChunkRingMulticastStrategy);

};



/**
   A strategy that sends along a tree with user specified branching factor.
*/
class ChunkTreeMulticastStrategy: public ChunkMulticastStrategy {
 private:
  int degree;
  
 public:
  
  void determineNextHopPEs(const int totalDestPEs, const ComlibMulticastIndexCount* destPEs, const int myIndex, int * &pelist, int &npes, int chunkNumber, int numChunks );
  
 ChunkTreeMulticastStrategy(CkMigrateMessage *m): ChunkMulticastStrategy(m) {}

 ChunkTreeMulticastStrategy(int treeDegree=4): ChunkMulticastStrategy(), degree(treeDegree) {}

  ~ChunkTreeMulticastStrategy() {}
  
  void pup(PUP::er &p){ 
    ChunkMulticastStrategy::pup(p); 
    p | degree;
  }
  
  PUPable_decl(ChunkTreeMulticastStrategy);
};


/**
   A strategy that sends along a tree with user specified branching factor.
*/
class ChunkPipeTreeMulticastStrategy: public ChunkMulticastStrategy {
 private:
  int degree;
  
 public:
  
  void determineNextHopPEs(const int totalDestPEs, const ComlibMulticastIndexCount* destPEs, const int myIndex, int * &pelist, int &npes, int chunkNumber, int numChunks );
  
 ChunkPipeTreeMulticastStrategy(CkMigrateMessage *m): ChunkMulticastStrategy(m) {}

 ChunkPipeTreeMulticastStrategy(int treeDegree=4): ChunkMulticastStrategy(), degree(treeDegree) {}

  ~ChunkPipeTreeMulticastStrategy() {}
  
  void pup(PUP::er &p){ 
    ChunkMulticastStrategy::pup(p); 
    p | degree;
  }
  
  PUPable_decl(ChunkPipeTreeMulticastStrategy);
};







/**
   A node-aware strategy that sends along a node-based tree with user specified branching factor. Once the message reaches the PE representative for each node, it is forwarded from the PE to all other destination PEs on the node. This strategy can result in imbalanced loads. The PEs along the tree have higher load than the other PEs.
*/
/*class ChunkNodeTreeMulticastStrategy: public ChunkMulticastStrategy {
 private:
  int degree;
  
 public:
  
  void determineNextHopPEs(const int totalDestPEs, const ComlibMulticastIndexCount* destPEs, const int myIndex, int * &pelist, int &npes );
  
 ChunkNodeTreeMulticastStrategy(CkMigrateMessage *m): ChunkMulticastStrategy(m) {}
  
 ChunkNodeTreeMulticastStrategy(int treeDegree=4): ChunkMulticastStrategy(), degree(treeDegree) {}
  
  ~ChunkNodeTreeMulticastStrategy() {}
  
  void pup(PUP::er &p){ 
    ChunkMulticastStrategy::pup(p); 
    p | degree;
  }
  
  PUPable_decl(ChunkNodeTreeMulticastStrategy);
};*/



/**
   A node-aware strategy that sends along a node-based tree with user specified branching factor. Once the message arrives at the first PE on the node, it is forwarded to the other PEs on the node through a ring.
*/
/*class ChunkNodeTreeRingMulticastStrategy: public ChunkMulticastStrategy {
 private:
  int degree;
  
 public:
  
  void determineNextHopPEs(const int totalDestPEs, const ComlibMulticastIndexCount* destPEs, const int myIndex, int * &pelist, int &npes );
  
 ChunkNodeTreeRingMulticastStrategy(CkMigrateMessage *m): ChunkMulticastStrategy(m) {}
  
 ChunkNodeTreeRingMulticastStrategy(int treeDegree=4): ChunkMulticastStrategy(), degree(treeDegree) {}
  
  ~ChunkNodeTreeRingMulticastStrategy() {}
  
  void pup(PUP::er &p){ 
    ChunkMulticastStrategy::pup(p); 
    p | degree;
  }
  
  PUPable_decl(ChunkNodeTreeRingMulticastStrategy);
};*/





#endif

/*@}*/

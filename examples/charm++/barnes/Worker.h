#ifndef __WORKER_H__
#define __WORKER_H__

#include "Node.h"
#include "util.h"
#include "MultipoleMoments.h"
#include "Descriptor.h"

#include <map>
using namespace std;

// worker abstract class
template<typename T>
class CutoffWorker {
  public:
  virtual int work(Node<T> *) = 0; 
  virtual void work(ExternalParticle *) {}
  virtual void bucketDone(Key k) {}
  virtual void *getContext() {return NULL;}
  virtual void setContext(void *context) {}
  virtual void done() {}
};

class DataManager;
class ParticleFlushWorker : public CutoffWorker<NodeDescriptor> {
  int leafCnt;
  DataManager *dataManager;

  public:
  ParticleFlushWorker(DataManager *dm) : 
    dataManager(dm),
    leafCnt(0)
  {
  }

  int work(Node<NodeDescriptor> *node);
  int getNumLeaves(){
    return leafCnt;
  }
};

class MomentsWorker : public CutoffWorker<ForceData> {

  // tree pieces on this PE, in sorted order
  // this allows us to mark the nodes that are
  // internal to each PE. The last PE in this list
  // is a dummy value, and is equal to the number of
  // useful treepieces+1
  CkVec<TreePieceDescriptor> &peTreePieces;
  map<Key,Node<ForceData>*> &nodeTable;
  CkVec<Node<ForceData>*> &buckets;

  int curTP;

  public: 
  MomentsWorker(CkVec<TreePieceDescriptor> &petps,
                map<Key,Node<ForceData>*> &tab,
                CkVec<Node<ForceData>*> &bucks
                ) : 
    peTreePieces(petps),
    nodeTable(tab),
    buckets(bucks),
    curTP(0)
  {
  }

  int work(Node<ForceData> *node);
  void setLeafType(Node<ForceData> *leaf);
  void setTypeFromChildren(Node<ForceData> *node);
};

class State;
class TraversalWorker : public CutoffWorker<ForceData> {
  protected:
  TreePiece *ownerTreePiece;
  State *state;

  Node<ForceData> *currentBucket;

#ifdef GPU_GRAVITY
  // Where the current bucket's particles sit in the PE's particle array.
  // Cached because it is needed once per interaction, and recomputing it means
  // going through the tree piece to the DataManager to subtract two pointers.
  int bucketPartStart;
  int bucketPartCount;

  // Out of line: only TreePiece.h knows how to reach the DataManager, and this
  // runs once per bucket rather than once per interaction.
  void cacheBucketRange();
#endif

  TraversalWorker() :
    ownerTreePiece(NULL),
    currentBucket(NULL)
  {
#ifdef GPU_GRAVITY
    bucketPartStart = 0;
    bucketPartCount = 0;
#endif
  }

  public:

  void reset(TreePiece *owner, State *s, Node<ForceData> *bucket){
    ownerTreePiece = owner;
    state = s;
    currentBucket = bucket;
#ifdef GPU_GRAVITY
    cacheBucketRange();
#endif
  }

  Node<ForceData> *getCurrentBucket(){
    return currentBucket;
  }

  void *getContext(){
    return (void *) currentBucket;
  }

  void setContext(void *context){
    currentBucket = (Node<ForceData> *) context;
#ifdef GPU_GRAVITY
    cacheBucketRange();
#endif
  }

  int work(Node<ForceData> *node);
  void work(ExternalParticle *particle);
  void bucketDone(Key k);
  
  virtual void done() {}
  virtual bool getKeep(NodeType type) = 0;
};

class LocalTraversalWorker : public TraversalWorker {
  static const bool keep[];
  public:
  LocalTraversalWorker() : TraversalWorker() {}
  bool getKeep(NodeType type);
};

class RemoteTraversalWorker : public TraversalWorker {
  static const bool keep[];
  public:
  RemoteTraversalWorker() : TraversalWorker() {}
  void done();
  bool getKeep(NodeType type);
};

class TreeSizeWorker : public CutoffWorker<ForceData> {

  int numNodes;
  int cutoffDepth;

  public:
  TreeSizeWorker(int d) : 
    cutoffDepth(d),
    numNodes(0)
  {
  }

  int work(Node<ForceData> *node);
  int getNumNodes(){
    return numNodes;
  }
};

template<typename T>
class FreeTreeWorker : public CutoffWorker<T> {
  public:
  int work(Node<T> *node){
    if(node->getNumChildren() > 0){
      /*
         CkPrintf("(%d) deleting %d children of %lu ptr %lx\n", 
         CkMyPe(), 
         node->getNumChildren(),
         node->getKey(),
         node->getChildren()
         );
         */
      delete[] node->getChildren();
    }
    return 1;
  }
};

class TreeChecker : public CutoffWorker<ForceData> {
  ostream &os;
  public: 
  TreeChecker(ostream &o) : os(o) {}
  int work(Node<ForceData> *node);
};

class InteractionChecker : public CutoffWorker<ForceData> {
  public:
  InteractionChecker() {}
  int work(Node<ForceData> *node);
};

#endif

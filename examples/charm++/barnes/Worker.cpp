#include "Worker.h"
#include "DataManager.h"
#include "util.h"
#include "TreePiece.h"
#include "gravity.h"

extern CProxy_TreePiece treePieceProxy;

int ParticleFlushWorker::work(Node<NodeDescriptor> *node){
  if(node->getNumChildren() > 0) return 1;

  // Record the leaf rather than sending it. The send is deferred until the
  // tree-piece-to-PE map arrives, because the leaves have to be grouped by
  // destination PE before anything goes on the wire -- see the note on
  // ParticleBlockMsg.
  dataManager->recordLeaf(node,leafCnt);
  leafCnt++;
  return 0;
}

int MomentsWorker::work(Node<ForceData> *node){
  
  int numChildren = node->getNumChildren();
  Particle *particles = node->getParticles();
  Node<ForceData> *parent = node->getParent();
  int numParticles = node->getNumParticles();

  if(numChildren == 0){
    // this node has no children, i.e. is a leaf
    // on this PE.
    setLeafType(node);

    if(node->getType() == Bucket || node->getType() == EmptyBucket){
      node->getMomentsFromParticles();
      if(parent != NULL) parent->childMomentsReady();
    }
  }
  else if(numChildren > 0){
    setTypeFromChildren(node);
    // don't have moments of remote nodes yet
    if(node->getType() == Internal){ 
      node->getMomentsFromChildren();
      if(parent != NULL) parent->childMomentsReady();
    }
  }
  nodeTable[node->getKey()] = node;
  return 1;
}

void MomentsWorker::setLeafType(Node<ForceData> *leaf){
  int ownerStart = leaf->getOwnerStart();
  int ownerEnd = leaf->getOwnerEnd();

  beg:
  // is the current PE the one that owns it?
  if(peTreePieces[curTP].index == ownerStart) {
    if(leaf->getNumParticles() > 0) leaf->setType(Bucket);
    else leaf->setType(EmptyBucket);

    buckets.push_back(leaf);
    return;
  }
  else if(ownerStart < peTreePieces[curTP].index){
    // type could have been set to EmptyBucket during construction
    // Just the type. The moments arrive with everyone else's in the frontier
    // reduction; there is no per-node moment request any more.
    if(leaf->getType() == Invalid) leaf->setType(Remote);
    return;
  }
  else do {
    curTP++;
  } while(ownerStart > peTreePieces[curTP].index);
  goto beg;
}


void MomentsWorker::setTypeFromChildren(Node<ForceData> *node){
  Node<ForceData> *child = node->getChildren();
  int numChildren = node->getNumChildren();
  bool isInternal = true;
  for(int i = 0; i < numChildren; i++){
    NodeType childType = child->getType();
    if(childType != Internal 
       && childType != Bucket 
       && childType != EmptyBucket){
      isInternal = false; 
      break;
    }
    child++;
  }

  if(isInternal){
    // all children were either buckets or internal
    node->setType(Internal);
  }else{
    // at least one child was not a bucket and not internal
    // note that not all children can be non-bucket and non-internal
    // since we wouldn't have split a node on this PE if we
    // didn't have at least some particles under it
    // therefore, the node can't be remote
    node->setType(Boundary);
  }
}

#ifdef GPU_GRAVITY
void TraversalWorker::cacheBucketRange(){
  bucketPartStart = 0;
  bucketPartCount = 0;
  if(currentBucket == NULL || ownerTreePiece == NULL) return;
  // An EmptyBucket is a legitimate target: it is in the PE's bucket list and
  // the traversal walks the tree for it. Its particleStart is NULL, so there
  // is nothing to take an offset from, and closeRun drops the run anyway.
  const int np = currentBucket->getNumParticles();
  if(np == 0) return;
  bucketPartStart = ownerTreePiece->particleOffset(currentBucket->getParticles());
  bucketPartCount = np;
}
#endif

// Every source a chosen bucket accepts, summed. A correct walk sees each
// particle exactly once, directly or through one multipole, so this must come
// to the universe mass. More than that is double counting.
thread_local double _srcMass = 0.0;
thread_local const void *_srcBucket = NULL;

int TraversalWorker::work(Node<ForceData> *node){
  NodeType type = node->getType();
  state->nodeEncountered(currentBucket->getKey(),node);
  bool keep = getKeep(type);

  if(!keep){
    state->nodeDiscarded(currentBucket->getKey(),node);
    return 0;
  }

  // basic opening criterion
  bool open = openCriterionBucket(node,currentBucket);
  state->incrOpenCriterion();
  if(open){
    state->nodeOpened(currentBucket->getKey(),node);
    return 1;
  }

#ifdef GPU_GRAVITY
  // The whole expansion, not just the point mass: the kernel evaluates the
  // quadrupole term too unless it was built -DMONOPOLE_ONLY.
  ownerTreePiece->getBatch().addSource(currentBucket,
                                       bucketPartStart, bucketPartCount,
                                       node->data.moments);
  int computed = currentBucket->getNumParticles();
#else
  int computed = nodeBucketForce(node,currentBucket);
#endif
  if(currentBucket == _srcBucket) _srcMass += node->data.moments.totalMass;
  state->nodeComputed(currentBucket,node->getKey());
  state->incrPartNodeInteractions(currentBucket->getKey(),computed);
  return 0;
}

void TraversalWorker::work(ExternalParticle *particle){
  if(currentBucket == _srcBucket) _srcMass += particle->mass;
#ifdef GPU_GRAVITY
  ownerTreePiece->getBatch().addSource(currentBucket,
                                       bucketPartStart, bucketPartCount,
                                       particle->mass,
                                       particle->position);
  int computed = currentBucket->getNumParticles();
#else
  int computed = partBucketForce(particle,currentBucket);
#endif
  state->incrPartPartInteractions(currentBucket->getKey(),computed);
}

void TraversalWorker::bucketDone(Key k){
  state->bucketComputed(currentBucket,k);
}

bool LocalTraversalWorker::getKeep(NodeType type){
  return keep[type];
}

bool RemoteTraversalWorker::getKeep(NodeType type){
  return keep[type];
}

void RemoteTraversalWorker::done(){
  ownerTreePiece->remoteGravityDone();
}

const bool LocalTraversalWorker::keep[] = {false,true,true,false,true,false,false,false};
const bool RemoteTraversalWorker::keep[] = {false,false,false,false,true,true,true,false};

int TreeSizeWorker::work(Node<ForceData> *node){
  int depth = node->getDepth();
  if(depth+1 <= cutoffDepth){
    numNodes += node->getNumChildren();
    return 1;
  }
  return 0;
}

int TreeChecker::work(Node<ForceData> *node){
  CkAssert(node != NULL);
  if(node->getParent() != NULL){
    os << "CHECK: " << node->getKey() << " parent " << node->getParent()->getKey() << endl;
  }
  if(node->getChildren() != 0){
    for(int i = 0; i < node->getNumChildren(); i++){
      os << "CHECK: " << node->getKey() << " child " << i << " " << node->getChild(i)->getKey() << endl;
    }
  }
  else{
    os << "CHECK: " << node->getKey() << " 0 children" << endl;
  }
  return 1;
}

int InteractionChecker::work(Node<ForceData> *node){
  if(node->getNumChildren() == 0) return 1;

  for(int i = 0; i < BRANCH_FACTOR; i++){
  }

  return 0;
}


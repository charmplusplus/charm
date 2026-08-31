#ifndef __TREE_PIECE_H__
#define __TREE_PIECE_H__

#include "defines.h"
#include "OrientedBox.h"
#include "barnes.decl.h"
#include "Particle.h"
#include "Messages.h"
#include "Node.h"
#include "Vector3D.h"
#include "State.h"

#include "Worker.h"

#include "MultipoleMoments.h"
#include "Traversal_decls.h"

#ifdef GPU_GRAVITY
#include "GpuBatch.h"
#endif

class TreePiece : public CBase_TreePiece {
  int numDecompMsgsRecvd;
  CkVec<ParticleMsg *> decompMsgsRecvd;
  int myNumParticles;

  int iteration;

  int myNumBuckets;
  Node<ForceData> **myBuckets;
  Node<ForceData> *root;

  State localTraversalState;
  State remoteTraversalState;

  LocalTraversalWorker localTraversalWorker;
  RemoteTraversalWorker remoteTraversalWorker;

  Traversal<ForceData> trav;

  Key smallestKey;
  Key largestKey;
  DataManager *myDM;

  void submitParticles();

  int localStateID;
  int remoteStateID;
  int totalNumTraversals;
  int numTraversalsDone;

  void traversalDone();
  void finishIteration();
  void reportTraversalsDone();

  // Is the iteration just finished one at which the balancer should run?
  bool isLbIteration() const;
  void lbBarrierDone();

  void checkTraversals();

#ifdef GPU_GRAVITY
  GpuTraversalBatch batch;
  // Launches issued this iteration whose HAPI callback has not yet fired.
  int outstandingKernels;
  bool traversalsComplete;

  void ensureDevice();
  void flushGpu();
  void maybeReportDone();
  void initGpuState();
#endif

  void initIterationState();

  public:
  TreePiece();
  TreePiece(CkMigrateMessage *m);

  int getIndex() {return thisIndex;}

  void receiveParticles(ParticleMsg *msg);
  void receiveParticles();

  void prepare(Node<ForceData> *_root, Node<ForceData> **buckets, int bucketStart, int bucketEnd);
  void startTraversal();

  void doLocalGravity(RescheduleMsg *);
  void doRemoteGravity(RescheduleMsg *);

  void requestParticles(RequestMsg *msg);
  void requestNode(RequestMsg *msg);

  void localGravityDone();
  void remoteGravityDone();
  void requestMoments(Key k, int replyTo);

#ifdef GPU_GRAVITY
  GpuTraversalBatch &getBatch(){ return batch; }
  // Offset of a bucket's particles within this PE's particle array. The tree,
  // and therefore the bucket, belongs to the local DataManager.
  int particleOffset(Particle *p) const;

  // Both exist so that every kernel launch happens inside an entry method of
  // this chare: that is what makes CUPTI attribute the GPU time to it. The
  // last sources of a remote traversal are appended from the DataManager's
  // reply-delivery path, which would otherwise charge the launch to the group.
  void finishGpuWork();
  void gpuWorkDone();
#endif

  void ResumeFromSync();

  void quiescence();
  int getIteration();

  void pup(PUP::er &p);
};

#endif

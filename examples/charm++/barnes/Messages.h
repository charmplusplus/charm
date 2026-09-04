#ifndef __MESSAGES_H__
#define __MESSAGES_H__

#include "Particle.h"
#include "MultipoleMoments.h"
#include "Node.h" 

struct SplitterMsg : public CMessage_SplitterMsg {
  int *splitBins;
  int nSplitBins;
};

struct ParticleMsg : public CMessage_ParticleMsg {
  Particle *part;
  int numParticles;
};

// One PE's entire contribution to another PE's tree pieces, in one message.
//
// The exchange used to be one message per (sending PE, tree piece) pair --
// numTreePieces * numPes of them, 10432 at 652 pieces on 16 PEs, and most
// carrying nothing, because a tree piece's key range is covered by only one or
// two PEs. Aggregating per destination PE takes that to numPes^2, and it is
// also the only shape a device transfer can take: at the ~80us IPC floor,
// 10432 device sends per iteration would cost far more than the host path it
// replaces.
//
// parts holds the tree pieces' particles back to back in the order tpIndex
// lists them; tpCount says how many belong to each, and tpKeys holds the
// (smallest, largest) key pair per tree piece.
struct ParticleBlockMsg : public CMessage_ParticleBlockMsg {
  int *tpIndex;
  int *tpCount;
  Key *tpKeys;
  Particle *parts;
  int numTps;
  int numParticles;
  int fromPe;
};

struct RangeMsg : public CMessage_RangeMsg {
  Key *keys;
  int numTreePieces;
};

struct RequestMsg : public CMessage_RequestMsg {
  Key key;
  int replyTo;

  RequestMsg(Key k, int reply) : 
    key(k), replyTo(reply)
  {
  }
};

struct ParticleReplyMsg : public CMessage_ParticleReplyMsg {
  Key key;
  ExternalParticle *data;
  int np;
};

struct NodeReplyMsg : public CMessage_NodeReplyMsg {
  Key key;
  Node<ForceData> *data;
  int nn;
};

struct MomentsExchangeStruct;
struct MomentsMsg : public CMessage_MomentsMsg {
  MomentsExchangeStruct data;

  MomentsMsg(Node<ForceData> *node) 
  {
    data = (*node);
  }
};

struct RescheduleMsg : public CMessage_RescheduleMsg {
};
#endif

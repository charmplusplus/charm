#ifndef PIPE_BROADCAST_STRATEGY
#define PIPE_BROADCAST_STRATEGY
#include "ComlibManager.h"

#define DEFAULT_PIPE   8196

struct PipeBcastInfo {
  short bcastPe;     // pe who is doing the broadcast, used for the hash key
  short seqNumber;
  int chunkSize;   // it is the size of the data part of the message (without the converse header)
  int chunkNumber;
  int messageSize;   // the entire message size, all included
  short srcPe;       // pe from whom I'm receiving the message
};

class PipeBcastHashKey{
 public:

    int srcPe;
    int seq;
    PipeBcastHashKey(int _pe, int _seq):srcPe(_pe), seq(_seq){};

    //These routines allow PipeBcastHashKey to be used in
    //  a CkHashtableT
    CkHashCode hash(void) const;
    static CkHashCode staticHash(const void *a,size_t);
    int compare(const PipeBcastHashKey &ind) const;
    static int staticCompare(const void *a,const void *b,size_t);
};

// sequential numbers must be below 2^16, so the number of processors must
inline CkHashCode PipeBcastHashKey::hash(void) const
{
    register int _seq = seq;
    register int _pe = srcPe;
    
    register CkHashCode ret = (_seq << 16) + _pe;
    return ret;
}

inline int PipeBcastHashKey::compare(const PipeBcastHashKey &k2) const
{
    if(seq == k2.seq && srcPe == k2.srcPe)
        return 1;
    
    return 0;
}

class PipeBcastHashObj{
 public:
  char *message;
  int dimension;
  int remaining;
  PipeBcastHashObj (int dim, int rem, char *msg) :dimension(dim),remaining(rem),message(msg) {};

};

class PipeBroadcastStrategy : public Strategy {
 protected:
  int pipeSize; // this is the size of the splitted messages, including the converse header
  int topology;
  double log_of_2_inv;
  int seqNumber;
  int isArrayDestination;
  CkArrayID destArrayID;
  CkQ <CharmMessageHolder*> *messageBuf;
  CkHashtableT<PipeBcastHashKey, PipeBcastHashObj *> fragments;
  int propagateHandle;
  int propagateHandle_frag;
  CkVec<CkArrayIndexMax> *localDest;

  void commonInit();
  void deliverer(envelope *env, int isFrag);

 public:
  PipeBroadcastStrategy();
  PipeBroadcastStrategy(int _topology);
  PipeBroadcastStrategy(int _topology, int _pipeSize);
  PipeBroadcastStrategy(int _topology, CkArrayID aid);
  PipeBroadcastStrategy(int _topology, CkArrayID aid, int _pipeSize);
  PipeBroadcastStrategy(CkMigrateMessage *){}
  void insertMessage(CharmMessageHolder *msg);
  void doneInserting();
  void conversePipeBcast(envelope *env, int size, int forceSplit);

  void propagate(envelope *env, int isFrag);

  virtual void pup(PUP::er &p);
  PUPable_decl(PipeBroadcastStrategy);
};
#endif

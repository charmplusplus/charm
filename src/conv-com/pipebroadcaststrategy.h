#ifndef PIPE_BROADCAST_CONVERSE
#define PIPE_BROADCAST_CONVERSE
#include "ckhashtable.h"
#include "charm++.h"
#include "envelope.h"
#include "convcomlibstrategy.h"
#include "convcomlibmanager.h"

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

typedef const UInt  constUInt;
typedef void (*setFunction)(char*, constUInt);

class PipeBroadcastConverse : public Strategy {
 protected:

  int pipeSize; // this is the size of the splitted messages, including the converse header
  int topology;
  double log_of_2_inv;
  int seqNumber;
  CkQ <MessageHolder*> *messageBuf;
  CkHashtableT<PipeBcastHashKey, PipeBcastHashObj *> fragments;
  //int propagateHandle;
  int propagateHandle_frag;

 public:
  PipeBroadcastConverse(int, int, Strategy*);
  PipeBroadcastConverse(CkMigrateMessage *) {};
  int getPipeSize() { return pipeSize; };
  void commonInit();
  void deliverer(char *msg);
  void storing(char *msg, int isFrag);
  void propagate(char *msg, int isFrag, int srcPeNumber, int totalSendingSize, setFunction setPeNumber);

  void conversePipeBcast(char *env, int size);
  void insertMessage(MessageHolder *msg);
  void doneInserting();

  virtual void pup(PUP::er &p);
  PUPable_decl(PipeBroadcastConverse);
};

#endif

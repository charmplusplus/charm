/**
   @addtogroup ComlibConverseStrategy
   @{
   @file 
   @brief A pipeline router strategy. 
*/


#ifndef PIPELINE_CONVERSE
#define PIPELINE_CONVERSE
#include "ckhashtable.h"
#include "charm++.h"
#include "convcomlibstrategy.h"
#include "convcomlibmanager.h"

#define DEFAULT_PIPE   8196

struct PipelineInfo {
  short bcastPe;     // pe who is doing the broadcast, used for the hash key
  short seqNumber;
  int chunkSize;   // it is the size of the data part of the message (without the converse header)
  int chunkNumber;
  int messageSize;   // the entire message size, all included
  short srcPe;       // pe from whom I'm receiving the message
};

class PipelineHashKey{
 public:

    int srcPe;
    int seq;
    PipelineHashKey(int _pe, int _seq):srcPe(_pe), seq(_seq){};

    //These routines allow PipelineHashKey to be used in
    //  a CkHashtableT
    CkHashCode hash(void) const;
    static CkHashCode staticHash(const void *a,size_t);
    int compare(const PipelineHashKey &ind) const;
    static int staticCompare(const void *a,const void *b,size_t);
};

// sequential numbers must be below 2^16, so the number of processors must
inline CkHashCode PipelineHashKey::hash(void) const
{
    register int _seq = seq;
    register int _pe = srcPe;
    
    register CkHashCode ret = (_seq << 16) + _pe;
    return ret;
}

inline int PipelineHashKey::compare(const PipelineHashKey &k2) const
{
    if(seq == k2.seq && srcPe == k2.srcPe)
        return 1;
    
    return 0;
}

class PipelineHashObj{
 public:
  char *message;
  int dimension;
  int remaining;
  PipelineHashObj (int dim, int rem, char *msg) :dimension(dim),remaining(rem),message(msg) {};

};

typedef const UInt  constUInt;
typedef void (*setFunction)(char*, constUInt);

class PipelineStrategy : public Strategy {
 protected:

  int pipeSize; // this is the size of the splitted messages, including the converse header
  //double log_of_2_inv;
  int seqNumber;
  CkQ <MessageHolder*> *messageBuf;
  CkHashtableT<PipelineHashKey, PipelineHashObj *> fragments;
  int deliverHandle;

 public:
  PipelineStrategy(int size=DEFAULT_PIPE, Strategy* st=NULL);
  PipelineStrategy(CkMigrateMessage *) {};
  int getPipeSize() { return pipeSize; };
  void commonInit();
  void deliverer(char *msg, int dim);
  void storing(char *msg);

  void conversePipeline(char *env, int size, int destination);
  void insertMessage(MessageHolder *msg);
  void doneInserting();

  virtual void pup(PUP::er &p);
  PUPable_decl(PipelineStrategy);
};

#endif
/*@}*/

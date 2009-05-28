/**
   @addtogroup ComlibConverseStrategy
   @{
   @file
   @brief Header for the PipeBroadcastConverse strategy
*/

#ifndef PIPE_BROADCAST_CONVERSE
#define PIPE_BROADCAST_CONVERSE
#include "ckhashtable.h"
//#include "charm++.h"
//#include "envelope.h"
#include "convcomlibmanager.h"

#define DEFAULT_PIPE   8196

CkpvExtern(int, pipeline_handler);
extern void PipelineHandler(void *msg);
CkpvExtern(int, pipeline_frag_handler);
extern void PipelineFragmentHandler(void *msg);

/**
 * Header used to split messages for the pipelining. Due to the usage of short
 * types, this will not work in machines like BG/L.
 */
struct PipeBcastInfo {
  short bcastPe;     ///< pe who is doing the broadcast, used for the hash key
  short seqNumber;   ///< timestamp of the message from processor bcastPe, the other part of the hash key
  int chunkSize;     ///< it is the size of the data part of the message (without the converse header)
  int chunkNumber;   ///< which chunk is this inside the whole message
  int messageSize;   ///< the entire message size, all included
  short srcPe;       ///< pe from whom I'm receiving the message
};

/**
 * The hash key for indexing incoming fragmented messages while waiting to
 * reassemble them. It is composed by the sourcePe and the timestamp it gave to
 * the message (sequential number per processor)
 */
class PipeBcastHashKey {
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

/// sequential numbers must be below 2^16, so the number of processors must
inline CkHashCode PipeBcastHashKey::hash(void) const {
    register int _seq = seq;
    register int _pe = srcPe;
    
    register CkHashCode ret = (_seq << 16) + _pe;
    return ret;
}

inline int PipeBcastHashKey::compare(const PipeBcastHashKey &k2) const {
    if(seq == k2.seq && srcPe == k2.srcPe)
        return 1;
    
    return 0;
}

/**
 * The message in reassembling
 */
class PipeBcastHashObj {
 public:
  char *message;
  int dimension; ///< total size of the reassembled message
  int remaining; ///< how many messages we are still waiting before reassembly is done
  PipeBcastHashObj (int dim, int rem, char *msg) :dimension(dim),remaining(rem),message(msg) {};

};

//typedef const UInt  constUInt;
//typedef void (*setFunction)(char*, constUInt);

/**
 * PipeBroadcastConverse streams broadcast messages to all processors, and
 * delivers them using the "deliver" method. These messages are splitted into
 * multiple packets and pipelined throughout the network. The typical routing
 * algorithm is hypercube (linear array is also available) and the information
 * of the next hops is computed at runtime without any overhead to store data in
 * the memory.
 */
class PipeBroadcastConverse : public Strategy {
 protected:

  int pipeSize; ///< this is the size of the splitted messages, including the converse header
  short topology; ///< which topology to use: Hypercube vs. Linear
  //double log_of_2_inv;
  CmiUInt2 seqNumber; ///< the sequential numbering in this processor
  //CkQ <MessageHolder*> *messageBuf;

  /// All the messages which are currently fragmented and we are reassembling
  CkHashtableT<PipeBcastHashKey, PipeBcastHashObj *> fragments;
  //int propagateHandle;
  //int propagateHandle_frag;

  // WARNING: All pure converse messages need to insert a "CmiFragmentHeader"
  // right after the ConverseHeader at the beginning of the user data.

  /// return the pointer to where the structure CmiFragmentHeader is in the message
  virtual CmiFragmentHeader *getFragmentHeader(char *msg);

 public:
  PipeBroadcastConverse(short top=USE_HYPERCUBE, int size=DEFAULT_PIPE);
  PipeBroadcastConverse(CkMigrateMessage *m): Strategy(m) {};
  int getPipeSize() { return pipeSize; };

  //void commonInit();
  // the deliver function deletes the message afterwards
  virtual void deliver(char *msg, int dim);

  void handleMessage(void*) {
    CmiAbort("PipeBroadcastConverse::handleMessage, this should never be used!\n");
  }

  /** Store the fragment of the message into the hashtable, and deliver it when
      the storing fragment is the last of the message */
  void store(char *msg);

  /** Forward the message to the next processors in the list */
  void propagate(char *msg, int isFrag);//, int srcPeNumber, int totalSendingSize, setFunction setPeNumber);

  //void conversePipeBcast(char *env, int size);
  void insertMessage(MessageHolder *msg);
  //void doneInserting();

  virtual void pup(PUP::er &p);
  PUPable_decl(PipeBroadcastConverse);
};

#endif

/*@}*/

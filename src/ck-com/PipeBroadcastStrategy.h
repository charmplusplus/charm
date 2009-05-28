/**
   @addtogroup ComlibCharmStrategy
   @{
   @file
   
   Header for the PipeBroadcastStrategy strategy
*/

#ifndef PIPE_BROADCAST_STRATEGY
#define PIPE_BROADCAST_STRATEGY
#include "ComlibManager.h"
#include "pipebroadcastconverse.h"

/**
 * Strategy that performs a broadcast over an entire group or array of chares.
 * This strategy utilized the lower level PipeBroadcastConverse to perform the
 * operation.
 */
class PipeBroadcastStrategy : public PipeBroadcastConverse, public CharmStrategy {
 protected:


 public:
  PipeBroadcastStrategy(int _topology, CkArrayID _aid, int _pipeSize=DEFAULT_PIPE);
  PipeBroadcastStrategy(CkGroupID _gid, int _topology=USE_HYPERCUBE, int _pipeSize=DEFAULT_PIPE);
  PipeBroadcastStrategy(CkMigrateMessage *m): PipeBroadcastConverse(m), CharmStrategy(m) {}
  void insertMessage(CharmMessageHolder *msg);

  virtual CmiFragmentHeader *getFragmentHeader(char*);

  void deliver(char *msg, int dim);

  virtual void pup(PUP::er &p);
  PUPable_decl(PipeBroadcastStrategy);
};
#endif

/*@}*/

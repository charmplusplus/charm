#ifndef PIPE_BROADCAST_STRATEGY
#define PIPE_BROADCAST_STRATEGY
#include "ComlibManager.h"
#include "pipebroadcastconverse.h"

class PipeBroadcastStrategy : public CharmStrategy {
 protected:

  int propagateHandle;
  CkQ <CharmMessageHolder*> *messageBuf;
  CkVec<CkArrayIndexMax> *localDest;

  void commonInit(int _top, int _pipe);

 public:
  PipeBroadcastStrategy(int _topology=USE_HYPERCUBE, int _pipeSize=DEFAULT_PIPE);
  PipeBroadcastStrategy(int _topology, CkArrayID _aid, int _pipeSize=DEFAULT_PIPE);
  PipeBroadcastStrategy(CkGroupID _gid, int _topology=USE_HYPERCUBE, int _pipeSize=DEFAULT_PIPE);
  PipeBroadcastStrategy(CkMigrateMessage *){}
  void insertMessage(CharmMessageHolder *msg);
  void doneInserting();
  void conversePipeBcast(envelope *env, int size);

  void deliverer(char *msg, int dim);

  virtual void pup(PUP::er &p);
  PUPable_decl(PipeBroadcastStrategy);
};
#endif

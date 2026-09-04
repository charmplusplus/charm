#ifndef __REQUEST_H__
#define __REQUEST_H__

#include "Node.h"


template<typename T> class CutoffWorker;
class State;
template<typename T> class Traversal;

struct Requestor {
  CutoffWorker<ForceData> *worker;
  State *state;
  Traversal<ForceData> *traversal;
  void *context;

  Requestor(CutoffWorker<ForceData> *w, State *s, Traversal<ForceData> * t, void *ctx) :
    worker(w),
    state(s),
    traversal(t),
    context(ctx)
  {
  }

  Requestor() : 
    worker(NULL), state(NULL), traversal(NULL), context(NULL) 
  {
  }
};

struct Request {
  // don't need data field
  void *data;
  bool sent;
  CkVec<Requestor> requestors;
  void *msg;

  Node<ForceData> *parent; 
  bool parentCached;

  // Stage 0 instrumentation: when the request went on the wire, so recvNode /
  // recvParticles can charge the round trip. Only read under BARNES_PHASE_REPORT.
  double sentAt;

  Request() : 
    data(NULL),
    sent(false),
    msg(NULL),
    parent(NULL),
    sentAt(0.0)
  {
  }

  void reset(){
    data = NULL;
    sent = false;
    sentAt = 0.0;
  }

  void deliverParticles(int num);
  void deliverNode();
};

#endif

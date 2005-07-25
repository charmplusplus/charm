#include "comlib.h"

#include "pgm.decl.h"

extern CProxy_Main mainProxy;
extern CProxy_Source src;
extern CProxy_Destination dest;

extern ComlibInstanceHandle strat_direct;
extern ComlibInstanceHandle strat_ring;
extern ComlibInstanceHandle strat_multiring;

class MyMulticastMessage : public CkMcastBaseMsg, public CMessage_MyMulticastMessage {
 public:
  char *data;
};

class Main : public CBase_Main {
 private:
  int nsrc, ndest;
  int count;
  int iteration;
 public:
  Main(CkArgMsg *m);
  void done();
};

class Source : public CBase_Source {
 private:
  /* local variables used to keep the sections for the multicast.
     each of them will be associated with a specific strategy */
  CProxySection_Destination direct_section;
  CProxySection_Destination ring_section;
  CProxySection_Destination multiring_section;
 public:
  Source(int n, int *list);
  Source(CkMigrateMessage *m) {}
  void start(int i);
};

class Destination : public CBase_Destination {
 private:
  int waiting;
  int nsrc;
 public:
  Destination(int senders);
  Destination(CkMigrateMessage *m) {}
  void receive(MyMulticastMessage *m);
};

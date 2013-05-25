#include <stdlib.h>
#include <alloca.h>
#include <time.h>
#include <string.h>

#include "pgm.h"
#include "rand48_replacement.h"

/* Proxies to allow charm structures to talk to each other */
CProxy_Main mainProxy;
CProxy_Source src;
CProxy_Destination dest;

/* Declaration of the three readonly variables used to hold the instances of
   multicast strategies. In this way the association with the ProxySections can
   be performed on any processor. */
ComlibInstanceHandle strat_direct;
ComlibInstanceHandle strat_ring;
ComlibInstanceHandle strat_multiring;

char *strategyName[3] = {"DirectMulticastStrategy", "RingMulticastStrategy", "MultiRingStrategy"};

Main::Main(CkArgMsg *m) {
  count = 0;

  // some hardcoded values
  nsrc = 1000;
  ndest = 1000;
  int mcastFactor = 50;
  iteration = 0;

  mainProxy = thishandle;

  // create the random mapping of sender-receiver
  char **matrix = (char **) alloca(nsrc * sizeof(char*));
  matrix[0] = (char *) malloc(nsrc * ndest * sizeof(char));
  memset(matrix[0], 0, nsrc * ndest * sizeof(char));
  for (int i=1; i<nsrc; ++i) matrix[i] = matrix[i-1] + ndest;

  srand48(time(NULL));
  for (int i=0; i<nsrc; ++i) {
    for (int j=0; j<mcastFactor; ++j) {
      int k = int(drand48() * ndest);
      matrix[i][k] = 1;
    }
  }

  // create the source and destination proxies
  src = CProxy_Source::ckNew();
  int *indices = (int *) alloca (mcastFactor * sizeof(int));
  for (int i=0; i<nsrc; ++i) {
    int idx = 0;
    //CkPrintf("%d: sending to ",i);
    for (int j=0; j<ndest; ++j) {
      if (matrix[i][j] > 0) {
	indices[idx++] = j;
	//CkPrintf("%d ",j);
      }
    }
    //CkPrintf("\n");
    src[i].insert(idx, indices);
  }
  src.doneInserting();

  dest = CProxy_Destination::ckNew();
  int empty = 0;
  for (int i=0; i<ndest; ++i) {
    int cnt = 0;
    for (int j=0; j<nsrc; ++j)
      if (matrix[j][i] > 0) cnt++;
    //CkPrintf("dest %d: receiving from %d\n",i,cnt);
    dest[i].insert(cnt);
    if (cnt == 0) empty++;
  }
  dest.doneInserting();
  ndest -= empty;

  // create the strategies and register them to commlib
  CharmStrategy *strategy = new DirectMulticastStrategy(dest);
  strat_direct = ComlibRegister(strategy);

  strategy = new RingMulticastStrategy(dest);
  strat_ring = ComlibRegister(strategy);

  strategy = new MultiRingMulticast(dest);
  strat_multiring = ComlibRegister(strategy);

  CkPrintf("Starting new iteration\n");
  src.start(++iteration);
}

/* invoked when the multicast has finished, will start next iteration, or end the program */
void Main::done() {
  if (++count == ndest) {
    CkPrintf("Iteration %d (%s) finished\n", iteration,strategyName[iteration-1]);
    count = 0;
    if (iteration == 3) CkExit();
    else {
      CkPrintf("Starting new iteration\n");
      src.start(++iteration);
    }
  }
}


Source::Source(int n, int *indices) {
  CkArrayIndex *elems = (CkArrayIndex*) alloca(n * sizeof(CkArrayIndex));
  for (int i=0; i<n; ++i) elems[i] = CkArrayIndex1D(indices[i]);

  /* Create the ProxySections and associate them with the three different
     instances of commlib multicasts. */
  direct_section = CProxySection_Destination::ckNew(dest, elems, n);
  ComlibAssociateProxy(&strat_direct, direct_section);

  ring_section = CProxySection_Destination::ckNew(dest, elems, n);
  ComlibAssociateProxy(&strat_ring, ring_section);

  multiring_section = CProxySection_Destination::ckNew(dest, elems, n);
  ComlibAssociateProxy(&strat_multiring, multiring_section);
}

void Source::start(int i) {
  MyMulticastMessage *msg = new (1000) MyMulticastMessage();
  //CkPrintf("Source %d: starting multicast %d\n",thisIndex,i);
  // Perform a multicast, using one of the strategies through the associated proxies
  switch (i) {
  case 1:
    direct_section.receive(msg);
    break;
  case 2:
    ring_section.receive(msg);
    break;
  case 3:
    multiring_section.receive(msg);
    break;
  default:
    CkAbort("Invalid iteration");
  }
}


Destination::Destination(int senders) {
  nsrc = senders;
  waiting = senders;
  //CkPrintf("dest %d: waiting for %d messages\n",thisIndex,senders);
}

void Destination::receive(MyMulticastMessage *m) {
  delete m;
  //CkPrintf("Destination %d: received, remaining %d\n",thisIndex,waiting);
  if (--waiting == 0) {
    // all messages received, send message to main
    //CkPrintf("Destination %d: received all\n",thisIndex);
    waiting = nsrc;
    mainProxy.done();
  }
}

#include "pgm.def.h"

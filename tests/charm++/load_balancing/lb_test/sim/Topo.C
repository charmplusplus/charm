#include <math.h>
#include <charm++.h>

#include "Topo.h"
#include "Topo.def.h"

Topo* Topo::Create(const int _elem, const char* _topology, 
		       const int min_us, const int max_us)
{
  int topo = Select(_topology);
  if (topo == -1)
    //The user's topology name wasn't in the table-- also bad!
    CkAbort("ERROR! Topology not found!  \n");

  TopoInitMsg* tmsg = new TopoInitMsg;
  tmsg->elements = _elem;
  tmsg->topology = topo;
  tmsg->seed = 12345;
  tmsg->min_us = min_us;
  tmsg->max_us = max_us;

  return new Topo(tmsg);
}

int Topo::Select(const char* _topo)
{
  int i=0;
  while (*TopoTable[i].name) {
    if (strcasecmp(_topo,TopoTable[i].name) == 0) {
      CkPrintf("Selecting Topology %s\n",TopoTable[i].name);
      return TopoTable[i].id;
    }
    i++;
  }
  CkPrintf("Unknown topology %s\n",_topo);
  return TopoError;
}

Topo::Topo(TopoInitMsg* _m)
{
  CkPrintf("Topo Constructed\n");
  elements = _m->elements;
  topo = TopoType(_m->topology);
  seed = _m->seed;
  min_us = _m->min_us;
  max_us = _m->max_us;

  // Do this first, to make sure everyone gets the same seed
  CrnSrand(seed);

  if (CkMyPe()==0)
    CkPrintf("Generating topology %d for %d elements\n",topo,elements);
  elemlist = new Elem[elements];

  FindComputeTimes();

  // Re-seed, so we can change the graph independent of computation
  CrnSrand(seed);

  switch (topo) {
  case TopoRing:
    ConstructRing();
    break;
  case  TopoMesh2D:
    ConstructMesh2D();
    break;
  case  TopoRandGraph:
    ConstructRandGraph();
    break;
  };

  delete _m;
}

void Topo::FindComputeTimes()
{
  int i;
  int total_work = 0;
  const double em1 = exp(1.) - 1;
  for(i=0; i < elements; i++) {
    double work;
    do {  
      // Gaussian doesn't give a bad enough distribution
      // work = gasdev() * devms + meanms;
      //work = (int)(((2.*devms*rand()) / RAND_MAX + meanms - devms) + 0.5);
      // Randomly select 10% to do 4x more work
//      if ((10.*rand())/RAND_MAX < 1.)
//	work *= 4;
//      work = meanms-devms + 2*devms*(exp((double)rand()/RAND_MAX)-1) / em1;
      work = min_us + (max_us-min_us)*pow((double)CrnRand()/RAND_MAX,4.);
    } while (work < 0);
    elemlist[i].work = work;
    total_work += work;
  }
  if (CkMyPe() == 0)
    CkPrintf("[%d] Total work/step = %f sec\n",CkMyPe(),total_work*1e-6);
      
}

float Topo::gasdev()
{
  // Based on Numerical Recipes, but throw away extra deviate.
  float fac,r,v1,v2;

  do {
    v1 = (CrnRand() * 2.)/RAND_MAX - 1.;
    v2 = (CrnRand() * 2.)/RAND_MAX - 1.;
    r = v1*v1 + v2*v2;
  } while (r >= 1.0);
  fac = sqrt(-2.0*log(r)/r);
  return v2 * fac;
}

void Topo::ConstructRing()
{
  int i;
  for(i=0;i<elements;i++) {
    elemlist[i].receivefrom = new MsgInfo;
    elemlist[i].sendto = new MsgInfo;
 
    int from = i-1;
    if (from < 0) from = elements-1;
    elemlist[i].receivefrom->obj = from;
    elemlist[i].receivefrom->bytes = N_BYTES;
    elemlist[i].receiving = 1;

    int to = i+1;
    if (to == elements) to = 0;
    elemlist[i].sendto->obj = to;
    elemlist[i].sendto->bytes = N_BYTES;
    elemlist[i].sending = 1;
  }
}

void Topo::ConstructMesh2D()
{
  // How should I build a mesh?  I'll make it close to square, and not
  // communicate with nonexistent elements

  int nrows = sqrt(elements) + 0.5; // Round it
  if (nrows < 1) nrows = 1;
  int ncols = elements / nrows;
  while (nrows * ncols < elements) ncols++;

  if (CkMyPe() == 0)
    CkPrintf("Building a %d x %d mesh, with %d missing elements\n",
	     nrows,ncols,nrows*ncols-elements);

  int i;
  for(i=0;i<elements;i++) {
    elemlist[i].receivefrom = new MsgInfo[4];
    elemlist[i].sendto = new MsgInfo[4];
    elemlist[i].receiving = elemlist[i].sending = 0;
  }

  for(i=0;i<elements;i++) {
    const int r = i / ncols;
    const int c = i % ncols;

    const int to_r[4] = { r+1, r,   r-1, r   };
    const int to_c[4] = { c,   c+1, c,   c-1 };

    for(int nbor = 0; nbor < 4; nbor++) {
      int dest_r = to_r[nbor];
      int dest_c = to_c[nbor];
      if (   dest_r >= nrows || dest_r < 0
	  || dest_c >= ncols || dest_c < 0 )
	continue;

      int dest = dest_r * ncols + dest_c;
      if (dest >= elements || dest < 0) 
	continue;

      // CkPrintf("[%d]Element %d (%d,%d) is sending to element %d(%d,%d)\n",
      //           CkMyPe(),i,r,c,dest,dest_r,dest_c);

      elemlist[i].sendto[elemlist[i].sending].obj = dest;
      elemlist[i].sendto[elemlist[i].sending].bytes = N_BYTES;
      elemlist[i].sending++;

      elemlist[dest].receivefrom[elemlist[dest].receiving].obj = i;
      elemlist[dest].receivefrom[elemlist[dest].receiving].bytes = N_BYTES;
      elemlist[dest].receiving++;
    }
  }
}

void Topo::ConstructRandGraph()
{
  // First, build a ring.  Then add more random links on top of those
  // To build the links, we will make a big temporary array for connections
  const int num_connections = elements * (elements - 1);
  int connections_made = 0;
  int connections_tried = 0;

  const double ratio = .01;

  int i;
  for(i=0;i<elements;i++)
    elemlist[i].receiving = elemlist[i].sending = 0;

  // To save memory, I will use this slightly more complicated algorithm
  // A) For each from element
  //    1) Compute the from links for each processor
  //    2) Allocate and store enough memory for that list
  //    3) Keep track of how many each receiver will get
  // B) For each to element
  //    1) Allocate enough elements for the to list
  // C) For each from element
  //    1) Copy the from list to the to list

  int* receivefrom = new int[elements];
  for(i=0;i<elements;i++)
    receivefrom[i] = 0;

  int from;
  for(from=0; from < elements; from++) {
    int n_sends = 0;
    MsgInfo* sendto = new MsgInfo[elements];

    // First, build the ring link
    int to = (from+1) % elements;
    receivefrom[to]++;
    sendto[n_sends].obj = to;
    sendto[n_sends].bytes = N_BYTES;
    n_sends++;
    connections_made++;
    connections_tried++;

    // Now check out the rest of the links for this processor
    // Examine each possible destination
    for(int j=2; j < elements; j++) {
      const int to = (from + j) % elements;
      const int to_make = ratio * num_connections - connections_made;
      const int to_try = num_connections - connections_tried;
      const double myrand = ((double) CrnRand() * to_try) / RAND_MAX;

      if (myrand < to_make) {
	int findx = n_sends++;
	sendto[findx].obj = to;
	sendto[findx].bytes = N_BYTES;
	receivefrom[to]++;
	connections_made++;
      }
      connections_tried++;
    }
    // Okay, now we have all of the outgoing links for this processor,
    // so we just have to copy them into the elemlist
    if (n_sends > elements)
      CkPrintf("%s:%d Too many sends attempted %d %d\n",
	       __FILE__,__LINE__,n_sends,elements);
    elemlist[from].sending = n_sends;
    elemlist[from].sendto = new MsgInfo[n_sends];
    int i;
    for(i=0;i<n_sends;i++)
      elemlist[from].sendto[i] = sendto[i];

    delete [] sendto;
  }

  // Now that we've created all of the send lists, and we know how many
  // elements we will receive, we can create the receivefrom lists
  for(int to=0; to < elements; to++)
    elemlist[to].receivefrom = new MsgInfo[receivefrom[to]];

  for(from=0;from<elements;from++) {
    for(int i=0; i < elemlist[from].sending; i++) {
      int to = elemlist[from].sendto[i].obj;
      if (elemlist[to].receiving < receivefrom[to]) {
	int tindex = elemlist[to].receiving++;
	elemlist[to].receivefrom[tindex].obj = from;
	elemlist[to].receivefrom[tindex].bytes = N_BYTES;
      } else {
	CkPrintf("%s:%d Too many receives going to %d: %d\n",
		 __FILE__,__LINE__,to,elemlist[to].receiving);
      }
    }
  }

  delete [] receivefrom;

  if (CkMyPe() == 0) {
    CkPrintf(
      "Built random graph with %d of %d possible links (%f percent)\n",
      connections_made,num_connections,
      (100.0*connections_made)/num_connections);
  }
}

int Topo::SendCount(int index)
{
  return elemlist[index].sending;
}

void Topo::SendTo(int index, MsgInfo* who)
{
  for(int i=0; i < elemlist[index].sending; i++)
    who[i] = elemlist[index].sendto[i];
}

int Topo::RecvCount(int index)
{
  return elemlist[index].receiving;
}

void Topo::RecvFrom(int index, MsgInfo* who)
{
  for(int i=0; i < elemlist[index].receiving; i++)
    who[i] = elemlist[index].receivefrom[i];
}

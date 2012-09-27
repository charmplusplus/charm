#include <math.h>
#include <charm++.h>

#if defined(_WIN32)
#define strcasecmp stricmp
#endif

#include "Topo.h"
#include "Topo.def.h"
#include "lb_test.decl.h"

#define LINEARLY_GRADED                            0

/* readonly*/ extern CProxy_main mainProxy;

CkGroupID Topo::Create(const int _elem, const char* _topology, 
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
  CkGroupID topo_id = CProxy_Topo::ckNew(tmsg);
  return topo_id;
}

Topo::Elem* Topo::elemlist = NULL;

int Topo::Select(const char* _topo)
{
  int i=0;
  while (TopoTable[i].name) {
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
  elements = _m->elements;
  topo = TopoType(_m->topology);
  seed = _m->seed;
  min_us = _m->min_us;
  max_us = _m->max_us;

  // Do this first, to make sure everyone gets the same seed
  srand(seed);

  if (CkMyPe()==0)
    CkPrintf("Generating topology %d for %d elements\n",topo,elements);

  if (elemlist) {
    delete _m;
    return; 
  }

  elemlist = new Elem[elements];

  FindComputeTimes();

  // Re-seed, so we can change the graph independent of computation
  srand(seed);

	switch (topo) {
	case TopoRing:
		ConstructRing();
		break;
	case  TopoMesh2D:
		ConstructMesh2D();
		break;
	case TopoMesh3D:
		ConstructMesh3D();
		break;
	case TopoRandGraph:
		ConstructRandGraph();
		break;
	};

  delete _m;
}

// Function to change loads in all the elements
void Topo::shuffleLoad(){

	//printf("[%d] At shuffleLoad\n",CkMyPe());
	
	// calling function to assign new loads
	FindComputeTimes();

	// reduction to continue execution
	contribute(CkCallback(CkIndex_main::resume(),mainProxy));
}

void Topo::FindComputeTimes()
{
  int i;
  double total_work = 0;
  const double em1 = exp(1.) - 1;
  for(i=0; i < elements; i++) {
    double work;
    do {  
#if LINEARLY_GRADED
      int mype = i/(elements/CkNumPes());
      double max_on_pe = min_us + 1.0*(max_us - min_us)*(mype+1)/CkNumPes();
      double min_on_pe = min_us + 1.0*(max_us - min_us)*(mype)/CkNumPes();
      work = min_on_pe + (max_on_pe-min_on_pe)*pow((double)rand()/RAND_MAX,4.);
#else
      // Gaussian doesn't give a bad enough distribution
      // work = gasdev() * devms + meanms;
      //work = (int)(((2.*devms*rand()) / RAND_MAX + meanms - devms) + 0.5);
      // Randomly select 10% to do 4x more work
//      if ((10.*rand())/RAND_MAX < 1.)
//	work *= 4;
//      work = meanms-devms + 2*devms*(exp((double)rand()/RAND_MAX)-1) / em1;
      work = min_us + (max_us-min_us)*pow((double)rand()/RAND_MAX,4.);
#endif
    } while (work < 0);
    elemlist[i].work = work;
    // CkPrintf("%d work %f\n", i, work);
    total_work += work;
  }
  if (CkMyNode() == 0)
    CkPrintf("[%d] Total work/step = %f sec\n",CkMyPe(),total_work*1e-6);
      
}

float Topo::gasdev()
{
  // Based on Numerical Recipes, but throw away extra deviate.
  float fac,r,v1,v2;

  do {
    v1 = (rand() * 2.)/RAND_MAX - 1.;
    v2 = (rand() * 2.)/RAND_MAX - 1.;
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

  int nrows = (int)(sqrt(1.0*elements) + 0.5); // Round it
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

/**
 * Builds a 3D mesh by creating the smallest cube containing all the objects.
 * The cube might have "holes" depending on the number of objects. We avoid
 * communication with non-existent elements.
 */
void Topo::ConstructMesh3D()
{
	// variable to store the length of one side of the cube
	int length, i;

	// computing the size of one side of the cube
	length = (int)ceil(pow((double)elements,(double)1/3));

	if (CkMyPe() == 0)
    	CkPrintf("Building a %d x %d x %d mesh, with %d missing elements\n",
	    	 length,length,length,length*length*length-elements);

	// initializing elemlist array, each element will potentially 
	// communicate with other 6 neighbors
	for(i=0;i<elements;i++) {
    	elemlist[i].receivefrom = new MsgInfo[6];
    	elemlist[i].sendto = new MsgInfo[6];
    	elemlist[i].receiving = elemlist[i].sending = 0;
	}

	// filling up neihgbor list for each element
	for(i=0; i<elements; i++) {
		
		// transforming linear index into (x,y,z) coordinates
		const int x = i % length;
		const int y = ((i - x) % (length*length)) / length;
		const int z = (i - y*length -x) / (length*length);
		const int to_x[6] = { x-1, x+1,   x,   x,   x,   x };
		const int to_y[6] = {   y,   y, y-1, y+1,   y,   y };
		const int to_z[6] = {   z,   z,   z,   z, z-1, z+1 };

		//DEBUG CkPrintf("[%d] Element %d -> (%d,%d,%d)\n",CkMyPe(),i,x,y,z);

		//  adding each neighbor to list
		for(int nbor = 0; nbor < 6; nbor++) {
			int dest_x = to_x[nbor];
			int dest_y = to_y[nbor];
			int dest_z = to_z[nbor];
			if (dest_x >= length || dest_x < 0 || dest_y >= length || dest_y < 0 || dest_z >= length || dest_z < 0 )
				continue;

			int dest = dest_z * length*length + dest_y*length + dest_x;
			if (dest >= elements || dest < 0) 
				continue;

 			//DEBUG CkPrintf("[%d]Element %d (%d,%d,%d) is sending to element %d (%d,%d,%d)\n",
 				//DEBUG CkMyPe(),i,x,y,z,dest,dest_x,dest_y,dest_z);

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
      const int to_make = (int)(ratio * num_connections - connections_made);
      const int to_try = num_connections - connections_tried;
      const double myrand = ((double) rand() * to_try) / RAND_MAX;

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

#include "user.hh"

start::start(CkArgMsg* msg)
{
  if (msg->argc != 5)
  {
    ckout << "Usage: " << msg->argv[0]
          << " <chare_array_size> <num_data_points_per_chare_array_element> "
             "<num_bits_for_pe> <num_bits_for_data_points>"
          << endl;
    CkExit();
  }

  n = atoi(msg->argv[1]);
  k = atoi(msg->argv[2]);
  x = atoi(msg->argv[3]);
  y = atoi(msg->argv[4]);
  delete msg;

  sim = CProxy_simBox::ckNew(thisProxy, k, n, x, y, n);

#ifdef FLOODING
  AllGather = CProxy_AllGather::ckNew(k, (int)allGatherType::ALL_GATHER_FLOODING);
#endif

#ifdef HYPERCUBE
  AllGather = CProxy_AllGather::ckNew(k, (int)allGatherType::ALL_GATHER_HYPERCUBE);
#endif

#ifdef RING
  AllGather = CProxy_AllGather::ckNew(k, (int)allGatherType::ALL_GATHER_RING);
#endif

  sim.begin(AllGather);
}

void start::fini(int numDone)
{
  if (numDone == n)
  {
    ckout << "[STATUS] Completed the AllGather Simulation" << endl;
    CkExit();
  }
}

simBox::simBox(CProxy_start startProxy, int k, int n, int x, int y)
    : startProxy(startProxy), k(k), n(n), x(x), y(y)
{
  result = (long int*)CkRdmaAlloc(k * n * sizeof(long int));
  data = (long int*)CkRdmaAlloc(k * sizeof(long int));
  long int max_serial = (1 << y) - 1;
  long int base = thisIndex;
  while (max_serial > 0)
  {
    base = base * 10;
    max_serial = max_serial / 10;
  }
  for (int i = 0; i < k; i++)
  {
    data[i] = base + i;
  }
}

void simBox::begin(CProxy_AllGather AllGatherGroup)
{
  CkCallback cb(CkIndex_simBox::done(NULL), CkArrayIndex1D(thisIndex), thisProxy);
  AllGather* libptr = AllGatherGroup.ckLocalBranch();
  libptr->init(result, data, thisIndex, cb);
}

void simBox::done(allGatherMsg* msg)
{
  bool success = true;
  for (int i = 0; i < n; i++)
  {
    long int max_serial = (1 << y) - 1;
    long int base = i;
    while (max_serial > 0)
    {
      base = base * 10;
      max_serial = max_serial / 10;
    }
    for (int j = 0; j < k; j++)
    {
      if (result[i * k + j] != base + j)
      {
        success = false;
        break;
      }
    }
    if (!success)
      break;
  }

  if (success)
    ckout << "[STATUS] Correct result for Chare " << thisIndex << endl;
  else
  {
    ckout << "[STATUS] Incorrect result for Chare " << thisIndex << endl;
    for (int i = 0; i < n * k; i++)
    {
      ckout << result[i] << " ";
    }
    ckout << endl;
  }
  int cnt = 1;
  CkCallback cbfini(CkReductionTarget(start, fini), startProxy);
  contribute(sizeof(int), &cnt, CkReduction::sum_int, cbfini);
}

#include "user.def.h"

#include "allGather.hh"

double alpha;
double beta;

int AllGather::gen_rand() {
  std::mt19937_64 gen(randCounter++);
  std::uniform_int_distribution<int> dis(0, n - 1);
  return dis(gen);
}

AllGather::AllGather(int k, int n, int type) : k(k), n(n) {
  this->type = (allGatherType)type;
  switch (type) {
  case allGatherType::ALL_GATHER_HYPERCUBE: {
    if ((n & (n - 1)) != 0) {
      HypercubeRecursiveDoubling = true;
    }
    numHypercubeIter = std::ceil(std::log2(n));
  } break;
  case allGatherType::ALL_GATHER_FLOODING: {
    graph.resize(n);
    for (int i = 0; i < n; i++) {
      graph[i].resize(n);
    }
    // Create a connected graph
    // Ring
    for (int i = 0; i < n; i++) {
      graph[i][(n + i + 1) % n] = 1;
      graph[i][(n + i - 1) % n] = 1;
    }
    // Random [n/2] connections
    for (int i = 0; i < 6; i++) {
      int x = gen_rand();
      int y = gen_rand();
      if (x != y) {
        graph[x][y] = 1;
        graph[y][x] = 1;
      }
    }
  } break;
  case allGatherType::ALL_GATHER_RING: {
  } break;
  }
}

// will be called only for index 0
void AllGather::initdone() {
  static int num_init_done = 0;
  num_init_done++;
  if (num_init_done == n) {
    thisProxy.startGather();
  }
}

// TODO: remove this broadcast
void AllGather::init(long int *result, long int *data, int idx, CkCallback cb) {
  this->lib_done_callback = cb;
  this->idx = idx;
  zero_copy_callback =
      CkCallback(CkIndex_AllGather::local_buff_done(NULL), thisProxy[CkMyPe()]);
  dum_dum = CkCallback(CkCallback::ignore);
  this->store = result;
  this->data = data;
  thisProxy[0].initdone();
}

void AllGather::local_buff_done(CkDataMsg *m) {
  numRecvMsg++;
  if (numRecvMsg == n - 1) {
    lib_done_callback.send(msg);
  }
}

void AllGather::startGather() {
  for (int i = 0; i < k; i++) {
    store[k * idx + i] = data[i];
  }
  CkNcpyBuffer src(data, k * sizeof(long int), dum_dum, CK_BUFFER_UNREG);

  switch (type) {
  case allGatherType::ALL_GATHER_RING: {
    thisProxy[(idx + 1) % n].recvRing(idx, src);
  } break;
  case allGatherType::ALL_GATHER_HYPERCUBE: {
    hyperCubeIndx.push_back(idx);
    hyperCubeStore.push_back(src);
    thisProxy[idx].Hypercube();
  } break;
  case allGatherType::ALL_GATHER_FLOODING: {
    recvFloodMsg[idx] = true;
    for (int i = 0; i < n; i++) {
      if (graph[idx][i] == 1) {
        thisProxy[i].Flood(idx, src);
      }
    }
  } break;
  }
}

void AllGather::recvRing(int sender, CkNcpyBuffer src) {
  CkNcpyBuffer dst(store + sender * k, k * sizeof(long int), zero_copy_callback,
                   CK_BUFFER_UNREG);
  dst.get(src);
  if (((CkMyPe() + 1) % n) != sender) {
    thisProxy[(CkMyPe() + 1) % n].recvRing(sender, src);
  }
}

void AllGather::Flood(int sender, CkNcpyBuffer src) {
  if (recvFloodMsg[sender]) {
    return;
  }
  recvFloodMsg[sender] = true;
  CkNcpyBuffer dst(store + sender * k, k * sizeof(long int), zero_copy_callback,
                   CK_BUFFER_UNREG);
  dst.get(src);
  for (int i = 0; i < n; i++) {
    if (graph[CkMyPe()][i] == 1 and i != sender) {
      thisProxy[i].Flood(sender, src);
    }
  }
}

#include "allGather.def.h"

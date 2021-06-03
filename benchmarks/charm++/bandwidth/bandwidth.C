#include "bandwidth.decl.h"

#include <cstring>
#define MAX_ITERS 10000
#define LARGE_MESSAGE_SIZE 8192

/* readonly */ CProxy_Main main_proxy;
/* readonly */ CProxy_Block block_proxy;
/* readonly */ size_t min_size;
/* readonly */ size_t max_size;
/* readonly */ int n_iters_reg;
/* readonly */ int n_iters_large;
/* readonly */ int warmup_iters;
/* readonly */ int window_size;

class Main : public CBase_Main {
  int cur_size;
  double start_time;

 public:
  Main(CkArgMsg* m) {
    main_proxy = thisProxy;
    min_size = 1;
    max_size = 4194304;
    n_iters_reg = 1000;
    n_iters_large = 100;
    warmup_iters = 10;
    window_size = 64;

    if (CkNumPes() != 2) {
      CkPrintf("Error: there should be 2 PEs");
      CkExit(1);
    }

    // Process command line arguments
    int c;
    while ((c = getopt(m->argc, m->argv, "s:x:i:l:w:d:z")) != -1) {
      switch (c) {
        case 's':
          min_size = atoi(optarg);
          break;
        case 'x':
          max_size = atoi(optarg);
          break;
        case 'i':
          n_iters_reg = atoi(optarg);
          break;
        case 'l':
          n_iters_large = atoi(optarg);
          break;
        case 'w':
          warmup_iters = atoi(optarg);
          break;
        case 'd':
          window_size = atoi(optarg);
          break;
        default:
          CkPrintf("Unknown command line argument detected");
          CkExit(1);
      }
    }
    delete m;

    if (n_iters_reg > MAX_ITERS || n_iters_large > MAX_ITERS) {
      CkPrintf("Number of iterations must be less than %d\n", MAX_ITERS);
      CkExit(1);
    }

    // Print info
    CkPrintf("# Charm++ GPU Bandwidth Test\n"
        "# Message sizes: %lu - %lu bytes\n# Window size: %d\n"
        "# Iterations: %d regular, %d large\n# Warmup: %d\n#\n",
        min_size, max_size, window_size, n_iters_reg, n_iters_large, warmup_iters
        );

    // Create block group chare
    block_proxy = CProxy_Block::ckNew();
    start_time = CkWallTimer();
    block_proxy.init();
  }

  void initDone() {
    CkPrintf("Starting %s test...\n", "regular");
    cur_size = min_size;
    testSetup();
  }

  void testSetup() {
    // Tell chares to memset their GPU data, will reduce back to testStart
    block_proxy.memset(cur_size);
  }

  void testStart() {
    // Start ping
    block_proxy[0].send(cur_size);
  }

  void testEnd() {
    cur_size *= 2;
    if (cur_size <= max_size) {
      // Proceed to next message size
      thisProxy.testSetup();
    } else {
        terminate();
    }
  }

  void terminate() {
    CkPrintf("Elapsed: %.6lf s\n", CkWallTimer() - start_time);
    CkExit();
  }
};

class Block : public CBase_Block {
public:
  int iter;
  int peer;
  int recv_count;

  double start_time;
  double* times;

  char* local_data;
  char* remote_data;
  bool memory_allocated;

  Block() {
    memory_allocated = false;
  }

  ~Block() {
    if (memory_allocated) {
      if (CkMyPe() == 0) free(times);
      free(local_data);
      free(remote_data);
    }
  }

  void init() {
    // Reset iteration counter and recv count
    iter = 1;
    recv_count = 0;

    // Allocate memory for timers
    if (CkMyPe() == 0) {
      if (memory_allocated) free(times);
      times = (double*)malloc(MAX_ITERS * sizeof(double));
    }

    // Determine the peer index
    peer = (thisIndex == 0) ? 1 : 0;

    // Allocate memory
    if (memory_allocated) {
      free(local_data);
      free(remote_data);
    }

    local_data = (char*) malloc(sizeof(char) * max_size);
    remote_data = (char*) malloc(sizeof(char) * max_size);
    memory_allocated = true;

    // Reduce back to main
    contribute(CkCallback(CkReductionTarget(Main, initDone), main_proxy));
  }

  void memset(size_t size) {
    std::memset(local_data, 'a', size);
    std::memset(remote_data, 'b', size);

    // Reduce back to main
    contribute(CkCallback(CkReductionTarget(Main, testStart), main_proxy));
  }

  void send(size_t size) {
    CkAssert(CkMyPe() == 0);

    start_time = CkWallTimer();
    for (int i = 0; i < window_size; i++) {
      thisProxy[peer].receiveReg(size, local_data);
    }
  }

  void receiveReg(size_t size, char* data) {
    memcpy(remote_data, data, size);
    afterReceive(size);
  }

  void afterReceive(size_t size) {
    if (++recv_count == window_size) {
      CkAssert(CkMyPe() == 1);
      recv_count = 0;
      thisProxy[peer].ack(size);
    }
  }

  void ack(size_t size) {
    CkAssert(CkMyPe() == 0);

    int n_iters = (size > LARGE_MESSAGE_SIZE) ? n_iters_large : n_iters_reg;

    if (iter > warmup_iters) {
      times[iter-warmup_iters-1] = CkWallTimer() - start_time;
    }

    // Start next iteration or end test for current size
    if (iter++ == warmup_iters + n_iters) {
      // Reset iteration
      iter = 1;

      // Calculate average/mean time
      double times_sum = 0;
      for (int i = 0; i < n_iters; i++) {
        times_sum += times[i];
      }
      double times_mean = times_sum / n_iters;

      CkPrintf("Bandwidth for %lu bytes: %.3lf MB/s\n",
          size, size / 1e6 * window_size / times_mean);
      main_proxy.testEnd();
    } else {
      send(size);
    }
  }
};

#include "bandwidth.def.h"

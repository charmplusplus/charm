#include "latency.decl.h"
#include "hapi.h"

#define MAX_ITERS 10000
#define LARGE_MESSAGE_SIZE 8192

/* readonly */ CProxy_Main main_proxy;
/* readonly */ CProxy_Block block_proxy;
/* readonly */ size_t min_size;
/* readonly */ size_t max_size;
/* readonly */ int n_iters_reg;
/* readonly */ int n_iters_large;
/* readonly */ int warmup_iters;

class Main : public CBase_Main {
  bool zerocopy;
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
    zerocopy = false;

    if (CkNumPes() != 2) {
      CkPrintf("Error: there should be 2 PEs");
      CkExit(1);
    }

    // Process command line arguments
    int c;
    while ((c = getopt(m->argc, m->argv, "s:x:i:l:w:")) != -1) {
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
    CkPrintf("# Charm++ GPU Latency Test\n"
        "# Message sizes: %lu - %lu bytes\n# Iterations: %d regular, %d large\n# Warmup: %d\n",
        min_size, max_size, n_iters_reg, n_iters_large, warmup_iters);

    // Create block group chare
    block_proxy = CProxy_Block::ckNew();
    start_time = CkWallTimer();
    block_proxy.init();
  }

  void initDone() {
    CkPrintf("Starting test...\n");
    cur_size = min_size;
    testBegin(cur_size);
  }

  void testBegin(size_t size) {
    // Start ping
    block_proxy[0].send(size);
  }

  void testEnd() {
    cur_size *= 2;
    if (cur_size <= max_size) {
      // Proceed to next message size
      thisProxy.testBegin(cur_size);
    } else {
      CkPrintf("Elapsed: %.6lf s\n", CkWallTimer() - start_time);
      CkExit();
    }
  }
};

class Block : public CBase_Block {
public:
  int iter;
  int peer;

  double start_time;
  double* times;

  size_t size;
  char* h_local_data;
  char* h_remote_data;
  char* d_local_data;
  char* d_remote_data;
  bool memory_allocated;

  cudaStream_t stream;
  bool stream_created;

  CkDevicePersistent my_send_buf;
  CkDevicePersistent my_recv_buf;
  CkDevicePersistent peer_send_buf;
  CkDevicePersistent peer_recv_buf;

  Block() {
    memory_allocated = false;
    stream_created = false;
  }

  ~Block() {
    if (memory_allocated) {
      if (CkMyPe() == 0) free(times);
      hapiCheck(cudaFreeHost(h_local_data));
      hapiCheck(cudaFreeHost(h_remote_data));
      hapiCheck(cudaFree(d_local_data));
      hapiCheck(cudaFree(d_remote_data));
    }

    if (stream_created) cudaStreamDestroy(stream);
  }

  void init() {
    // Reset iteration counter
    iter = 1;

    // Allocate memory for timers
    if (CkMyPe() == 0) {
      if (memory_allocated) free(times);
      times = (double*)malloc(MAX_ITERS * sizeof(double));
    }

    // Determine the peer index
    peer = (thisIndex == 0) ? 1 : 0;

    // Allocate memory
    if (memory_allocated) {
      hapiCheck(cudaFreeHost(h_local_data));
      hapiCheck(cudaFreeHost(h_remote_data));
      hapiCheck(cudaFree(d_local_data));
      hapiCheck(cudaFree(d_remote_data));
    }
    hapiCheck(cudaMallocHost(&h_local_data, max_size));
    hapiCheck(cudaMallocHost(&h_remote_data, max_size));
    hapiCheck(cudaMalloc(&d_local_data, max_size));
    hapiCheck(cudaMalloc(&d_remote_data, max_size));
    memory_allocated = true;

    // Create CUDA stream
    if (!stream_created) {
      cudaStreamCreate(&stream);
      stream_created = true;
    }

    // Create persistent buffers
    my_send_buf = CkDevicePersistent(d_local_data, max_size, stream);
    my_recv_buf = CkDevicePersistent(d_remote_data, max_size,
        CkCallback(CkIndex_Block::recv(), thisProxy[thisIndex]), stream);
    my_send_buf.open();
    my_recv_buf.open();
    thisProxy[peer].initRecv(my_send_buf, my_recv_buf);
  }

  void initRecv(CkDevicePersistent send_buf, CkDevicePersistent recv_buf) {
    peer_send_buf = send_buf;
    peer_recv_buf = recv_buf;

    // Reduce back to main
    contribute(CkCallback(CkReductionTarget(Main, initDone), main_proxy));
  }

  void send(size_t size) {
    this->size = size;
    if (CkMyPe() == 0) start_time = CkWallTimer();

    my_send_buf.cnt = size;
    peer_recv_buf.cnt = size;
    my_send_buf.put(peer_recv_buf);
  }

  void recv() {
    int n_iters = (size > LARGE_MESSAGE_SIZE) ? n_iters_large : n_iters_reg;

    if (CkMyPe() == 1) {
      // PE 1: send pong
      send(size);
    } else {
      // PE 0: received pong
      if (iter > warmup_iters) {
        times[iter-warmup_iters-1] = (CkWallTimer() - start_time) / 2.0;
      }

      // Start next iteration or end test for current size
      if (iter++ == warmup_iters + n_iters) {
        // Reset iteration
        iter = 1;

        // Calculate average/mean pingpong time
        double times_sum = 0;
        for (int i = 0; i < n_iters; i++) {
          times_sum += times[i];
        }
        double times_mean = times_sum / n_iters;

        // Calculate standard deviation
        double stdev = 0;
        for (int i = 0; i < n_iters; i++) {
          stdev += (times[i] - times_mean) * (times[i] - times_mean);
        }
        stdev = sqrt(stdev / n_iters);

        CkPrintf("Latency for %lu bytes: %.3lf += %.3lf us\n",
            size, times_mean * 1e6, stdev * 1e6);
        main_proxy.testEnd();
      } else {
        send(size);
      }
    }
  }
};

#include "latency.def.h"

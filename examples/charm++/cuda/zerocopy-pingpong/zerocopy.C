#include "zerocopy.decl.h"
#include "hapi.h"

#define VALIDATE 0

/* readonly */ CProxy_Main main_proxy;
/* readonly */ CProxy_Block block_proxy;
/* readonly */ int min_count;
/* readonly */ int max_count;
/* readonly */ int n_iters;
/* readonly */ bool use_zerocopy;

extern void invokeInitKernel(double*, int, double, cudaStream_t);

class Main : public CBase_Main {
  int cur_count;
  double start_time;

 public:
  Main(CkArgMsg* m) {
    main_proxy = thisProxy;
    min_count = 1;
    max_count = 1024 * 1024;
    n_iters = 100;
    use_zerocopy = false;

    if (CkNumPes() != 2) {
      CkAbort("There should be 2 PEs");
    }

    // Process command line arguments
    int c;
    while ((c = getopt(m->argc, m->argv, "s:x:i:z")) != -1) {
      switch (c) {
        case 's':
          min_count = atoi(optarg);
          break;
        case 'x':
          max_count = atoi(optarg);
          break;
        case 'i':
          n_iters = atoi(optarg);
          break;
        case 'z':
          use_zerocopy = true;
          break;
        default:
          CkAbort("Unknown command line argument detected");
      }
    }
    delete m;

    // Print info
    CkPrintf("[GPU zero-copy pingpong]\n"
        "Min count: %d doubles (%lu B), Max count: %d doubles (%lu B), "
        "Iters: %d, Zerocopy: %d\n",
        min_count, min_count * sizeof(double), max_count, max_count * sizeof(double),
        n_iters, use_zerocopy);

    // Create block group chare
    block_proxy = CProxy_Block::ckNew();
    start_time = CkWallTimer();
    block_proxy.init();
  }

  void initDone() {
    CkPrintf("Data initialized, starting test...\n");
    cur_count = min_count;
    thisProxy.testBegin(cur_count);
  }

  void testBegin(int count) {
    // Start ping
    block_proxy[0].send(count);
  }

  void testEnd() {
    cur_count *= 2;
    if (cur_count <= max_count) {
      thisProxy.testBegin(cur_count);
    }
    else {
      thisProxy.terminate();
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

  double pingpong_start_time;
  double pingpong_time_sum;

  double* h_local_data;
  double* h_remote_data;
  double* d_local_data;
  double* d_remote_data;

  cudaStream_t stream;

  Block() {}

  ~Block() {
    // Free memory and destroy CUDA stream
    hapiCheck(cudaFreeHost(h_local_data));
    hapiCheck(cudaFreeHost(h_remote_data));
    hapiCheck(cudaFree(d_local_data));
    hapiCheck(cudaFree(d_remote_data));
    cudaStreamDestroy(stream);
  }

  void init() {
    iter = 1;

    pingpong_time_sum = 0;

    // Determine the peer index
    peer = (thisIndex < CkNumPes() / 2) ? (thisIndex + CkNumPes() / 2) :
      (thisIndex - CkNumPes() / 2);

    // Allocate memory and create CUDA stream
    hapiCheck(cudaMallocHost(&h_local_data, max_count * sizeof(double)));
    hapiCheck(cudaMallocHost(&h_remote_data, max_count * sizeof(double)));
    hapiCheck(cudaMalloc(&d_local_data, max_count * sizeof(double)));
    hapiCheck(cudaMalloc(&d_remote_data, max_count * sizeof(double)));
    cudaStreamCreate(&stream);

    // Initialize data
    invokeInitKernel(d_local_data, max_count, (double)thisIndex, stream);
    invokeInitKernel(d_remote_data, max_count, (double)thisIndex, stream);
    cudaStreamSynchronize(stream);

    // Reduce back to main once data is initialized
    contribute(CkCallback(CkReductionTarget(Main, initDone), main_proxy));
    /*
    CkCallback* cb = new CkCallback(CkReductionTarget(Main, initDone), main_proxy);
    hapiAddCallback(stream, cb);
    */
  }

  void send(int count) {
    if (CkMyPe() == 0) {
      pingpong_start_time = CkWallTimer();
    }

    if (use_zerocopy) {
      thisProxy[peer].receive_zc(count, CkSendBuffer(d_local_data));
    }
    else {
      hapiCheck(cudaMemcpy(h_local_data, d_local_data, count * sizeof(double), cudaMemcpyDeviceToHost));
      thisProxy[peer].receive_reg(count, h_local_data);
    }
  }

  void receive_reg(int count, double *data) {
    // XXX: Do cudaMemcpy straight from data?
    memcpy(h_remote_data, data, count * sizeof(double));
    hapiCheck(cudaMemcpy(d_remote_data, h_remote_data, count * sizeof(double), cudaMemcpyHostToDevice));

#if VALIDATE
    validateData(count);
#endif

    afterReceive(count);
  }

  // First receive, user should set the destination buffer
  void receive_zc(int &count, double *&data, CkNcpyBufferPost *ncpyPost) {
    // Inform the runtime where the incoming data should be stored
    data = d_remote_data;
  }

  // Second receive, invoked after the data transfer is complete
  void receive_zc(int count, double *data) {
#if VALIDATE
    validateData(count);
#endif

    afterReceive(count);
  }

  void afterReceive(int count) {
    if (CkMyPe() == 1) {
      // Send pong
      thisProxy[thisIndex].send(count);
    }
    else {
      // Received pong
      double pingpong_time = CkWallTimer() - pingpong_start_time;
      pingpong_time_sum += pingpong_time;

      // Start next iteration or end test for current count
      if (iter++ == n_iters) {
        iter = 1;
        CkPrintf("Average roundtrip time for %d doubles (%lu B): %.3lf us\n",
            count, count * sizeof(double), (pingpong_time_sum / n_iters) * 1000000);
        main_proxy.testEnd();
      }
      else {
        thisProxy[thisIndex].send(count);
      }
    }
  }

  void validateData(int count) {
    // Move the data to the host for validation
    hapiCheck(cudaMemcpy(h_remote_data, d_remote_data, count * sizeof(double), cudaMemcpyDeviceToHost));

    // Validate data
    bool validated = true;
    for (int i = 0; i < count; i++) {
      if (h_remote_data[i] != (double)peer) {
        CkPrintf("h_remote_data[%d] = %lf invalid! Expected %lf\n", i,
            h_remote_data[i], (double)peer);
        validated = false;
      }
    }

    if (!validated) {
      CkPrintf("PE %d: Validation failed\n", CkMyPe());
    }
  }
};

#include "zerocopy.def.h"

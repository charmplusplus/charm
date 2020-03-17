#include "pingpong.decl.h"
#include "hapi.h"

/* readonly */ CProxy_Main main_proxy;
/* readonly */ CProxy_Block block_proxy;
/* readonly */ int min_count;
/* readonly */ int max_count;
/* readonly */ int n_iters;
/* readonly */ bool use_zerocopy;
/* readonly */ bool validate;

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
    validate = false;

    if (CkNumPes() != 2) {
      CkAbort("There should be 2 PEs");
    }

    // Process command line arguments
    int c;
    while ((c = getopt(m->argc, m->argv, "s:x:i:zv")) != -1) {
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
        case 'v':
          validate = true;
          break;
        default:
          CkAbort("Unknown command line argument detected");
      }
    }
    delete m;

    // Print info
    CkPrintf("[GPU zero-copy pingpong]\n"
        "Min count: %d doubles (%lu B), Max count: %d doubles (%lu B), "
        "Iters: %d, Zerocopy: %d, Validate: %d\n",
        min_count, min_count * sizeof(double), max_count, max_count * sizeof(double),
        n_iters, use_zerocopy, validate);

    // Create block group chare
    block_proxy = CProxy_Block::ckNew(CkNumPes());
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
  double* pingpong_times;

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

    if (CkMyPe() == 0) {
      pingpong_times = (double*)malloc(n_iters * sizeof(double));
    }

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
      hapiCheck(cudaMemcpyAsync(h_local_data, d_local_data, count * sizeof(double), cudaMemcpyDeviceToHost, stream));
      cudaStreamSynchronize(stream);
      thisProxy[peer].receive_reg(count, h_local_data);
    }
  }

  void receive_reg(int count, double *data) {
    // XXX: Do cudaMemcpy straight from data?
    memcpy(h_remote_data, data, count * sizeof(double));
    hapiCheck(cudaMemcpyAsync(d_remote_data, h_remote_data, count * sizeof(double), cudaMemcpyHostToDevice, stream));
    cudaStreamSynchronize(stream);

    if (validate) validateData(count);

    afterReceive(count);
  }

  // First receive, user should set the destination buffer
  void receive_zc(int &count, double *&data, CkNcpyBufferPost *ncpyPost) {
    // Inform the runtime where the incoming data should be stored
    // and which CUDA stream should be used for the transfer
    data = d_remote_data;
    ncpyPost[0].cuda_stream = stream;
  }

  // Second receive, invoked after the data transfer is initiated
  // The user can either wait for it to complete or offload other operations
  // into the stream (that may be dependent on the arriving data)
  void receive_zc(int count, double *data) {
    // Wait for data transfer to complete
    cudaStreamSynchronize(stream);

    if (validate) validateData(count);

    afterReceive(count);
  }

  void afterReceive(int count) {
    if (CkMyPe() == 1) {
      // Send pong
      send(count);
    }
    else {
      // Received pong
      pingpong_times[iter-1] = CkWallTimer() - pingpong_start_time;

      // Start next iteration or end test for current count
      if (iter++ == n_iters) {
        // Reset iteration
        iter = 1;

        // Calculate average/mean pingpong time
        double pingpong_times_sum = 0;
        for (int i = 0; i < n_iters; i++) {
          pingpong_times_sum += pingpong_times[i];
        }
        double pingpong_times_mean = pingpong_times_sum / n_iters;

        // Calculate standard deviation
        double stdev = 0;
        for (int i = 0; i < n_iters; i++) {
          stdev += (pingpong_times[i] - pingpong_times_mean) * (pingpong_times[i] - pingpong_times_mean);
        }
        stdev = sqrt(stdev / n_iters);

        // Calculate confidence interval
        double stderror = stdev / sqrt(n_iters);

        CkPrintf("Roundtrip time for %d doubles (%lu B): %.3lf += %.3lf us (95%% confidence)\n",
            count, count * sizeof(double), pingpong_times_mean * 1000000, 2 * stderror * 1000000);
        main_proxy.testEnd();
      }
      else {
        send(count);
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

#include "pingpong.def.h"

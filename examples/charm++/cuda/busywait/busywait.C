#include <unistd.h>
#include <string>
#include "busywait.decl.h"
#ifdef USE_NVTX
#include "hapi_nvtx.h"
#endif
#include "hapi.h"

#define CHARM_MODE 10
#define CUDA_MODE 11
#define HAPI_MODE 12

extern void blockingKernel(char*, char*, char*, char*, int, int, int, cudaStream_t, void*, void*);

/* readonly */ CProxy_Main mainProxy;
/* readonly */ int num_chares;
/* readonly */ int num_iters;
/* readonly */ float cpu_time; // sleep time in seconds
/* readonly */ float gpu_time;
/* readonly */ int data_size; // data transfer size in bytes
/* readonly */ int num_threads;
/* readonly */ int global_exec_mode;
/* readonly */ float offload_ratio;
/* readonly */ bool gpu_prio;
/* readonly */ int gpu_pes;
/* readonly */ bool sync_mode;
/* readonly */ int kernel_clock_count;

class CustomMap : public CkArrayMap {
  public:
    CustomMap() {}

    int registerArray(CkArrayIndex& numElements, CkArrayID aid) {
      return 0;
    }

    int procNum(int, const CkArrayIndex &idx) {
      int elem = *(int*)idx.data();
      int penum;
      int gpu_chares = num_chares * offload_ratio;
      if (elem < gpu_chares) {
        penum = elem % gpu_pes + (CkNumPes() - gpu_pes);
      }
      else {
        if (gpu_pes == CkNumPes()) {
          penum = 0;
        }
        else {
          penum = elem % (CkNumPes() - gpu_pes);
        }
      }

      return penum;
    }
};

class CallbackMsg : public CMessage_CallbackMsg {
  public:
    CallbackMsg() {}
};

class Main : public CBase_Main {
  private:
    CProxy_Worker workers;
    double start_time;
    double agg_time;
    int iter; // used only in sync mode

  public:
    Main(CkArgMsg* m) {
#ifdef USE_NVTX
      NVTXTracer nvtx_range("Main::Main", NVTXColor::Turquoise);
#endif
      mainProxy = thisProxy;

      // default values
      num_chares = 1;
      num_iters = 1;
      cpu_time = 1.0f;
      gpu_time = 1.0f;
      data_size = 0;
      num_threads = 1024;
      global_exec_mode = CHARM_MODE;
      offload_ratio = 0.0f;
      gpu_prio = false;
      gpu_pes = 0;
      sync_mode = false;

      // handle arguments
      int c;
      bool bFlag = false;
      while ((c = getopt(m->argc, m->argv, "c:i:a:b:r:pd:t:uyg:s")) != -1) {
        switch (c) {
          case 'c':
            num_chares = atoi(optarg);
            break;
          case 'i':
            num_iters = atoi(optarg);
            break;
          case 'a':
            cpu_time = atof(optarg);
            break;
          case 'b':
            gpu_time = atof(optarg);
            bFlag = true;
            break;
          case 'r':
            offload_ratio = atof(optarg);
            break;
          case 'p':
            gpu_prio = true;
            break;
          case 'd':
            data_size = atoi(optarg);
            break;
          case 't':
            num_threads = atoi(optarg);
            break;
          case 'u':
            global_exec_mode = CUDA_MODE;
            break;
          case 'y':
            global_exec_mode = HAPI_MODE;
            break;
          case 'g':
            gpu_pes = atoi(optarg);
            break;
          case 's':
            sync_mode = true;
            break;
          default:
            CkPrintf("Usage: %s -c [chares] -i [iterations] -a [CPU time] -b [GPU time] -r [offload ratio] -d [data size] -t [threads per kernel] -u/y: CUDA/HAPI -p: higher priority for GPU -g [GPU handler PEs] -s: sync mode\n", m->argv[0]);
            CkExit();
        }
      }

      if (!bFlag)
        gpu_time = cpu_time;
      if (offload_ratio > 0.0f && global_exec_mode == CHARM_MODE) {
        CkPrintf("Offload ratio set higher than 0 but GPU mode not set!\n"
                 "Reverting offload ratio to 0...\n");
        offload_ratio = 0.0f;
      }
      if (gpu_pes > CkNumPes()) {
        CkPrintf("More GPU handler PEs than total number of PEs!\n"
                 "Setting it to equal the total number of PEs...\n");
        gpu_pes = CkNumPes();
      }

      // print info
      CkPrintf("\n[CUDA busywait example]\n");
      CkPrintf("Execution mode: %s\n", (global_exec_mode == CHARM_MODE) ? "Charm++ only" : ((global_exec_mode == CUDA_MODE) ? "Charm++ with CUDA" : "Charm++ with HAPI"));
      CkPrintf("Chares: %d\n", num_chares);
      CkPrintf("Iterations: %d\n", num_iters);
      CkPrintf("CPU time: %f, GPU time: %f\n", cpu_time, gpu_time);
      CkPrintf("Offload ratio: %.2f\n", offload_ratio);
      CkPrintf("Data size: %d\n", data_size);
      CkPrintf("Threads per kernel: %d\n", num_threads);
      CkPrintf("Higher priority for GPU methods and callbacks: %s\n", (gpu_prio) ? "ON" : "OFF");
      CkPrintf("GPU handler PEs: %d\n", gpu_pes);
      CkPrintf("Sync mode: %s\n\n", (sync_mode) ? "ON" : "OFF");
      delete m;

      // calculate kernel clock count
      int cuda_device = 0;
      cudaDeviceProp deviceProp;
      cudaGetDevice(&cuda_device);
      cudaGetDeviceProperties(&deviceProp, cuda_device);
      kernel_clock_count = gpu_time * deviceProp.clockRate * 1000;

      // initialize values
      iter = 0;
      agg_time = 0.0;

      // create workers
      if (gpu_pes > 0) {
        CkArrayOptions opts(num_chares);
        CProxy_CustomMap cmap = CProxy_CustomMap::ckNew();
        opts.setMap(cmap);
        workers = CProxy_Worker::ckNew(opts);
      }
      else {
        workers = CProxy_Worker::ckNew(num_chares);
      }

      // initialize workers
      workers.init();
    }

    void start() {
      // begin working
      workers.begin();

      // record start time
      start_time = CkWallTimer();
    }

    void done(double time) {
#ifdef USE_NVTX
      NVTXTracer nvtx_range("Main::done", NVTXColor::Emerald);
#endif
      if (sync_mode) {
        agg_time += time;
        if (++iter < num_iters) {
          workers.begin();
        }
        else {
          CkPrintf("Average time per iteration: %lf s\n\n", (agg_time / num_chares) / num_iters);
          CkPrintf("Aggregated execution time of all chares: %lf s\n", agg_time);
          CkPrintf("Elapsed time: %lf s\n", CkWallTimer() - start_time);
          CkExit();
        }
      }
      else {
        CkPrintf("\nAggregated execution time of all chares: %lf s\n", time);
        CkPrintf("Elapsed time: %lf s\n", CkWallTimer() - start_time);
        CkExit();
      }
    }
};

class Worker: public CBase_Worker {
  private:
    int num_chares_pe;
    int this_rank;
    int local_exec_mode;
    char* h_A; char* h_B;
    char* d_A; char* d_B;
    int iter;
    double start_time;
    double iter_time;
    double agg_time;
    cudaStream_t stream;

  public:
    Worker() {}

    void init() {
#ifdef USE_NVTX
      NVTXTracer nvtx_range(std::to_string(thisIndex) + " Worker::init", NVTXColor::PeterRiver);
#endif
      if (gpu_pes == 0) {
        // compute this chare's rank for hybrid mode
        num_chares_pe = num_chares / CkNumPes();
        int leftover = num_chares % CkNumPes();
        int start_index = num_chares_pe * CkMyPe();
        if (CkMyPe() < leftover) {
          num_chares_pe++;
          start_index += CkMyPe();
        }
        else {
          start_index += leftover;
        }
        this_rank = thisIndex - start_index;

        // determine execution mode
        local_exec_mode = global_exec_mode;
        if (this_rank >= num_chares_pe * offload_ratio) {
          local_exec_mode = CHARM_MODE;
        }
      }
      else {
        local_exec_mode = global_exec_mode;
        if (thisIndex >= num_chares * offload_ratio) {
          local_exec_mode = CHARM_MODE;
        }
      }

      CkPrintf("[%4d] Mode %d, PE %d\n", thisIndex, local_exec_mode, CkMyPe());

      // allocate memory for data transfers
      if (local_exec_mode != CHARM_MODE) {
        if (data_size > 0) {
          hapiCheck(cudaMallocHost(&h_A, data_size));
          hapiCheck(cudaMallocHost(&h_B, data_size));
          hapiCheck(cudaMalloc(&d_A, data_size));
          hapiCheck(cudaMalloc(&d_B, data_size));
        }
        cudaStreamCreate(&stream);
      }

      iter = 0;
      agg_time = 0.0;

      contribute(CkCallback(CkReductionTarget(Main, start), mainProxy));
    }

    ~Worker() {
#ifdef USE_NVTX
      NVTXTracer nvtx_range(std::to_string(thisIndex) + " Worker::~Worker", NVTXColor::WetAsphalt);
#endif
      // free memory
      if (local_exec_mode != CHARM_MODE) {
        if (data_size > 0) {
          hapiCheck(cudaFreeHost(h_A));
          hapiCheck(cudaFreeHost(h_B));
          hapiCheck(cudaFree(d_A));
          hapiCheck(cudaFree(d_B));
        }
        cudaStreamDestroy(stream);
      }
    }

    void begin() {
#ifdef USE_NVTX
      NVTXTracer nvtx_range(std::to_string(thisIndex) + " Worker::begin", NVTXColor::Amethyst);
#endif
      start_time = CkWallTimer();

      CallbackMsg* cb_msg = new CallbackMsg();
      if (local_exec_mode == CHARM_MODE) {
        usleep(cpu_time * 1000000);
        thisProxy[thisIndex].end(cb_msg);
      }
      else if (local_exec_mode == CUDA_MODE) {
        blockingKernel(h_A, h_B, d_A, d_B, data_size, num_threads, kernel_clock_count, stream, NULL, NULL);
        thisProxy[thisIndex].end(cb_msg);
      }
      else {
        CkArrayIndex1D myIndex = CkArrayIndex1D(thisIndex);
        if (gpu_prio)
          CkSetQueueing(cb_msg, CK_QUEUEING_LIFO);
        CkCallback* cb = new CkCallback(CkIndex_Worker::end(NULL), myIndex, thisArrayID);
        blockingKernel(h_A, h_B, d_A, d_B, data_size, num_threads, kernel_clock_count, stream, cb, cb_msg);
      }
    }

    void end(CallbackMsg* m) {
#ifdef USE_NVTX
      NVTXTracer nvtx_range(std::to_string(thisIndex) + " Worker::end", NVTXColor::Concrete);
#endif
      delete m;
      iter_time = CkWallTimer() - start_time;

      if (sync_mode) {
        contribute(sizeof(double), &iter_time, CkReduction::sum_double, CkCallback(CkReductionTarget(Main, done), mainProxy));
      }
      else {
        agg_time += iter_time;
        if (++iter < num_iters) {
          CkEntryOptions opts;
          if (local_exec_mode == CHARM_MODE) {
            if (gpu_prio)
              opts.setQueueing(CK_QUEUEING_FIFO);
            thisProxy[thisIndex].begin(&opts);
          }
          else {
            if (gpu_prio) {
              // set GPU work to be higher priority
              opts.setQueueing(CK_QUEUEING_LIFO);
            }
            thisProxy[thisIndex].begin(&opts);
          }
        } else {
          CkPrintf("[%4d] Average time per iteration: %lf s\n", thisIndex, agg_time / num_iters);
          contribute(sizeof(double), &agg_time, CkReduction::sum_double, CkCallback(CkReductionTarget(Main, done), mainProxy));
        }
      }
    }
};

#include "busywait.def.h"

#include "fft1d.decl.h"
#include <fftw3.h>
#include <limits>
#include "fileio.h"
#include "CkLoopAPI.h"
#include "rand48_replacement.h"

#define TWOPI 6.283185307179586

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ int numChunks;
/*readonly*/ int numTasks;
/*readonly*/ uint64_t N;
static CmiNodeLock fft_plan_lock;
#include "fftmacro.h"

CProxy_FuncCkLoop ckLoopProxy;

extern "C" void doCalc(int first,int last, void *result, int paramNum, void * param)
{
  //result=first;
  for(int i=first; i<=last; i++)
    fft_execute(((fft_plan*)param)[i]);
}

void initplanlock ()

{
  fft_plan_lock=CmiCreateLock();
}

struct fftMsg : public CMessage_fftMsg {
  int source;
  fft_complex *data;
};

struct Main : public CBase_Main {
  double start;
  CProxy_fft fftProxy;

  Main(CkArgMsg* m) {
    numChunks = atoi(m->argv[1]); //#1D partitions
    N = atol(m->argv[2]); //matrix size
    if(m->argc>=4)
      numTasks = atol(m->argv[3]); //the number of tasks that 1D partition is splitted into
    else
      numTasks = CmiMyNodeSize();  //default to 1/core
    delete m;
    
    mainProxy = thisProxy;

    /* how to nodify this computation? */
    /* We make one block alloc per chare and divide the work evenly
       across the number of threads.
     * cache locality issues... 

     *       The CkLoop scheme presents a problem in cache
     *       ignorance.  We push tasks into the queue as the remote
     *       message dependencies are met, however the execution of
     *       dequeued tasks performance will have significant cache
     *       limitations obfuscated to our scheduler.  Our helper
     *       threads will block while fetching data into cache local
     *       to the thread.  If we only have 1 thread per core, we
     *       have no way to self overlap those operations.  This
     *       implies that there are probably conditions under which
     *       more than one ckloop thread per core will result in
     *       better performance.  A natural sweet spot for these
     *       should be explored in the SMT case where one thread per
     *       SMT will allow for natural overlap of execution based on
     *       cache availability, as controlled by the OS without
     *       additional pthread context switching overhead.  A further
     *       runtime based virtualized overthreading may provide
     *       further benefits depending on thread overhead.
     */
    if (N % numChunks != 0)
      CkAbort("numChunks not a factor of N\n");

    // Construct an array of fft chares to do the calculation
    fftProxy = CProxy_fft::ckNew(numChunks);

    // Construct a ckloop to do the calculation
    // WARNING: In non-SMP mode, the number of threads to be created
    // should be passed in as a parameter for CkLoop initialization!
    // -Chao Mei
    ckLoopProxy = CkLoop_Init();
    
    CkStartQD(CkIndex_Main::initDone((CkQdMsg *)0), &thishandle);
  }

  void initDone(CkQdMsg *msg){
    delete msg;
    startFFT();
  }
  
  void startFFT() {
    start = CkWallTimer();
    // Broadcast the 'go' signal to the fft chare array
    fftProxy.doFFT();
  }

  void doneFFT() {
    double time = CkWallTimer() - start;
    double gflops = 5 * (double)N*N * log2((double)N*N) / (time * 1000000000);
    CkPrintf("chares: %d\ncores: %d\nTasks: %d\nsize: %ld\ntime: %f sec\nrate: %f GFlop/s\n",
             numChunks, CkNumPes(), numTasks, N*N, time, gflops);

    fftProxy.initValidation();
  }

  void printResidual(realType r) {
    CkPrintf("residual = %g\n", r);
    CkExit();
  }

};

struct fft : public CBase_fft {
  fft_SDAG_CODE

  int iteration, count;
  uint64_t n;
  fft_plan *plan;
  fft_plan p1;
  fftMsg **msgs;
  fft_complex *in, *out;
  bool validating;
  int nPerThread;
  fft() {
    validating = false;

    n = N*N/numChunks;

    in = (fft_complex*) fft_malloc(sizeof(fft_complex) * n);
    out = (fft_complex*) fft_malloc(sizeof(fft_complex) * n);
    nPerThread= n/numTasks;
    
    int length[] = {N};
    
    /** Basically, we want to parallelize the following fftw function call
     * which is to do fft on each row of a 2D array with #rows=N/numChunks, #cols=N;
     * 1. create a plan: singlePlan = fft_plan_many_dft(1, len, N/numChunks, out, len,
     *                                                  1, N, out, len, 1,
     *                                                  N, FFTW_FORWARD, FFTW_ESTIMATE)
     * where len is defined as int len[]={N}
     * 2. execute the plan: fft_execute(singlePlan).
     * 
     * It's not a loop, we transformed it into a loop with N/numTasks plans so that 
     * each task execute one plan. Each plan has N/numChunks/numTasks rows for fftw
     * processing.
     */
    
    CmiLock(fft_plan_lock);
    size_t offset=0;
    plan= new fft_plan[numTasks];
    for(int i=0; i < numTasks; i++,offset+=nPerThread)
      {
	/* ??? should the dist be nPerThread as the fft is performed as 1d of length nPerThread?? */
	//plan[i] = fft_plan_many_dft(1, length, N/numChunks/numTasks, out+offset, length, 1, N/numTasks,
    //                        out+offset, length, 1, N/numTasks, FFTW_FORWARD, FFTW_ESTIMATE);
    
    plan[i] = fft_plan_many_dft(1, length, N/numChunks/numTasks, out+offset, length, 1, N,
                            out+offset, length, 1, N, FFTW_FORWARD, FFTW_ESTIMATE);                        
      }
    CmiUnlock(fft_plan_lock);
    
    srand48(thisIndex);
    for(int i = 0; i < n; i++) {
      in[i][0] = drand48();
      in[i][1] = drand48();
    }

    msgs = new fftMsg*[numChunks];
    for(int i = 0; i < numChunks; i++) {
      msgs[i] = new (n/numChunks) fftMsg;
      msgs[i]->source = thisIndex;
    }

    // Reduction to the mainchare to signal that initialization is complete
    //contribute(CkCallback(CkIndex_Main::startFFT(), mainProxy));
  }

  void sendTranspose(fft_complex *src_buf) {
    // All-to-all transpose by constructing and sending
    // point-to-point messages to each chare in the array.
    for(int i = thisIndex; i < thisIndex+numChunks; i++) {
      //  Stagger communication order to avoid hotspots and the
      //  associated contention.
      int k = i % numChunks;
      for(int j = 0, l = 0; j < N/numChunks; j++)
        memcpy(msgs[k]->data[(l++)*N/numChunks], src_buf[k*N/numChunks+j*N], sizeof(fft_complex)*N/numChunks);

      // Tag each message with the iteration in which it was
      // generated, to prevent mis-matched messages from chares that
      // got all of their input quickly and moved to the next step.
      CkSetRefNum(msgs[k], iteration);
      thisProxy[k].getTranspose(msgs[k]);
      // Runtime system takes ownership of messages once they're sent
      msgs[k] = NULL;
    }
  }

  void applyTranspose(fftMsg *m) {
    int k = m->source;
    for(int j = 0, l = 0; j < N/numChunks; j++)
      for(int i = 0; i < N/numChunks; i++) {
        out[k*N/numChunks+(i*N+j)][0] = m->data[l][0];
        out[k*N/numChunks+(i*N+j)][1] = m->data[l++][1];
      }

    // Save just-received messages to reuse for later sends, to
    // avoid reallocation
    delete msgs[k];
    msgs[k] = m;
    msgs[k]->source = thisIndex;
  }

  void twiddle(realType sign) {
    realType a, c, s, re, im;

    int k = thisIndex;
    for(int i = 0; i < N/numChunks; i++)
      for(int j = 0; j < N; j++) {
        a = sign * (TWOPI*(i+k*N/numChunks)*j)/(N*N);
        c = cos(a);
        s = sin(a);

        int idx = i*N+j;

        re = c*out[idx][0] - s*out[idx][1];
        im = s*out[idx][0] + c*out[idx][1];
        out[idx][0] = re;
        out[idx][1] = im;
      }
  }
  void fftHelperLaunch()
  {
    //kick off thread computation
    //FuncCkLoop *nth = ckLoopProxy[CkMyNode()].ckLocalBranch();
    //nth->parallelizeFunc(doCalc, numTasks, numTasks, thisIndex, numTasks, 1, 1, plan, 0, NULL);
    double ffttime = CkWallTimer();
    CkLoop_Parallelize(doCalc, 1, plan, numTasks, 0, numTasks-1);    
    CkPrintf("FFT time: %.3f (ms)\n", (CkWallTimer()-ffttime)*1e3);
  }

  void initValidation() {
    memcpy(in, out, sizeof(fft_complex) * n);

    validating = true;
    int length[] = {N};
    CmiLock(fft_plan_lock);
    size_t offset=0;
    plan= new fft_plan[numTasks];
    for(int i=0; i < numTasks; i++,offset+=nPerThread)
      {
	//	fft_destroy_plan(plan[i]);
	//plan[i] = fft_plan_many_dft(1, length, N/numChunks/numTasks, out+offset, length, 1, N/numTasks,
    //                        out+offset, length, 1, N/numTasks, FFTW_BACKWARD, FFTW_ESTIMATE);
    plan[i] = fft_plan_many_dft(1, length, N/numChunks/numTasks, out+offset, length, 1, N,
                            out+offset, length, 1, N, FFTW_BACKWARD, FFTW_ESTIMATE);
      }
    CmiUnlock(fft_plan_lock);
    contribute(CkCallback(CkIndex_Main::startFFT(), mainProxy));
  }

  void calcResidual() {
    double infNorm = 0.0;

    srand48(thisIndex);
    for(int i = 0; i < n; i++) {
      out[i][0] = out[i][0]/(N*N) - drand48();
      out[i][1] = out[i][1]/(N*N) - drand48();

      double mag = sqrt(pow(out[i][0], 2) + pow(out[i][1], 2));
      if(mag > infNorm) infNorm = mag;
    }

    double r = infNorm / (std::numeric_limits<double>::epsilon() * log((double)N * N));

    CkCallback cb(CkReductionTarget(Main, printResidual), mainProxy);
    contribute(sizeof(double), &r, CkReduction::max_double, cb);
  }

  fft(CkMigrateMessage* m) {}
  ~fft() {}
};

#include "fft1d.def.h"

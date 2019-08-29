#include "stdio.h"
#include "AmpiInterop.h"
/*readonly*/ CProxy_AmpiInterop ampiInteropProxy;
/*readonly*/ int nRanks;

void dummy_mpi_fn(void* in, void* out) {}

class AmpiInterop : public CBase_AmpiInterop {
  AmpiInterop_SDAG_CODE
  CProxy_ampi ampiProxy;
  bool finished;

  public:
  AmpiInterop() {
    finished = false;
  }
  bool isFinished() {
    return finished;
  }
  void call_mpi_fn(int pe, int chareIndex, MpiCallData buf) {
    // Wrap chare array index around number of AMPI ranks?
    //if (chareIndex >= nRanks) chareIndex = chareIndex % nRanks;

    // Send a message
    ampiProxy[chareIndex].injectMsg(sizeof(buf), (char*)&buf);
  }
  void finish() {
    // target called on processor 0 only
    finished = true;
    CkCallback cb(CkReductionTarget(AmpiInterop, finalize), 0, thisProxy);
    contribute(cb);
  }
  void finalize() {
    // send a dummy message
    MpiCallData buf;
    buf.fn = dummy_mpi_fn;
    buf.cb = CkCallback(CkCallback::ignore);
    ampiProxy.injectMsg(sizeof(buf), (char*)&buf);
  }
};

void AmpiInteropInit() {
  ampiInteropProxy = CProxy_AmpiInterop::ckNew();
  ampiInteropProxy.run();
}

int main(int argc, char **argv) {
  int rank;
  MPI_Status status;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &nRanks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    CkPrintf("[%d] In AMPI: running on %d PEs with %d AMPI ranks\n", CkMyPe(), CkNumPes(), nRanks);
    ampi *ptr = getAmpiInstance(MPI_COMM_WORLD);
    const ampiCommStruct &dest = ptr->comm2CommStruct(MPI_COMM_WORLD);
    ampiInteropProxy.init(dest.getProxy());
  }
  CkPrintf("[%d] In AMPI: created AMPI rank %d\n", CkMyPe(), rank);

  MPI_Barrier(MPI_COMM_WORLD);

  while (!ampiInteropProxy.ckLocalBranch()->isFinished()) {
    MpiCallData buf;
    MPI_Recv((void*)&buf, sizeof(buf), MPI_BYTE, rank, 0, MPI_COMM_WORLD, &status);
    buf.fn(buf.in, buf.out);
    buf.cb.send(NULL);
  }

  CkPrintf("[%d] In AMPI: finalizing AMPI rank %d\n", CkMyPe(), rank);
  MPI_Finalize();
  return 0;
}

#include "AmpiInterop.def.h"

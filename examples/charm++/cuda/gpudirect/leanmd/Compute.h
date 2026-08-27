#ifndef __COMPUTE_H__
#define __COMPUTE_H__

#include "defs.h"
#include "hapi.h"
#include "leanmd_cuda.h"

// The interaction agent between a pair of cells.
//
// It is stateless across steps apart from its device scratch: two position
// buffers posted directly by arriving messages, two force buffers sent straight
// back out. Nothing is staged through host memory except the pair potential, and
// only on the two steps that report energy.
class Compute : public CBase_Compute {
  private:
    Compute_SDAG_CODE
    int stepCount;  //current step number
    double energy[2]; //store potential energy

    // The two cells I pair, recovered from my own 6D index. For a self-compute
    // they are the same cell and only one message arrives.
    int cellA[3], cellB[3];
    bool selfCompute;

    // ---- device-resident scratch, one slot per cell ----
    int cap[2];
    vec3* d_pos[2];
    vec3* d_force[2];
    double* d_energyPartial;
    double* d_energyScalar;
    double* h_energy;       // pinned
    cudaStream_t stream;

    // Filled by the post entry method as messages land.
    int nPart[2];
    int ordinal[2];
    int cellIdx[2][3];

    // Force sends whose receiving cell has not finished pulling d_force yet.
    // run() drains these at the end of every step, so neither the next step's
    // kernels nor a migration can touch the buffer while a cell is reading it.
    int pendingForceSends;
    int ackCount;   // SDAG loop counter for draining them

    void updateInstrumentation();
    void deriveCells();
    void ensureDevice();
    void ensureSlot(int s, int n);
    void freeDevice();
    int slotFor(int cx, int cy, int cz) const;
    vec3 periodicShift() const;

  public:
    Compute();
    Compute(CkMigrateMessage *msg);
    ~Compute();
    void pup(PUP::er &p);

    void launchForces();
    void sendForces();

    // Device-zerocopy post entry method.
    void calculateForces(int ref, int ord, int cx, int cy, int cz, int& n,
                         vec3*& pos, CkDeviceBufferPost* devicePost);
};

#endif

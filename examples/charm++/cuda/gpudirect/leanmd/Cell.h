#ifndef __CELL_H__
#define __CELL_H__

#include <string>
#include <vector>

#include "hapi.h"
#include "leanmd_cuda.h"

// CUDA streams handed out per PE. A 3x3x3 decomposition already has 27 Cells and
// 378 Computes; one stream apiece would be hundreds per PE, buying overlap a
// single GPU cannot deliver anyway.
#define NUM_STREAMS 8

// The (0,0,0) neighbour offset -- a particle that stays where it is.
#define SELF_NBR (KAWAY_X * NBRS_Y * NBRS_Z + KAWAY_Y * NBRS_Z + KAWAY_Z)

class StreamPool : public CBase_StreamPool {
  std::vector<cudaStream_t> streams;
  int next;

 public:
  StreamPool() : streams(NUM_STREAMS), next(0) {
    for (int i = 0; i < NUM_STREAMS; i++)
      hapiCheck(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
  }
  StreamPool(CkMigrateMessage* m) : streams(0), next(0) {}
  cudaStream_t acquire() { return streams[(next++) % NUM_STREAMS]; }
};

class CellMap : public CkArrayMap {
  int num_x, num_y, num_z, num_yz;
  float ratio;
  public:
  CellMap(int num_x_, int num_y_, int num_z_) :
    num_x(num_x_), num_y(num_y_), num_z(num_z_)
  {
    num_yz = num_y * num_z;
    ratio = ((float)CkNumPes())/(num_x * num_yz);
  }
  CellMap(CkMigrateMessage *m) {}
  int registerArray(CkArrayIndex& numElements, CkArrayID aid) {
    return 0;
  }
  int procNum(int arrayHdl, const CkArrayIndex &idx) {
    int *index =(int *)idx.data();
    int patchID = index[2] + index[1] * num_z + index[0] * num_yz;
    return (int)(patchID * ratio);
  }
};

// A cell of the simulation box.
//
// Its particles live on the GPU from the end of the constructor to the end of the
// run: forces arrive as device buffers, are folded in by a kernel, the integrator
// is a kernel, and the migration sort is a kernel. Nothing but counts and two
// energy scalars ever crosses back to the host.
class Cell : public CBase_Cell {
private:
  Cell_SDAG_CODE;

  // Host copy of the particles. Populated by the constructor to lay out the
  // initial lattice, uploaded once, and thereafter used only by pup(). It is NOT
  // kept in step with the device array -- the device array is the live state.
  std::vector<Particle> particles;
  // my compute locations
  std::vector<CkArrayIndex6D> computesList;
  // to count the number of steps, and decide when to stop
  int stepCount;
  // live particle count; the host mirror of how much of d_particles is in use
  int myNumParts;
  // number of interacting neighbors
  int inbrs;
  double stepTime;
  int forceCount;
  int updateCount;
  // store kinetic energy - initial and final
  double energy[2];
  int numReadyCheckpoint;

  // ---- device-resident state ----
  int part_capacity;   // allocation size of d_particles / d_stay
  int exch_capacity;   // per-bucket allocation for migration traffic
  Particle* d_particles;
  Particle* d_stay;        // compaction target; swapped with d_particles
  Particle* d_send_parts;  // inbrs buckets of outgoing particles
  Particle* d_recv_parts;  // inbrs landing slots for incoming particles
  vec3*     d_pos;         // positions, gathered each step and sent to Computes
  vec3*     d_force;        // force accumulator
  vec3*     d_recv_force;   // inbrs landing slots for returning force arrays
  double*   d_kePartial;
  double*   d_energy;
  int*      d_counts;
  int*      h_counts;      // pinned, inbrs ints, read after binning
  double*   h_energy;      // pinned, one double
  cudaStream_t stream;

  void allocateDevice();
  void freeDevice();

public:
  Cell();
  Cell(CkMigrateMessage *msg);
  ~Cell();
  void createComputes();  //add my computes
  void beginStep();
  void sendPositions();
  void accumulateForce(int ordinal, int n, vec3* f);
  void computeKineticEnergy();
  void integrate();
  void binParticles();
  void sendMigrants();
  void appendMigrants(int ordinal, int n, Particle* parts);
  void startCheckpoint(int);
  void pup(PUP::er &p);

  // Device-zerocopy post entry methods: run when the message lands, and name the
  // device buffer the payload should be written into.
  void receiveForces(int ref, int ordinal, int& n, vec3*& f,
                     CkDeviceBufferPost* devicePost);
  void receiveMigrants(int ref, int ordinal, int n, int& m, Particle*& parts,
                       CkDeviceBufferPost* devicePost);
};

#endif

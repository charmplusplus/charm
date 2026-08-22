#include "defs.h"
#include "leanmd.decl.h"
#include "Cell.h"

#include <algorithm>

extern /* readonly */ CProxy_StreamPool streamPool;

Cell::Cell() : inbrs(NUM_NEIGHBORS), stepCount(1), updateCount(0), forceCount(0),
               computesList(NUM_NEIGHBORS), part_capacity(0), exch_capacity(0),
               d_particles(NULL), d_stay(NULL), d_send_parts(NULL),
               d_recv_parts(NULL), d_pos(NULL), d_force(NULL), d_recv_force(NULL),
               d_kePartial(NULL), d_energy(NULL), d_counts(NULL), h_counts(NULL),
               h_energy(NULL), stream(NULL) {
  // stream is read by the post entry methods, which run at message arrival and
  // are not ordered against run(). A garbage handle there surfaces far away, as
  // CUDA error 400 inside the runtime's IPC path, so it starts null.
  //load balancing to be called when AtSync is called
  usesAtSync = true;

  int myid = thisIndex.z+cellArrayDimZ*(thisIndex.y+thisIndex.x*cellArrayDimY);
  myNumParts = cellParticleCount(thisIndex.x, thisIndex.y, thisIndex.z);

  // Per-cell RNG state.
  //
  // The original called srand48(myid) then drand48(), which share one global
  // state. Every PE of an SMP build is a thread of one process, so Cell
  // constructors on different PEs interleave and corrupt each other's sequence:
  // the initial velocities, and therefore the initial energy, came out different
  // on every run at +p4 and varied with the PE count. That makes any A/B
  // comparison between balancers meaningless, since the two runs would not be
  // simulating the same system.
  //
  // erand48 carries its state in the caller's buffer. Seeded the way glibc's
  // srand48 seeds the global one -- X = (seed << 16) | 0x330E -- so the sequence
  // is identical to what a single-PE stock run produced.
  unsigned short xsub[3];
  xsub[0] = 0x330E;
  xsub[1] = (unsigned short)(myid & 0xFFFF);
  xsub[2] = (unsigned short)((myid >> 16) & 0xFFFF);

  // Particle initialization. This is the one place particles exist on the host:
  // the lattice and the RNG are host code, and the result is uploaded once in
  // allocateDevice().
  for(int i = 0; i < myNumParts; i++) {
    particles.push_back(Particle());
    particles[i].mass = HYDROGEN_MASS;

    // Lattice site for this atom. The stock fill takes sites 0..N-1 in order,
    // which is fine at 800-1000 of the 1000 available sites but would bunch a
    // sparse cell's atoms against one face -- an artificial density spike inside
    // the cell, on top of the one between cells that the profile is meant to
    // create. Stride the sites for the imbalanced profiles so a sparse cell still
    // fills its own volume evenly. DENSITY_UNIFORM keeps the stock fill exactly,
    // so the stock benchmark's initial energy is unchanged.
    const int site = (densityMode == DENSITY_UNIFORM)
                       ? i
                       : (int)(((long)i * PERDIM * PERDIM * PERDIM) / myNumParts);

    //uniformly place particles, avoid close distance among them
    particles[i].pos.x = (GAP/(float)2) + thisIndex.x * CELL_SIZE_X + ((site*KAWAY_Y*KAWAY_Z)/(PERDIM*PERDIM))*GAP;
    particles[i].pos.y = (GAP/(float)2) + thisIndex.y * CELL_SIZE_Y + (((site*KAWAY_Z)/PERDIM)%(PERDIM/KAWAY_Y))*GAP;
    particles[i].pos.z = (GAP/(float)2) + thisIndex.z * CELL_SIZE_Z + (site%(PERDIM/KAWAY_Z))*GAP;
    //give random values for velocity
    particles[i].vel.x = (erand48(xsub) - 0.5) * .2 * MAX_VELOCITY;
    particles[i].vel.y = (erand48(xsub) - 0.5) * .2 * MAX_VELOCITY;
    particles[i].vel.z = (erand48(xsub) - 0.5) * .2 * MAX_VELOCITY;
    particles[i].acc = vec3(0.0);
  }

  energy[0] = energy[1] = 0;
  setMigratable(false);
}

//constructor for chare object migration
Cell::Cell(CkMigrateMessage *msg): CBase_Cell(msg),
               part_capacity(0), exch_capacity(0),
               d_particles(NULL), d_stay(NULL), d_send_parts(NULL),
               d_recv_parts(NULL), d_pos(NULL), d_force(NULL), d_recv_force(NULL),
               d_kePartial(NULL), d_energy(NULL), d_counts(NULL), h_counts(NULL),
               h_energy(NULL), stream(NULL) {
  // Mirror the default constructor: everything device-side starts null so it is
  // reallocated on first use. Leaving these uninitialized meant a reconstructed
  // Cell held garbage -- a garbage stream surfaces far away as CUDA error 400
  // (invalid resource handle) inside the runtime's send path, and a garbage
  // d_force faults in cuLaunchKernel. Note this only makes the state *defined*:
  // pup() still does not carry the device-resident particles, so Cells remain
  // setMigratable(false) and a migrated Cell would resume with empty arrays.
  usesAtSync = true;
  setMigratable(false);
  delete msg;
}

Cell::~Cell() { freeDevice(); }

// Allocate every device buffer the cell will need, and upload the initial
// particles. Capacities carry headroom because migration moves particles between
// cells and an imbalanced run makes some of them grow.
void Cell::allocateDevice() {
  stream = streamPool.ckLocalBranch()->acquire();

  part_capacity = 2 * maxCellParts + 64;
  // A particle drifts at most MAX_VELOCITY * DEFAULT_DELTA * MIGRATE_STEPCOUNT
  // = 2 Angstroms between migrations, against a 30 Angstrom cell, so only atoms
  // within a thin shell of a face can leave. An eighth of the cell is generous.
  exch_capacity = std::max(64, part_capacity / 8);

  hapiCheck(hapiMalloc((void**)&d_particles, sizeof(Particle) * part_capacity));
  hapiCheck(hapiMalloc((void**)&d_stay, sizeof(Particle) * part_capacity));
  hapiCheck(hapiMalloc((void**)&d_send_parts,
                       sizeof(Particle) * (size_t)inbrs * exch_capacity));
  hapiCheck(hapiMalloc((void**)&d_recv_parts,
                       sizeof(Particle) * (size_t)inbrs * exch_capacity));
  hapiCheck(hapiMalloc((void**)&d_pos, sizeof(vec3) * part_capacity));
  hapiCheck(hapiMalloc((void**)&d_force, sizeof(vec3) * part_capacity));
  hapiCheck(hapiMalloc((void**)&d_recv_force,
                       sizeof(vec3) * (size_t)inbrs * part_capacity));
  hapiCheck(hapiMalloc((void**)&d_kePartial, sizeof(double) * 64));
  hapiCheck(hapiMalloc((void**)&d_energy, sizeof(double)));
  hapiCheck(hapiMalloc((void**)&d_counts, sizeof(int) * inbrs));
  hapiCheck(hapiMallocHost((void**)&h_counts, sizeof(int) * inbrs));
  hapiCheck(hapiMallocHost((void**)&h_energy, sizeof(double)));

  hapiCheck(cudaMemcpyAsync(d_particles, &particles[0],
                            sizeof(Particle) * myNumParts,
                            cudaMemcpyHostToDevice, stream));
  hapiCheck(cudaStreamSynchronize(stream));
  // The device array is the live state from here on.
  particles.clear();
  particles.shrink_to_fit();
}

void Cell::freeDevice() {
  if (d_particles)   { hapiCheck(hapiFree(d_particles));   d_particles = NULL; }
  if (d_stay)        { hapiCheck(hapiFree(d_stay));        d_stay = NULL; }
  if (d_send_parts)  { hapiCheck(hapiFree(d_send_parts));  d_send_parts = NULL; }
  if (d_recv_parts)  { hapiCheck(hapiFree(d_recv_parts));  d_recv_parts = NULL; }
  if (d_pos)         { hapiCheck(hapiFree(d_pos));         d_pos = NULL; }
  if (d_force)       { hapiCheck(hapiFree(d_force));       d_force = NULL; }
  if (d_recv_force)  { hapiCheck(hapiFree(d_recv_force));  d_recv_force = NULL; }
  if (d_kePartial)   { hapiCheck(hapiFree(d_kePartial));   d_kePartial = NULL; }
  if (d_energy)      { hapiCheck(hapiFree(d_energy));      d_energy = NULL; }
  if (d_counts)      { hapiCheck(hapiFree(d_counts));      d_counts = NULL; }
  if (h_counts)      { hapiCheck(hapiFreeHost(h_counts));  h_counts = NULL; }
  if (h_energy)      { hapiCheck(hapiFreeHost(h_energy));  h_energy = NULL; }
}

//function to create my computes
void Cell::createComputes() {
  int x = thisIndex.x, y = thisIndex.y, z = thisIndex.z;
  int px1, py1, pz1, dx, dy, dz, px2, py2, pz2;

  /*  The computes X are inserted by a given cell:
   *
   *	^  X  X  X
   *	|  0  X  X
   *	y  0  0  0
   *	   x ---->
   */

  // for round robin insertion
  int currPe = CkMyPe();

  for (int num = 0; num < inbrs; num++) {
    dx = num / (NBRS_Y * NBRS_Z)                - NBRS_X/2;
    dy = (num % (NBRS_Y * NBRS_Z)) / NBRS_Z     - NBRS_Y/2;
    dz = num % NBRS_Z                           - NBRS_Z/2;

    if (num >= inbrs / 2){
      px1 = x + KAWAY_X;
      py1 = y + KAWAY_Y;
      pz1 = z + KAWAY_Z;
      px2 = px1+dx;
      py2 = py1+dy;
      pz2 = pz1+dz;
      CkArrayIndex6D index(px1, py1, pz1, px2, py2, pz2);
      // Round-robin spreads each cell's Computes over every PE, which balances
      // by construction and averages any density profile away. COMPUTEMAP_LOCAL
      // keeps a Compute on the PE of the cell that owns it -- cell one of the
      // pair is this cell -- so the box's spatial structure reaches the PEs and
      // a density profile becomes a real imbalance for the balancer to work on.
      computeArray[index].insert(computeMapMode == COMPUTEMAP_LOCAL
                                     ? CkMyPe()
                                     : ((++currPe) % CkNumPes()));
      computesList[num] = index;
    } else {
      // these computes will be created by pairing cells
      px1 = WRAP_X(x + dx) + KAWAY_X;
      py1 = WRAP_Y(y + dy) + KAWAY_Y;
      pz1 = WRAP_Z(z + dz) + KAWAY_Z;
      px2 = px1 - dx;
      py2 = py1 - dy;
      pz2 = pz1 - dz;
      CkArrayIndex6D index(px1, py1, pz1, px2, py2, pz2);
      computesList[num] = index;
    }
  } // end of for loop
  contribute(CkCallback(CkReductionTarget(Main,run),mainProxy));
}

// Atoms currently in this cell, binned by x -- the axis the density profiles vary
// along. Summed across all cells this says whether a profile is still there, which
// is the question of whether the imbalance it creates is static or decaying.
void Cell::reportDensity() {
  std::vector<int> slab(cellArrayDimX, 0);
  slab[thisIndex.x] = myNumParts;
  contribute(sizeof(int) * cellArrayDimX, slab.data(), CkReduction::sum_int,
             CkCallback(CkReductionTarget(Main, densityReport), mainProxy));
}

// Zero the accumulator and pull this step's positions out of the particle array.
// sendPositions runs from the completion callback, so the buffer that goes on the
// wire is the one the gather just filled.
void Cell::beginStep() {
  invokeZeroForces(d_force, myNumParts, stream);
  invokeGatherPositions(d_particles, d_pos, myNumParts, stream);
  CkCallback* cb = new CkCallback(CkIndex_Cell::positionsReady(),
                                  thisProxy[thisIndex]);
  hapiAddCallback(stream, cb);
}

// One device-to-device send per compute, carrying the ordinal so the force array
// that comes back can be routed to its own landing slot without a rendezvous.
//
// This replaces a CkMulticast section send. The spanning tree is gone, but a
// device-buffer multicast does not exist, and the point-to-point edges are what
// the load balancer's communication graph needs to see anyway.
void Cell::sendPositions() {
  for (int num = 0; num < inbrs; num++)
    computeArray[computesList[num]].calculateForces(
        stepCount, num, thisIndex.x, thisIndex.y, thisIndex.z, myNumParts,
        CkDeviceBuffer(d_pos, stream));
}

void Cell::receiveForces(int ref, int ordinal, int& n, vec3*& f,
                         CkDeviceBufferPost* devicePost) {
  f = d_recv_force + (size_t)ordinal * part_capacity;
  devicePost[0].hapi_stream = stream;
}

// Fold one compute's contribution into the accumulator. This is what used to be a
// CkReduction::sum_double up the section tree; that reduction ran on host buffers,
// so keeping it would have meant a D2H on every compute and an H2D here, every
// step. Summation order is now arrival order rather than tree order, so the total
// is no longer bit-identical run to run -- well inside ENERGY_VAR, but worth
// knowing when comparing balancer configurations.
void Cell::accumulateForce(int ordinal, int n, vec3* f) {
  invokeAccumulateForces(d_force, f, n, stream);
}

// Kinetic energy is sampled before the velocity update, matching the original
// updateProperties, which accumulated it at the top of the loop body.
void Cell::computeKineticEnergy() {
  const int nb = std::min(64, (myNumParts + 255) / 256);
  invokeKineticEnergy(d_particles, myNumParts, d_kePartial, nb, stream);
  invokeReduceDoubles(d_kePartial, nb, d_energy, stream);
  hapiCheck(cudaMemcpyAsync(h_energy, d_energy, sizeof(double),
                            cudaMemcpyDeviceToHost, stream));
}

void Cell::integrate() {
  if (stepCount == 1 || stepCount == finalStepCount) computeKineticEnergy();

  // Two deltas: acceleration is in m/s^2 and velocity in A/fs, so the velocity
  // update carries the 10^-20 and the position update does not.
  invokeIntegrate(d_particles, d_force, myNumParts, DEFAULT_DELTA * 1.0e-20,
                  DEFAULT_DELTA, MAX_VELOCITY, stream);

  CkCallback* cb = new CkCallback(CkIndex_Cell::integrateDone(),
                                  thisProxy[thisIndex]);
  hapiAddCallback(stream, cb);
}

// Sort the particles into per-neighbour buckets on the device. Only the bucket
// counts come back to the host.
void Cell::binParticles() {
  hapiCheck(cudaMemsetAsync(d_counts, 0, sizeof(int) * inbrs, stream));
  invokeBinParticles(d_particles, myNumParts, thisIndex.x, thisIndex.y, thisIndex.z,
                     CELL_SIZE_X, CELL_SIZE_Y, CELL_SIZE_Z,
                     cellArrayDimX, cellArrayDimY, cellArrayDimZ,
                     d_stay, d_send_parts, exch_capacity, d_counts, stream);
  hapiCheck(cudaMemcpyAsync(h_counts, d_counts, sizeof(int) * inbrs,
                            cudaMemcpyDeviceToHost, stream));
  CkCallback* cb = new CkCallback(CkIndex_Cell::binDone(), thisProxy[thisIndex]);
  hapiAddCallback(stream, cb);
}

void Cell::sendMigrants() {
  // The stayers are already compacted into d_stay, so adopting them is a pointer
  // swap rather than a copy.
  std::swap(d_particles, d_stay);
  myNumParts = h_counts[SELF_NBR];

  for (int num = 0; num < inbrs; num++) {
    if (num == SELF_NBR) continue;

    const int cnt = h_counts[num];
    if (cnt > exch_capacity)
      CkAbort("Cell (%d,%d,%d): %d particles leaving toward neighbour %d exceeds "
              "exchange capacity %d at step %d\n", thisIndex.x, thisIndex.y,
              thisIndex.z, cnt, num, exch_capacity, stepCount);

    const int dx = num / (NBRS_Y * NBRS_Z)            - KAWAY_X;
    const int dy = (num % (NBRS_Y * NBRS_Z)) / NBRS_Z - KAWAY_Y;
    const int dz = num % NBRS_Z                       - KAWAY_Z;

    // Tag with the mirrored ordinal, so the receiver can use it directly as a
    // landing slot: two different neighbours never mirror to the same value.
    cellArray(WRAP_X(thisIndex.x+dx), WRAP_Y(thisIndex.y+dy), WRAP_Z(thisIndex.z+dz))
        .receiveMigrants(stepCount, (inbrs - 1) - num, cnt, std::max(cnt, 1),
                         CkDeviceBuffer(d_send_parts + (size_t)num * exch_capacity,
                                        stream));
  }
}

void Cell::receiveMigrants(int ref, int ordinal, int n, int& m, Particle*& parts,
                           CkDeviceBufferPost* devicePost) {
  parts = d_recv_parts + (size_t)ordinal * exch_capacity;
  devicePost[0].hapi_stream = stream;
}

void Cell::appendMigrants(int ordinal, int n, Particle* parts) {
  if (n == 0) return;
  if (myNumParts + n > part_capacity)
    CkAbort("Cell (%d,%d,%d): %d particles exceed capacity %d at step %d; the "
            "cell has grown past its headroom\n", thisIndex.x, thisIndex.y,
            thisIndex.z, myNumParts + n, part_capacity, stepCount);

  // Appends are enqueued in arrival order and the host-side offset advances with
  // them, so each lands on its own stretch of the array.
  invokeAppendParticles(d_particles, myNumParts, parts, n, stream);
  myNumParts += n;
}

//pack important data when I move/checkpoint
void Cell::pup(PUP::er &p) {
  CBase_Cell::pup(p);
  __sdag_pup(p);

  // Cells do not migrate yet (setMigratable(false)); when they do, the particle
  // array has to come down from the device here and go back up on unpacking.
  p | particles;
  p | stepCount;
  p | myNumParts;
  p | updateCount;
  p | forceCount;
  p | stepTime;
  p | inbrs;
  p | numReadyCheckpoint;
  PUParray(p, energy, 2);

  p | computesList;
}

#include "NDMeshStreamer.h"

typedef CmiUInt8 dtype;
#include "randomAccess.decl.h"
#include "TopoManager.h"

#define POLY 0x0000000000000007ULL
#define PERIOD 1317624576693539401LL

// log_2 of the table size per PE
int N;
// The local table size
CmiInt8 localTableSize;
// Handle to the test driver (chare)
CProxy_TestDriver driverProxy;
// Handle to the communication library (group)
CProxy_ArrayMeshStreamer<dtype, int, Updater,
                         SimpleMeshRouter> aggregator;
// Number of chares per PE
int numElementsPerPe;
// Max number of keys buffered by communication library
const int numMsgsBuffered = 1024;

CmiUInt8 HPCC_starts(CmiInt8 n);

class TestDriver : public CBase_TestDriver {
private:
  CProxy_Updater  updater_array;
  double starttime;
  CmiInt8 tableSize;

public:
  TestDriver(CkArgMsg* args) {
    N = atoi(args->argv[1]);
    numElementsPerPe = atoi(args->argv[2]);
    localTableSize = (1l << N) / numElementsPerPe;
    tableSize = localTableSize * CkNumPes() * numElementsPerPe;

    CkPrintf("Global table size   = 2^%d * %d = %lld words\n",
             N, CkNumPes(), tableSize);
    CkPrintf("Number of processors = %d\nNumber of updates = %lld\n",
             CkNumPes(), 4 * tableSize);

    driverProxy = thishandle;
    // Create the chares storing and updating the global table
    updater_array   = CProxy_Updater::ckNew(CkNumPes() * numElementsPerPe);
    int dims[2] = {CkNumNodes(), CkNumPes() / CkNumNodes()};
    CkPrintf("Aggregation topology: %d %d\n", dims[0], dims[1]);

    // Instantiate communication library group with a handle to the client
    aggregator =
      CProxy_ArrayMeshStreamer<dtype, int, Updater, SimpleMeshRouter>
      ::ckNew(numMsgsBuffered, 2, dims, updater_array, 1);

    delete args;
  }

  void start() {
    starttime = CkWallTimer();
    CkCallback startCb(CkIndex_Updater::generateUpdates(), updater_array);
    CkCallback endCb(CkIndex_TestDriver::startVerificationPhase(), thisProxy);
    // Initialize the communication library, which, upon readiness,
    //  will initiate the test via startCb
    aggregator.init(updater_array.ckGetArrayID(), startCb, endCb, -1, false);
  }

  void startVerificationPhase() {
    double update_walltime = CkWallTimer() - starttime;
    double gups = 1e-9 * tableSize * 4.0/update_walltime;
    CkPrintf("CPU time used = %.6f seconds\n", update_walltime);
    CkPrintf("%.9f Billion(10^9) Updates    per second [GUP/s]\n", gups);
    CkPrintf("%.9f Billion(10^9) Updates/PE per second [GUP/s]\n",
             gups / CkNumPes());

    // Repeat the update process to verify
    // At the end of the second update phase, check the global table
    //  for errors in Updater::checkErrors()
    CkCallback startCb(CkIndex_Updater::generateUpdates(), updater_array);
    CkCallback endCb(CkIndex_Updater::checkErrors(), updater_array);
    // Initialize the communication library, which, upon readiness,
    //  will initiate the verification via startCb
    aggregator.init(updater_array.ckGetArrayID(), startCb, endCb, -1, false);
  }

  void reportErrors(CmiInt8 globalNumErrors) {
    CkPrintf("Found %lld errors in %lld locations (%s).\n", globalNumErrors,
             tableSize, globalNumErrors <= 0.01 * tableSize ?
             "passed" : "failed");
    CkExit();
  }
};

// Chare Array with multiple chares on each PE
// Each chare: owns a portion of the global table
//             performs updates on its portion
//             generates random keys and sends them to the appropriate chares
class Updater : public CBase_Updater {
private:
  CmiUInt8 *HPCC_Table;
  CmiUInt8 globalStartmyProc;

public:
  Updater() {
    // Compute table start for this chare
    globalStartmyProc = thisIndex * localTableSize;
    // Create table;
    HPCC_Table = (CmiUInt8*)malloc(sizeof(CmiUInt8) * localTableSize);
    // Initialize
    for(CmiInt8 i = 0; i < localTableSize; i++)
      HPCC_Table[i] = i + globalStartmyProc;
    // Contribute to a reduction to signal the end of the setup phase
    contribute(CkCallback(CkReductionTarget(TestDriver, start), driverProxy));
  }

  Updater(CkMigrateMessage *msg) {}

  // Communication library calls this to deliver each randomly generated key
  inline void process(const dtype  &key) {
    CmiInt8  localOffset = key & (localTableSize - 1);
    // Apply update
    HPCC_Table[localOffset] ^= key;
  }

  void generateUpdates() {
    int arrayN = N - (int) log2((double) numElementsPerPe);
    int numElements = CkNumPes() * numElementsPerPe;
    CmiUInt8 key = HPCC_starts(4 * globalStartmyProc);
    // Get a pointer to the local communication library object
    //  from its proxy handle
    ArrayMeshStreamer<dtype, int, Updater, SimpleMeshRouter>
      * localAggregator = aggregator.ckLocalBranch();

    // Generate this chare's share of global updates
    for(CmiInt8 i = 0; i < 4 * localTableSize; i++) {
      key = key << 1 ^ ((CmiInt8) key < 0 ? POLY : 0);
      int destinationIndex = key >> arrayN & numElements - 1;
      // Submit generated key to chare owning that portion of the table
      localAggregator->insertData(key, destinationIndex);
    }
    // Indicate to the communication library that this chare is done sending
    localAggregator->done();
  }

  void checkErrors() {
    CmiInt8 numErrors = 0;
    // The second verification phase should have returned the table
    //  to its initial state
    for (CmiInt8 j = 0; j < localTableSize; j++)
      if (HPCC_Table[j] != j + globalStartmyProc)
        numErrors++;
    // Sum the errors observed across the entire system
    contribute(sizeof(CmiInt8), &numErrors, CkReduction::sum_long,
               CkCallback(CkReductionTarget(TestDriver, reportErrors),
                          driverProxy));
  }
};

/** random number generator */
CmiUInt8 HPCC_starts(CmiInt8 n) {
  int i, j;
  CmiUInt8 m2[64];
  CmiUInt8 temp, ran;
  while (n < 0) n += PERIOD;
  while (n > PERIOD) n -= PERIOD;
  if (n == 0) return 0x1;
  temp = 0x1;
  for (i = 0; i < 64; i++) {
    m2[i] = temp;
    temp = temp << 1 ^ ((CmiInt8) temp < 0 ? POLY : 0);
    temp = temp << 1 ^ ((CmiInt8) temp < 0 ? POLY : 0);
  }
  for (i = 62; i >= 0; i--)
    if (n >> i & 1)
      break;

  ran = 0x2;
  while (i > 0) {
    temp = 0;
    for (j = 0; j < 64; j++)
      if (ran >> j & 1)
        temp ^= m2[j];
    ran = temp;
    i -= 1;
    if (n >> i & 1)
      ran = ran << 1 ^ ((CmiInt8) ran < 0 ? POLY : 0);
  }
  return ran;
}

#include "randomAccess.def.h"

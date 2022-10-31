#include "NDMeshStreamer.h"

typedef CmiUInt8 dtype;
#include "histo.decl.h"
#include "TopoManager.h"
#include "htram.h"

#include <assert.h>
// Handle to the test driver (chare)
CProxy_TestDriver driverProxy;
/* readonly */ CProxy_HTram htramProxy;
/* readonly */ CProxy_HTramRecv nodeGrpProxy;
int l_num_ups = 1000000;     // per thread number of requests (updates)
int lnum_counts = 1000;       // per thread size of the table

void deliverCallback(CkGroupID gid, void* objPtr, int payload);
void deliverCallbackVerify(CkGroupID gid, void* objPtr, int payload);

class TestDriver : public CBase_TestDriver {
private:
  CProxy_Updater  updater_array;
  double starttime;

public:
  TestDriver(CkArgMsg* args) {
    int64_t printhelp = 0;
    int opt;
    while( (opt = getopt(args->argc, args->argv, "hn:T:")) != -1 ) {
      switch(opt) {
      case 'h': printhelp = 1; break;
      case 'n': sscanf(optarg,"%d" ,&l_num_ups);  break;
      case 'T': sscanf(optarg,"%d" ,&lnum_counts);  break;
      default:  break;
      }
    }
    assert(sizeof(CmiInt8) == sizeof(int64_t));
    CkPrintf("Running histo on %d PEs\n", CkNumPes());
    CkPrintf("Number updates / PE              (-n)= %d\n", l_num_ups);
    CkPrintf("Table size / PE                  (-T)= %d\n", lnum_counts);

    driverProxy = thishandle;
    CkGroupID updater_array_gid;
    updater_array = CProxy_Updater::ckNew(CkNumPes());
    updater_array_gid = updater_array.ckLocalBranch()->getGroupID();
    nodeGrpProxy = CProxy_HTramRecv::ckNew();
    
    htramProxy = CProxy_HTram::ckNew(updater_array_gid, CkCallback(CkReductionTarget(TestDriver, reportErrors), thisProxy));
//    CProxy_Updater updater_array(updater_array_gid);
//    updater_array = CProxy_Updater::ckNew(CkNumPes());
    // Create the chares storing and updating the global table
    //
    //updater_array = CProxy_Updater::ckNew(CkNumPes() * numElementsPerPe);
    int dims[2] = {CkNumNodes(), CkNumPes() / CkNumNodes()};
    CkPrintf("Aggregation topology: %d %d\n", dims[0], dims[1]);

    delete args;
  }

  void start() {
    starttime = CkWallTimer();
    //CkCallback endCb(CkIndex_TestDriver::startVerificationPhase(), thisProxy);
    updater_array.preGenerateUpdates();
    //CkStartQD(endCb);
  }

  void startVerificationPhase() {
    double update_walltime = CkWallTimer() - starttime;
    CkPrintf("  %8.3lf seconds\n", update_walltime);

    // Repeat the update process to verify
    // At the end of the second update phase, check the global table
    //  for errors in Updater::checkErrors()
    CkCallback endCb(CkIndex_Updater::checkErrors(), updater_array);
    updater_array.preGenerateverify();//generateUpdatesVerify();
    CkStartQD(endCb);
  }

  void reportErrors(CmiInt8 globalNumErrors) {
    CkPrintf("Found %" PRId64 " errors in %" PRId64 " locations (%s).\n", globalNumErrors,
             lnum_counts*CkNumPes(), globalNumErrors == 0 ?
             "passed" : "failed");
    CkExit();
  }
};


#define PPN_COUNT 64

// Chare Array with multiple chares on each PE
// Each chare: owns a portion of the global table
//             performs updates on its portion
//             generates random keys and sends them to the appropriate chares
class Updater : public CBase_Updater {
private:
  CmiInt8 *counts;
  CmiInt8 *index;
  CmiInt8 *pckindx;
public:
  Updater() {
    // Compute table start for this chare
    CkPrintf("[PE%d] Update (thisIndex=%d) created: lnum_counts = %d, l_num_ups =%d\n", CkMyPe(), thisIndex, lnum_counts, l_num_ups);

    srand(thisIndex + 120348);
    // Create table;
    counts = (CmiInt8*)malloc(sizeof(CmiInt8) * lnum_counts); assert(counts != NULL);
    // Initialize
    for(CmiInt8 i = 0; i < lnum_counts; i++) {
      counts[i] = 0;
    }
    index = (CmiInt8 *) malloc(l_num_ups * sizeof(CmiInt8)); assert(index != NULL);
    pckindx = (CmiInt8 *) malloc(l_num_ups * sizeof(CmiInt8)); assert(pckindx != NULL);
  
    CmiInt8 num_counts = lnum_counts * CkNumPes();
    CmiInt8 indx, lindx, pe;
    for(CmiInt8 i = 0; i < l_num_ups; i++) {
      //indx = i % num_counts;          //might want to do this for debugging
      indx = rand() % num_counts;
      index[i] = indx;
      lindx = indx / CkNumPes();
      pe  = indx % CkNumPes();
      pckindx[i]  =  (lindx << 16L) | (pe & 0xffff);
    }
    // Contribute to a reduction to signal the end of the setup phase
    contribute(CkCallback(CkReductionTarget(TestDriver, start), driverProxy));
  }

  Updater(CkMigrateMessage *msg) {}

#if 0
  // Communication library calls this to deliver each randomly generated key
  inline void insertData(const CmiInt8& key) {
    counts[key]++;
  }
#endif

  inline void insertData2(const CmiInt8& key) {
    counts[key]--;
  }

  inline void userDeliver(int key) {
//    if(counts==NULL) CkPrintf("\NULL on PE-%d\n", CkMyPe());
//    counts[0]++;
//    CkPrintf("\nUpdatig value [%d] = %d on pe%d\n", key, counts[key], CkMyPe());
    counts[key]++;//index_n[CkMyRank()]++] = key;
  }

  inline void userDeliver2(int key) {
    counts[key]--;
  }

  void preGenerateUpdates() {
    HTram* htram = htramProxy.ckLocalBranch();
    htram->setCb(&deliverCallback, this);
    contribute(CkCallback(CkReductionTarget(Updater, generateUpdates), thisProxy));
  }

  void generateUpdates() {
    // Generate this chare's share of global updates
    CmiInt8 pe, col;
    HTram* htram = htramProxy.ckLocalBranch();

    //CkPrintf("[%d] Hi from generateUpdates %d, l_num_ups: %d\n", CkMyPe(),thisIndex, l_num_ups);
    for(CmiInt8 i = 0; i < l_num_ups; i++) {
      col = pckindx[i] >> 16;
      pe  = pckindx[i] & 0xffff;
      // Submit generated key to chare owning that portion of the table
//      thisProxy(pe).insertData(col);
      htram->insertValue(col, pe);
//      userDeliver(0);
    }
    htram->tflush();
    //contribute(CkCallback(CkReductionTarget(TestDriver, startVerificationPhase), driverProxy));
    if (thisIndex == 0)
      CkStartQD(CkCallback(CkIndex_TestDriver::startVerificationPhase(), driverProxy));
  }

  void preGenerateverify() {
    HTram* htram = htramProxy.ckLocalBranch();
    htram->setCb(&deliverCallbackVerify, this);
    contribute(CkCallback(CkReductionTarget(Updater, generateUpdatesVerify), thisProxy));
  }

  void generateUpdatesVerify() {
  
    // Generate this chare's share of global updates
    CmiInt8 pe, col;
    HTram* htram = htramProxy.ckLocalBranch();
    //CkPrintf("[%d] Hi from generateUpdatesVerify %d, l_num_ups: %d\n", CkMyPe(),thisIndex, l_num_ups);
    for(CmiInt8 i = 0; i < l_num_ups; i++) {
      col = pckindx[i] >> 16;
      pe  = pckindx[i] & 0xffff;
      // Submit generated key to chare owning that portion of the table
      //thisProxy(pe).insertData2(col);
      htram->insertValue(col, pe);
      if  ((i % 10000) == 9999) CthYield();
    }
    htram->tflush();
  }

  void checkErrors() {
    CmiInt8 numErrors = 0;

    for(CmiInt8 i = 0; i < lnum_counts; i++) {
      if(counts[i] != 0L) {
        numErrors++;
        if(numErrors < 5)  // print first five errors, report number of errors below
          fprintf(stderr,"ERROR: Thread %d error at %ld (= %ld)\n", CkMyPe(), i, counts[i]);
      }
    }
    // Sum the errors observed across the entire system
    contribute(sizeof(CmiInt8), &numErrors, CkReduction::sum_long,
               CkCallback(CkReductionTarget(TestDriver, reportErrors),
                          driverProxy));
  }
};

void deliverCallback(CkGroupID gid, void* objPtr, int payload){
//  ((Updater*)CkLocalBranch(gid))->userDeliver(payload);
    ((Updater*)objPtr)->userDeliver(payload);
}

void deliverCallbackVerify(CkGroupID gid, void* objPtr, int payload){
//  ((Updater*)CkLocalBranch(gid))->userDeliver(payload);
    ((Updater*)objPtr)->userDeliver2(payload);
}
#include "histo.def.h"

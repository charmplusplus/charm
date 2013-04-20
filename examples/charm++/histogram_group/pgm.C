#include "pgm.decl.h"
#include <iostream>
#include <cstdlib>
#include <algorithm>

// Readonly proxy that is used by the HistogramMerger group
CProxy_Main mainProxy;
// Proxy to the Histogram chare array
CProxy_Histogram histogramProxy;
// The proxy to the merger group is used by the Histogram chare array to submit results
CProxy_HistogramMerger histogramMergerProxy;

// Initiates and controls program flow
class Main : public CBase_Main {
  int nChares;
  int nElementsPerChare;
  int maxElementValue;
  int nBins;

  public:
  Main(CkArgMsg *m){
    if(m->argc != 5){
      CkPrintf("[main] Usage: pgm <nChares> <nElementsPerChare> <maxElementValue> <nBins>\n");
      CkExit();
    }

    // Process command-line arguments
    nChares = atoi(m->argv[1]);
    nElementsPerChare = atoi(m->argv[2]);
    maxElementValue = atoi(m->argv[3]);
    nBins = atoi(m->argv[4]);

    // Create Histogram chare array 
    histogramProxy = CProxy_Histogram::ckNew(nElementsPerChare, maxElementValue, nChares);
    // Create HistogramMerger group
    histogramMergerProxy = CProxy_HistogramMerger::ckNew(nBins);
    // set readonly 
    mainProxy = thisProxy;

    // Tell Histogram chare array elements to register themselves with their groups.
    // This is done so that each local branch of the HistogramMerger group knows the
    // number of chares from which to expect submissions.
    histogramProxy.registerWithMerger(CkCallback(CkReductionTarget(Main, charesRegistered), mainProxy));
  }

  void charesRegistered(){
    // This entry method is the target of a reduction that finishes when 
    // each chare has registered itself with its local branch of the HistogramMerger group


    // Now, create a number of bins
    CkVec<int> bins(nBins);
    int delta = maxElementValue/nBins;
    if(nBins*delta < maxElementValue) delta++;

    int currentKey = 0;

    // Set the key of each bin
    for(int i = 0; i < nBins; i++, currentKey += delta) bins[i] = currentKey;

    // Broadcast these bin keys to the Histogram chare array. This will cause 
    // each chare to iterate through its set of values and count the number of
    // values that falls into the range implied by each bin.
    histogramProxy.count(&bins[0], nBins);
  }

  void receiveHistogram(int *binCounts, int nBins){
    // This entry method receives the results of the histogramming operation
    int nTotalElements = 0;

    // Print out number of values in each bin, check for sanity and exit
    CkPrintf("[main] histogram: \n");
    for(int i = 0; i < nBins; i++){
      CkPrintf("bin %d count %d\n", i, binCounts[i]);
      nTotalElements += binCounts[i];
    }

    CkAssert(nTotalElements == nChares*nElementsPerChare);
    CkExit();
  }
};

class Histogram : public CBase_Histogram {
  CkVec<int> myValues;

  public:
  Histogram(int nElementsPerChare, int maxElementValue){
    // Create a set of random values within the range [0,maxElementValue)
    myValues.resize(nElementsPerChare);
    std::srand(thisIndex);

    for(int i = 0; i < nElementsPerChare; i++){
      myValues[i] = std::rand() % maxElementValue;
    }
  }

  Histogram(CkMigrateMessage *m) {}

  void registerWithMerger(const CkCallback &replyToCb);
  void count(int *binKeys, int nKeys);
};

class HistogramMerger : public CBase_HistogramMerger {
  int nCharesOnMyPe;
  int nSubmissionsReceived;
  CkVec<int> mergedCounts;

  public:
  HistogramMerger(int nKeys){
    // Initialize counters.
    nCharesOnMyPe = 0;
    nSubmissionsReceived = 0;

    // Allocate an array large enough to hold the counts of all bins and initialize it.
    mergedCounts.resize(nKeys);
    for(int i = 0; i < mergedCounts.size(); i++) mergedCounts[i] = 0;
  }

  
  // Another chare on my PE called the registerMe() method (not an entry).
  // Therefore, I should expect a submission from one more chare.
  void registerMe(){
    nCharesOnMyPe++; 
  }

  // This method (also not an entry) is called by a chare on its local branch 
  // of the HistogramMerger group to submit a partial histogram
  void submitCounts(int *binCounts, int nBins){
    CkAssert(nBins == mergedCounts.size());
    for(int i = 0; i < nBins; i++) mergedCounts[i] += binCounts[i];

    nSubmissionsReceived++;
    // If the number of submissions received equals the number of registered chares, 
    // we can contribute the partial results to a reduction that will convey the final
    // result to the main chare
    if(nSubmissionsReceived == nCharesOnMyPe){
      CkCallback cb(CkReductionTarget(Main, receiveHistogram), mainProxy);
      contribute(mergedCounts.size()*sizeof(int), &mergedCounts[0], CkReduction::sum_int, cb);
    }
  }
};

void Histogram::registerWithMerger(const CkCallback &replyToCb){
  histogramMergerProxy.ckLocalBranch()->registerMe();
  contribute(replyToCb);
}

// Invoked on the Histogram array by the main chare. Given a number of keys,
// a chare goes through its set of random values and for each value from the set,
// increments the count of values in the appropriate bin. This partial result is
// submitted to the HistogramMerger group
void Histogram::count(int *binKeys, int nKeys){
  int *begin = binKeys;
  int *end = begin + nKeys;
  int *search = NULL;

  // Allocate an array for the histogram and initialize the counts
  CkVec<int> myCounts(nKeys);
  for(int i = 0; i < nKeys; i++) myCounts[i] = 0;

  // Iterate over values in myValues
  for(int i = 0; i < myValues.size(); i++){
    int value = myValues[i];

    // Search for appropriate bin for value
    search = std::upper_bound(begin, end, value);
    if(search != begin) search--; 
    int bin = std::distance(begin, search);

    // One more value falls in the range implied by 'bin'
    myCounts[bin]++;
  }

  // Submit partial results to HistogramMerger
  histogramMergerProxy.ckLocalBranch()->submitCounts(&myCounts[0], myCounts.size());
}


#include "pgm.def.h"

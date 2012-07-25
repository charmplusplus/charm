#include "pgm.decl.h"
#include <iostream>
#include <cstdlib>
#include <algorithm>

CProxy_Main mainProxy;
CProxy_Histogram histogramProxy;
CProxy_HistogramMerger histogramMergerProxy;

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

    nChares = atoi(m->argv[1]);
    nElementsPerChare = atoi(m->argv[2]);
    maxElementValue = atoi(m->argv[3]);
    nBins = atoi(m->argv[4]);

    mainProxy = thisProxy;
    histogramProxy = CProxy_Histogram::ckNew(nElementsPerChare, maxElementValue, nChares);
    histogramMergerProxy = CProxy_HistogramMerger::ckNew(nBins);

    histogramProxy.registerWithMerger(CkCallback(CkReductionTarget(Main, charesRegistered), mainProxy));
  }

  void charesRegistered(){
    CkVec<int> bins(nBins);
    int delta = maxElementValue/nBins;
    if(nBins*delta < maxElementValue) delta++;

    int currentKey = 0;

    for(int i = 0; i < nBins; i++, currentKey += delta) bins[i] = currentKey;

    histogramProxy.count(&bins[0], nBins);
  }

  void receiveHistogram(int *binCounts, int nBins){
    int nTotalElements = 0;

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
    nCharesOnMyPe = 0;
    nSubmissionsReceived = 0;

    mergedCounts.resize(nKeys);
    for(int i = 0; i < mergedCounts.size(); i++) mergedCounts[i] = 0;
  }

  void registerMe(){
    nCharesOnMyPe++; 
  }

  void submitCounts(int *binCounts, int nBins){
    CkAssert(nBins == mergedCounts.size());
    for(int i = 0; i < nBins; i++) mergedCounts[i] += binCounts[i];

    nSubmissionsReceived++;
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

void Histogram::count(int *binKeys, int nKeys){
  int *begin = binKeys;
  int *end = begin + nKeys;
  int *search = NULL;

  CkVec<int> myCounts(nKeys);
  for(int i = 0; i < nKeys; i++) myCounts[i] = 0;

  for(int i = 0; i < myValues.size(); i++){
    int value = myValues[i];
    search = std::upper_bound(begin, end, value);

    if(search != begin) search--; 
    int bin = std::distance(begin, search);
    myCounts[bin]++;
  }

  histogramMergerProxy.ckLocalBranch()->submitCounts(&myCounts[0], myCounts.size());
}


#include "pgm.def.h"

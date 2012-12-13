/**
 * \addtogroup CkLdb
*/
/*@{*/

#ifndef _GREEDYLB_H_
#define _GREEDYLB_H_

#include "CentralLB.h"
#include "RefinerTemp.h"
#include "TempAwareRefineLB.decl.h"

void CreateTempAwareRefineLB();
BaseLB * AllocateTempAwareRefineLB();

class TempAwareRefineLB : public CentralLB {
  friend void printCurrentTemperature(void *LB, double curWallTime);
  FILE* logFD;
public: 
 struct HeapData {
    double load;
    int    pe;
    int    id;
  };
	void populateEffectiveFreq(int numProcs);
  int procsPerNode,*freqsEffect,*procFreq,*procFreqEffect,*procFreqNewEffect,*procFreqNew,numProcs,coresPerChip,*freqs,numAvailFreqs;
  int numChips,*procFreqPtr;
  float *procTemp,*avgChipTemp;
  void changeFreq(int);
  TempAwareRefineLB(const CkLBOptions &);
  TempAwareRefineLB(CkMigrateMessage *m):CentralLB(m) { lbname = "TempAwareRefineLB"; }
  void work(LDStats* stats);
  float getTemp(int);
private:
	enum           HeapCmp {GT = '>', LT = '<'};
    	void           Heapify(HeapData*, int, int, HeapCmp);
	void           HeapSort(HeapData*, int, HeapCmp);
	void           BuildHeap(HeapData*, int, HeapCmp);
	CmiBool        Compare(double, double, HeapCmp);
	HeapData*      BuildCpuArray(BaseLB::LDStats*, int, int *);  
	HeapData*      BuildObjectArray(BaseLB::LDStats*, int, int *);      
	CmiBool        QueryBalanceNow(int step);
};

#endif /* _HEAPCENTLB_H_ */

/*@}*/

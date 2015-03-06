
#ifndef _COMMAWARELB_H_
#define _COMMAWARELB_H_

#include "CentralLB.h"
#include "TempAwareCommLB.decl.h"

void CreateTempAwareCommLB();
BaseLB * AllocateTempAwareCommLB();

class TempAwareCommLB : public CBase_TempAwareCommLB {
friend void printCurrentTemperature(void *LB, double curWallTime);
public:
  struct HeapData {
    double load;
    int    pe;
    int    id;

  };
	void populateEffectiveFreq(int numProcs);
	void convertToInsts();
  int procsPerNode,*freqsEffect,*procFreq,*procFreqEffect,*procFreqNewEffect,*procFreqNew,numProcs,coresPerChip,*freqs,numAvailFreqs;
  int numChips,*procFreqPtr;
	void initStructs(LDStats *s);
	void tempControl();
	FILE *migFile;
	double starting;

  float *procTemp,*avgChipTemp;
  TempAwareCommLB(const CkLBOptions &);
  TempAwareCommLB(CkMigrateMessage *m):CBase_TempAwareCommLB(m) {
    lbname = "TempAwareCommLB";
  }
  void work(LDStats* stats);
private:
	enum           HeapCmp {GT = '>', LT = '<'};
    	void           Heapify(HeapData*, int, int, HeapCmp);
	void           HeapSort(HeapData*, int, HeapCmp);
	void           BuildHeap(HeapData*, int, HeapCmp);
	bool        Compare(double, double, HeapCmp);
	HeapData*      BuildCpuArray(BaseLB::LDStats*, int, int *);  
	HeapData*      BuildObjectArray(BaseLB::LDStats*, int, int *);      
	bool        QueryBalanceNow(int step);
};

#endif /* _COMMAWARELB_H_ */

/*@}*/

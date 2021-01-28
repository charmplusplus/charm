/**
 * \addtogroup CkLdb
*/
/*@{*/

#ifndef _TEMPAWAREFINELB_H_
#define _TEMPAWAREFINELB_H_

#include "CentralLB.h"
#include "RefinerTemp.h"
#include "TempAwareRefineLB.decl.h"

void CreateTempAwareRefineLB();
BaseLB * AllocateTempAwareRefineLB();

class TempAwareRefineLB : public CBase_TempAwareRefineLB {
  friend void printCurrentTemperature(void *LB, double curWallTime);
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
  TempAwareRefineLB(CkMigrateMessage *m):CBase_TempAwareRefineLB(m) { lbname = "TempAwareRefineLB"; }
  void work(LDStats* stats);
  float getTemp(int);
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

#endif /* _HEAPCENTLB_H_ */

/*@}*/

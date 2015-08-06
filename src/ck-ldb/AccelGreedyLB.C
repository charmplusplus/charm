/** \file AccelGreedyLB.C
 *
 *  Written by Gengbin Zheng
 *  Updated by Abhinav Bhatele, 2010-12-09 to use ckgraph
 *  Updated by David Kunzman, 2012-03-14 (copy of Greedy and modified for Accel)
 *
 *  Status:
 *    -- Does not support pe_speed's currently
 *    -- Does not support nonmigratable attribute
 */

/**
 * \addtogroup CkLdb
*/
/*@{*/

#include "AccelGreedyLB.h"
#include "ckgraph.h"
#include <algorithm>

#include <map>
#include <stack>

#include "ckaccel.h"


CreateLBFunc_Def(AccelGreedyLB, "always assign the heaviest obj onto lightest loaded processor.")

AccelGreedyLB::AccelGreedyLB(const CkLBOptions &opt) : CentralLB(opt) {
  lbname = "AccelGreedyLB";
  if (CkMyPe()==0)
    CkPrintf("[%d] AccelGreedyLB created\n",CkMyPe());
}

bool AccelGreedyLB::QueryBalanceNow(int _step) {
  //  CkPrintf("[%d] Balancing on step %d\n",CkMyPe(),_step);
  return true;
}

class ProcLoadGreater {
  public:
    bool operator()(ProcInfo p1, ProcInfo p2) {
      return (p1.getTotalLoad() > p2.getTotalLoad());
    }
};

class ObjLoadGreater {
  public:
    bool operator()(Vertex v1, Vertex v2) {
      return (v1.getVertexLoad() > v2.getVertexLoad());
    }
};

class ProcOMIDKey {

 private:

 public:

  int pe;
  int idx;

  ProcOMIDKey(const LDOMid &omid, const int pe) {
    this->idx = omid.id.idx;
    this->pe = pe;
  }

  bool operator<(const ProcOMIDKey &k) const {
    return ((pe < k.pe) ? (true) : (idx < k.idx));
  }

  bool operator==(const ProcOMIDKey &k) const {
    return ((idx == k.idx) && (pe == k.pe));
  }

  bool operator!=(const ProcOMIDKey &k) const {
    return ((idx != k.idx) || (pe != k.pe));
  }

};


class ProcOMIDData {

 private:

 public:

  LBRealType sum;
  int count;

  LBRealType avg;
  LBRealType accumAvg;
  int startCount;    // Inclusive
  int endCount;     // Exclusive

  ProcOMIDData() { sum = 0.0; count = 0; }
  ProcOMIDData(LBRealType wt, int c) { sum = wt; count = c; }

  void add(LBRealType wt) { sum += wt; count++; }
};

//typedef std::map<ProcOMIDKey, ProcOMIDData> ProcOMIDMap;
typedef std::map<int, ProcOMIDData*> ProcOMIDMap;

class AccelData {

 private:

  int numPEs;
  ProcOMIDMap *procObjSums;

  void initialize(int peCount) {
    numPEs = peCount;
    procObjSums = new ProcOMIDMap;
  }

  void cleanup() {
    if (procObjSums != NULL) { delete procObjSums; }
  }

 public:

  AccelData(int peCount) { initialize(peCount); }
  ~AccelData() { cleanup(); }

  void addObj(const LDObjData &obj, const int pe) {

    ProcOMIDMap::iterator i;
    std::pair<ProcOMIDMap::iterator, bool> rtn;

    //ProcOMIDKey key(obj.omID(), pe);
    int key = obj.omID().id.idx;
    i = procObjSums->find(key);
    if (i == procObjSums->end()) {
      //ProcOMIDData data(obj.wallTime, 1);
      ProcOMIDData *data = new ProcOMIDData[numPEs];
      data[pe].add(obj.wallTime);
      rtn = procObjSums->insert(std::pair<int, ProcOMIDData*>(key, data));
    } else {
      (i->second)[pe].add(obj.wallTime);
    }

  }

  void dumpData() {
    ProcOMIDMap::iterator i;
    for (i = procObjSums->begin(); i != procObjSums->end(); i++) {
      //printf("[ACCELDATA-DEBUG] :: PE %d ::  [ %d, %d ] --> ( %lf, %d : %lf )\n",
      //       CkMyPe(), i->first.idx, i->first.pe, i->second.sum, i->second.count, i->second.avg
      //      );
      printf("[LB-DEBUG] ::: PE %d :: ( %d ) --> [ ", CkMyPe(), i->first);
      for (int pe = 0; pe < numPEs; pe++) {
	printf("%lf(%d) ", (i->second)[pe].sum, (i->second)[pe].count);
      }
      printf("]\n");
    }
  }

  ProcOMIDMap::iterator begin() { return procObjSums->begin(); }
  ProcOMIDMap::iterator end() { return procObjSums->end(); }
};


void AccelGreedyLB::work(LDStats* stats)
{

  // DMK - DEBUG
  CkPrintf("[LB-DEBUG] :: PE %d :: AccelGreedyLB::work() - Started @ %lf...\n", CkMyPe(), CmiWallTimer());
  //CkPrintf("[LB-DEBUG] :: PE %d :: stats->count = %d, stats->n_objs = %d...\n", CkMyPe(), stats->count, stats->n_objs);

  AccelData *accelData = new AccelData(stats->count);

  for (int i = 0; i < stats->n_objs; i++) {
    accelData->addObj(stats->objData[i], stats->from_proc[i]);
  }

  //accelData->dumpData();

  ProcOMIDMap::iterator iter;
  for (iter = accelData->begin(); iter != accelData->end(); iter++) {

    //printf("processing %d...\n", iter->first);

    ProcOMIDData *data = iter->second;

    #if 1

    const int numPEs = stats->count;
    const int numObjs = stats->n_objs;

    // Calculate and average time per PE for this object type, tracking the max
    LBRealType avgMax = 0.0;
    int totalCount = 0;
    for (int i = 0; i < numPEs; i++) {
      if (stats->procs[i].available) {
        data[i].avg = ((data[i].count > 0) ? (data[i].sum / data[i].count) : (-1.0));
        if (data[i].avg > avgMax) { avgMax = data[i].avg; }
        totalCount += data[i].count;
      } else {
        data[i].avg = 0.0;
      }
    }

    // Correct for any empty PEs by assuming they are as bad as the worst PE with objects
    // NOTE: Unavailable PEs are set to zero above and should be left at zero here
    for (int i = 0; i < numPEs; i++) { if (data[i].avg < 0.0) { data[i].avg = avgMax; } }

    // Invert the averages based on their max, so that lower averages get higher object counts
    LBRealType avgSum = 0.0;
    for (int i = 0; i < numPEs; i++) {
      if (stats->procs[i].available) {
        data[i].avg = avgMax / data[i].avg;
        avgSum += data[i].avg;
      }
    }

    // TODO : Correct for empty PEs

    // Calculate the target object counts for each PE based on the avg times per PE
    int *targetCount = new int[numPEs];
    int *actualCount = new int[numPEs];
    for (int i = 0; i < numPEs; i++) {
      if (stats->procs[i].available) {
        targetCount[i] = (int)(((LBRealType)totalCount) * (data[i].avg / avgSum));
      } else {
        targetCount[i] = 0;
      }
      actualCount[i] = 0;
    }

    // Apply a damping effect
    const float damping = 0.25f;
    int targetTotal = 0;
    for (int i = 0; i < numPEs; i++) {
      targetCount[i] = (int)(((1.0f - damping) * targetCount[i]) + (damping * data[i].count));
      targetTotal += targetCount[i];
    }
    for (int i = 0; i < numPEs; i++) {
      targetCount[i] = (int)( (((float)targetCount[i]) / ((float)targetTotal)) * totalCount );
    }

    std::list<int> extras;

    // Loop through the objects, counting up to the target value (listing extras as found)
    for (int i = 0; i < numObjs; i++) {
      if (stats->objData[i].omID().id.idx == iter->first) {  // If an object of the correct type
        int fromPE = stats->from_proc[i];
        if (actualCount[fromPE] < targetCount[fromPE]) {
          actualCount[fromPE] += 1;  // Leave this object where it is
	} else {
          extras.push_back(i);  // 'i' is an extra that can be used if needed
	}
      }
    }

    // Loop through the PEs, moving extras to any PEs that are short their target counts
    #if 0  // Fill each PE in turn
      for (int i = 0; i < numPEs && !(extras.empty()); i++) {
        if (stats->procs[i].available) {
          while ((actualCount[i] < targetCount[i]) && (!(extras.empty()))) {
            int objIndex = extras.front();
            extras.pop_front();
            stats->to_proc[objIndex] = i;
            actualCount[i] += 1;
          }
        }
      }
    #else  // Place extra objects in round-robin order
      std::list<int> toPEs;
      for (int i = 0; i < numPEs; i++) {
        if ((stats->procs[i].available) && (actualCount[i] < targetCount[i])) {
          toPEs.push_back(i);
	}
      }
      while ((!(extras.empty())) && (!(toPEs.empty()))) {
        int peI = toPEs.front(); toPEs.pop_front();
        int objI = extras.front(); extras.pop_front();
        stats->to_proc[objI] = peI;
        (actualCount[peI])++;
        if (actualCount[peI] < targetCount[peI]) { toPEs.push_back(peI); }
      }
    #endif

    // DMK - DEBUG - Print calculated distribution
    for (int pe = 0; pe < numPEs; pe++) {
      actualCount[pe] = 0;
    }
    for (int i = 0; i < numObjs; i++) {
      if (stats->objData[i].omID().id.idx == iter->first) {  // If an object of the correct type
        actualCount[stats->to_proc[i]]++;
      }
    }
    printf("[LB-DEBUG] :: PE %d :: %d --->", CkMyPe(), iter->first);
    for (int i = 0; i < numPEs; i++) {
      printf(" %d:%d:%d", i, actualCount[i], targetCount[i]);
    }
    printf("\n");

    // DMK - DEBUG - Print stats
    #if 0
    CkPrintf("[LB-DEBUG] :: PE %d :: %d ===>\n", CkMyPe(), iter->first);
    LBRealType *min = new LBRealType[numPEs * 4];
    LBRealType *max = min + numPEs;
    LBRealType *sum = max + numPEs;
    LBRealType *avg = sum + numPEs;
    int *cnt = new int[numPEs];
    for (int pe = 0; pe < numPEs; pe++) { min[pe] = -1.0; max[pe] = sum[pe] = avg[pe] = 0.0; cnt[pe] = 0; }
    for (int i = 0; i < numObjs; i++) {
      if (stats->objData[i].omID().id.idx == iter->first) {  // If an object of the correct type
        int pe = stats->from_proc[i];
        if (pe < 0 || pe >= numPEs) { CkPrintf("<<bad pe detected>>\n"); }
        LBRealType wt = stats->objData[i].wallTime;
        if (min[pe] < 0.0 || min[pe] > wt) { min[pe] = wt; }
        if (max[pe] < wt) { max[pe] = wt; }
        sum[pe] += wt;
        cnt[pe] += 1;
      }
    }
    for (int pe = 0; pe < numPEs; pe++) {
      LBRealType stdDev = 0.0;
      if (cnt[pe] <= 0) {
        avg[pe] = -1.0;
        stdDev = -1.0;
      } else {
        avg[pe] = sum[pe] / cnt[pe];
        for (int i = 0; i < numObjs; i++) {
          if (stats->objData[i].omID().id.idx == iter->first) {  // If an object of the correct type
            int pe = stats->from_proc[i];
            if (pe < 0 || pe >= numPEs) { CkPrintf("<<bad pe detected>>\n"); }
            LBRealType wt = stats->objData[i].wallTime;
            stdDev += (wt - avg[pe]) * (wt - avg[pe]);
	  }
        }
        stdDev /= cnt[pe];
        stdDev = sqrt(stdDev);
      }
      CkPrintf("                  %d:(%d->%d) %f/%f/%f/%f\n",
               pe, cnt[pe], actualCount[pe], (float)(min[pe]), (float)(avg[pe]), (float)(max[pe]), (float)(stdDev)
              );
    }
    delete [] min;
    delete [] cnt;
    #endif

    delete [] actualCount;
    delete [] targetCount;

    #elif 1

    LBRealType avgSum = 0.0;
    LBRealType avgMax = 0.0;
    int countSum = 0;
    for (int i = 0; i < stats->count; i++) {
      data[i].avg = data[i].sum / data[i].count;
      countSum += data[i].count;
      if (i == 0) {
        avgMax = data[i].avg;
      } else {
        if (data[i].avg > avgMax) { avgMax = data[i].avg; }
      }
    }

    for (int i = 0; i < stats->count; i++) {
      data[i].avg = avgMax / data[i].avg;
      avgSum += data[i].avg;
      data[i].accumAvg = avgSum;
    }

    for (int i = 0; i < stats->count; i++) {
      data[i].accumAvg /= avgSum;
      data[i].startCount = ((i == 0) ? (0) : ( data[i-1].endCount ));
      data[i].endCount = (int)(countSum * data[i].accumAvg);
    }

    for (int i = 0; i < stats->count; i++) {
      data[i].endCount = data[i].endCount - data[i].startCount;  // Becomes target count
      data[i].startCount = 0;                                    // Becomes count of assigned so far
    }

    std::stack<int> extras;

    // Place elements on the extra stack
    for (int i = 0; i < stats->n_objs; i++) {
      if (stats->objData[i].omID().id.idx == iter->first) {
        int fromPE = stats->from_proc[i];
        if (data[fromPE].startCount > 1 && data[fromPE].startCount >= data[fromPE].endCount) {
          extras.push(i);
	} else {
          data[fromPE].startCount++;
	}
      }
    }

    //// DMK - DEBUG
    //printf("[LB-DEBUG] :: PE %d :: extras.size() = %d...\n", CkMyPe(), extras.size());

    // Loop through the PEs, using the extra elements to fill in under utilized PEs
    for (int i = 0; i < stats->count; i++) {
      while (data[i].startCount < data[i].endCount) {
        if (extras.empty()) { break; } // Ran out of extra elements to move around
        int objIndex = extras.top();
        extras.pop();
        stats->to_proc[objIndex] = i;
        data[i].startCount++;
      }
    }
    // NOTE: If there are still elements in extras, then just ignore them and they will not be moved to new PEs

    // DMK - DEBUG
    printf("[LB-DEBUG] :: PE %d :: %d --->", CkMyPe(), iter->first);
    for (int i = 0; i < stats->count; i++) {
      printf(" %d:%d:%d", i, data[i].startCount, data[i].endCount);
    }
    printf("\n");

    #else

    // Correct for PEs with zero elements (make it at least one)
    for (int i = 0; i < stats->count; i++) {
      if (data[i].endCount - data[i].startCount <= 0) {
        int maxIndex = -1;
        int maxCount = -1;
        for (int j = 0; j < stats->count; j++) {
          int jCount = data[j].endCount - data[j].startCount;
          if (i != j && jCount >= 2 && (maxIndex < 0 || maxCount < jCount)) {
            maxIndex = j;
            maxCount = jCount;
          }
	}
        if (maxIndex >= 0) {
          if (maxIndex < i) {
            data[maxIndex].endCount--;
            for (int j = maxIndex + 1; j < i; j++) {
              data[j].startCount--;
              data[j].endCount--;
	    }
            data[i].startCount--;
	  } else {
            data[i].endCount++;
            for (int j = i + 1; j < maxIndex; j++) {
              data[j].startCount++;
              data[j].endCount++;
	    }
            data[maxIndex].startCount++;
	  }
	}
      }
    }

    printf("[LB-DEBUG] :: PE %d :: %3d --->", CkMyPe(), iter->first);
    for (int i = 0; i < stats->count; i++) {
      printf(" %d:%d:%d", i, data[i].endCount, data[i].endCount - data[i].startCount);
    }
    printf("\n");

    int objCount = 0;
    int pe = 0;
    for (int i = 0; i < stats->n_objs; i++) {
      if (stats->objData[i].omID().id.idx == iter->first) {
        while (objCount >= data[pe].endCount) { pe++; }
        stats->to_proc[i] = pe;
        objCount++;
      }
    }

    #endif

  }

  //accelData->dumpData();

  // DMK - DEBUG
  int migrateCount = 0;
  for (int i = 0; i < stats->n_objs; i++) {
    if (stats->from_proc[i] != stats->to_proc[i]) { migrateCount++; }
  }
  CkPrintf("[LB-DEBUG] :: PE %d :: migrateCount = %d...\n", CkMyPe(), migrateCount);

  delete accelData;

  // DMK - DEBUG
  CkPrintf("[LB-DEBUG] :: PE %d :: AccelGreedyLB::work() - Finished @ %lf...\n", CkMyPe(), CmiWallTimer());


  #if 0
  /** ========================== INITIALIZATION ============================= */
  ProcArray *parr = new ProcArray(stats);       // Processor Array
  ObjGraph *ogr = new ObjGraph(stats);          // Object Graph


  /** ============================= STRATEGY ================================ */

  parr->resetTotalLoad();

  if (_lb_args.debug()>1)
    CkPrintf("[%d] In AccelGreedyLB strategy\n",CkMyPe());

  int vert;

  // max heap of objects
  std::sort(ogr->vertices.begin(), ogr->vertices.end(), ObjLoadGreater());
  // min heap of processors
  std::make_heap(parr->procs.begin(), parr->procs.end(), ProcLoadGreater());

  for(vert = 0; vert < ogr->vertices.size(); vert++) {
    // Pop the least loaded processor
    ProcInfo p = parr->procs.front();
    std::pop_heap(parr->procs.begin(), parr->procs.end(), ProcLoadGreater());
    parr->procs.pop_back();

    // Increment the load of the least loaded processor by the load of the
    // 'heaviest' unmapped object
    p.setTotalLoad(p.getTotalLoad() + ogr->vertices[vert].getVertexLoad());
    ogr->vertices[vert].setNewPe(p.getProcId());

    // Insert the least loaded processor with load updated back into the heap
    parr->procs.push_back(p);
    std::push_heap(parr->procs.begin(), parr->procs.end(), ProcLoadGreater());
  }


  /** ============================== CLEANUP ================================ */
  ogr->convertDecisions(stats);         // Send decisions back to LDStats

  #endif
}

#include "AccelGreedyLB.def.h"

/*@}*/


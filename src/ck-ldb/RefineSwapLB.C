/** \file RefineSwapLB.C
 *
 *  Written by Harshitha Menon 
 *
 *  Status:
 *    -- Does not support pe_speed's currently
 *    -- Does not support nonmigratable attribute
 */

/**
 * \addtogroup CkLdb
*/
/*@{*/

#include "RefineSwapLB.h"
#include "ckgraph.h"
#include <algorithm>
#include <iostream>

using std::cout;
using std::endl;

CreateLBFunc_Def(RefineSwapLB,
    "always assign the heaviest obj onto lightest loaded processor.")

RefineSwapLB::RefineSwapLB(const CkLBOptions &opt): CBase_RefineSwapLB(opt)
{
  lbname = "RefineSwapLB";
  if (CkMyPe()==0)
    CkPrintf("[%d] RefineSwapLB created\n",CkMyPe());
}

bool RefineSwapLB::QueryBalanceNow(int _step)
{
  //  CkPrintf("[%d] Balancing on step %d\n",CkMyPe(),_step);
  return true;
}

class ProcLoadGreater {
  public:
    bool operator()(ProcInfo p1, ProcInfo p2) {
      return (p1.getTotalLoad() > p2.getTotalLoad());
    }
};

class ProcLoadGreaterIndex {
 public: 
  ProcLoadGreaterIndex(ProcArray * parr) : parr(parr) {}
  bool operator()(int lhs, int rhs) {
    return (parr->procs[lhs].getTotalLoad() < parr->procs[rhs].getTotalLoad());
  }
 private:
  ProcArray *parr;
};

class ObjLoadGreater {
  public:
    ObjLoadGreater(ObjGraph* ogr) : ogr(ogr) {}
    bool operator()(int lhs, int rhs) {
      return (ogr->vertices[lhs].getVertexLoad() < ogr->vertices[rhs].getVertexLoad());
    }
  private:
    ObjGraph* ogr;
};

inline void addObjToProc(ProcArray* parr, ObjGraph* ogr, std::vector<int>*
    pe_obj, int pe_index, int obj_index) {

  // Set the new pe
  ogr->vertices[obj_index].setNewPe(pe_index);

  // Add obj to the pe obj list
  pe_obj[pe_index].push_back(obj_index);

  // Update load
  parr->procs[pe_index].totalLoad() += ogr->vertices[obj_index].getVertexLoad();
}

inline void removeObjFromProc(ProcArray* parr, ObjGraph* ogr, std::vector<int>*
    pe_obj, int pe_index, int arr_index) {

  // Update load
  parr->procs[pe_index].totalLoad() -=
      ogr->vertices[pe_obj[pe_index][arr_index]].getVertexLoad();

  // Remove from pe_obj
  pe_obj[pe_index].erase(pe_obj[pe_index].begin() + arr_index);
}


inline int getMax(ProcArray* parr, std::vector<int>& max_pe_heap) {
  int p_index = max_pe_heap.front();
  std::pop_heap(max_pe_heap.begin(), max_pe_heap.end(),
      ProcLoadGreaterIndex(parr));
  max_pe_heap.pop_back();
  return p_index;
}

bool refine(ProcArray* parr, ObjGraph* ogr, std::vector<int>& max_pe_heap, 
    std::vector<int>& min_pe_heap, std::vector<int>* pe_obj, int max_pe,
    double avg_load, double threshold) {

  int best_p, best_p_iter, arr_index;
  bool allocated = false;
  int pe_considered;
  int obj_considered;
  double best_size = 0.0;
  std::sort(pe_obj[max_pe].begin(), pe_obj[max_pe].end(), ObjLoadGreater(ogr));

  // Iterate over all the min pes and see which is the best object to
  // transfer.

  for (int i = (pe_obj[max_pe].size()-1); i >= 0; i--) {
    for (int j = 0; j < min_pe_heap.size(); j++) {
      obj_considered = pe_obj[max_pe][i];
      pe_considered = min_pe_heap[j];
   
      if (parr->procs[pe_considered].getTotalLoad() +
          ogr->vertices[obj_considered].getVertexLoad() < (avg_load + threshold)) {
    //    if (ogr->vertices[obj_considered].getVertexLoad() > best_size) {
          best_size = ogr->vertices[obj_considered].getVertexLoad();
          best_p = pe_considered;
          best_p_iter = j;
          arr_index = i;
          allocated = true;
          break;
    //    }
      }
    }
  }

  if (allocated) {

    int best_obj = pe_obj[max_pe][arr_index];
    addObjToProc(parr, ogr, pe_obj, best_p, best_obj);
    removeObjFromProc(parr, ogr, pe_obj, max_pe, arr_index);

    // Update the max heap and min list
    if (parr->procs[max_pe].getTotalLoad() > (avg_load + threshold)) {
      // Reinsert
      max_pe_heap.push_back(max_pe);
      std::push_heap(max_pe_heap.begin(), max_pe_heap.end(),
          ProcLoadGreaterIndex(parr));
    } else if (parr->procs[max_pe].getTotalLoad() < (avg_load - threshold)) {
      // Insert into the list of underloaded procs
      min_pe_heap.push_back(max_pe);
    }

    if (parr->procs[best_p].getTotalLoad() > (avg_load - threshold)) {
      // Remove from list of underloaded procs
      min_pe_heap.erase(min_pe_heap.begin() + best_p_iter);
    }
  }
  return allocated;
}

bool IsSwapPossWithPe(ProcArray* parr, ObjGraph* ogr, std::vector<int>* pe_obj,
    std::vector<int>& max_pe_heap, std::vector<int>& min_pe_heap,
    int max_pe, int pe_considered, int pe_cons_iter, double diff,
    double avg_load, double threshold) {

  bool set = false;
  for (int i = pe_obj[max_pe].size() - 1; i >= 0; i--) {
    for (int j = 0; j < pe_obj[pe_considered].size(); j++) {
      int pe_cons = pe_obj[pe_considered][j];
      int max_pe_obj = pe_obj[max_pe][i];
     // CkPrintf("\tCandidates %d(%lf) with %d(%lf) : diff (%lf)\n", max_pe_obj,
     //     ogr->vertices[max_pe_obj].getVertexLoad(), pe_cons,
     //     ogr->vertices[pe_cons].getVertexLoad(), diff);

      if (ogr->vertices[pe_cons].getVertexLoad() <
          ogr->vertices[max_pe_obj].getVertexLoad()) {
        if ((diff + ogr->vertices[pe_cons].getVertexLoad()) >
            ogr->vertices[max_pe_obj].getVertexLoad()) {
          //CkPrintf("\tSwapping %d with %d\n", max_pe_obj, pe_cons);
          set = true;

          addObjToProc(parr, ogr, pe_obj, pe_considered, max_pe_obj);
          removeObjFromProc(parr, ogr, pe_obj, max_pe, i);

          addObjToProc(parr, ogr, pe_obj, max_pe, pe_cons);
          removeObjFromProc(parr, ogr, pe_obj, pe_considered, j);

          // Update the max heap and min list
          if (parr->procs[max_pe].getTotalLoad() > (avg_load + threshold)) {
            // Reinsert
            max_pe_heap.push_back(max_pe);
            std::push_heap(max_pe_heap.begin(), max_pe_heap.end(),
                ProcLoadGreaterIndex(parr));
          } else if (parr->procs[max_pe].getTotalLoad() < (avg_load - threshold)) {
            // Insert into the list of underloaded procs
            min_pe_heap.push_back(max_pe);
          }

          if (parr->procs[pe_considered].getTotalLoad() > (avg_load - threshold)) {
            // Remove from list of underloaded procs
            min_pe_heap.erase(min_pe_heap.begin() + pe_cons_iter);
          }
          break;
        }
      }
    }

    if (set) {
      break;
    }
  }
  return set;
}

bool refineSwap(ProcArray* parr, ObjGraph* ogr, std::vector<int>& max_pe_heap, 
    std::vector<int>& min_pe_heap, std::vector<int>* pe_obj, int max_pe,
    double avg_load, double threshold) {

  double diff = 0;
  bool is_possible = false;
  int pe_considered;
  int pe_cons_iter;
  for (int i = 0; i < min_pe_heap.size(); i++) {
    pe_considered = min_pe_heap[i];
    pe_cons_iter = i;
    std::sort(pe_obj[pe_considered].begin(), pe_obj[pe_considered].end(), ObjLoadGreater(ogr));
    diff = avg_load - parr->procs[pe_considered].getTotalLoad();

//    CkPrintf("Checking to swap maxload pe %d  with minpe %d  + diff %lf \n",
//        max_pe, pe_considered, diff);
    is_possible = IsSwapPossWithPe(parr, ogr, pe_obj, max_pe_heap, min_pe_heap, max_pe,
        pe_considered, pe_cons_iter, diff, avg_load, threshold); 
    if (is_possible) {
      break;
    }
  }

  if (!is_possible) {
    return false;
  }

  return true;
}

void RefineSwapLB::work(LDStats* stats)
{
  /** ========================== INITIALIZATION ============================= */
  ProcArray *parr = new ProcArray(stats);       // Processor Array
  ObjGraph *ogr = new ObjGraph(stats);          // Object Graph


  /** ============================= STRATEGY ================================ */

  if (_lb_args.debug()>1) 
    CkPrintf("[%d] In RefineSwapLB strategy\n",CkMyPe());

  int vert;
  double avg_load = parr->getAverageLoad();
  double threshold = avg_load * 0.01;
  double lower_bound_load = avg_load - threshold;
  double upper_bound_load = avg_load + threshold;
  cout <<"Average load " << avg_load << endl;
  
  std::vector<int> min_pe_heap;
  std::vector<int> max_pe_heap;

  std::vector<int>* pe_obj = new std::vector<int>[parr->procs.size()];


  // Create a datastructure to store the objects in a processor
  for (int i = 0; i < ogr->vertices.size(); i++) {
    pe_obj[ogr->vertices[i].getCurrentPe()].push_back(i);
//    CkPrintf("%d pe %d: %lf\n", i, ogr->vertices[i].getCurrentPe(), ogr->vertices[i].getVertexLoad());
  }

  // Construct max heap of overloaded processors and min heap of underloaded
  // processors.
  for (int i = 0; i < parr->procs.size(); i++) {
    //CkPrintf("%d : %lf\n", i, parr->procs[i].getTotalLoad());
    if (parr->procs[i].getTotalLoad() > upper_bound_load) {
      max_pe_heap.push_back(i);
    } else if (parr->procs[i].getTotalLoad() < lower_bound_load) {
      min_pe_heap.push_back(i);
    }
  }

  std::make_heap(max_pe_heap.begin(), max_pe_heap.end(), ProcLoadGreaterIndex(parr));

  while (max_pe_heap.size() != 0 && min_pe_heap.size() != 0) {
    int p_index = getMax(parr, max_pe_heap);
    ProcInfo &pinfo = parr->procs[p_index];

    bool success = refine(parr, ogr, max_pe_heap, min_pe_heap, pe_obj, p_index, avg_load, threshold);
    

    if (!success) {
      // Swap with something. 

      if (!refineSwap(parr, ogr, max_pe_heap, min_pe_heap, pe_obj, p_index, avg_load,
            threshold)) {
        max_pe_heap.push_back(p_index);
        std::push_heap(max_pe_heap.begin(), max_pe_heap.end(),
            ProcLoadGreaterIndex(parr));
        break;
      }
    }
  }

  /** ============================== CLEANUP ================================ */
  ogr->convertDecisions(stats);         // Send decisions back to LDStats
  delete[] pe_obj;
  delete parr;
  delete ogr;
}

#include "RefineSwapLB.def.h"

/*@}*/


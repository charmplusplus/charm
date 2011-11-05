/** \file RefineSwapLB.C
 *
 *  Written by Gengbin Zheng
 *  Updated by Abhinav Bhatele, 2010-12-09 to use ckgraph
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

CreateLBFunc_Def(RefineSwapLB, "always assign the heaviest obj onto lightest loaded processor.")

RefineSwapLB::RefineSwapLB(const CkLBOptions &opt): CentralLB(opt)
{
  lbname = "RefineSwapLB";
  if (CkMyPe()==0)
    CkPrintf("[%d] RefineSwapLB created\n",CkMyPe());
}

CmiBool RefineSwapLB::QueryBalanceNow(int _step)
{
  //  CkPrintf("[%d] Balancing on step %d\n",CkMyPe(),_step);
  return CmiTrue;
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
    bool operator()(Vertex v1, Vertex v2) {
      return (v1.getVertexLoad() > v2.getVertexLoad());
    }
};


void RefineSwapLB::work(LDStats* stats)
{
  /** ========================== INITIALIZATION ============================= */
  ProcArray *parr = new ProcArray(stats);       // Processor Array
  ObjGraph *ogr = new ObjGraph(stats);          // Object Graph


  /** ============================= STRATEGY ================================ */
  //parr->resetTotalLoad();

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

  // Construct max heap of overloaded processors and min heap of underloaded
  // processors.
  for (int i = 0; i < parr->procs.size(); i++) {
    CkPrintf("%d : %lf\n", i, parr->procs[i].getTotalLoad());
    if (parr->procs[i].getTotalLoad() > upper_bound_load) {
      max_pe_heap.push_back(i);
    } else if (parr->procs[i].getTotalLoad() < lower_bound_load) {
      min_pe_heap.push_back(i);
    }
  }

  // Create a datastructure to store the objects in a processor
  CkPrintf("Object load\n");
  for (int i = 0; i < ogr->vertices.size(); i++) {
    pe_obj[ogr->vertices[i].getCurrentPe()].push_back(i);
    CkPrintf("%d: %lf\n", ogr->vertices[i].getCurrentPe(), ogr->vertices[i].getVertexLoad());
  }

  std::make_heap(max_pe_heap.begin(), max_pe_heap.end(), ProcLoadGreaterIndex(parr));

  int best_p;
  int best_obj;
  int iter_location;
  int obj_iter_location;
  int min_pe_iter;
  int min_weight_obj;
  int min_iter_location;
  while (max_pe_heap.size() != 0 && min_pe_heap.size() != 0) {
    int ideal_transfer_pe = 0;
    int p_index = max_pe_heap.front();
    double best_size = 0.0;
    double obj_wg_min = 100.0;
    int allocated = false;
    int obj_considered;
    int pe_considered;
    ProcInfo &pinfo = parr->procs[p_index];
    std::pop_heap(max_pe_heap.begin(), max_pe_heap.end(),
        ProcLoadGreaterIndex(parr));
    max_pe_heap.pop_back();
    cout << "Picked max pe " << p_index << " (" <<
        parr->procs[p_index].getTotalLoad() << ")" << endl;
    int second_phase_pe_considered_iter = 0;

    // Iterate over all the min pes and see which is the best object to
    // transfer.
    for (int j = 0; j < min_pe_heap.size(); j++) {
      for (int i = 0; i < pe_obj[p_index].size(); i++) {
        obj_considered = pe_obj[p_index][i];
        pe_considered = min_pe_heap[j];

        if (parr->procs[pe_considered].getTotalLoad() < parr->procs[ideal_transfer_pe].getTotalLoad()) {
          ideal_transfer_pe = pe_considered;
          second_phase_pe_considered_iter = j;
        }
        if (ogr->vertices[obj_considered].getVertexLoad() < obj_wg_min) {
          min_weight_obj = obj_considered;
          min_iter_location = i;
        }

        if (parr->procs[pe_considered].getTotalLoad() + ogr->vertices[obj_considered].getVertexLoad() < (avg_load + threshold)) {
          if (ogr->vertices[obj_considered].getVertexLoad() > best_size) {
            best_obj = obj_considered;
            best_size = ogr->vertices[obj_considered].getVertexLoad();
            best_p = pe_considered;
            iter_location = i;
            min_pe_iter = j;
            allocated = true;
          }
        }
      }
    }

    if (allocated) {
      // Set the new pe
      ogr->vertices[best_obj].setNewPe(best_p);

      // Remove from pe_obj
      pe_obj[p_index].erase(pe_obj[p_index].begin() + iter_location);
      pe_obj[best_p].push_back(best_obj);

      // Update load of underloaded and overloaded
      parr->procs[p_index].setTotalLoad(parr->procs[p_index].getTotalLoad() -
          best_size);
      parr->procs[best_p].setTotalLoad(parr->procs[best_p].getTotalLoad() +
          best_size);

      std::cout << " Moving obj " << best_obj << " (" << best_size << ") from " << p_index << " to " <<
            best_p << " New load " << p_index << ":" << parr->procs[p_index].getTotalLoad()
            << " " << best_p << ":" << parr->procs[best_p].getTotalLoad()<< std::endl; 

      // Update the max heap and min list
      if (parr->procs[p_index].getTotalLoad() > (avg_load + threshold)) {
        // Reinsert
        max_pe_heap.push_back(p_index);
        std::push_heap(max_pe_heap.begin(), max_pe_heap.end(),
            ProcLoadGreaterIndex(parr));
      } else if (parr->procs[p_index].getTotalLoad() < (avg_load - threshold)) {
        // Insert into the list of underloaded procs
        min_pe_heap.push_back(p_index);
      }

      if (parr->procs[best_p].getTotalLoad() > (avg_load - threshold)) {
        // Remove from list of underloaded procs
        min_pe_heap.erase(min_pe_heap.begin() + min_pe_iter);
      }
    } else {
      // Swap with something. 
      // TODO:
//      cout << " Swapping needs to be done min weight pe : " << ideal_transfer_pe
//      << " load " << parr->procs[ideal_transfer_pe].getTotalLoad() << " diff "
//      << avg_load - parr->procs[ideal_transfer_pe].getTotalLoad() << endl;
    //  max_pe_heap.push_back(p_index);
    //  std::push_heap(max_pe_heap.begin(), max_pe_heap.end(),
    //      ProcLoadGreaterIndex(parr));
      double diff_load = avg_load - parr->procs[ideal_transfer_pe].getTotalLoad(); 

      int possibility_x = 0;
      int possibility_y = 0;
      bool is_possible = false;
      for (int i = 0; i < pe_obj[p_index].size(); i++) {
        for (int j = 0; j < pe_obj[ideal_transfer_pe].size(); j++) {
          if ((ogr->vertices[pe_obj[p_index][i]].getVertexLoad() >
                ogr->vertices[pe_obj[ideal_transfer_pe][j]].getVertexLoad())) {
           // CkPrintf("%d (%lf) : %d(%lf) \n", pe_obj[p_index][i],
           //     ogr->vertices[pe_obj[p_index][i]].getVertexLoad(),
           //     pe_obj[ideal_transfer_pe][j],
           //     ogr->vertices[pe_obj[ideal_transfer_pe][j]].getVertexLoad());
           // CkPrintf("\t %lf : %lf \n",
           // ogr->vertices[pe_obj[p_index][i]].getVertexLoad(),
           // ogr->vertices[pe_obj[ideal_transfer_pe][j]].getVertexLoad() + diff_load);

            if ((ogr->vertices[pe_obj[p_index][i]].getVertexLoad() - diff_load) <
                ogr->vertices[pe_obj[ideal_transfer_pe][j]].getVertexLoad()){
              is_possible = true;
              possibility_x = i;
              possibility_y = j;
            }
          }
        }
      }
      if (!is_possible) {
        max_pe_heap.push_back(p_index);
        std::push_heap(max_pe_heap.begin(), max_pe_heap.end(),
            ProcLoadGreaterIndex(parr));
        //for (int i = 0; i < pe_obj[p_index].size(); i++) {
        //  for (int j = 0; j < pe_obj[ideal_transfer_pe].size(); j++) {
        //    CkPrintf("\t :( %d (%lf) : %d(%lf) \n", pe_obj[p_index][i],
        //        ogr->vertices[pe_obj[p_index][i]].getVertexLoad(),
        //        pe_obj[ideal_transfer_pe][j],
        //        ogr->vertices[pe_obj[ideal_transfer_pe][j]].getVertexLoad());
        //    CkPrintf("\t\t %lf : %lf \n",
        //        ogr->vertices[pe_obj[p_index][i]].getVertexLoad(),
        //        ogr->vertices[pe_obj[ideal_transfer_pe][j]].getVertexLoad() + diff_load);
        //  }
        //}

        break;
      }
     // CkPrintf(" Possibility of swap %d (%lf) : %d(%lf) \n",
     //     pe_obj[p_index][possibility_x],
     //     ogr->vertices[pe_obj[p_index][possibility_x]].getVertexLoad(),
     //     pe_obj[ideal_transfer_pe][possibility_y],
     //     ogr->vertices[pe_obj[ideal_transfer_pe][possibility_y]].getVertexLoad());


      pe_obj[ideal_transfer_pe].push_back(pe_obj[p_index][possibility_x]);
      parr->procs[p_index].setTotalLoad(parr->procs[p_index].getTotalLoad() -
          ogr->vertices[pe_obj[p_index][possibility_x]].getVertexLoad());
      parr->procs[ideal_transfer_pe].setTotalLoad(parr->procs[ideal_transfer_pe].getTotalLoad() +
          ogr->vertices[pe_obj[p_index][possibility_x]].getVertexLoad());
      pe_obj[p_index].erase(pe_obj[p_index].begin() + possibility_x);

      pe_obj[p_index].push_back(pe_obj[ideal_transfer_pe][possibility_y]);
      // Update load of underloaded and overloaded
      parr->procs[p_index].setTotalLoad(parr->procs[p_index].getTotalLoad() +
          ogr->vertices[pe_obj[ideal_transfer_pe][possibility_y]].getVertexLoad());
      parr->procs[ideal_transfer_pe].setTotalLoad(parr->procs[ideal_transfer_pe].getTotalLoad() -
          ogr->vertices[pe_obj[ideal_transfer_pe][possibility_y]].getVertexLoad());
      pe_obj[ideal_transfer_pe].erase(pe_obj[ideal_transfer_pe].begin() + possibility_y);


      // Update the max heap and min list
      if (parr->procs[p_index].getTotalLoad() > (avg_load + threshold)) {
        // Reinsert
        max_pe_heap.push_back(p_index);
        std::push_heap(max_pe_heap.begin(), max_pe_heap.end(),
            ProcLoadGreaterIndex(parr));
      } else if (parr->procs[p_index].getTotalLoad() < (avg_load - threshold)) {
        // Insert into the list of underloaded procs
        min_pe_heap.push_back(p_index);
      }

      if (parr->procs[ideal_transfer_pe].getTotalLoad() > (avg_load - threshold)) {
        // Remove from list of underloaded procs
        min_pe_heap.erase(min_pe_heap.begin() + second_phase_pe_considered_iter);
      }

    }
  }

  std::cout << "Overloaded Processor load"<< endl;
  for (int p_index = 0; p_index < max_pe_heap.size(); p_index++) {
    std::cout << max_pe_heap[p_index] << ": " << parr->procs[max_pe_heap[p_index]].getTotalLoad() << std::endl;

       
  }

  std::cout << "Processor load"<< endl;
  for (int i = 0; i < parr->procs.size(); i++) {
    CkPrintf("%d : %lf\n", i, parr->procs[i].getTotalLoad());
  }

  /** ============================== CLEANUP ================================ */
  ogr->convertDecisions(stats);         // Send decisions back to LDStats
  delete[] pe_obj;
  delete parr;
  delete ogr;
}

#include "RefineSwapLB.def.h"

/*@}*/


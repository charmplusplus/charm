/** \file TempAwareCommLB.C
 *
 *  Written by Osman Sarood
 *  This load balancing strategy is meant for temperature control. It is mandotry to 
 *  have access to frequency control to use it. It is a mixture of Refine and Comm LBs
 *  in addition to normalization being done according to frequencies. 
 *  This load balancer can also cater for sensitivity to frequency that an applicaiton 
 *  has to frequency.  
 */

/*@{*/
//#define NO_TEMP_LB
//#define ORG_VERSION
//#define MAX_MIN
#define MAX_TEMP 49
//#define tolerance 0.03

#include "TempAwareCommLB.h"
#include "ckgraphTemp.h"
#include <algorithm>
#include <map>

#include <time.h>

#define THRESHOLD 0.02
#define SWAP_MULTIPLIER 5 
#ifdef TEMP_LDB


static int cpufreq_sysfs_write (
                     const char *setting,int proc
                     )
{
char path[100];
sprintf(path,"/sys/devices/system/cpu/cpu%d/cpufreq/scaling_setspeed",proc);
                FILE *fd = fopen (path, "w");

                if (!fd) {
                        printf("PROC#%d ooooooo666 FILE OPEN ERROR file=%s\n",CkMyPe(),path);
                        return -1;
                }
//                else CkPrintf("PROC#%d opened freq file=%s\n",proc,path);

        fseek ( fd , 0 , SEEK_SET );
        int numw=fprintf (fd, setting);
        if (numw <= 0) {

                fclose (fd);
                printf("FILE WRITING ERROR\n");
                return 0;
        }
//        else CkPrintf("Freq for Proc#%d set to %s numw=%d\n",proc,setting,numw);
        fclose(fd);
        return 1;
}

static int cpufreq_sysfs_read (int proc)
{
        FILE *fd;
        char path[100];
        int i=proc;
        sprintf(path,"/sys/devices/system/cpu/cpu%d/cpufreq/scaling_setspeed",i);

        fd = fopen (path, "r");

        if (!fd) {
                printf("22 FILE OPEN ERROR file=%s\n",path);
                return 0;
        }
        char val[10];
        fgets(val,10,fd);
        int ff=atoi(val);
        fclose (fd);

        return ff;
}

void printCurrentTemperature(void *LB, double curWallTime)
{
  TempAwareCommLB *taalb = static_cast<TempAwareCommLB *>(LB);
  int pe = CkMyPe();
  float temp = taalb->getTemp(pe % taalb->physicalCoresPerNode);
  int freq = cpufreq_sysfs_read (pe % taalb->logicalCoresPerNode);
  fprintf(taalb->logFD, "%f, %d, %f, %d\n", curWallTime, pe, temp, freq);
}

#endif
inline void eraseObjFromParrObjs(std::vector<int> & parr_objs, int remove_objid);
inline void printMapping(std::vector<Vertex> &vertices);
inline void removeFromArray(int pe_id, std::vector<int> &array);
inline int popFromProcHeap(std::vector<int> & parr_above_avg, ProcArray *parr);
inline void handleTransfer(int randomly_obj_id, ProcInfo& p, int possible_pe, std::vector<int> *parr_objs, ObjGraph *ogr, ProcArray* parr);
inline void updateLoadInfo(int p_index, int possible_pe, double upper_threshold_temp, double lower_threshold_temp,
                           std::vector<int> &parr_above_avg, std::vector<int> &parr_below_avg,
                           std::vector<bool> &proc_load_info, ProcArray *parr);
inline void getPossiblePes(std::vector<int>& possible_pes, int randomly_obj_id,
    ObjGraph *ogr, ProcArray* parr);

double upper_threshold_temp;
double lower_threshold_temp;

CreateLBFunc_Def(TempAwareCommLB, "always assign the heaviest obj onto lightest loaded processor.")

TempAwareCommLB::TempAwareCommLB(const CkLBOptions &opt): CBase_TempAwareCommLB(opt)
{
#ifdef TEMP_LDB
  lbname = "TempAwareCommLB";
  if (CkMyPe()==0)
    CkPrintf("[%d] TempAwareCommLB created\n",CkMyPe());

	starting=CmiWallTimer();
	migFile=fopen("migInfo","w");
  numAvailFreqs = 11;
	freqs=new int[numAvailFreqs];
	freqsEffect=new int[numAvailFreqs];
// for tarekc cluster
  freqs[0] = 2395000;
  freqs[1] = 2394000;
  freqs[2] = 2261000;
  freqs[3] = 2128000;
  freqs[4] = 1995000;
  freqs[5] = 1862000;
  freqs[6] = 1729000;
  freqs[7] = 1596000;
  freqs[8] = 1463000;
  freqs[9] = 1330000;
  freqs[10] = 1197000;

  freqsEffect[0] = 1979886;
  freqsEffect[1] = 1943017;
  freqsEffect[2] = 1910989;
  freqsEffect[3] = 1876619;
  freqsEffect[4] = 1824126;
  freqsEffect[5] = 1763990;
  freqsEffect[6] = 1666773;
  freqsEffect[7] = 1560224;
  freqsEffect[8] = 1443154;
  freqsEffect[9] = 1317009;
  freqsEffect[10] = 1200000;

  procFreqPtr = new int[CkNumPes()];

  for(int i=0;i<CkNumPes();i++)
  {
        char newfreq[10];
        sprintf(newfreq,"%d",freqs[0]);
  cpufreq_sysfs_write(newfreq,i%physicalCoresPerNode);
  procFreqPtr[i]=0;
  }

  procFreq=NULL;
  procTemp=NULL;
  procFreqNew=NULL;
  procFreqNewEffect = NULL;
  avgChipTemp=NULL;

  char logFile[100];
  snprintf(logFile, sizeof(logFile), "temp_freq.log.%d", CkMyPe());
  if ((logFD = fopen(logFile, "a"))) {
    fprintf(logFD, "Time, PE, Temperature, Frequency\n");
  } else {
    CkAbort("Couldn't open temperature/frequency log file");
  }


  CcdCallOnConditionKeep(CcdPERIODIC_1second, &printCurrentTemperature, this);
#else
#endif
}

bool TempAwareCommLB::QueryBalanceNow(int _step)
{
  //  CkPrintf("[%d] Balancing on step %d\n",CkMyPe(),_step);
  return true;
}

class ProcLoadGreater {
  public:
    ProcLoadGreater(ProcArray *parr) : parr(parr) {
    }
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

class PeCommInfo {
  public:
    PeCommInfo() : num_msg(0), num_bytes(0) {
    }

    PeCommInfo(int pe_id) : pe_id(pe_id), num_msg(0) , num_bytes(0) {
    }
    int pe_id;
    int num_msg;
    int num_bytes;
    // TODO: Should probably have a communication cost
};

// Consists of communication information of an object with is maintained
// as a list of PeCommInfo containing the processor id and the bytes transferred
class ObjPeCommInfo {
  public:
    ObjPeCommInfo() {
    }

    int obj_id;
    std::vector<PeCommInfo> pcomm;
};

class ProcCommGreater {
  public:
    bool operator()(PeCommInfo p1, PeCommInfo p2) {
      // TODO(Harshitha): Should probably consider total communication cost
      return (p1.num_bytes > p2.num_bytes);
    }
};

void PrintProcLoadTemp(ProcArray *parr) {
  int vert;
  double pe_load;
  for (vert = 0; vert < parr->procs.size(); vert++) {
    pe_load = parr->procs[vert].getTotalLoad();
    if (pe_load > upper_threshold_temp) {
      CkPrintf("Above load : %d load : %E overhead : %E\n",
        parr->procs[vert].getProcId(), parr->procs[vert].getTotalLoad(),
        parr->procs[vert].overhead());
    } else if (pe_load < lower_threshold_temp) {
      CkPrintf("Below load : %d load : %E overhead : %E\n",
        parr->procs[vert].getProcId(), parr->procs[vert].getTotalLoad(),
        parr->procs[vert].overhead());
    } else {
      CkPrintf("Within avg load : %d load : %E overhead : %E\n",
        parr->procs[vert].getProcId(), parr->procs[vert].getTotalLoad(),
        parr->procs[vert].overhead());
    }
  }
}

void PrintProcObjTemp(ProcArray *parr, std::vector<int>* parr_objs) {
  int i, j;
  CkPrintf("---------------------\n");
  for (i = 0; i < parr->procs.size(); i++) {
    CkPrintf("[%d] contains ", i);
    for (j = 0; j < parr_objs[i].size(); j++) {
      CkPrintf(" %d, ", parr_objs[i][j]);
    }
    CkPrintf("\n");
  }
  CkPrintf("---------------------\n");
}

void TempAwareCommLB::populateEffectiveFreq(int numProcs)
{
#ifdef TEMP_LDB
  for(int i=0;i<numProcs;i++)
  {
    for(int j=0;j<numAvailFreqs;j++)
    {
      if(freqs[j] == procFreqNew[i]) // same freq . copy effective freq
      {
        procFreqNewEffect[i] = freqsEffect[j];
//        CkPrintf("** Proc%d j:%d NEWFreq:%d\n",i,j,procFreqNewEffect[i]);
      }
      if(freqs[j] == procFreq[i])
      {
        procFreqEffect[i] = freqsEffect[j];
//        CkPrintf("-- Proc%d j:%d OLDFreq:%d procFreq:%d \n",i,j,procFreqEffect[i],procFreq[i]);
      }
    }
  }
#endif
}


void TempAwareCommLB::initStructs(LDStats* stats)
{
#ifdef TEMP_LDB
  numProcs=stats->nprocs();
  numChips=numProcs/logicalCoresPerChip;
  avgChipTemp=new float[numChips];
  if(procFreq!=NULL) delete [] procFreq;
  if(procFreqEffect!=NULL) delete [] procFreqEffect;
  if(procTemp!=NULL) delete [] procTemp;
  if(procFreqNew!=NULL) delete [] procFreqNew;
  if(procFreqNewEffect!=NULL) delete [] procFreqNewEffect;
  if(avgChipTemp!=NULL) delete [] avgChipTemp;

  procFreq = new int[numProcs];
  procFreqEffect = new int[numProcs];
  procTemp = new float[numProcs];
  procFreqNew = new int[numProcs];
  procFreqNewEffect = new int[numProcs];
  avgChipTemp = new float[numChips];

  for(int i=0;i<numChips;i++) avgChipTemp[i]=0;

  for(int i=0;i<numProcs;i++)
  {
        procFreq[i] = stats->procs[i].pe_speed;
        procTemp[i] = stats->procs[i].pe_temp;
        avgChipTemp[i/logicalCoresPerChip] += procTemp[i];
  }
  for(int i=0;i<numChips;i++)
  {
        avgChipTemp[i]/=logicalCoresPerChip;
//CkPrintf("---- CHIP#%d has temp=%f ----------\n",i,avgChipTemp[i]);
  }
#endif
}

void TempAwareCommLB::tempControl()
{
#ifdef TEMP_LDB
	CkPrintf("----------- TempAwareCommLB::tempControl ---------\n");
  for(int i=0;i<numChips;i++)
  {
  int over=0,under=0;
        if(avgChipTemp[i] > MAX_TEMP)
        {
    over=1;
                if(procFreqPtr[i*logicalCoresPerChip]==numAvailFreqs-1)
                {
                        for(int j=i*logicalCoresPerChip;j<i*logicalCoresPerChip+logicalCoresPerChip;j++) procFreqNew[j] = freqs[procFreqPtr[j]];
                        CkPrintf("CHIP#%d RUNNING HOT EVEN WITH MIN FREQUENCY!!\n",i);
                }
                else
                {
                        for(int j=i*logicalCoresPerChip;j<i*logicalCoresPerChip+logicalCoresPerChip;j++)
                        {
                                if(procFreqPtr[j]<numAvailFreqs-1) procFreqPtr[j]++;
#ifdef MAX_MIN
/// PLEASE COMMENT OUT .. TESTING ONLY
if(i==0) {procFreqPtr[j] = numAvailFreqs-1;/*CkPrintf("C for i:%d\n",j);*/}
//if(i<numChips-1) procFreqPtr[j]=0;
else  procFreqPtr[j]=0;
/////////////////////////
#endif
                                procFreqNew[j] = freqs[procFreqPtr[j]];
                        }
#ifndef ORG_VERSION
                        CkPrintf("!!!!! Chip#%d running HOT shifting from %d to %d temp=%f\n",i,procFreq[i*logicalCoresPerChip],procFreqNew[i*logicalCoresPerChip],avgChipTemp[i]);
#endif
                }
        }
        else
//  if(avgChipTemp[i] < MAX_TEMP-1)
        {
    under=1;
                if(procFreqPtr[i*logicalCoresPerChip]>0)
                {
                        for(int j=i*logicalCoresPerChip;j<i*logicalCoresPerChip+logicalCoresPerChip;j++)
                        {
                                if(procFreqPtr[j]>0)
                                        procFreqPtr[j]--;
#ifdef MAX_MIN
/// PLEASE COMMENT OUT .. TESTING ONLY
if(i==0) procFreqPtr[j] = numAvailFreqs-1;
//if(i<numChips-1) procFreqPtr[j]=0;
else  procFreqPtr[j]=0;
/////////////////////////
#endif
                                procFreqNew[j] = freqs[procFreqPtr[j]];
                        }
#ifndef ORG_VERSION
                        CkPrintf("!!!!! Chip#%d running COLD shifting from %d to %d temp=%f\n",i,procFreq[i*logicalCoresPerChip],procFreqNew[i*logicalCoresPerChip],avgChipTemp[i]);
#endif
                }
                else
                {
                        for(int j=i*logicalCoresPerChip;j<i*logicalCoresPerChip+logicalCoresPerChip;j++) procFreqNew[j] = freqs[procFreqPtr[j]];
                }
        }
#ifdef ORG_VERSION
	for(int j=i*logicalCoresPerChip;j<i*logicalCoresPerChip+logicalCoresPerChip;j++) procFreqNew[j] = freqs[0];
#endif
  }
	for(int x=0;x<numProcs;x++)
  {
//CkPrintf("--------- Proc#%d %d numProcs=%d\n",x,procFreqNew[x],numProcs);
		if(procFreq[x]!=procFreqNew[x]) thisProxy[x].changeFreq(procFreqNew[x]);
	}
#endif
}

void TempAwareCommLB::work(LDStats* stats) {
  /** ========================== INITIALIZATION ============================= */
#ifdef TEMP_LDB
//////////////////////////////////////////////////////
// initialize structures for TempLBs
	initStructs(stats);
	tempControl();
	populateEffectiveFreq(stats->nprocs());
//////////////////////////////////////////////////////
	CkPrintf(" ================== in TempAwareCommLB::work() ===========\n");
  ProcArrayTemp *parr = new ProcArrayTemp(stats,procFreq,procFreqNew);       // Processor Array
	parr->convertToInsts(stats);
  ObjGraphTemp *ogr = new ObjGraphTemp(stats,procFreq,procFreqNew);          // Object Graph
	ogr->convertToInsts(stats);
  double avgload = parr->getAverageLoad();      // Average load of processors

  // Sets to false if it is overloaded, else to true
  vector<bool> proc_load_info(parr->procs.size(), false);

  // Create an array of vectors for each processor mapping to the objects in
  // that processor
  std::vector<int>* parr_objs = new std::vector<int>[parr->procs.size()];

  upper_threshold_temp = avgload + (avgload * THRESHOLD);
  //lower_threshold = avgload - (avgload * THRESHOLD * THRESHOLD);
  lower_threshold_temp = avgload;

  int less_loaded_counter = 0;

  srand(time(NULL));
  /** ============================= STRATEGY ================================ */

  if (_lb_args.debug()>1) 
    CkPrintf("[%d] In TempAwareCommLB strategy\n",CkMyPe());

  CkPrintf("Average load %E\n", avgload);

  int vert, i, j;
  int curr_pe;

  // Iterate over all the chares and construct the peid, vector<chareid> array
  for(vert = 0; vert < ogr->vertices.size(); vert++) {
    curr_pe = ogr->vertices[vert].getCurrentPe();
    parr_objs[curr_pe].push_back(vert);
    ogr->vertices[vert].setNewPe(curr_pe);
  }

  std::vector<int> parr_above_avg;
  std::vector<int> parr_below_avg;

  double pe_load;  

  // Insert into parr_above_avg if the processor fits under the criteria of
  // overloaded processor.
  // Insert the processor id into parr_below_avg if the processor is underloaded 
  for (vert = 0; vert < parr->procs.size(); vert++) {
    pe_load = parr->procs[vert].getTotalLoad();
    if (pe_load > upper_threshold_temp) {
      // Pushing ProcInfo into this list
      parr_above_avg.push_back(vert);
    } else if (pe_load < lower_threshold_temp) {
      parr_below_avg.push_back(parr->procs[vert].getProcId());
      proc_load_info[parr->procs[vert].getProcId()] = true;
      less_loaded_counter++;
    }
  }

  std::make_heap(parr_above_avg.begin(), parr_above_avg.end(),
      ProcLoadGreater(parr));

  int random;
  int randomly_obj_id;
  bool obj_allocated;
  int num_tries;
  // Allow as many swaps as there are chares
  int total_swaps = ogr->vertices.size() * SWAP_MULTIPLIER;
  int possible_pe;
  double obj_load;

  // Keep on loadbalancing until the number of above avg processors is 0
  while (parr_above_avg.size() != 0 && total_swaps > 0 && parr_below_avg.size() != 0) {
    // CkPrintf("Above avg : %d Below avg : %d Total swaps: %d\n", parr_above_avg.size(),
    //    parr_below_avg.size(), total_swaps);
    obj_allocated = false;
    num_tries = 0;

    // Pop the heaviest processor
    int p_index = popFromProcHeap(parr_above_avg, parr);
    ProcInfo& p = parr->procs[p_index];

    while (!obj_allocated && num_tries < parr_objs[p.getProcId()].size()) {

      // It might so happen that due to overhead load, it might not have any
      // more objects in its list
      if (parr_objs[p.getProcId()].size() == 0) {
        // CkPrintf("No obj left to be allocated\n");
        obj_allocated = true;
        break;
      }

      int randd = rand();
      random = randd % parr_objs[p.getProcId()].size();
      randomly_obj_id = parr_objs[p.getProcId()][random];
//need to update the load below .. account for freqs
      obj_load = ogr->vertices[randomly_obj_id].getVertexLoad();

      // CkPrintf("Heavy %d: Parr obj size : %d random : %d random obj id : %d\n", p_index,
      //     parr_objs[p.getProcId()].size(), randd, randomly_obj_id);
      std::vector<int> possible_pes;
      getPossiblePes(possible_pes, randomly_obj_id, ogr, parr);
      for (i = 0; i < possible_pes.size(); i++) {

        // If the heaviest communicating processor is there in the list, then
        // assign it to that.
        possible_pe = possible_pes[i];

        if ((parr->procs[possible_pe].getTotalLoad() + obj_load) < upper_threshold_temp) {
         // CkPrintf("**  Transfered %d(Load %lf) from %d:%d(Load %lf) to %d:%d(Load %lf)\n",
         //     randomly_obj_id, obj_load, CkNodeOf(p.getProcId()), p.getProcId(), p.getTotalLoad(),
         //     CkNodeOf(possible_pe), possible_pe,
         //     parr->procs[possible_pe].getTotalLoad());

          handleTransfer(randomly_obj_id, p, possible_pe, parr_objs, ogr, parr);
          obj_allocated = true;
          total_swaps--;
          updateLoadInfo(p_index, possible_pe, upper_threshold_temp, lower_threshold_temp,
              parr_above_avg, parr_below_avg, proc_load_info, parr);

          break;
        }
      }

      // Since there is no processor in the least loaded list with which this
      // chare communicates, pick a random least loaded processor.
      if (!obj_allocated) {
        //CkPrintf(":( Could not transfer to the nearest communicating ones\n");
        for (int x = 0; x < parr_below_avg.size(); x++) {
          int random_pe = parr_below_avg[x];
          if ((parr->procs[random_pe].getTotalLoad() + obj_load) < upper_threshold_temp) {
            obj_allocated = true;
            total_swaps--;
            handleTransfer(randomly_obj_id, p, random_pe, parr_objs, ogr, parr);
            updateLoadInfo(p_index, random_pe, upper_threshold_temp, lower_threshold_temp,
                           parr_above_avg, parr_below_avg, proc_load_info, parr);
            break;
          }
          num_tries++;
        }
      }
    }

    if (!obj_allocated) {
      //CkPrintf("!!!! Could not handle the heavy proc %d so giving up\n", p_index);
      // parr_above_avg.push_back(p_index);
      // std::push_heap(parr_above_avg.begin(), parr_above_avg.end(),
      //     ProcLoadGreater(parr));
    }
  }

  //CkPrintf("CommAwareRefine> After lb max load: %lf avg load: %lf\n", max_load, avg_load/parr->procs.size());

  /** ============================== CLEANUP ================================ */
  ogr->convertDecisions(stats);         // Send decisions back to LDStats
  delete parr;
  delete ogr;
  delete[] parr_objs;
#else
	CmiAbort("TempLBs are not supported without the TEMP_LDB flag\n"); 
#endif
}

inline void eraseObjFromParrObjs(std::vector<int> & parr_objs, int remove_objid) {
  for (int i = 0; i < parr_objs.size(); i++) {
    if (parr_objs[i] == remove_objid) {
      parr_objs.erase(parr_objs.begin() + i);
      return;
    }
  }
}

inline void printMapping(std::vector<Vertex> &vertices) {
  for (int i = 0; i < vertices.size(); i++) {
    CkPrintf("%d: old map : %d new map : %d\n", i, vertices[i].getCurrentPe(),
        vertices[i].getNewPe());
  }
}

inline void removeFromArray(int pe_id, std::vector<int> &array) {
  for (int i = 0; i < array.size(); i++) {
    if (array[i] == pe_id) {
      array.erase(array.begin() + i);
    }
  }
}

inline int popFromProcHeap(std::vector<int> & parr_above_avg, ProcArray *parr) {
  int p_index = parr_above_avg.front();
  std::pop_heap(parr_above_avg.begin(), parr_above_avg.end(),
      ProcLoadGreater(parr));
  parr_above_avg.pop_back();
  return p_index;
}

    
inline void handleTransfer(int randomly_obj_id, ProcInfo& p, int possible_pe, std::vector<int>* parr_objs, ObjGraph *ogr, ProcArray* parr) {
  ogr->vertices[randomly_obj_id].setNewPe(possible_pe);
  parr_objs[possible_pe].push_back(randomly_obj_id);
  ProcInfo &possible_pe_procinfo = parr->procs[possible_pe];

  p.totalLoad() -= ogr->vertices[randomly_obj_id].getVertexLoad();
  possible_pe_procinfo.totalLoad() += ogr->vertices[randomly_obj_id].getVertexLoad();
  eraseObjFromParrObjs(parr_objs[p.getProcId()], randomly_obj_id);
  //CkPrintf("After transfered %d from %d : Load %E to %d : Load %E\n", randomly_obj_id, p.getProcId(), p.getTotalLoad(),
  //    possible_pe, possible_pe_procinfo.getTotalLoad());
}

inline void updateLoadInfo(int p_index, int possible_pe, double upper_threshold_temp, double lower_threshold_temp,
                           std::vector<int>& parr_above_avg, std::vector<int>& parr_below_avg,
                           std::vector<bool> &proc_load_info, ProcArray *parr) {

  ProcInfo& p = parr->procs[p_index];
  ProcInfo& possible_pe_procinfo = parr->procs[possible_pe];

  // If the updated load is still greater than the average by the
  // threshold value, then push it back to the max heap
  if (p.getTotalLoad() > upper_threshold_temp) {
    parr_above_avg.push_back(p_index);
    std::push_heap(parr_above_avg.begin(), parr_above_avg.end(),
        ProcLoadGreater(parr));
    //CkPrintf("\t Pushing pe : %d to max heap\n", p.getProcId());
  } else if (p.getTotalLoad() < lower_threshold_temp) {
    parr_below_avg.push_back(p_index);
    proc_load_info[p_index] = true;
    //CkPrintf("\t Adding pe : %d to less loaded\n", p.getProcId());
  }

  // If the newly assigned processor's load is greater than the average
  // by the threshold value, then push it into the max heap.
  if (possible_pe_procinfo.getTotalLoad() > upper_threshold_temp) {
    // TODO: It should be the index in procarray :(
    parr_above_avg.push_back(possible_pe);
    std::push_heap(parr_above_avg.begin(), parr_above_avg.end(),
        ProcLoadGreater(parr));
    removeFromArray(possible_pe, parr_below_avg);
    proc_load_info[possible_pe] = false;
    //CkPrintf("\t Pusing pe : %d to max heap\n", possible_pe);
  } else if (possible_pe_procinfo.getTotalLoad() < lower_threshold_temp) {
  } else {
    removeFromArray(possible_pe, parr_below_avg);
    proc_load_info[possible_pe] = false;
    //CkPrintf("\t Removing from lower list pe : %d\n", possible_pe);
  }

}

inline void getPossiblePes(std::vector<int>& possible_pes, int vert,
    ObjGraph *ogr, ProcArray* parr) {
  std::map<int, int> tmp_map_pid_index;
  int counter = 0;
  int index;
  int i, j, nbrid;
  ObjPeCommInfo objpcomm;
 // CkPrintf("%d sends msgs to %d and recv msgs from %d\n", vert,
 //   ogr->vertices[vert].sendToList.size(),
 //   ogr->vertices[vert].recvFromList.size());
  
  for (i = 0; i < ogr->vertices[vert].sendToList.size(); i++) {
    nbrid = ogr->vertices[vert].sendToList[i].getNeighborId();
    j = ogr->vertices[nbrid].getNewPe(); // Fix me!! New PE
    // TODO: Should it index with vertexId?
    if (tmp_map_pid_index.count(j) == 0) {
      tmp_map_pid_index[j] = counter;
      PeCommInfo pecomminf(j);
      // TODO: Shouldn't it use vertexId instead of vert?
      objpcomm.pcomm.push_back(pecomminf);
      counter++;
    }
    index = tmp_map_pid_index[j];

    objpcomm.pcomm[index].num_msg +=
      ogr->vertices[vert].sendToList[i].getNumMsgs();
    objpcomm.pcomm[index].num_bytes +=
      ogr->vertices[vert].sendToList[i].getNumBytes();
  }

  for (i = 0; i < ogr->vertices[vert].recvFromList.size(); i++) {
    nbrid = ogr->vertices[vert].recvFromList[i].getNeighborId();
    j = ogr->vertices[nbrid].getNewPe();

    if (tmp_map_pid_index.count(j) == 0) {
      tmp_map_pid_index[j] = counter;
      PeCommInfo pecomminf(j);
      // TODO: Shouldn't it use vertexId instead of vert?
      objpcomm.pcomm.push_back(pecomminf);
      counter++;
    }
    index = tmp_map_pid_index[j];

    objpcomm.pcomm[index].num_msg +=
      ogr->vertices[vert].sendToList[i].getNumMsgs();
    objpcomm.pcomm[index].num_bytes +=
      ogr->vertices[vert].sendToList[i].getNumBytes();
  }

  // Sort the pe communication vector for this chare
  std::sort(objpcomm.pcomm.begin(), objpcomm.pcomm.end(),
      ProcCommGreater());
/*
  int pe_id;
  int node_id;
  int node_size;
  int node_first;
  //CkPrintf("%d talks to %d pes and possible pes are :\n", vert,
  //    objpcomm.pcomm.size());
  for (i = 0; i < objpcomm.pcomm.size(); i++) {
    pe_id = objpcomm.pcomm[i].pe_id;
    node_id = CkNodeOf(pe_id);
    node_size = CkNodeSize(node_id);
    node_first = CkNodeFirst(node_id);
   // CkPrintf("smp details pe_id %d, node_id %d, node_size %d, node_first %d\n",
   //   pe_id, node_id, node_size, node_first);
    for (j = 0; j < node_size; j++) {
      possible_pes.push_back(node_first + j);
      //CkPrintf("\t %d:%d (comm: %d)\n",node_id, node_first+j, objpcomm.pcomm[i].num_bytes); 
    }
  }
*/
}


#include "TempAwareCommLB.def.h"

/*@}*/


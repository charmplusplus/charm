/**
 * \addtogroup CkLdb
*/
/*@{*/

#include "elements.h"
#include "ckheap.h"
#include "RefineKLB.h"

#define _USE_APPROX_ALGO_ 1
#define _USE_RESIDUAL_MOVES_ 1
//#include "heap.h"

CreateLBFunc_Def(RefineKLB, "Move objects away from overloaded processor to reach average")

RefineKLB::RefineKLB(const CkLBOptions &opt): CBase_RefineKLB(opt)
{
  lbname = (char *)"RefineKLB";
  if (CkMyPe() == 0)
    CkPrintf("[%d] RefineKLB created\n",CkMyPe());
}

void RefineKLB::work(LDStats* stats)
{
  int obj;
  int n_pes = stats->nprocs();

  //  CkPrintf("[%d] RefineKLB strategy\n",CkMyPe());

  // RemoveNonMigratable(stats, n_pes);

  // get original object mapping
  int* from_procs = RefinerApprox::AllocProcs(n_pes, stats);
  for(obj=0;obj<stats->n_objs;obj++)  {
    int pe = stats->from_proc[obj];
    from_procs[obj] = pe;
  }

  // Get a new buffer to refine into
  int* to_procs = RefinerApprox::AllocProcs(n_pes, stats);

  RefinerApprox refiner(1.003);  // overload tolerance=1.003

  if(_lb_args.percentMovesAllowed()>0 && _USE_APPROX_ALGO_)
  {
    refiner.Refine(n_pes, stats, from_procs, to_procs, _lb_args.percentMovesAllowed());
  }
  else
  {
    for(obj=0;obj<stats->n_objs;obj++)  
    {
      to_procs[obj] = stats->from_proc[obj];
    }
  }

  // Save output
  int numMoves=0;
  for(obj=0;obj<stats->n_objs;obj++) 
  {
    int pe = stats->from_proc[obj];
    if (to_procs[obj] != pe) 
    {
      // CkPrintf("[%d] Obj %d migrating from %d to %d\n",
      //	 CkMyPe(),obj,pe,to_procs[obj]);
      stats->to_proc[obj] = to_procs[obj];
      numMoves++;
    }
  }
  int maxMoves=0.01*(stats->n_objs)*(_lb_args.percentMovesAllowed());
  int availableMoves=maxMoves-numMoves;

  //Perform Additional Moves in Greedy Fashion
  if(availableMoves>0 && _USE_RESIDUAL_MOVES_)
  {
    int *to_procs2=new int[stats->n_objs];
    performGreedyMoves(n_pes, stats, to_procs, to_procs2, availableMoves);

    int nmoves2=0;
    for(obj=0;obj<stats->n_objs;obj++)
    {
      if(to_procs2[obj]!=to_procs[obj])
      {
        stats->to_proc[obj]=to_procs2[obj];
        nmoves2++;
      }
    }
    delete[] to_procs2;
  }

  // Free the refine buffers
  RefinerApprox::FreeProcs(from_procs);
  RefinerApprox::FreeProcs(to_procs);
}

void RefineKLB::performGreedyMoves(int count, BaseLB::LDStats* stats,int *from_procs, int *to_procs, int numMoves)
{
  //Calculate load per proc and objs per proc
  int *objPerProc=new int[count];
  double *loadPerProc=new double[count];
  int i;
  for(i=0;i<count;i++)
  {
    objPerProc[i]=0;
    loadPerProc[i]=stats->procs[i].bg_walltime;
  }
  for(i=0;i<stats->n_objs;i++)
  {
    to_procs[i]=from_procs[i];
    objPerProc[from_procs[i]]++;
    loadPerProc[from_procs[i]]+=(stats->objData[i]).wallTime;
  }

  //Create a MaxHeap to select most-loaded procs
  maxHeap *procLoad=new maxHeap(count);
  for(i=0;i<count;i++)
  {
    InfoRecord *rec=new InfoRecord();
    rec->load=loadPerProc[i];
    rec->Id=i;
    procLoad->insert(rec);
  }

  //Create a MaxHeap(for every proc) for selecting heaviest computes from each proc
  maxHeap **procObjs=new maxHeap*[count];
  for(i=0;i<count;i++)
  {
    procObjs[i]=new maxHeap(objPerProc[i]);
  }
  for(i=0;i<stats->n_objs;i++)
  {
    if((stats->objData[i]).migratable == false)
      continue;
    InfoRecord *rec=new InfoRecord();
    rec->load=(stats->objData[i]).wallTime;
    rec->Id=i;
    procObjs[from_procs[i]]->insert(rec);
  }

  //Pick k'(=numMoves) computes one-by-one by picking largest computes from most -loaded procs;
  //Place in unassignedHeap;
  maxHeap *unassignedComputes=new maxHeap(numMoves);
  for(i=0;i<numMoves;i++)
  {
    InfoRecord *maxProc=procLoad->deleteMax();
    
    InfoRecord *maxObj=procObjs[maxProc->Id]->deleteMax();
    if(!maxObj)
    {
      procLoad->insert(maxProc);
      break;
    }
    unassignedComputes->insert(maxObj);

    maxProc->load-=maxObj->load;
    loadPerProc[maxProc->Id]=maxProc->load;

    procLoad->insert(maxProc);
  }

  //Assign one-by-one to least-loaded proc
  minHeap *leastLoadedP=new minHeap(count);
  for(i=0;i<count;i++)
  {
    leastLoadedP->insert(procLoad->deleteMax());
  }

  for(i=0;i<numMoves;i++)
  {
    InfoRecord *c=unassignedComputes->deleteMax();
    if(!c)
      break;

    InfoRecord *proc=leastLoadedP->deleteMin();
    proc->load+=c->load;
    leastLoadedP->insert(proc);

    to_procs[c->Id]=proc->Id;
    delete c;
  }

  //free up memory
  delete[] objPerProc;
  delete[] loadPerProc;
  for(i=0;i<count;i++)
  {
    delete procObjs[i];
  }
  delete procObjs;
  delete unassignedComputes;
  delete leastLoadedP;
}

#include "RefineKLB.def.h"

/*@}*/

#include <charm++.h>

#if CMK_LBDB_ON

#if CMK_STL_USE_DOT_H
#include <deque.h>
#include <queue.h>
#else
#include <deque>
#include <queue>
#endif

#include "RefineLB.h"
#include "RefineLB.def.h"

#if CMK_STL_USE_DOT_H
template class deque<CentralLB::MigrateInfo>;
#else
template class std::deque<CentralLB::MigrateInfo>;
#endif

void CreateRefineLB()
{
  CkPrintf("[%d] creating RefineLB %d\n",CkMyPe(),loadbalancer);
  loadbalancer = CProxy_RefineLB::ckNew();
  CkPrintf("[%d] created RefineLB %d\n",CkMyPe(),loadbalancer);
}

RefineLB::RefineLB()
{
  CkPrintf("[%d] RefineLB created\n",CkMyPe());
}

CmiBool RefineLB::QueryBalanceNow(int _step)
{
  CkPrintf("[%d] Balancing on step %d\n",CkMyPe(),_step);
  return CmiTrue;
}

void RefineLB::create(CentralLB::LDStats* stats, int count)
{
  int i,j;

  P = count;

  numComputes = 0;
  for(j=0; j < P; j++) numComputes+= stats[j].n_objs;
  computes = new computeInfo[numComputes];

  processors = new processorInfo[count];

  int index = 0;
  for(j=0; j < count; j++) {
    processors[j].Id = i;
    processors[j].backgroundLoad = 0;
    processors[j].load = processors[i].backgroundLoad;
    processors[j].computeLoad = 0;
    processors[j].computeSet = new Set();

    LDObjData *odata = stats[j].objData;
    const int osz = stats[j].n_objs;  
    for(i=0; i < osz; i++) {
//      computes[index].omID = odata[i].omID;
//      computes[index].id = odata[i].id;
      computes[index].id = odata[i].id;
      computes[index].handle = odata[i].handle;
      computes[index].load = odata[i].cpuTime;
      computes[index].processor = -1;
      computes[index].oldProcessor = j;
      index ++;
    }
  }

//  for (i=0; i < numComputes; i++)
//      processors[computes[i].oldProcessor].computeLoad += computes[i].load;
}

void RefineLB::assign(computeInfo *c, int processor)
{
   assign(c, &(processors[processor]));
}

void RefineLB::assign(computeInfo *c, processorInfo *p)
{
   c->processor = p->Id;
   p->computeSet->insert((InfoRecord *) c);
   p->computeLoad += c->load;
   p->load = p->computeLoad + p->backgroundLoad;
}

void  RefineLB::deAssign(computeInfo *c, processorInfo *p)
{
   c->processor = -1;
   p->computeSet->remove(c);
   p->computeLoad -= c->load;
   p->load = p->computeLoad + p->backgroundLoad;
}

void RefineLB::computeAverage()
{
   int i;
   double total = 0;
   for (i=0; i<numComputes; i++)
      total += computes[i].load;

   for (i=0; i<P; i++)
      total += processors[i].backgroundLoad;

   averageLoad = total/P;
}

double RefineLB::computeMax()
{
   int i;
   double max = processors[0].load;
   for (i=1; i<P; i++)
   {
      if (processors[i].load > max)
         max = processors[i].load;
   }
   return max;
}

int RefineLB::refine()
{
   int finish = 1;
   maxHeap *heavyProcessors = new maxHeap(P);

   Set *lightProcessors = new Set();
   int i;
   for (i=0; i<P; i++)
   {
      if (processors[i].load > overLoad*averageLoad)
      {
CkPrintf("Processor %d is HEAVY: load:%f averageLoad:%f!\n", i, processors[i].load, averageLoad);
         heavyProcessors->insert((InfoRecord *) &(processors[i]));
      }
      else if (processors[i].load < averageLoad)
      {
CkPrintf("Processor %d is LIGHT: load:%f averageLoad:%f!\n", i, processors[i].load, averageLoad);
	      lightProcessors->insert((InfoRecord *) &(processors[i]));
      }
   }
   int done = 0;

   while (!done)
   {
      double bestSize;
      computeInfo *bestCompute;
      processorInfo *bestP;
    
      processorInfo *donor = (processorInfo *) heavyProcessors->deleteMax();
      if (!donor) break;

      //find the best pair (c,receiver)
      Iterator nextProcessor;
      processorInfo *p = (processorInfo *) 
             lightProcessors->iterator((Iterator *) &nextProcessor);
      bestSize = 0;
      bestP = 0;
      bestCompute = 0;

      while (p)
      {
         Iterator nextCompute;
         nextCompute.id = 0;
         computeInfo *c = (computeInfo *) 
            donor->computeSet->iterator((Iterator *)&nextCompute);
         // iout << iINFO << "Considering Procsessor : " << p->Id << "\n" << endi;
         while (c)
         {
//CkPrintf("c->load: %f p->load:%f overLoad*averageLoad:%f \n", c->load, p->load, overLoad*averageLoad);
            if ( c->load + p->load < overLoad*averageLoad) 
            {
               // iout << iINFO << "Considering Compute : " << c->Id << " with load " 
               //      << c->load << "\n" << endi;
               if(c->load > bestSize) 
               {
                  bestSize = c->load;
                  bestCompute = c;
                  bestP = p;
               }
            }
            nextCompute.id++;
            c = (computeInfo *) donor->computeSet->next((Iterator *)&nextCompute);
         }
         p = (processorInfo *) 
         lightProcessors->next((Iterator *) &nextProcessor);
      }

      if (bestCompute)
      {
CkPrintf("Assign: [%d] with load: %f from %d to %d \n", bestCompute->id.id[0], bestCompute->load, donor->Id, bestP->Id);
        deAssign(bestCompute, donor);      
        assign(bestCompute, bestP);
      }
      else {
	finish = 0;
	break;
      }

      if (bestP->load > averageLoad)
         lightProcessors->remove(bestP);
    
      if (donor->load > overLoad*averageLoad)
         heavyProcessors->insert((InfoRecord *) donor);
      else if (donor->load < averageLoad)
         lightProcessors->insert((InfoRecord *) donor);
   }  
   return finish;
}

CLBMigrateMsg* RefineLB::Strategy(CentralLB::LDStats* stats, int count)
{
  CkPrintf("[%d] RefineLB strategy\n",CkMyPe());

  create(stats, count);

  int i;
  for (i=0; i<numComputes; i++)
    assign((computeInfo *) &(computes[i]),
           (processorInfo *) &(processors[computes[i].oldProcessor]));

  computeAverage();
  overLoad = 1.02;

  refine();

#if CMK_STL_USE_DOT_H
  queue<MigrateInfo> migrateInfo;
#else
  std::queue<MigrateInfo> migrateInfo;
#endif

  for (int pe=0; pe < P; pe++) {
    Iterator nextCompute;
    nextCompute.id = 0;
    computeInfo *c = (computeInfo *)
         processors[pe].computeSet->iterator((Iterator *)&nextCompute);
    while(c)
    {
      if (c->oldProcessor != c->processor)
      {
CkPrintf("Migrate: from %d to %d\n", c->oldProcessor, c->processor);
	MigrateInfo migrateMe;
	migrateMe.obj = c->handle;
	migrateMe.from_pe = c->oldProcessor;
	migrateMe.to_pe = c->processor;
	migrateInfo.push(migrateMe);
      }

        nextCompute.id++;
        c = (computeInfo *) processors[pe].computeSet->next((Iterator *)&nextCompute);
    }
  }

/*
  for(int pe=0; pe < count; pe++) {
    CkPrintf("[%d] PE %d : %d Objects : %d Communication\n",
	     CkMyPe(),pe,stats[pe].n_objs,stats[pe].n_comm);
    for(int obj=0; obj < stats[pe].n_objs; obj++) {
      const int dest = static_cast<int>(drand48()*(CmiNumPes()-1) + 0.5);
      if (dest != pe) {
	CkPrintf("[%d] Obj %d migrating from %d to %d\n",
		 CkMyPe(),obj,pe,dest);
	MigrateInfo migrateMe;
	migrateMe.obj = stats[pe].objData[obj].handle;
	migrateMe.from_pe = pe;
	migrateMe.to_pe = dest;
	migrateInfo.push(migrateMe);
      }
    }
  }
*/

  int migrate_count=migrateInfo.size();
  CLBMigrateMsg* msg = new(&migrate_count,1) CLBMigrateMsg;
  msg->n_moves = migrate_count;
  for(i=0; i < migrate_count; i++) {
    msg->moves[i] = migrateInfo.front();
    migrateInfo.pop();
  }

  return msg;
};

#endif

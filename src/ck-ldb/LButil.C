
#include "elements.h"
#include "ckheap.h"

#include "BaseLB.h"

LBVectorMigrateMsg * VectorStrategy(BaseLB::LDStats *stats)
{
   int i;
   int n_pes = stats->nprocs();

   processorInfo *processors = new processorInfo[n_pes];

   for(i=0; i < n_pes; i++) {
    processors[i].Id = i;
    processors[i].backgroundLoad = stats->procs[i].bg_walltime;
    processors[i].computeLoad = stats->procs[i].total_walltime;
    processors[i].load = processors[i].computeLoad + processors[i].backgroundLoad;
    processors[i].pe_speed = stats->procs[i].pe_speed;
    processors[i].available = stats->procs[i].available;
  }

  // compute average
  double total = 0.0;
  for (i=0; i<n_pes; i++)
    total += processors[i].load;
                                                                                
  double averageLoad = total/n_pes;

  if (_lb_args.debug()>1) CkPrintf("Average load: %f (total: %f, n_pes: %d)\n", averageLoad, total, n_pes);

  maxHeap *heavyProcessors = new maxHeap(n_pes);
  Set *lightProcessors = new Set();

  double overload_factor = 1.01;
  for (i=0; i<n_pes; i++) {
    if (processors[i].load > averageLoad*overload_factor) {
      //      CkPrintf("Processor %d is HEAVY: load:%f averageLoad:%f!\n",
      //               i, processors[i].load, averageLoad);
      heavyProcessors->insert((InfoRecord *) &(processors[i]));
    } else if (processors[i].load < averageLoad) {
      //      CkPrintf("Processor %d is LIGHT: load:%f averageLoad:%f!\n",
      //               i, processors[i].load, averageLoad);
      lightProcessors->insert((InfoRecord *) &(processors[i]));
    }
  }

  if (_lb_args.debug()>1) {
    CkPrintf("Before migration: (%d) ", n_pes);
    for (i=0; i<n_pes; i++) CkPrintf("%f (%f %f) ", processors[i].load, processors[i].computeLoad, processors[i].backgroundLoad);
    CkPrintf("\n");
  }

  int done = 0;
  CkVec<VectorMigrateInfo *> miginfo;
  while (!done) {
    processorInfo *donor = (processorInfo *) heavyProcessors->deleteMax();
    if (!donor) break;
    if (donor->computeLoad == 0.0) continue;    // nothing to move
    Iterator nextProcessor;
    processorInfo *p = (processorInfo *)
      lightProcessors->iterator((Iterator *) &nextProcessor);
    double load = donor->load - averageLoad;
    while (load > 0.0 && p) {
      double needed = averageLoad - p->load;
      double give;
      if (load > needed) give = needed;
      else give = load;
      if (give > donor->computeLoad) give = donor->computeLoad;
      donor->load -= give;
      donor->computeLoad -= give;
      p->load += give;
      p->computeLoad += give;
      VectorMigrateInfo *move = new VectorMigrateInfo;
      move->from_pe = donor->Id;
      move->to_pe = p->Id;
      move->load = give;
      miginfo.push_back(move);
      if (give < needed) 
        break;
      else
        lightProcessors->remove(p);
      p = (processorInfo *)lightProcessors->next((Iterator *) &nextProcessor);
      load -= give;
    }
  }

  int migrate_count = miginfo.length();
  LBVectorMigrateMsg* msg = new(migrate_count,0) LBVectorMigrateMsg;
  msg->n_moves = migrate_count;
  for(i=0; i < migrate_count; i++) {
    VectorMigrateInfo* item = (VectorMigrateInfo*) miginfo[i];
    msg->moves[i] = *item;
    if (_lb_args.debug()>1)
      CkPrintf("Processor %d => %d load: %f.\n", item->from_pe, item->to_pe, item->load);
    delete item;
    miginfo[i] = 0;
  }

  if (_lb_args.debug()>1) {
    CkPrintf("After migration: (%d) ", n_pes);
    for (i=0; i<n_pes; i++) CkPrintf("%f (%f %f) ", processors[i].load, processors[i].computeLoad, processors[i].backgroundLoad);
    CkPrintf("\n");
  }

  if (_lb_args.debug())
    CkPrintf("VectorStrategy: %d processor vector migrating.\n", migrate_count);

  delete heavyProcessors;
  delete lightProcessors;

  delete [] processors;

  return msg;
}



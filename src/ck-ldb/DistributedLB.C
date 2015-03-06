/**
 * Author: gplkrsh2@illinois.edu (Harshitha Menon)
 * A distributed load balancer.
*/

#include "DistributedLB.h"

#include "elements.h"

CreateLBFunc_Def(DistributedLB, "The distributed load balancer")

using std::vector;

DistributedLB::DistributedLB(CkMigrateMessage *m) : CBase_DistributedLB(m) {
}

DistributedLB::DistributedLB(const CkLBOptions &opt) : CBase_DistributedLB(opt) {
  lbname = "DistributedLB";
  if (CkMyPe() == 0)
    CkPrintf("[%d] DistributedLB created\n",CkMyPe());
  InitLB(opt);
}


void DistributedLB::InitLB(const CkLBOptions &opt) {
  thisProxy = CProxy_DistributedLB(thisgroup);
}

void DistributedLB::Strategy(const DistBaseLB::LDStats* const stats) {
	if (CkMyPe() == 0)
		CkPrintf("[%d] In DistributedLB strategy\n", CkMyPe());

  // Initialize constants
  kUseAck = true;
  kTransferThreshold = 1.05;
  // Indicates use all the information present
  kPartialInfoCount = -1;
  // Maximum number of times we will try to find a PE to transfer an object
  // successfully
  kMaxTrials = CkNumPes();
  // Max gossip messages sent from each PE
  kMaxGossipMsgCount = 2 * CmiLog2(CkNumPes());
 
  my_stats = stats;

	my_load = 0.0;
	for (int i = 0; i < my_stats->n_objs; i++) {
		my_load += my_stats->objData[i].wallTime; 
  }
  b_load = my_stats->total_walltime - (my_stats->idletime + my_load);
  //my_load += b_load;

  // Reset member variables
	pe_no.clear();
	loads.clear();
	distribution.clear();
  lb_started = false;
  gossip_msg_count = 0;
  negack_count = 0;

  srand((unsigned)(CmiWallTimer()*1.0e06) + CkMyPe());
  // Use reduction to obtain the average load in the system
  CkCallback cb(CkReductionTarget(DistributedLB, AvgLoadReduction), thisProxy);
  contribute(sizeof(double), &my_load, CkReduction::sum_double, cb);
}

/*
* Once the reduction callback is obtained for average load in the system, the
* gossiping starts. Only the underloaded processors gossip.
* Termination of gossip is via QD and callback is DoneGossip.
*/
void DistributedLB::AvgLoadReduction(double x) {
  avg_load = x/CkNumPes();
  // Calculate the average load by considering the threshold for imbalance
  thr_avg = ceil(kTransferThreshold * avg_load);

  // If my load is less than the avg_load, I am underloaded. Initiate the gossip
  // by sending my load information to random neighbors.
  if (my_load < avg_load) {
		double r_loads[1];
		int r_pe_no[1];
    r_loads[0] = my_load;
    r_pe_no[0] = CkMyPe();
    // Initialize the req_hop of the message to 0
		req_hop = 0;
    GossipLoadInfo(req_hop, CkMyPe(), 1, r_pe_no, r_loads);
  }

  // Start quiescence detection at PE 0.
  if (CkMyPe() == 0) {
    CkCallback cb(CkIndex_DistributedLB::DoneGossip(), thisProxy);
    CkStartQD(cb);
  }
}

/*
* Gossip load information between peers. Receive the gossip message.
*/
void DistributedLB::GossipLoadInfo(int req_h, int from_pe, int n,
    int remote_pe_no[], double remote_loads[]) {
  // Placeholder temp vectors for the sorted pe and their load 
  vector<int> p_no;
  vector<double> l;

  int i = 0;
  int j = 0;
  int m = pe_no.size();

  // Merge (using merge sort) information received with the information at hand
  // Since the initial list is sorted, the merging is linear in the size of the
  // list. 
  while (i < m && j < n) {
    if (pe_no[i] < remote_pe_no[j]) {
      p_no.push_back(pe_no[i]);
      l.push_back(loads[i]);
      i++;
    } else {
      p_no.push_back(remote_pe_no[j]);
      l.push_back(remote_loads[j]);
      if (pe_no[i] == remote_pe_no[j]) {
        i++;
      }
      j++;
    }
  }

  if (i == m && j != n) {
    while (j < n) {
      p_no.push_back(remote_pe_no[j]);
      l.push_back(remote_loads[j]);
      j++;
    }
  } else if (j == n && i != m) {
    while (i < m) {
      p_no.push_back(pe_no[i]);
      l.push_back(loads[i]);
      i++;
    }
  }

  // After the merge sort, swap. Now pe_no and loads have updated information
  pe_no.swap(p_no);
  loads.swap(l);
	req_hop = req_h + 1;

  SendLoadInfo();
}

/*
* Construct the gossip message and send to peers
*/
void DistributedLB::SendLoadInfo() {
  // TODO: Keep it 0.8*log
  // This PE has already sent the maximum set threshold for gossip messages.
  // Hence don't send out any more messages. This is to prevent flooding.
  if (gossip_msg_count > kMaxGossipMsgCount) {
    return;
  }

  // Pick two random neighbors to send the message to
  int rand_nbor1;
  int rand_nbor2 = -1;
  do {
    rand_nbor1 = rand() % CkNumPes();
  } while (rand_nbor1 == CkMyPe());
  // Pick the second neighbor which is not the same as the first one.
  do {
    rand_nbor2 = rand() % CkNumPes();
  } while ((rand_nbor2 == CkMyPe()) || (rand_nbor2 == rand_nbor1));

  // kPartialInfoCount indicates how much information is send in gossip. If it
  // is set to -1, it means use all the information available.
  int info_count = (kPartialInfoCount >= 0) ? kPartialInfoCount : pe_no.size();
  int* p = new int[info_count];
  double* l = new double[info_count];
  for (int i = 0; i < info_count; i++) {
    p[i] = pe_no[i];
    l[i] = loads[i];
  }

  thisProxy[rand_nbor1].GossipLoadInfo(req_hop, CkMyPe(), info_count, p, l);
  thisProxy[rand_nbor2].GossipLoadInfo(req_hop, CkMyPe(), info_count, p, l);

  // Increment the outgoind msg count
  gossip_msg_count++;

  delete[] p;
  delete[] l;
}

/*
* Callback invoked when gossip is done and QD is detected
*/
void DistributedLB::DoneGossip() {
  // The gossip is done, now perform load balance based on the information
  // received in the information propagation stage (gossip)
  LoadBalance();
}

/*
* Perform load balancing based on the partial information obtained from the
* information propagation stage (gossip).
*/
void DistributedLB::LoadBalance() {
  if (lb_started) {
    return;
  }
  lb_started = true;
  underloaded_pe_count = pe_no.size();

  CkVec<int> obj_no;
  CkVec<int> obj_pe_no;

  // Balance load and add objs to be transferred to obj_no and pe to be
  // transferred to in obj_pe_no
  LoadBalance(obj_no, obj_pe_no);
  total_migrates = obj_no.length();
  total_migrates_ack = total_migrates;

  // If there is no migration, then this is done
  if (total_migrates == 0) {	
    // Total migrates will be 0
    msg = new(total_migrates,CkNumPes(),CkNumPes(),0) LBMigrateMsg;
    msg->n_moves = total_migrates;
    // Wait for all acks before starting to transfer
    CkCallback cb(CkIndex_DistributedLB::SendAfterBarrier(NULL), thisProxy);
    contribute(0, NULL, CkReduction::nop, cb);
	}
}

/*
* Callback for the barrier after the load balance decisions are made and acks
* received.
* Once all that has been done, now process the migration decision.
*/
void DistributedLB::SendAfterBarrier(CkReductionMsg *red_msg) {
  ProcessMigrationDecision(msg);
}

/*
* Map the objects present in this PE to other PEs for load balance and store the
* result in obj_no and obj_pe_no. obj_no contains the index of the object to be
* transferred and corresponding entry in obj_pe_no indicates the PE to which it
* is transferred.
*/
void DistributedLB::LoadBalance(CkVec<int> &obj_no, CkVec<int> &obj_pe_no) {
  objs_count = 0;
  // Count the number of objs that are migratable and whose load is not 0.
  for(int i=0; i < my_stats->n_objs; i++) {
    if (my_stats->objData[i].migratable &&
        my_stats->objData[i].wallTime > 0.0001) {
      objs_count++;
    }
  }
  // If I am underloaded or if I did not receive information about any
  // underloaded pes in the system, then return
  if (underloaded_pe_count <= 0 || my_load < thr_avg) {
    return;
  }
 
  // Create a min heap of objs. The idea is to transfer smaller objs. The reason
  // is that since we are making probabilistic transfer of load, sending small
  // objs will result in better load balance.
  minHeap objs(objs_count);
  for(int i=0; i < my_stats->n_objs; i++) {
    if (my_stats->objData[i].migratable &&
        my_stats->objData[i].wallTime > 0.0001) {
      InfoRecord* item = new InfoRecord;
      item->load = my_stats->objData[i].wallTime;
      item->Id = i;
      objs.insert(item);
    }
  }

  // Calculate the probabilities and cdf for PEs based on their load
  // distribution
	CalculateCumulateDistribution();

  // Map the objects in the objs mapping where obj_no contains the object number
  // and the corresponding entry in obj_pe_no is the new mapping.
  MapObjsToPe(objs, obj_no, obj_pe_no);

  // Delete the object records from the heap
  InfoRecord* obj;
  while (NULL!=(obj=objs.deleteMin())) {
    delete obj;
  }
}

/*
* Map objects to PE for load balance. It takes in a min heap of objects which
* can be transferred and finds suitable receiver PEs. The mapping is stored in
* obj_no and the corresponding entry in obj_pe_no indicates the receiver PE.
*/
void DistributedLB::MapObjsToPe(minHeap &objs, CkVec<int> &obj_no,
    CkVec<int> &obj_pe_no) {
  int p_id;
  double p_load;
  int rand_pe;

  // While my load is more than the threshold, try to transfer objs
  while (my_load > (thr_avg)) {
    // If there is only one object, then nothing can be done to balance it.
    if (objs_count < 2) break;

    // Flag to indicate whether successful in finding a transfer
    bool success = false;

    // Get the smallest object
    InfoRecord* obj = objs.deleteMin();
    // No more objects to retrieve
    if (obj == 0) break;

    // If transferring this object makes this PE underloaded, then don't
    // transfer
    if ((my_load - obj->load) < (thr_avg)) {
      break;
    }

    // Pick random PE based on the probability and the find is successful only
    // if on transferring the object, that PE does not become overloaded
    do {
      rand_pe = PickRandReceiverPeIdx();
      if (rand_pe == -1) break;
      p_id = pe_no[rand_pe];
      p_load = loads[rand_pe];
      if ((p_load + obj->load) < avg_load) {
        success = true;
      }
      kMaxTrials--;
    } while (!success && (kMaxTrials > 0));

    // No successful in finding a suitable PE to transfer the object
    if (!success) {
      break;
    }

    // Found an object and a suitable PE to transfer it to. Decrement the obj
    // count and update the loads.
    obj_no.insertAtEnd(obj->Id);
    obj_pe_no.insertAtEnd(p_id);
    objs_count--;
    loads[rand_pe] += obj->load;
    my_load -= obj->load;

    // Send information to the receiver PE about this obj. This is necessary for
    // ack as well as finding out how many objs are migrating in
		thisProxy[p_id].InformMigration(obj->Id, CkMyPe(),
        my_stats->objData[obj->Id].wallTime, false);

    // This object is assigned, so we delete it from the heap
    delete obj;
  }
}

/*
* Receive information about inbound object including the id, from_pe and its
* load. 
*
* obj_id is the index of the object in the original PE.
* from_pe is the originating PE
* obj_load is the load of this object
* force flag indicates that this PE is forced to accept the object after
* multiple trials and ack should not be sent.
*/
void DistributedLB::InformMigration(int obj_id, int from_pe, double obj_load,
    bool force) {
  // If not using ack based scheme or adding this obj does not make this PE
  // overloaded, then accept the migrated obj and return. 
  if (!kUseAck || (my_load + obj_load) <= (thr_avg)) {
    migrates_expected++;
    // add to my load and reply true
    my_load += obj_load;
    thisProxy[from_pe].RecvAck(obj_id, CkMyPe(), true);
    return;
  }

  // We are using ack based scheme and turns out accepting this obj will make me
  // overloaded but if it is a forced one, then accept it else return negative
  // acknowledgement.
  if (force) {
    migrates_expected++;
    // add to my load and reply with positive ack
    my_load += obj_load;
  } else {
    // If my_load + obj_load is > threshold, then reply with negative ack 
    thisProxy[from_pe].RecvAck(obj_id, CkMyPe(), false);
  }
}

/*
* Receive an ack message which the message whether the assigned object can be
* assigned or not. If all the acks have been received, then create migration
* message.
*/
void DistributedLB::RecvAck(int obj_id, int assigned_pe, bool can_accept) {
  total_migrates_ack--;

  // If it is a positive ack, then create a migrate msg for that object
  if (can_accept) {
    MigrateInfo* migrateMe = new MigrateInfo;
    migrateMe->obj = my_stats->objData[obj_id].handle;
    migrateMe->from_pe = CkMyPe();
    migrateMe->to_pe = assigned_pe;
    migrateInfo.push_back(migrateMe);
  } else if (negack_count > 0.1*underloaded_pe_count) {
    // If received negative acks more than the specified threshold, then drop it
    negack_count++;
    total_migrates--;
    objs_count++;
  } else {
    // Try to transfer again. Add the object back to a heap, update my load and
    // try to find a suitable PE now.
    total_migrates--;
    negack_count++;
    objs_count++;
    my_load += my_stats->objData[obj_id].wallTime;

    minHeap objs(1);
    InfoRecord* item = new InfoRecord;
    item->load = my_stats->objData[obj_id].wallTime;
    item->Id = obj_id;
    objs.insert(item);

    CkVec<int> obj_no;
    CkVec<int> obj_pe_no;
    MapObjsToPe(objs, obj_no, obj_pe_no);

    // If a PE could be found to transfer this object, MapObjsToPe sends a
    // message to it. Wait for the ack.
    // Maybe at this point we can try to force it or just drop it.
    if (obj_pe_no.size() > 0) {
      total_migrates_ack++;
      total_migrates++;
    }
    InfoRecord* obj;
    while (NULL!=(obj=objs.deleteMin())) {
      delete obj;
    }
  }

  // Whenever all the acks have been received, create migration msg, go into the
  // barrier and wait for everyone to finish their load balancing phase
  if (total_migrates_ack == 0) {
    msg = new(total_migrates,CkNumPes(),CkNumPes(),0) LBMigrateMsg;
    msg->n_moves = total_migrates;
    for(int i=0; i < total_migrates; i++) {
      MigrateInfo* item = (MigrateInfo*) migrateInfo[i];
      msg->moves[i] = *item;
      delete item;
      migrateInfo[i] = 0;
    }
    migrateInfo.clear();
    CkCallback cb(CkIndex_DistributedLB::SendAfterBarrier(NULL), thisProxy);
    contribute(0, NULL, CkReduction::nop, cb);
  }
}

/*
* Pick a random PE based on the probability distribution.
*/
int DistributedLB::PickRandReceiverPeIdx() const {
  // The min loaded PEs have probabilities inversely proportional to their load.
  // A cumulative distribution is calculated and a PE is randomly selected based
  // on the cdf.
  // Generate a random number and return the index of min loaded PE whose cdf is
  // greater than the random number.
	double no = (double) rand()/(double) RAND_MAX;
	for (int i = 0; i < underloaded_pe_count; i++) {
		if (distribution[i] >= no) {
			return i;
		}
	}
	return -1;
}

/*
* The PEs have probabilities inversely proportional to their load. Construct a
* CDF based on this.
*/
void DistributedLB::CalculateCumulateDistribution() {
  // The min loaded PEs have probabilities inversely proportional to their load.
	double cumulative = 0.0;
	for (int i = 0; i < underloaded_pe_count; i++) {
		cumulative += (thr_avg - loads[i])/thr_avg;
		distribution.push_back(cumulative);
	}

  for (int i = 0; i < underloaded_pe_count; i++) {
    distribution[i] = distribution[i]/cumulative;
  }
}

#include "DistributedLB.def.h"

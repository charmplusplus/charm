/**************************************************************************
** Greg Koenig (koenig@uiuc.edu)
** March 1, 2006
**
** This is GridCommRefineLB.C
**
** GridCommRefineLB is a load balancer for the Charm++ load balancing
** framework.  It is designed to work in a Grid computing environment
** consisting of two or more clusters separated by wide-area communication
** links.  Communication between objects within a cluster is assumed to be
** light weight (measured in microseconds) while communication between
** objects on different clusters is assumed to be heavy weight (measured in
** milliseconds).
**
** The load balancer examines all communications in a computation and
** attempts to spread the objects that communicate with objects on remote
** clusters evenly across the PEs in the local cluster.  No objects are
** ever migrated across cluster boundaries, they are simply distributed
** as evenly as possible across the PEs in the cluster in which they were
** originally placed.  The idea is that by spreading objects that
** communicate over the wide-area evenly, a relatively small number of
** WAN objects will be mixed with a relatively large number of LAN
** objects, allowing the message-driven characteristics of Charm++ the
** greatest possibility of overlapping the high-cost WAN communication
** with locally-driven work.
**
** The load balancer secondarily balances on scaled processor load
** (i.e., a processor that is 2x the speed of another processor in
** the local cluster will get 2x the work) as well as the number of
** LAN objects.
**
** This load balancer applies a "refinement" approach which attempts to
** avoid disrupting the object-to-PE mapping by causing large numbers of
** objects to migrate with each load balancing i®nvocation.  This may be
** undesirable in some cases.  (For example, if the vmi-linux "eager
** protocol" is used, eager channels may be pinned between two PEs, and
** migrating objects that communicate heavily with each other onto other
** PEs could actually slow the computationif they no longer communicate
** with each other over an eager channel.)  To prevent this, the balancer
** determines the average number of objects per PE that communicate with
** objects on remote clusters, and then migrates objects away from PEs
** that exceed this average plus some tolerance (e.g., 110% of average).
** This means that only the objects on the most overloaded PEs will be
** migrated.
*/

#include "GridCommRefineLB.decl.h"

#include "GridCommRefineLB.h"
#include "manager.h"

CreateLBFunc_Def (GridCommRefineLB, "Grid communication load balancer (refines object mapping within each cluster)")



/**************************************************************************
**
*/
GridCommRefineLB::GridCommRefineLB (const CkLBOptions &opt) : CBase_GridCommRefineLB (opt)
{
  char *value;


  lbname = (char *) "GridCommRefineLB";

  if (CkMyPe() == 0) {
    CkPrintf ("[%d] GridCommRefineLB created.\n", CkMyPe());
  }

  if (value = getenv ("CK_LDB_GRIDCOMMREFINELB_TOLERANCE")) {
    CK_LDB_GridCommRefineLB_Tolerance = atof (value);
  } else {
    CK_LDB_GridCommRefineLB_Tolerance = CK_LDB_GRIDCOMMREFINELB_TOLERANCE;
  }

  manager_init ();
}



/**************************************************************************
**
*/
GridCommRefineLB::GridCommRefineLB (CkMigrateMessage *msg) : CBase_GridCommRefineLB (msg)
{
  char *value;


  lbname = (char *) "GridCommRefineLB";

  if (value = getenv ("CK_LDB_GRIDCOMMREFINELB_TOLERANCE")) {
    CK_LDB_GridCommRefineLB_Tolerance = atof (value);
  } else {
    CK_LDB_GridCommRefineLB_Tolerance = CK_LDB_GRIDCOMMREFINELB_TOLERANCE;
  }

  manager_init ();
}



/**************************************************************************
** The Charm++ load balancing framework invokes this method to determine
** whether load balancing can be performed at a specified time.
*/
bool GridCommRefineLB::QueryBalanceNow (int step)
{
  if (_lb_args.debug() > 2) {
    CkPrintf ("[%d] GridCommRefineLB is balancing on step %d.\n", CkMyPe(), step);
  }

  return (true);
}



/**************************************************************************
** The vmi-linux machine layer incorporates the idea that PEs are located
** within identifiable clusters.  This information can be supplied by the
** user or can be probed automatically by the machine layer.  The exposed
** API call CmiGetCluster() returns the integer cluster number for a
** specified PE or -1 if the information is unknown.
**
** For machine layers other than vmi-linux, simply return the constant 0.
** GridCommRefineLB will assume a single-cluster computation and will
** balance on the scaled processor load and number of LAN messages.
*/
int GridCommRefineLB::Get_Cluster (int pe)
{
  return (0);
}



/**************************************************************************
** Instantiate and initialize the PE_Data[] data structure.
**
** While doing this...
**    - ensure that there is at least one available PE
**    - ensure that all PEs are mapped to a cluster
**    - determine the maximum cluster number (gives the number of clusters)
**    - determine the minimum speed PE (used to compute relative PE speeds)
*/
void GridCommRefineLB::Initialize_PE_Data (CentralLB::LDStats *stats)
{
  int min_speed;
  int i;


  PE_Data = new PE_Data_T[Num_PEs];

  min_speed = MAXINT;
  for (i = 0; i < Num_PEs; i++) {
    (&PE_Data[i])->available      = stats->procs[i].available;
    (&PE_Data[i])->cluster        = Get_Cluster (i);
    (&PE_Data[i])->num_objs       = 0;
    (&PE_Data[i])->num_lan_objs   = 0;
    (&PE_Data[i])->num_lan_msgs   = 0;
    (&PE_Data[i])->num_wan_objs   = 0;
    (&PE_Data[i])->num_wan_msgs   = 0;
    (&PE_Data[i])->relative_speed = 0.0;
    (&PE_Data[i])->scaled_load    = 0.0;

    if (stats->procs[i].pe_speed < min_speed) {
      min_speed = stats->procs[i].pe_speed;
    }
  }

  // Compute the relative PE speeds.
  // Also add background CPU time to each PE's scaled load.
  for (i = 0; i < Num_PEs; i++) {
    (&PE_Data[i])->relative_speed = (double) (stats->procs[i].pe_speed / min_speed);
    (&PE_Data[i])->scaled_load += stats->procs[i].bg_walltime;
  }
}



/**************************************************************************
**
*/
int GridCommRefineLB::Available_PE_Count ()
{
  int available_pe_count;
  int i;


  available_pe_count = 0;
  for (i = 0; i < Num_PEs; i++) {
    if ((&PE_Data[i])->available) {
      available_pe_count += 1;
    }
  }
  return (available_pe_count);
}



/**************************************************************************
**
*/
int GridCommRefineLB::Compute_Number_Of_Clusters ()
{
  int max_cluster;
  int i;


  max_cluster = 0;
  for (i = 0; i < Num_PEs; i++) {
    if ((&PE_Data[i])->cluster < 0) {
      return (-1);
    }

    if ((&PE_Data[i])->cluster > max_cluster) {
      max_cluster = (&PE_Data[i])->cluster;
    }
  }
  return (max_cluster + 1);
}



/**************************************************************************
**
*/
void GridCommRefineLB::Initialize_Object_Data (CentralLB::LDStats *stats)
{
  int i;


  Object_Data = new Object_Data_T[Num_Objects];

  for (i = 0; i < Num_Objects; i++) {
    (&Object_Data[i])->migratable   = (&stats->objData[i])->migratable;
    (&Object_Data[i])->cluster      = Get_Cluster (stats->from_proc[i]);
    (&Object_Data[i])->from_pe      = stats->from_proc[i];
    (&Object_Data[i])->to_pe        = stats->from_proc[i];
    (&Object_Data[i])->num_lan_msgs = 0;
    (&Object_Data[i])->num_wan_msgs = 0;
    (&Object_Data[i])->load         = (&stats->objData[i])->wallTime;

    //(&PE_Data[(&Object_Data[i])->from_pe])->num_objs += 1;
    //(&PE_Data[(&Object_Data[i])->from_pe])->scaled_load += (&Object_Data[i])->load / (&PE_Data[(&Object_Data[i])->from_pe])->relative_speed;
  }
}



/**************************************************************************
**
*/
void GridCommRefineLB::Examine_InterObject_Messages (CentralLB::LDStats *stats)
{
  int i;
  int j;
  LDCommData *com_data;
  int send_object;
  int send_pe;
  int send_cluster;
  int recv_object;
  int recv_pe;
  int recv_cluster;
  LDObjKey *recv_objects;
  int num_objects;


  for (i = 0; i < stats->n_comm; i++) {
    com_data = &(stats->commData[i]);
    if ((!com_data->from_proc()) && (com_data->recv_type() == LD_OBJ_MSG)) {
      send_object = stats->getHash (com_data->sender);
      recv_object = stats->getHash (com_data->receiver.get_destObj());

      if ((send_object < 0) || (send_object > Num_Objects) || (recv_object < 0) || (recv_object > Num_Objects)) {
        continue;
      }

      send_pe = (&Object_Data[send_object])->from_pe;
      recv_pe = (&Object_Data[recv_object])->from_pe;

      send_cluster = Get_Cluster (send_pe);
      recv_cluster = Get_Cluster (recv_pe);

      if (send_cluster == recv_cluster) {
        (&Object_Data[send_object])->num_lan_msgs += com_data->messages;
      } else {
        (&Object_Data[send_object])->num_wan_msgs += com_data->messages;
      }
    } else if (com_data->receiver.get_type() == LD_OBJLIST_MSG) {
      send_object = stats->getHash (com_data->sender);

      if ((send_object < 0) || (send_object > Num_Objects)) {
        continue;
      }

      send_pe = (&Object_Data[send_object])->from_pe;
      send_cluster = Get_Cluster (send_pe);

      recv_objects = com_data->receiver.get_destObjs (num_objects);   // (num_objects is passed by reference)

      for (j = 0; j < num_objects; j++) {
        recv_object = stats->getHash (recv_objects[j]);

        if ((recv_object < 0) || (recv_object > Num_Objects)) {
          continue;
        }

        recv_pe = (&Object_Data[recv_object])->from_pe;
        recv_cluster = Get_Cluster (recv_pe);

        if (send_cluster == recv_cluster) {
          (&Object_Data[send_object])->num_lan_msgs += com_data->messages;
        } else {
          (&Object_Data[send_object])->num_wan_msgs += com_data->messages;
        }
      }
    }
  }
}



/**************************************************************************
**
*/
void GridCommRefineLB::Place_Objects_On_PEs ()
{
  int i;


  for (i = 0; i < Num_Objects; i++) {
    Assign_Object_To_PE (i, (&Object_Data[i])->from_pe);
  }
}



/**************************************************************************
**
*/
void GridCommRefineLB::Remap_Objects_To_PEs (int cluster)
{
  int num_cluster_pes;
  int num_wan_msgs;
  int avg_wan_msgs;
  int target_object;
  int target_pe;
  int i;


  // Compute average number of objects per PE for this cluster.
  num_cluster_pes = 0;
  num_wan_msgs = 0;
  for (i = 0; i < Num_PEs; i++) {
    if (cluster == (&PE_Data[i])->cluster) {
      num_cluster_pes += 1;
      num_wan_msgs += (&PE_Data[i])->num_wan_msgs;
    }
  }
  avg_wan_msgs = num_wan_msgs / num_cluster_pes;

  // Move objects away from PEs that exceed the average.
  for (i = 0; i < Num_PEs; i++) {
    if (cluster == (&PE_Data[i])->cluster) {
      while ((&PE_Data[i])->num_wan_msgs > (avg_wan_msgs * CK_LDB_GridCommRefineLB_Tolerance)) {
	target_object = Find_Maximum_WAN_Object (i);
	target_pe = Find_Minimum_WAN_PE (cluster);

	if ((target_object == -1) || (target_pe == -1)) {
	  break;
	}

	Remove_Object_From_PE (target_object, i);
	Assign_Object_To_PE (target_object, target_pe);
      }
    }
  }

/*
  // Compute average number of objects per PE for this cluster.
  num_cluster_pes = 0;
  num_wan_objs = 0;
  for (j = 0; j < Num_PEs; j++) {
    if (cluster == (&PE_Data[j])->cluster) {
      num_cluster_pes += 1;
      num_wan_objs += (&PE_Data[j])->num_wan_objs;
    }
  }
  avg_wan_objs = num_wan_objs / num_cluster_pes;

  // Move objects away from PEs that exceed the average.
  for (j = 0; j < Num_PEs; j++) {
    if (cluster == (&PE_Data[j])->cluster) {
      while ((&PE_Data[j])->num_wan_objs > (avg_wan_objs * CK_LDB_GridCommRefineLB_Tolerance)) {
	target_object = Find_Maximum_WAN_Object (j);
	target_pe = Find_Minimum_WAN_PE (i);

	if ((target_object == -1) || (target_pe == -1)) {
	  break;
	}

	Remove_Object_From_PE (target_object, j);
	Assign_Object_To_PE (target_object, target_pe);
      }
    }
  }
*/
}



/**************************************************************************
** This method locates the maximum WAN object in terms of number of
** messages that traverse a wide-area connection.  The search is
** constrained to objects on the specified PE.
**
** The method returns -1 if no matching object is found.
*/
int GridCommRefineLB::Find_Maximum_WAN_Object (int pe)
{
  int i;
  int max_index;
  int max_wan_msgs;


  max_index = -1;
  max_wan_msgs = -1;

  for (i = 0; i < Num_Objects; i++) {
    if ((&Object_Data[i])->from_pe == pe) {
      if ((&Object_Data[i])->migratable) {
	if ((&Object_Data[i])->num_wan_msgs > max_wan_msgs) {
	  max_index = i;
	  max_wan_msgs = (&Object_Data[i])->num_wan_msgs;
	}
      }
    }
  }

  return (max_index);
}



/**************************************************************************
** This method locates the minimum WAN PE in terms of number of objects
** that communicate with objects across a wide-area connection.  The search
** is constrained to PEs within the specified cluster.
**
** In the event of a "tie" (i.e., the number of WAN objects on a candidate
** PE is equal to the minimum number of WAN objects discovered so far) the
** tie is broken by considering the scaled CPU loads on the PEs.  The PE
** with the smaller scaled load is the better candidate.  In the event of
** a secondary tie, the secondary tie is broken by considering the number
** of LAN objects on the two PEs.
**
** The method returns -1 if no matching PE is found.
*/
int GridCommRefineLB::Find_Minimum_WAN_PE (int cluster)
{
  int i;
  int min_index;
  int min_wan_msgs;


  min_index = -1;
  min_wan_msgs = MAXINT;

  for (i = 0; i < Num_PEs; i++) {
    if (((&PE_Data[i])->available) && ((&PE_Data[i])->cluster == cluster)) {
      if ((&PE_Data[i])->num_wan_msgs < min_wan_msgs) {
        min_index = i;
        min_wan_msgs = (&PE_Data[i])->num_wan_msgs;
      } else if (((&PE_Data[i])->num_wan_msgs == min_wan_msgs) &&
                 ((&PE_Data[i])->scaled_load < (&PE_Data[min_index])->scaled_load)) {
        min_index = i;
        min_wan_msgs = (&PE_Data[i])->num_wan_msgs;
      } else if (((&PE_Data[i])->num_wan_msgs == min_wan_msgs) &&
                 ((&PE_Data[i])->scaled_load == (&PE_Data[min_index])->scaled_load) &&
                 ((&PE_Data[i])->num_objs < (&PE_Data[min_index])->num_objs)) {
        min_index = i;
        min_wan_msgs = (&PE_Data[i])->num_wan_msgs;
      }
    }
  }

  return (min_index);

/*
  int i;
  int min_index;
  int min_wan_objs;


  min_index = -1;
  min_wan_objs = MAXINT;

  for (i = 0; i < Num_PEs; i++) {
    if (((&PE_Data[i])->available) && ((&PE_Data[i])->cluster == cluster)) {
      if ((&PE_Data[i])->num_wan_objs < min_wan_objs) {
	min_index = i;
	min_wan_objs = (&PE_Data[i])->num_wan_objs;
      } else if (((&PE_Data[i])->num_wan_objs == min_wan_objs) &&
		 ((&PE_Data[i])->scaled_load < (&PE_Data[min_index])->scaled_load)) {
	min_index = i;
	min_wan_objs = (&PE_Data[i])->num_wan_objs;
      } else if (((&PE_Data[i])->num_wan_objs == min_wan_objs) &&
		 ((&PE_Data[i])->scaled_load == (&PE_Data[min_index])->scaled_load) &&
		 ((&PE_Data[i])->num_lan_objs < (&PE_Data[min_index])->num_lan_objs)) {
	min_index = i;
	min_wan_objs = (&PE_Data[i])->num_wan_objs;
      }
    }
  }

  return (min_index);
*/
}



/**************************************************************************
** This method removes target_object from target_pe.  The data structure
** entry for target_pe is updated appropriately with measurements from
** target_object.
*/
void GridCommRefineLB::Remove_Object_From_PE (int target_object, int target_pe)
{
  (&Object_Data[target_object])->to_pe = -1;

  (&PE_Data[target_pe])->num_objs -= 1;

  if ((&Object_Data[target_object])->num_lan_msgs > 0) {
    (&PE_Data[target_pe])->num_lan_objs -= 1;
    (&PE_Data[target_pe])->num_lan_msgs -= (&Object_Data[target_object])->num_lan_msgs;
  }

  if ((&Object_Data[target_object])->num_wan_msgs > 0) {
    (&PE_Data[target_pe])->num_wan_objs -= 1;
    (&PE_Data[target_pe])->num_wan_msgs -= (&Object_Data[target_object])->num_wan_msgs;
  }

  (&PE_Data[target_pe])->scaled_load -= (&Object_Data[target_object])->load / (&PE_Data[target_pe])->relative_speed;
}



/**************************************************************************
** This method assigns target_object to target_pe.  The data structure
** entry for target_pe is updated appropriately with measurements from
** target_object.
*/
void GridCommRefineLB::Assign_Object_To_PE (int target_object, int target_pe)
{
  (&Object_Data[target_object])->to_pe = target_pe;

  (&PE_Data[target_pe])->num_objs += 1;

  if ((&Object_Data[target_object])->num_lan_msgs > 0) {
    (&PE_Data[target_pe])->num_lan_objs += 1;
    (&PE_Data[target_pe])->num_lan_msgs += (&Object_Data[target_object])->num_lan_msgs;
  }

  if ((&Object_Data[target_object])->num_wan_msgs > 0) {
    (&PE_Data[target_pe])->num_wan_objs += 1;
    (&PE_Data[target_pe])->num_wan_msgs += (&Object_Data[target_object])->num_wan_msgs;
  }

  (&PE_Data[target_pe])->scaled_load += (&Object_Data[target_object])->load / (&PE_Data[target_pe])->relative_speed;
}



/**************************************************************************
** The Charm++ load balancing framework invokes this method to cause the
** load balancer to migrate objects to "better" PEs.
*/
void GridCommRefineLB::work (LDStats *stats)
{
  int i;
  // int j;
  // bool available;
  // bool all_pes_mapped;
  // int max_cluster;
  // int min_speed;
  // int send_object;
  // int send_pe;
  // int send_cluster;
  // int recv_object;
  // int recv_pe;
  // int recv_cluster;
  // LDCommData *com_data;


  if (_lb_args.debug() > 0) {
    CkPrintf ("[%d] GridCommRefineLB is working.\n", CkMyPe());
  }

  // Since this load balancer looks at communications data, it must call stats->makeCommHash().
  stats->makeCommHash ();

  // Initialize object variables for the number of PEs and number of objects.
  Num_PEs = stats->nprocs();
  Num_Objects = stats->n_objs;

  if (_lb_args.debug() > 0) {
    CkPrintf ("[%d] GridCommRefineLB is examining %d PEs and %d objects.\n", CkMyPe(), Num_PEs, Num_Objects);
  }

  // Initialize the PE_Data[] data structure.
  Initialize_PE_Data (stats);

  // If at least one available PE does not exist, return from load balancing.
  if (Available_PE_Count() < 1) {
    if (_lb_args.debug() > 0) {
      CkPrintf ("[%d] GridCommRefineLB finds no available PEs -- no balancing done.\n", CkMyPe());
    }

    delete [] PE_Data;

    return;
  }

  // Determine the number of clusters.
  // If any PE is not mapped to a cluster, return from load balancing.
  Num_Clusters = Compute_Number_Of_Clusters ();
  if (Num_Clusters < 1) {
    if (_lb_args.debug() > 0) {
      CkPrintf ("[%d] GridCommRefineLB finds incomplete PE cluster map -- no balancing done.\n", CkMyPe());
    }

    delete [] PE_Data;

    return;
  }

  if (_lb_args.debug() > 0) {
    CkPrintf ("[%d] GridCommRefineLB finds %d clusters.\n", CkMyPe(), Num_Clusters);
  }

  // Initialize the Object_Data[] data structure.
  Initialize_Object_Data (stats);

  // Examine all object-to-object messages for intra-cluster and inter-cluster communications.
  Examine_InterObject_Messages (stats);

  // Place objects on the PE they are currently assigned to.
  Place_Objects_On_PEs ();

  // Remap objects to PEs in each cluster.
  for (i = 0; i < Num_Clusters; i++) {
    Remap_Objects_To_PEs (i);
  }

  // Make the assignment of objects to PEs in the load balancer framework.
  for (i = 0; i < Num_Objects; i++) {
    stats->to_proc[i] = (&Object_Data[i])->to_pe;

    if (_lb_args.debug() > 2) {
      CkPrintf ("[%d] GridCommRefineLB migrates object %d from PE %d to PE %d.\n", CkMyPe(), i, stats->from_proc[i], stats->to_proc[i]);
    } else if (_lb_args.debug() > 1) {
      if (stats->to_proc[i] != stats->from_proc[i]) {
	CkPrintf ("[%d] GridCommRefineLB migrates object %d from PE %d to PE %d.\n", CkMyPe(), i, stats->from_proc[i], stats->to_proc[i]);
      }
    }
  }

  // Free memory.
  delete [] Object_Data;
  delete [] PE_Data;
}

#include "GridCommRefineLB.def.h"

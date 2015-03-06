/**************************************************************************
** Greg Koenig (koenig@uiuc.edu)
** June 14, 2007
**
** This is GridHybridLB.C
**
*/

#include "GridHybridLB.decl.h"

#include "GridHybridLB.h"
#include "manager.h"

CreateLBFunc_Def (GridHybridLB, "Grid load balancer that uses hybrid technique to optimize communication graph")



/**************************************************************************
**
*/
GridHybridLB::GridHybridLB (const CkLBOptions &opt) : CBase_GridHybridLB (opt)
{
  char *value;


  lbname = (char *) "GridHybridLB";

  if (CkMyPe() == 0) {
    CkPrintf ("[%d] GridHybridLB created.\n", CkMyPe());
  }

  if (value = getenv ("CK_LDB_GRIDHYBRIDLB_MODE")) {
    CK_LDB_GridHybridLB_Mode = atoi (value);
  } else {
    CK_LDB_GridHybridLB_Mode = CK_LDB_GRIDHYBRIDLB_MODE;
  }

  if (value = getenv ("CK_LDB_GRIDHYBRIDLB_BACKGROUND_LOAD")) {
    CK_LDB_GridHybridLB_Background_Load = atoi (value);
  } else {
    CK_LDB_GridHybridLB_Background_Load = CK_LDB_GRIDHYBRIDLB_BACKGROUND_LOAD;
  }

  if (value = getenv ("CK_LDB_GRIDHYBRIDLB_LOAD_TOLERANCE")) {
    CK_LDB_GridHybridLB_Load_Tolerance = atof (value);
  } else {
    CK_LDB_GridHybridLB_Load_Tolerance = CK_LDB_GRIDHYBRIDLB_LOAD_TOLERANCE;
  }

  manager_init ();
}



/**************************************************************************
**
*/
GridHybridLB::GridHybridLB (CkMigrateMessage *msg) : CBase_GridHybridLB (msg)
{
  char *value;


  lbname = (char *) "GridHybridLB";

  if (value = getenv ("CK_LDB_GRIDHYBRIDLB_MODE")) {
    CK_LDB_GridHybridLB_Mode = atoi (value);
  } else {
    CK_LDB_GridHybridLB_Mode = CK_LDB_GRIDHYBRIDLB_MODE;
  }

  if (value = getenv ("CK_LDB_GRIDHYBRIDLB_BACKGROUND_LOAD")) {
    CK_LDB_GridHybridLB_Background_Load = atoi (value);
  } else {
    CK_LDB_GridHybridLB_Background_Load = CK_LDB_GRIDHYBRIDLB_BACKGROUND_LOAD;
  }

  if (value = getenv ("CK_LDB_GRIDHYBRIDLB_LOAD_TOLERANCE")) {
    CK_LDB_GridHybridLB_Load_Tolerance = atof (value);
  } else {
    CK_LDB_GridHybridLB_Load_Tolerance = CK_LDB_GRIDHYBRIDLB_LOAD_TOLERANCE;
  }

  manager_init ();
}



/**************************************************************************
** The Charm++ load balancing framework invokes this method to determine
** whether load balancing can be performed at a specified time.
*/
bool GridHybridLB::QueryBalanceNow (int step)
{
  if (_lb_args.debug() > 2) {
    CkPrintf ("[%d] GridHybridLB is balancing on step %d.\n", CkMyPe(), step);
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
** GridHybridLB will assume a single-cluster computation and will balance
** on the scaled processor load and number of LAN messages.
*/
int GridHybridLB::Get_Cluster (int pe)
{
  return (0);
}



/**************************************************************************
**
*/
void GridHybridLB::Initialize_PE_Data (CentralLB::LDStats *stats)
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
    if (CK_LDB_GridHybridLB_Background_Load) {
      (&PE_Data[i])->scaled_load += stats->procs[i].bg_walltime;
    }
  }
}



/**************************************************************************
**
*/
int GridHybridLB::Available_PE_Count ()
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
int GridHybridLB::Compute_Number_Of_Clusters ()
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
void GridHybridLB::Initialize_Object_Data (CentralLB::LDStats *stats)
{
  int i;


  Object_Data = new Object_Data_T[Num_Objects];

  for (i = 0; i < Num_Objects; i++) {
    (&Object_Data[i])->migratable   = (&stats->objData[i])->migratable;
    (&Object_Data[i])->from_pe      = stats->from_proc[i];
    (&Object_Data[i])->num_lan_msgs = 0;
    (&Object_Data[i])->num_wan_msgs = 0;
    (&Object_Data[i])->load         = (&stats->objData[i])->wallTime;

    if ((&Object_Data[i])->migratable) {
      (&Object_Data[i])->to_pe = -1;
      (&Object_Data[i])->cluster = -1;
    } else {
      (&Object_Data[i])->to_pe = (&Object_Data[i])->from_pe;
      (&Object_Data[i])->cluster = Get_Cluster ((&Object_Data[i])->from_pe);
      if (_lb_args.debug() > 1) {
	CkPrintf ("[%d] GridHybridLB identifies object %d as non-migratable.\n", CkMyPe(), i);
      }
    }
  }
}



/**************************************************************************
**
*/
void GridHybridLB::Initialize_Cluster_Data ()
{
  int cluster;
  double min_total_cpu_power;
  int i;


  Cluster_Data = new Cluster_Data_T[Num_Clusters];

  for (i = 0; i < Num_Clusters; i++) {
    (&Cluster_Data[i])->num_pes = 0;
    (&Cluster_Data[i])->total_cpu_power = 0.0;
    (&Cluster_Data[i])->scaled_cpu_power = 0.0;
  }

  // Compute the relative speed of each cluster.
  for (i = 0; i < Num_PEs; i++) {
    cluster = (&PE_Data[i])->cluster;

    (&Cluster_Data[cluster])->num_pes += 1;
    (&Cluster_Data[cluster])->total_cpu_power += (&PE_Data[i])->relative_speed;
  }

  min_total_cpu_power = MAXDOUBLE;
  for (i = 0; i < Num_Clusters; i++) {
    if ((&Cluster_Data[i])->total_cpu_power < min_total_cpu_power) {
      min_total_cpu_power = (&Cluster_Data[i])->total_cpu_power;
    }
  }

  for (i = 0; i < Num_Clusters; i++) {
    (&Cluster_Data[i])->scaled_cpu_power = (double) ((&Cluster_Data[i])->total_cpu_power / min_total_cpu_power);
  }
}



/**************************************************************************
** This takes objects and partitions them into clusters.
*/
void GridHybridLB::Partition_Objects_Into_Clusters (CentralLB::LDStats *stats)
{
  int num_migratable_objects;
  int *migratable_objects;
  int index;
  int num_partitions;
  int *partition_to_cluster_map;
  int cluster;
  int partition;
  int partition_count;
  int *vertex_weights;
  int vertex;
  int **communication_matrix;
  LDCommData *com_data;
  int send_object;
  int recv_object;
  int send_index;
  int recv_index;
  LDObjKey *recv_objects;
  int num_objects;
  int *xadj;
  int num_edges;
  int *adjncy;
  int *edge_weights;
  int count;
  int weight_flag;
  int numbering_flag;
  int options[5];
  int edgecut;
  int *newmap;
  int i;
  int j;


  if (Num_Clusters == 1) {
    for (i = 0; i < Num_Objects; i++) {
      (&Object_Data[i])->cluster = 0;
    }

    return;
  }

  for (i = 0; i < Num_Objects; i++) {
    (&Object_Data[i])->secondary_index = -1;
  }

  // Count the number of migratable objects, which are the only candidates to give to Metis.
  // (The non-migratable objects have been placed onto the correct destination PEs earlier.)
  // After getting the count, create a migratable_objects[] array to keep track of them.
  num_migratable_objects = 0;
  for (i = 0; i < Num_Objects; i++) {
    if ((&Object_Data[i])->migratable) {
      num_migratable_objects += 1;
    }
  }

  migratable_objects = new int[num_migratable_objects];

  index = 0;
  for (i = 0; i < Num_Objects; i++) {
    if ((&Object_Data[i])->migratable) {
      (&Object_Data[i])->secondary_index = index;
      migratable_objects[index] = i;
      index += 1;
    }
  }

  // Compute the number of partitions for Metis, based on the scaled CPU power for each cluster.
  // Also create a partition-to-cluster mapping so the output of Metis can be mapped back to clusters.
  num_partitions = 0;
  for (i = 0; i < Num_Clusters; i++) {
    num_partitions += (int) ceil ((&Cluster_Data[i])->scaled_cpu_power);
  }

  partition_to_cluster_map = new int[num_partitions];

  cluster = 0;
  partition = 0;
  while (partition < num_partitions) {
    partition_count = (int) ceil ((&Cluster_Data[cluster])->scaled_cpu_power);

    for (i = partition; i < (partition + partition_count); i++) {
      partition_to_cluster_map[i] = cluster;
    }

    partition += partition_count;
    cluster += 1;
  }

  if ((CK_LDB_GridHybridLB_Mode == 1) || (CK_LDB_GridHybridLB_Mode == 3)) {
    vertex_weights = new int[num_migratable_objects];
    vertex = 0;
    for (i = 0; i < Num_Objects; i++) {
      if ((&Object_Data[i])->migratable) {
	vertex_weights[vertex] = (int) ceil ((&Object_Data[i])->load * 10000);
	vertex += 1;
      }
    }
  }

  // Create communication_matrix[] to hold all object-to-object message counts.
  communication_matrix = new int *[num_migratable_objects];
  for (i = 0; i < num_migratable_objects; i++) {
    communication_matrix[i] = new int[num_migratable_objects];
    for (j = 0; j < num_migratable_objects; j++) {
      communication_matrix[i][j] = 0;
    }
  }

  for (i = 0; i < stats->n_comm; i++) {
    com_data = &(stats->commData[i]);
    if ((!com_data->from_proc()) && (com_data->recv_type() == LD_OBJ_MSG)) {
      send_object = stats->getHash (com_data->sender);
      recv_object = stats->getHash (com_data->receiver.get_destObj());

      //if ((recv_object == -1) && (stats->complete_flag == 0)) {
      if ((send_object < 0) || (send_object > Num_Objects) || (recv_object < 0) || (recv_object > Num_Objects)) {
	continue;
      }

      if ((!(&Object_Data[send_object])->migratable) || (!(&Object_Data[recv_object])->migratable)) {
	continue;
      }

      send_index = (&Object_Data[send_object])->secondary_index;
      recv_index = (&Object_Data[recv_object])->secondary_index;

      communication_matrix[send_index][recv_index] += com_data->messages;
      communication_matrix[recv_index][send_index] += com_data->messages;
    } else if (com_data->receiver.get_type() == LD_OBJLIST_MSG) {
      send_object = stats->getHash (com_data->sender);

      if ((send_object < 0) || (send_object > Num_Objects)) {
	continue;
      }

      if (!(&Object_Data[send_object])->migratable) {
	continue;
      }

      recv_objects = com_data->receiver.get_destObjs (num_objects);   // (num_objects is passed by reference)

      for (j = 0; j < num_objects; j++) {
	recv_object = stats->getHash (recv_objects[j]);

	//if (recv_object == -1) {
	if ((recv_object < 0) || (recv_object > Num_Objects)) {
	  continue;
	}

	if (!(&Object_Data[recv_object])->migratable) {
	  continue;
	}

	send_index = (&Object_Data[send_object])->secondary_index;
	recv_index = (&Object_Data[recv_object])->secondary_index;

	communication_matrix[send_index][recv_index] += com_data->messages;
	communication_matrix[recv_index][send_index] += com_data->messages;
      }
    }
  }

  for (i = 0; i < num_migratable_objects; i++) {
    communication_matrix[i][i] = 0;
  }

  // Construct a graph in CSR format for input to Metis.
  xadj = new int[num_migratable_objects + 1];
  num_edges = 0;
  for (i = 0; i < num_migratable_objects; i++) {
    for (j = 0; j < num_migratable_objects; j++) {
      if (communication_matrix[i][j] > 0) {
	num_edges += 1;
      }
    }
  }
  adjncy = new int[num_edges];
  edge_weights = new int[num_edges];
  count = 0;
  xadj[0] = 0;
  for (i = 0; i < num_migratable_objects; i++) {
    for (j = 0; j < num_migratable_objects; j++) {
      if (communication_matrix[i][j] > 0) {
	adjncy[count] = j;
	edge_weights[count] = communication_matrix[i][j];
	count += 1;
      }
    }
    xadj[i+1] = count;
  }

  if ((CK_LDB_GridHybridLB_Mode == 0) || (CK_LDB_GridHybridLB_Mode == 2)) {
    // Call Metis to partition the communication graph.
    weight_flag = 1;      // weights on edges only
    numbering_flag = 0;   // C style numbering (base 0)
    options[0] = 0;
    newmap = new int[num_migratable_objects];

    METIS_PartGraphRecursive (&num_migratable_objects, xadj, adjncy, NULL, edge_weights, &weight_flag, &numbering_flag, &num_partitions, options, &edgecut, newmap);
  } else if ((CK_LDB_GridHybridLB_Mode == 1) || (CK_LDB_GridHybridLB_Mode == 3)) {
    // Call Metis to partition the communication graph.
    weight_flag = 3;      // weights on both vertices and edges
    numbering_flag = 0;   // C style numbering (base 0)
    options[0] = 0;
    newmap = new int[num_migratable_objects];

    METIS_PartGraphRecursive (&num_migratable_objects, xadj, adjncy, vertex_weights, edge_weights, &weight_flag, &numbering_flag, &num_partitions, options, &edgecut, newmap);
  } else {
    if (_lb_args.debug() > 0) {
      CkPrintf ("[%d] GridHybridLB was told to use bad mode (%d).\n", CkMyPe(), CK_LDB_GridHybridLB_Mode);
    }
  }

  // Place the partitioned objects into their correct clusters.
  for (i = 0; i < num_migratable_objects; i++) {
    partition = newmap[i];
    cluster = partition_to_cluster_map[partition];

    index = migratable_objects[i];

    (&Object_Data[index])->cluster = cluster;
  }

  // Free memory.
  delete [] newmap;
  delete [] edge_weights;
  delete [] adjncy;
  delete [] xadj;
  for (i = 0; i < num_migratable_objects; i++) {
    delete [] communication_matrix[i];
  }
  delete [] communication_matrix;
  if ((CK_LDB_GridHybridLB_Mode == 1) || (CK_LDB_GridHybridLB_Mode == 3)) {
    delete [] vertex_weights;
  }
  delete [] partition_to_cluster_map;
  delete [] migratable_objects;
}



/**************************************************************************
**
*/
void GridHybridLB::Examine_InterObject_Messages (CentralLB::LDStats *stats)
{
  int i;
  int j;
  LDCommData *com_data;
  int send_object;
  int send_cluster;
  int recv_object;
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

      send_cluster = (&Object_Data[send_object])->cluster;
      recv_cluster = (&Object_Data[recv_object])->cluster;

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

      send_cluster = (&Object_Data[send_object])->cluster;

      recv_objects = com_data->receiver.get_destObjs (num_objects);   // (num_objects is passed by reference)

      for (j = 0; j < num_objects; j++) {
	recv_object = stats->getHash (recv_objects[j]);

        if ((recv_object < 0) || (recv_object > Num_Objects)) {
          continue;
        }

	recv_cluster = (&Object_Data[recv_object])->cluster;

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
void GridHybridLB::Map_NonMigratable_Objects_To_PEs ()
{
  int i;


  for (i = 0; i < Num_Objects; i++) {
    if (!((&Object_Data[i])->migratable)) {
      if (_lb_args.debug() > 1) {
	CkPrintf ("[%d] GridHybridLB identifies object %d as non-migratable.\n", CkMyPe(), i);
      }

      Assign_Object_To_PE (i, (&Object_Data[i])->from_pe);
    }
  }
}



/**************************************************************************
**
*/
void GridHybridLB::Map_Migratable_Objects_To_PEs (int cluster)
{
  int target_object;
  int target_pe;


  while (1) {
    target_object = Find_Maximum_Object (cluster);
    target_pe = Find_Minimum_PE (cluster);

    if ((target_object == -1) || (target_pe == -1)) {
      break;
    }

    Assign_Object_To_PE (target_object, target_pe);
  }
}



/**************************************************************************
** This method locates the maximum WAN object in terms of number of
** messages that traverse a wide-area connection.  The search is
** constrained to objects within the specified cluster that have not yet
** been mapped (balanced) to a PE.
**
** The method returns -1 if no matching object is found.
*/
int GridHybridLB::Find_Maximum_Object (int cluster)
{
  int max_index;
  int max_load_index;
  double max_load;
  int max_wan_msgs_index;
  int max_wan_msgs;
  double load_tolerance;
  int i;


  max_index = -1;

  max_load_index = -1;
  max_load = -1.0;

  max_wan_msgs_index = -1;
  max_wan_msgs = -1;

  for (i = 0; i < Num_Objects; i++) {
    if (((&Object_Data[i])->cluster == cluster) && ((&Object_Data[i])->to_pe == -1)) {
      if ((&Object_Data[i])->load > max_load) {
	max_load_index = i;
	max_load = (&Object_Data[i])->load;
      }
      if ((&Object_Data[i])->num_wan_msgs > max_wan_msgs) {
	max_wan_msgs_index = i;
	max_wan_msgs = (&Object_Data[i])->num_wan_msgs;
      }
    }
  }

  if (max_load_index < 0) {
    return (max_load_index);
  }

  if ((&Object_Data[max_load_index])->num_wan_msgs >= (&Object_Data[max_wan_msgs_index])->num_wan_msgs) {
    return (max_load_index);
  }

  load_tolerance = (&Object_Data[max_load_index])->load * CK_LDB_GridHybridLB_Load_Tolerance;

  max_index = max_load_index;

  for (i = 0; i < Num_Objects; i++) {
    if (((&Object_Data[i])->cluster == cluster) && ((&Object_Data[i])->to_pe == -1)) {
      if (i != max_load_index) {
	if (fabs ((&Object_Data[max_load_index])->load - (&Object_Data[i])->load) <= load_tolerance) {
	  if ((&Object_Data[i])->num_wan_msgs > (&Object_Data[max_index])->num_wan_msgs) {
	    max_index = i;
	  }
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
int GridHybridLB::Find_Minimum_PE (int cluster)
{
  if ((CK_LDB_GridHybridLB_Mode == 0) || (CK_LDB_GridHybridLB_Mode == 1)) {
    int min_index;
    int min_objs;
    int i;


    min_index = -1;
    min_objs = MAXINT;

    for (i = 0; i < Num_PEs; i++) {
      if (((&PE_Data[i])->available) && ((&PE_Data[i])->cluster == cluster)) {
	if ((&PE_Data[i])->num_objs < min_objs) {
	  min_index = i;
	  min_objs = (&PE_Data[i])->num_objs;
	} else if (((&PE_Data[i])->num_objs == min_objs) &&
		   ((&PE_Data[i])->num_wan_objs < (&PE_Data[min_index])->num_wan_objs)) {
	  min_index = i;
	} else if (((&PE_Data[i])->num_objs == min_objs) &&
		   ((&PE_Data[i])->num_wan_objs == (&PE_Data[min_index])->num_wan_objs) &&
		   ((&PE_Data[i])->num_wan_msgs < (&PE_Data[min_index])->num_wan_msgs)) {
	  min_index = i;
	} else if (((&PE_Data[i])->num_objs == min_objs) &&
		   ((&PE_Data[i])->num_wan_objs == (&PE_Data[min_index])->num_wan_objs) &&
		   ((&PE_Data[i])->num_wan_msgs == (&PE_Data[min_index])->num_wan_msgs) &&
		   ((&PE_Data[i])->scaled_load < (&PE_Data[min_index])->scaled_load)) {
	  min_index = i;
	}
      }
    }

    return (min_index);
  } else if ((CK_LDB_GridHybridLB_Mode == 2) || (CK_LDB_GridHybridLB_Mode == 3)) {
    int min_index;
    int min_load_index;
    double min_scaled_load;
    int min_wan_msgs_index;
    int min_wan_msgs;
    double load_tolerance;
    int i;


    min_index = -1;

    min_load_index = -1;
    min_scaled_load = MAXDOUBLE;

    min_wan_msgs_index = -1;
    min_wan_msgs = MAXINT;

    for (i = 0; i < Num_PEs; i++) {
      if (((&PE_Data[i])->available) && ((&PE_Data[i])->cluster == cluster)) {
	if ((&PE_Data[i])->scaled_load < min_scaled_load) {
	  min_load_index = i;
	  min_scaled_load = (&PE_Data[i])->scaled_load;
	}
	if ((&PE_Data[i])->num_wan_msgs < min_wan_msgs) {
	  min_wan_msgs_index = i;
	  min_wan_msgs = (&PE_Data[i])->num_wan_msgs;
	}
      }
    }

    // If no PE at all was found, return a -1.
    if (min_load_index < 0) {
      return (min_load_index);
    }

    // If the number of WAN messages on the lightest loaded PE happens to match the minimum number
    // of WAN messages overall, we win because this target PE is overall the minimum PE in terms
    // of both load *and* WAN messages.
    if ((&PE_Data[min_load_index])->num_wan_msgs <= (&PE_Data[min_wan_msgs_index])->num_wan_msgs) {
      return (min_load_index);
    }

    // Otherwise, we now search for PEs that have loads +/- our tolerance.  If any PE has a load
    // within our tolerance, check its number of WAN messages.  The one of these that has the
    // fewest WAN messages is probably the best candidate for placing the next object onto.

    load_tolerance = (&PE_Data[min_load_index])->scaled_load * CK_LDB_GridHybridLB_Load_Tolerance;

    min_index = min_load_index;

    for (i = 0; i < Num_PEs; i++) {
      if (((&PE_Data[i])->available) && ((&PE_Data[i])->cluster == cluster)) {
	if (i != min_load_index) {
	  if (fabs ((&PE_Data[i])->scaled_load - (&PE_Data[min_load_index])->scaled_load) <= load_tolerance) {
	    if ((&PE_Data[i])->num_wan_msgs < (&PE_Data[min_index])->num_wan_msgs) {
	      min_index = i;
	    }
	  }
	}
      }
    }

    return (min_index);
  } else {
    if (_lb_args.debug() > 0) {
      CkPrintf ("[%d] GridHybridLB was told to use bad mode (%d).\n", CkMyPe(), CK_LDB_GridHybridLB_Mode);
    }
    return (-1);
  }
}



/**************************************************************************
** This method assigns target_object to target_pe.  The data structure
** entry for target_pe is updated appropriately with measurements from
** target_object.  This updated information is considered when placing
** successive objects onto PEs.
*/
void GridHybridLB::Assign_Object_To_PE (int target_object, int target_pe)
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
void GridHybridLB::work (LDStats *stats)
{
  int i;


  if (_lb_args.debug() > 0) {
    CkPrintf ("[%d] GridHybridLB is working (mode=%d, background load=%d, load tolerance=%f).\n", CkMyPe(), CK_LDB_GridHybridLB_Mode, CK_LDB_GridHybridLB_Background_Load, CK_LDB_GridHybridLB_Load_Tolerance);
  }

  // Since this load balancer looks at communications data, it must call stats->makeCommHash().
  stats->makeCommHash ();

  // Initialize object variables for the number of PEs and number of objects.
  Num_PEs = stats->nprocs();
  Num_Objects = stats->n_objs;

  if (_lb_args.debug() > 0) {
    CkPrintf ("[%d] GridHybridLB is examining %d PEs and %d objects.\n", CkMyPe(), Num_PEs, Num_Objects);
  }

  // Initialize the PE_Data[] data structure.
  Initialize_PE_Data (stats);

  // If at least one available PE does not exist, return from load balancing.
  if (Available_PE_Count() < 1) {
    if (_lb_args.debug() > 0) {
      CkPrintf ("[%d] GridHybridLB finds no available PEs -- no balancing done.\n", CkMyPe());
    }

    delete [] PE_Data;

    return;
  }

  // Determine the number of clusters.
  // If any PE is not mapped to a cluster, return from load balancing.
  Num_Clusters = Compute_Number_Of_Clusters ();
  if (Num_Clusters < 1) {
    if (_lb_args.debug() > 0) {
      CkPrintf ("[%d] GridHybridLB finds incomplete PE cluster map -- no balancing done.\n", CkMyPe());
    }

    delete [] PE_Data;

    return;
  }

  if (_lb_args.debug() > 0) {
    CkPrintf ("[%d] GridHybridLB finds %d clusters.\n", CkMyPe(), Num_Clusters);
  }

  // Initialize the Object_Data[] data structure.
  Initialize_Object_Data (stats);

  // Initialize the Cluster_Data[] data structure.
  Initialize_Cluster_Data ();

  // Partition objects into clusters.
  Partition_Objects_Into_Clusters (stats);

  // Examine all object-to-object messages for intra-cluster and inter-cluster communications.
  Examine_InterObject_Messages (stats);

  // Map non-migratable objects to PEs.
  Map_NonMigratable_Objects_To_PEs ();

  // Map migratable objects to PEs in each cluster.
  for (i = 0; i < Num_Clusters; i++) {
    Map_Migratable_Objects_To_PEs (i);
  }

  // Make the assignment of objects to PEs in the load balancer framework.
  for (i = 0; i < Num_Objects; i++) {
    stats->to_proc[i] = (&Object_Data[i])->to_pe;

    if (_lb_args.debug() > 2) {
      CkPrintf ("[%d] GridHybridLB migrates object %d from PE %d to PE %d.\n", CkMyPe(), i, stats->from_proc[i], stats->to_proc[i]);
    } else if (_lb_args.debug() > 1) {
      if (stats->to_proc[i] != stats->from_proc[i]) {
        CkPrintf ("[%d] GridHybridLB migrates object %d from PE %d to PE %d.\n", CkMyPe(), i, stats->from_proc[i], stats->to_proc[i]);
      }
    }
  }

  // Free memory.
  delete [] Cluster_Data;
  delete [] Object_Data;
  delete [] PE_Data;
}

#include "GridHybridLB.def.h"

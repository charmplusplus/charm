/**************************************************************************
** Greg Koenig (koenig@uiuc.edu)
** April 1, 2006
**
** This is GridMetisLB.C
**
*/

#include "GridMetisLB.decl.h"

#include "GridMetisLB.h"
#include "manager.h"

CreateLBFunc_Def (GridMetisLB, "Grid load balancer that uses Metis to optimize communication graph")



/**************************************************************************
**
*/
GridMetisLB::GridMetisLB (const CkLBOptions &opt) : CBase_GridMetisLB (opt)
{
  char *value;


  lbname = (char *) "GridMetisLB";

  if (CkMyPe() == 0) {
    CkPrintf ("[%d] GridMetisLB created.\n", CkMyPe());
  }

  if (value = getenv ("CK_LDB_GRIDMETISLB_MODE")) {
    CK_LDB_GridMetisLB_Mode = atoi (value);
  } else {
    CK_LDB_GridMetisLB_Mode = CK_LDB_GRIDMETISLB_MODE;
  }

  if (value = getenv ("CK_LDB_GRIDMETISLB_BACKGROUND_LOAD")) {
    CK_LDB_GridMetisLB_Background_Load = atoi (value);
  } else {
    CK_LDB_GridMetisLB_Background_Load = CK_LDB_GRIDMETISLB_BACKGROUND_LOAD;
  }

  manager_init ();
}



/**************************************************************************
**
*/
GridMetisLB::GridMetisLB (CkMigrateMessage *msg) : CBase_GridMetisLB (msg)
{
  char *value;


  lbname = (char *) "GridMetisLB";

  if (value = getenv ("CK_LDB_GRIDMETISLB_MODE")) {
    CK_LDB_GridMetisLB_Mode = atoi (value);
  } else {
    CK_LDB_GridMetisLB_Mode = CK_LDB_GRIDMETISLB_MODE;
  }

  if (value = getenv ("CK_LDB_GRIDMETISLB_BACKGROUND_LOAD")) {
    CK_LDB_GridMetisLB_Background_Load = atoi (value);
  } else {
    CK_LDB_GridMetisLB_Background_Load = CK_LDB_GRIDMETISLB_BACKGROUND_LOAD;
  }

  manager_init ();
}



/**************************************************************************
** The Charm++ load balancing framework invokes this method to determine
** whether load balancing can be performed at a specified time.
*/
bool GridMetisLB::QueryBalanceNow (int step)
{
  if (_lb_args.debug() > 2) {
    CkPrintf ("[%d] GridMetisLB is balancing on step %d.\n", CkMyPe(), step);
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
** GridCommLB will assume a single-cluster computation and will balance
** on the scaled processor load and number of LAN messages.
*/
int GridMetisLB::Get_Cluster (int pe)
{
  return (0);
}



/**************************************************************************
**
*/
void GridMetisLB::Initialize_PE_Data (CentralLB::LDStats *stats)
{
  int min_speed;
  int i;


  PE_Data = new PE_Data_T[Num_PEs];

  min_speed = MAXINT;
  for (i = 0; i < Num_PEs; i++) {
    (&PE_Data[i])->available      = stats->procs[i].available;
    (&PE_Data[i])->cluster        = Get_Cluster (i);
    (&PE_Data[i])->num_objs       = 0;
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
    if (CK_LDB_GridMetisLB_Background_Load) {
      (&PE_Data[i])->scaled_load += stats->procs[i].bg_walltime;
    }
  }
}



/**************************************************************************
**
*/
int GridMetisLB::Available_PE_Count ()
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
int GridMetisLB::Compute_Number_Of_Clusters ()
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
void GridMetisLB::Initialize_Object_Data (CentralLB::LDStats *stats)
{
  int i;


  Object_Data = new Object_Data_T[Num_Objects];

  for (i = 0; i < Num_Objects; i++) {
    (&Object_Data[i])->migratable = (&stats->objData[i])->migratable;
    //(&Object_Data[i])->cluster    = Get_Cluster (stats->from_proc[i]);
    (&Object_Data[i])->cluster    = -1;
    (&Object_Data[i])->from_pe    = stats->from_proc[i];
    (&Object_Data[i])->load       = (&stats->objData[i])->wallTime;

    if ((&Object_Data[i])->migratable) {
      (&Object_Data[i])->to_pe = -1;
    } else {
      (&Object_Data[i])->to_pe = (&Object_Data[i])->from_pe;
      //(&PE_Data[(&Object_Data[i])->to_pe])->scaled_load += (&Object_Data[i])->load;
      (&PE_Data[(&Object_Data[i])->to_pe])->scaled_load += (&Object_Data[i])->load / (&PE_Data[(&Object_Data[i])->to_pe])->relative_speed;
      if (_lb_args.debug() > 1) {
	CkPrintf ("[%d] GridMetisLB identifies object %d as non-migratable.\n", CkMyPe(), i);
      }
    }
  }
}



/**************************************************************************
**
*/
void GridMetisLB::Initialize_Cluster_Data ()
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
void GridMetisLB::Partition_Objects_Into_Clusters (CentralLB::LDStats *stats)
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

  if (CK_LDB_GridMetisLB_Mode == 1) {
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

  if (CK_LDB_GridMetisLB_Mode == 0) {
    // Call Metis to partition the communication graph.
    weight_flag = 1;      // weights on edges only
    numbering_flag = 0;   // C style numbering (base 0)
    options[0] = 0;
    newmap = new int[num_migratable_objects];

    METIS_PartGraphRecursive (&num_migratable_objects, xadj, adjncy, NULL, edge_weights, &weight_flag, &numbering_flag, &num_partitions, options, &edgecut, newmap);
  } else if (CK_LDB_GridMetisLB_Mode == 1) {
    // Call Metis to partition the communication graph.
    weight_flag = 3;      // weights on both vertices and edges
    numbering_flag = 0;   // C style numbering (base 0)
    options[0] = 0;
    newmap = new int[num_migratable_objects];

    METIS_PartGraphRecursive (&num_migratable_objects, xadj, adjncy, vertex_weights, edge_weights, &weight_flag, &numbering_flag, &num_partitions, options, &edgecut, newmap);
  } else {
    if (_lb_args.debug() > 0) {
      CkPrintf ("[%d] GridMetisLB was told to use bad mode (%d).\n", CkMyPe(), CK_LDB_GridMetisLB_Mode);
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
  if (CK_LDB_GridMetisLB_Mode == 1) {
    delete [] vertex_weights;
  }
  delete [] partition_to_cluster_map;
  delete [] migratable_objects;
}



/**************************************************************************
** This takes objects in a cluster and partitions them onto PEs.
*/
void GridMetisLB::Partition_ClusterObjects_Into_PEs (CentralLB::LDStats *stats, int cluster)
{
  int num_migratable_cluster_objects;
  int *migratable_cluster_objects;
  int index;
  int num_available_cluster_pes;
  int num_partitions;
  int *partition_to_pe_map;
  int pe;
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


  for (i = 0; i < Num_Objects; i++) {
    (&Object_Data[i])->secondary_index = -1;
  }

  // Count the number of migratable objects within this cluster, which are the only candidates to give to Metis.
  // (The non-migratable objects have been placed onto the correct destination PEs earlier.)
  // After getting the count, create a migratable_cluster_objects[] array to keep track of them.
  num_migratable_cluster_objects = 0;
  for (i = 0; i < Num_Objects; i++) {
    if (((&Object_Data[i])->migratable) && ((&Object_Data[i])->cluster == cluster)) {
      num_migratable_cluster_objects += 1;
    }
  }

  migratable_cluster_objects = new int[num_migratable_cluster_objects];

  index = 0;
  for (i = 0; i < Num_Objects; i++) {
    if (((&Object_Data[i])->migratable) && ((&Object_Data[i])->cluster == cluster)) {
      (&Object_Data[i])->secondary_index = index;
      migratable_cluster_objects[index] = i;
      index += 1;
    }
  }

  // Count the number of available PEs in the cluster.
  num_available_cluster_pes = 0;
  for (i = 0; i < Num_PEs; i++) {
    if (((&PE_Data[i])->available) && ((&PE_Data[i])->cluster == cluster)) {
      num_available_cluster_pes += 1;
    }
  }

  // Compute the number of partitions for Metis, based on the relative speed of each PE.
  // Also create the partition-to-PE mapping so the output of Metis can be mapped back to PEs.
  num_partitions = 0;
  for (i = 0; i < Num_PEs; i++) {
    if (((&PE_Data[i])->available) && ((&PE_Data[i])->cluster == cluster)) {
      num_partitions += (int) ceil ((&PE_Data[i])->relative_speed);
    }
  }

  partition_to_pe_map = new int[num_partitions];

  pe = 0;
  while (((!(&PE_Data[pe])->available) || ((&PE_Data[pe])->cluster != cluster)) && (pe < Num_PEs)) {
    pe += 1;
  }
  if (pe >= Num_PEs) {
    CmiAbort ("GridMetisLB: Error computing partition to PE map!\n");
  }
  partition = 0;
  while (partition < num_partitions) {
    partition_count = (int) ceil ((&PE_Data[pe])->relative_speed);

    for (i = partition; i < (partition + partition_count); i++) {
      partition_to_pe_map[i] = pe;
    }

    partition += partition_count;

    pe += 1;
    while (((!(&PE_Data[pe])->available) || ((&PE_Data[pe])->cluster != cluster)) && (pe < Num_PEs)) {
      pe += 1;
    }
    if (pe > Num_PEs) {
      CmiAbort ("GridMetisLB: Error computing partition to PE map!\n");
    }
  }

  // Compute vertex weights for the objects.
  vertex_weights = new int[num_migratable_cluster_objects];
  vertex = 0;
  for (i = 0; i < Num_Objects; i++) {
    if ((&Object_Data[i])->migratable && ((&Object_Data[i])->cluster == cluster)) {
      vertex_weights[vertex] = (int) ceil ((&Object_Data[i])->load * 10000);
      vertex += 1;
    }
  }

  // Create communication_matrix[] to hold all object-to-object message counts;
  communication_matrix = new int *[num_migratable_cluster_objects];
  for (i = 0; i < num_migratable_cluster_objects; i++) {
    communication_matrix[i] = new int[num_migratable_cluster_objects];
    for (j = 0; j < num_migratable_cluster_objects; j++) {
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

      if (((&Object_Data[send_object])->cluster != cluster) || ((&Object_Data[recv_object])->cluster != cluster)) {
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

      if ((&Object_Data[send_object])->cluster != cluster) {
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

	if ((&Object_Data[recv_object])->cluster != cluster) {
	  continue;
	}

	send_index = (&Object_Data[send_object])->secondary_index;
	recv_index = (&Object_Data[recv_object])->secondary_index;

	communication_matrix[send_index][recv_index] += com_data->messages;
	communication_matrix[recv_index][send_index] += com_data->messages;
      }
    }
  }

  for (i = 0; i < num_migratable_cluster_objects; i++) {
    communication_matrix[i][i] = 0;
  }

  // Construct a graph in CSR format for input to Metis.
  xadj = new int[num_migratable_cluster_objects + 1];
  num_edges = 0;
  for (i = 0; i < num_migratable_cluster_objects; i++) {
    for (j = 0; j < num_migratable_cluster_objects; j++) {
      if (communication_matrix[i][j] > 0) {
	num_edges += 1;
      }
    }
  }
  adjncy = new int[num_edges];
  edge_weights = new int[num_edges];
  count = 0;
  xadj[0] = 0;
  for (i = 0; i < num_migratable_cluster_objects; i++) {
    for (j = 0; j < num_migratable_cluster_objects; j++) {
      if (communication_matrix[i][j] > 0) {
	adjncy[count] = j;
	edge_weights[count] = communication_matrix[i][j];
	count += 1;
      }
    }
    xadj[i+1] = count;
  }

  // Call Metis to partition the communication graph.
  weight_flag = 3;      // weights on both vertices and edges
  numbering_flag = 0;   // C style numbering (base 0)
  options[0] = 0;
  newmap = new int[num_migratable_cluster_objects];

  CmiPrintf ("[%d] GridMetisLB is partitioning %d objects in cluster %d into %d partitions.\n", CmiMyPe(), num_migratable_cluster_objects, cluster, num_partitions);

  METIS_PartGraphRecursive (&num_migratable_cluster_objects, xadj, adjncy, vertex_weights, edge_weights, &weight_flag, &numbering_flag, &num_partitions, options, &edgecut, newmap);

  // Place the partitioned objects onto their correct PEs.
  for (i = 0; i < num_migratable_cluster_objects; i++) {
    partition = newmap[i];
    pe = partition_to_pe_map[partition];

    index = migratable_cluster_objects[i];

    /* WRONG!
    for (j = 0; j < Num_Objects; j++) {
      if ((&Object_Data[j])->secondary_index == index) {
	(&Object_Data[j])->to_pe = pe;
	break;
      }
    }
    */

    (&Object_Data[index])->to_pe = pe;
  }

  // Free memory.
  delete [] newmap;
  delete [] edge_weights;
  delete [] adjncy;
  delete [] xadj;
  for (i = 0; i < num_migratable_cluster_objects; i++) {
    delete [] communication_matrix[i];
  }
  delete [] communication_matrix;
  delete [] vertex_weights;
  delete [] partition_to_pe_map;
  delete [] migratable_cluster_objects;
}



/**************************************************************************
** The Charm++ load balancing framework invokes this method to cause the
** load balancer to migrate objects to "better" PEs.
*/
void GridMetisLB::work (LDStats *stats)
{
  int i;


  if (_lb_args.debug() > 0) {
    CkPrintf ("[%d] GridMetisLB is working (mode=%d, background load=%d).\n", CkMyPe(), CK_LDB_GridMetisLB_Mode, CK_LDB_GridMetisLB_Background_Load);
  }

  // Since this load balancer looks at communications data, it must call stats->makeCommHash().
  stats->makeCommHash ();

  // Initialize object variables for the number of PEs and number of objects.
  Num_PEs = stats->nprocs();
  Num_Objects = stats->n_objs;

  if (_lb_args.debug() > 0) {
    CkPrintf ("[%d] GridMetisLB is examining %d PEs and %d objects.\n", CkMyPe(), Num_PEs, Num_Objects);
  }

  // Initialize the PE_Data[] data structure.
  Initialize_PE_Data (stats);

  // If at least one available PE does not exist, return from load balancing.
  if (Available_PE_Count() < 1) {
    if (_lb_args.debug() > 0) {
      CkPrintf ("[%d] GridMetisLB finds no available PEs -- no balancing done.\n", CkMyPe());
    }

    delete [] PE_Data;

    return;
  }

  // Determine the number of clusters.
  // If any PE is not mapped to a cluster, return from load balancing.
  Num_Clusters = Compute_Number_Of_Clusters ();
  if (Num_Clusters < 1) {
    if (_lb_args.debug() > 0) {
      CkPrintf ("[%d] GridMetisLB finds incomplete PE cluster map -- no balancing done.\n", CkMyPe());
    }

    delete [] PE_Data;

    return;
  }

  if (_lb_args.debug() > 0) {
    CkPrintf ("[%d] GridMetisLB finds %d clusters.\n", CkMyPe(), Num_Clusters);
  }

  // Initialize the Object_Data[] data structure.
  Initialize_Object_Data (stats);

  // Initialize the Cluster_Data[] data structure.
  Initialize_Cluster_Data ();

  Partition_Objects_Into_Clusters (stats);
  for (i = 0; i < Num_Clusters; i++) {
    Partition_ClusterObjects_Into_PEs (stats, i);
  }

  // Make the assignment of objects to PEs in the load balancer framework.
  for (i = 0; i < Num_Objects; i++) {
    stats->to_proc[i] = (&Object_Data[i])->to_pe;

    if (_lb_args.debug() > 2) {
      CkPrintf ("[%d] GridMetisLB migrates object %d from PE %d to PE %d.\n", CkMyPe(), i, stats->from_proc[i], stats->to_proc[i]);
    } else if (_lb_args.debug() > 1) {
      if (stats->to_proc[i] != stats->from_proc[i]) {
        CkPrintf ("[%d] GridMetisLB migrates object %d from PE %d to PE %d.\n", CkMyPe(), i, stats->from_proc[i], stats->to_proc[i]);
      }
    }
  }

  // Free memory.
  delete [] Cluster_Data;
  delete [] Object_Data;
  delete [] PE_Data;
}

#include "GridMetisLB.def.h"

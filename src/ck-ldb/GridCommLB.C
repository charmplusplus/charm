/**************************************************************************
** Greg Koenig (koenig@uiuc.edu)
** November 4, 2004
*/
#include <stdio.h>

#include "charm++.h"
#include "cklists.h"

#include "GridCommLB.decl.h"

#include "GridCommLB.h"
#include "manager.h"


CreateLBFunc_Def (GridCommLB);


/**************************************************************************
**
*/
static void lbinit (void)
{
  LBRegisterBalancer ("GridCommLB", 
		      CreateGridCommLB, 
		      AllocateGridCommLB, 
		      "Load balancer for Grid computing environments");
}


/**************************************************************************
**
*/
GridCommLB::GridCommLB (const CkLBOptions &opt) : CentralLB (opt)
{
    lbname = (char *) "GridCommLB";

    if (CkMyPe () == 0) {
      CkPrintf ("[%d] GridCommLB created\n", CkMyPe ());
    }

    manager_init ();
}


/**************************************************************************
**
*/
GridCommLB::GridCommLB (CkMigrateMessage *m) : CentralLB (m)
{
  lbname = (char *) "GridCommLB";

  manager_init ();
}


/**************************************************************************
**
*/
CmiBool GridCommLB::QueryBalanceNow (int step)
{
  // CkPrintf ("[%d] Balancing on step %d\n", CkMyPe (), step);
  return (CmiTrue);
}


/**************************************************************************
**
*/
int GridCommLB::Get_Cluster (int pe)
{
  if (pe < (Num_PEs / 2)) {
    return (0);
  } else {
    return (1);
  }
}


/**************************************************************************
**
*/
int GridCommLB::Find_Maximum_WAN_Object (int cluster)
{
  int i;
  int max_index;
  int max_wan_msgs;


  max_index    = -1;
  max_wan_msgs = -1;

  for (i = 0; i < Num_Objects; i++) {
    if ((&Object_Data[i])->cluster == cluster) {
      if ((&Object_Data[i])->to_pe == -1) {
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
**
*/
int GridCommLB::Find_Minimum_WAN_PE (int cluster)
{
  int i;
  int min_index;
  int min_wan_objs;
  int min_lan_objs;


  min_index = -1;
  min_wan_objs = INT_MAX;
  min_lan_objs = INT_MAX;

  for (i = 0; i < Num_PEs; i++) {
    if (((&PE_Data[i])->available) && ((&PE_Data[i])->cluster == cluster)) {
      if ((&PE_Data[i])->num_wan_objs < min_wan_objs) {
	min_index = i;
	min_wan_objs = (&PE_Data[i])->num_wan_objs;
	min_lan_objs = (&PE_Data[i])->num_lan_objs;
      } else if ((&PE_Data[i])->num_wan_objs == min_wan_objs) {
	if ((&PE_Data[i])->num_lan_objs < min_lan_objs) {
	  min_index = i;
	  min_lan_objs = (&PE_Data[i])->num_lan_objs;
	}
      }
    }
  }

  return (min_index);
}


/**************************************************************************
**
*/
void GridCommLB::Assign_Object_To_PE (int target_object, int target_pe)
{
  (&Object_Data[target_object])->to_pe = target_pe;

  (&PE_Data[target_pe])->num_objs++;

  if ((&Object_Data[target_object])->num_lan_msgs > 0) {
    (&PE_Data[target_pe])->num_lan_objs++;
    (&PE_Data[target_pe])->num_lan_msgs +=
        (&Object_Data[target_object])->num_lan_msgs;
  }

  if ((&Object_Data[target_object])->num_wan_msgs > 0) {
    (&PE_Data[target_pe])->num_wan_objs++;
    (&PE_Data[target_pe])->num_wan_msgs +=
        (&Object_Data[target_object])->num_wan_msgs;
  }
}


/**************************************************************************
**
*/
void GridCommLB::work (CentralLB::LDStats *stats, int count)
{
  int i;
  int send_object;
  int send_pe;
  int send_cluster;
  int recv_object;
  int recv_pe;
  int recv_cluster;
  int target_object;
  int target_pe;
  LDCommData *com_data;


  stats->makeCommHash ();

  Num_PEs = count;
  Num_Objects = stats->n_objs;

  PE_Data = new PE_Data_T[Num_PEs];
  Object_Data = new Object_Data_T[Num_Objects];

  for (i = 0; i < Num_PEs; i++) {
    (&PE_Data[i])->available    = stats->procs[i].available;
    (&PE_Data[i])->cluster      = Get_Cluster (i);
    (&PE_Data[i])->num_objs     = 0;
    (&PE_Data[i])->num_lan_objs = 0;
    (&PE_Data[i])->num_lan_msgs = 0;
    (&PE_Data[i])->num_wan_objs = 0;
    (&PE_Data[i])->num_wan_msgs = 0;
  }

  for (i = 0; i < Num_Objects; i++) {
    (&Object_Data[i])->migratable   = (&stats->objData[i])->migratable;
    (&Object_Data[i])->cluster      = Get_Cluster (stats->from_proc[i]);
    (&Object_Data[i])->from_pe      = stats->from_proc[i];
    (&Object_Data[i])->num_lan_msgs = 0;
    (&Object_Data[i])->num_wan_msgs = 0;

    if ((&Object_Data[i])->migratable) {
      (&Object_Data[i])->to_pe = -1;
    } else {
      (&Object_Data[i])->to_pe = (&Object_Data[i])->from_pe;
    }
  }

  for (i = 0; i < stats->n_comm; i++) {
    com_data = &(stats->commData[i]);
    if ((!com_data->from_proc()) && (com_data->recv_type() == LD_OBJ_MSG)) {
      send_object = stats->getHash (com_data->sender);
      recv_object = stats->getHash (com_data->receiver.get_destObj());

      send_pe = (&Object_Data[send_object])->from_pe;
      recv_pe = (&Object_Data[recv_object])->from_pe;

      send_cluster = Get_Cluster (send_pe);
      recv_cluster = Get_Cluster (recv_pe);

      if (send_cluster == recv_cluster) {
	(&Object_Data[send_object])->num_lan_msgs += com_data->messages;
      } else {
	(&Object_Data[send_object])->num_wan_msgs += com_data->messages;
      }
    }
  }

  // Map objects to PEs in cluster 0.
  while (1) {
    target_object = Find_Maximum_WAN_Object (0);
    target_pe     = Find_Minimum_WAN_PE (0);

    if ((target_object == -1) || (target_pe == -1)) {
      break;
    }

    Assign_Object_To_PE (target_object, target_pe);
  }

  // Map objects to PEs in cluster 1.
  while (1) {
    target_object = Find_Maximum_WAN_Object (1);
    target_pe     = Find_Minimum_WAN_PE (1);

    if ((target_object == -1) || (target_pe == -1)) {
      break;
    }

    Assign_Object_To_PE (target_object, target_pe);
  }

  // Make the assignment of objects to PEs in the load balancer framework.
  for (i = 0; i < Num_Objects; i++) {
    stats->to_proc[i] = (&Object_Data[i])->to_pe;
  }
}


#include "GridCommLB.def.h"

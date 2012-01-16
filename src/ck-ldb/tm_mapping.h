#ifndef TM_MAPPING_H
#define TM_MAPPING_H

#ifdef __cplusplus
extern "C" {
#endif

#include "tm_tree.h"
int  build_comm(char *filename,double ***pcomm);
void TreeMatchMapping(int nb_obj, int nb_proc,double **comm_mat,  double * obj_weigth, double *com_speed, int d, int *sol);
double print_sol(int N,int *Value,double **comm, double **arch);

/*Map topology to cores: 
 sigma_i is such that  process i is mapped on core sigma_i
 k_i is such that core i exectutes process k_i

 size of sigma is the number of process (nb_objs)
 size of k is the number of cores/nodes (nb_proc)

 We must have numbe of process<=number of cores

 k_i =-1 if no process is mapped on core i
*/
void map_topology_simple(tm_topology_t *topology,tree_t *comm_tree, int *sigma,int *k);

int nb_nodes(tm_topology_t *topology);
void free_topology(tm_topology_t *topology);
void display_other_heuristics(tm_topology_t *topology,int N,double **comm,double **arch);
void print_1D_tab(int *tab,int N);
void   build_synthetic_proc_id(tm_topology_t *topology);
void display_topology(tm_topology_t *topology);
tm_topology_t  *build_synthetic_topology(int *arity, int nb_levels, int *core_numbering, int nb_core_per_node);


#ifdef __cplusplus
}
#endif

#endif

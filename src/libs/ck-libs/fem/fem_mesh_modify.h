/*!
 
This file contains a set of functions, which allow primitive operations upon meshes in parallel. The functions are defined in fem_mesh_modify.C.


Assumptions:

The mesh must be in a consistant state before and after these operations:
    - Any shared node must be in the IDXL table for both the local and 
      other chunks.
    - Exactly one ghost layer exists around all chunks. See definition below
    - All adjacency tables must be correct before any of these calls. 
      The calls will maintain the adjacency tables, both remotely
      and locally.
    - FEM_add_element can only be called with a set of existing local or shared nodes


A ghost element is one that is adjacent to at least one shared node. A ghost node is any node adjacent to a ghost element, but is itself not a shared node. Thus we have exactly one layer of ghosts.

 */



int FEM_add_node();
int FEM_add_shared_node(int* adjacent_nodes, int num_adjacent_nodes, int upcall);
void FEM_remove_node(int node);

void FEM_remove_element(int element, int elem_type);
int FEM_add_element(int* conn, int conn_size, int elem_type);

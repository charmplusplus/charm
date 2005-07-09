/* File: fem_mesh_modify.C
 * Authors: Isaac Dooley, Nilesh Choudhury
 * 
 * This file contains a set of functions, which allow primitive operations upon meshes in parallel.
 *
 * See the assumptions listed in fem_mesh_modify.h before using these functions.
 *
 */

#include "fem.h"
#include "fem_impl.h"

int FEM_add_node(){
  
  // lengthen node array, and any attributes if needed
  // return a new index
  
}


int FEM_add_shared_node(int* adjacent_nodes, int num_adjacent_nodes, int upcall){
  // add local node
  int newnode = FEM_add_node();

  // for each adjacent node, if the node is shared
  for(int i=0;i<num_adjacent_nodes;i++){
    
    // if node adjacent_nodes[i] is shared,
    {
      // call_shared_node_remote() on all chunks for which the shared node exists
    }



  }

}


// The function called by the entry method on the remote chunk
void FEM_add_shared_node_remote(){
  // create local node
  int newnode = FEM_add_node();
  
  // must negotiate the common IDXL number for the new node, 
  // and store it in appropriate IDXL tables

}




// remove a local or shared node, but NOT a ghost node
void FEM_remove_node(int node){
  
  if(FEM_Is_ghost_index(node))
    CkAbort("Cannot call FEM_remove_node on a ghost node\n");
  
  // if node is shared:
  //   verify it is not adjacent to any elements locally
  //   verify it is not adjacent to any elements on any of the associated chunks
  //   delete it locally and delete it on remote chunks, update IDXL tables

  // if node is local:
  //   verify it is not adjacent to any elements locally
  //   delete it locally
     
}




// Can be called on local or ghost elements
void FEM_remove_element(int element, int elem_type){
 
  if(FEM_Is_ghost_index(element)){
    // remove local copy from elem[elem_type]->ghost() table
    // call FEM_remove_element_remote on other chunk which owns the element
  }
  else {
    // delete the element from local elem[elem_type] table
  }

  
}

void FEM_remove_element_remote(int element, int elem_type){
  // remove local element from elem[elem_type] table
}




int FEM_add_element(int* conn, int conn_size, int elem_type){
  // if no shared or ghost nodes in conn
  //   grow local element and attribute tables if needed
  //   add to the elem[elem_type] table
  //   return new element id
  
  // else if any shared nodes but no ghosts in conn
  //   make this element ghost on all others, updating all IDXL's
  //   also in same remote entry method, update adjacencies on all others
  //   grow local element and attribute tables if needed
  //   add to local elem[elem_type] table, and update IDXL if needed
  //   update local adjacencies
  //   return the new element id

  // else if any ghosts in conn
  //   promote ghosts to shared on others, requesting new ghosts
  //   grow local element and attribute tables if needed
  //   add to local elem[elem_type] table, and update IDXL if needed
  //   update remote adjacencies
  //   update local adjacencies

}


int FEM_add_element_remote(){
  // promote ghosts to shared

  // find new ghosts for remote calling chunk by looking at new shared nodes
  // send these new ghosts to the remote calling chunk.

  // update my adjacencies

}

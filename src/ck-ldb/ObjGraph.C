#include "ObjGraph.h"

ObjGraph::ObjGraph(int count, CentralLB::LDStats* _stats)
{
  stats = _stats;
  // First, we need to make a linked list of objects.
  // Also, we'll construct a little hash table to improve
  // efficiency in finding the objects to build the edge lists

  // Initialize the linked list
  for(int i=0; i < hash_max; i++)
    node_table[i] = 0;
  
  nodelist = 0;
  n_objs = 0;
  int pe;
  int index;
  for(pe=count-1; pe >= 0; pe--)
    for(index = stats[pe].n_objs-1; index >= 0; index--) {
      Node* new_node = new Node;
      new_node->nxt = nodelist;
      nodelist = new_node;
      nodelist->proc = pe;
      nodelist->index = index;
      nodelist->n_out = 0;
      nodelist->outEdge = 0;
      nodelist->n_in = 0;
      nodelist->inEdge = 0;
      n_objs++;

      const int hashval = calc_hashval(stats[pe].objData[index].omID,
				       stats[pe].objData[index].id);
      nodelist->nxt_hash = node_table[hashval];
      node_table[hashval] = nodelist;
    }

  // Now go through the comm lists
  for(pe=count-1; pe >= 0; pe--)
    for(index = stats[pe].n_comm-1; index >= 0; index--) {
      const LDCommData newedgedata = stats[pe].commData[index];

      // If this isn't an object-to-object message, ignore it
      if (newedgedata.from_proc || newedgedata.to_proc)
	continue;
      
      Node* from_node = find_node(newedgedata.senderOM,newedgedata.sender);
      if (from_node == 0)
	CkPrintf("ObjGraph::find_node: Didn't locate from node match!\n");

      Node* to_node = find_node(newedgedata.receiverOM,newedgedata.receiver);
      if (to_node == 0)
	CkPrintf("ObjGraph::find_node: Didn't locate to node match!\n");

      // Store the edge data in correct outgoing and incoming lists.
      // Note that we should not free the edges twice, since they are in
      // two lists
      Edge* newedge = new Edge;
      newedge->proc = pe;
      newedge->index = index;
      newedge->from_proc = from_node->proc;
      newedge->from_index = from_node->index;
      newedge->to_proc = to_node->proc;
      newedge->to_index = to_node->index;
      newedge->nxt_out = from_node->outEdge;
      from_node->outEdge = newedge;
      from_node->n_out++;
      newedge->nxt_in = to_node->inEdge;
      to_node->inEdge = newedge;
      to_node->n_in++;
    }
}

ObjGraph::~ObjGraph()
{
  Node* node = nodelist;
  while (node != 0) {
    // I should only delete the from edges, since the same memory is
    // in both the from and to lists

    Edge* outEdge = node->edges_from();
    while (outEdge != 0) {
      Edge* nxt_edge = outEdge->next_from();
      delete outEdge;
      outEdge = nxt_edge;
    }
    Node* nxt_node = node->next();
    delete node;
    node = nxt_node;
  }
}

int ObjGraph::calc_hashval(LDOMid omid, LDObjid id)
{
  int hashval = omid.id;
  for(int i=0; i < OBJ_ID_SZ; i++)
    hashval +=  id.id[i];
  hashval %= hash_max;
  return hashval;
}

ObjGraph::Node* ObjGraph::find_node(LDOMid edge_omid, LDObjid edge_id)
{
  const int from_hashval = calc_hashval(edge_omid,edge_id);
  Node* from_node = node_table[from_hashval];

  while (from_node != 0) {
    const LDOMid omid =
      stats[from_node->proc].objData[from_node->index].omID;
    const LDObjid objid =
      stats[from_node->proc].objData[from_node->index].id;
    if (LDOMidEqual(omid,edge_omid) && LDObjIDEqual(objid,edge_id) )
      break;
  }

  return from_node;
}


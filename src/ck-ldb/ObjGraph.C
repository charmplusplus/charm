/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

/**
 * \addtogroup CkLdb
*/
/*@{*/

#include "ObjGraph.h"

static const double alpha = 30.e-6;
static const double beta = 3.e-9;

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
  // Count up the edges and the nodes, and allocate storage for
  // them all at once.
  n_objs = 0;
  n_edges = 0;
  int pe;
  for(pe=0; pe < count; pe++) {
    n_objs += stats[pe].n_objs;
    int index;
    // initialize node array
    for(index = 0; index < stats[pe].n_comm; index++) {
      const LDCommData newedgedata = stats[pe].commData[index];

      // If this isn't an object-to-object message, ignore it
      if (!newedgedata.from_proc() && !newedgedata.to_proc())
	n_edges++;
    }
  }
  nodelist = new Node[n_objs];
  edgelist = new Edge[n_edges];

  // Now initialize the node and the edge arrays
  int cur_node = 0;
  int cur_edge = 0;
  for(pe=0; pe < count; pe++) {
    int index;
    // initialize node array
    for(index = 0; index < stats[pe].n_objs; index++) {
      if(cur_node >= n_objs)
	CkPrintf("Error %d %d\n",cur_node,n_objs);
      Node* thisnode = nodelist + cur_node;
      thisnode->node_index = cur_node;
      thisnode->proc = pe;
      thisnode->index = index;
      thisnode->n_out = 0;
      thisnode->outEdge = 0;
      thisnode->n_in = 0;
      thisnode->inEdge = 0;
      cur_node++;
      const int hashval = calc_hashval(stats[pe].objData[index].omID(),
				       stats[pe].objData[index].id());
      thisnode->nxt_hash = node_table[hashval];
      node_table[hashval] = thisnode;
    }

    // initialize edge array
    for(index=0; index < stats[pe].n_comm; index++) {
      const LDCommData newedgedata = stats[pe].commData[index];

      // If this isn't an object-to-object message, ignore it
      if (newedgedata.from_proc() || newedgedata.to_proc())
	continue;

      if(cur_edge >= n_edges)
	CkPrintf("Error %d %d\n",cur_edge,n_edges);

      Edge* thisedge = edgelist + cur_edge;
      thisedge->edge_index = cur_edge;
      thisedge->proc = pe;
      thisedge->index = index;
      thisedge->from_node = -1;
      thisedge->to_node = -1;
      thisedge->nxt_out = 0;
      thisedge->nxt_in = 0;
      cur_edge++;
    }
  }
  if(cur_node != n_objs)
      CkPrintf("did not fill table %d %d\n",cur_node,n_objs);

  if(cur_edge != n_edges)
    CkPrintf("did not fill edge table %d %d\n",cur_edge,n_edges);

  // Now go through the comm lists
  for(cur_edge = 0; cur_edge < n_edges; cur_edge++) {
    Edge* newedge = edgelist + cur_edge;
    int pe = newedge->proc;
    int index = newedge->index;
    const LDCommData newedgedata = stats[pe].commData[index];

    Node* from_node = find_node(newedgedata.senderOM,newedgedata.sender);
    if (from_node == 0)
      CkPrintf("ObjGraph::find_node: Didn't locate from node match!\n");

    Node* to_node = find_node(newedgedata.receiverOM,newedgedata.receiver);
    if (to_node == 0)
      CkPrintf("ObjGraph::find_node: Didn't locate to node match!\n");

    // Store the edge data in correct outgoing and incoming lists.
    newedge->from_node = from_node->node_index;
    newedge->to_node = to_node->node_index;
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
  delete [] nodelist;
  delete [] edgelist;
}

double ObjGraph::EdgeWeight(Edge* e) {
  LDCommData commData = stats[e->proc].commData[e->index];
  return commData.messages * alpha + commData.bytes * beta;
}

int ObjGraph::calc_hashval(LDOMid omid, LDObjid id)
{
  int hashval = omid.id.idx;
  for(int i=0; i < OBJ_ID_SZ; i++)
    hashval +=  id.id[i];
  hashval %= hash_max;
  return hashval;
}

ObjGraph::Node* ObjGraph::find_node(LDOMid edge_omid, LDObjid edge_id)
{
  const int from_hashval = calc_hashval(edge_omid,edge_id);
  //  CkPrintf("From = %d\n",from_hashval);
  Node* from_node = node_table[from_hashval];

  while (from_node != 0) {
    const LDOMid omid =
      stats[from_node->proc].objData[from_node->index].omID();
    const LDObjid objid =
      stats[from_node->proc].objData[from_node->index].id();
    //    CkPrintf("Comparing %d to %d\n",objid.id[0],edge_id.id[0]);
    if (LDOMidEqual(omid,edge_omid) && LDObjIDEqual(objid,edge_id) )
      break;
    from_node = from_node->nxt_hash;
  }

  return from_node;
}





/*@}*/

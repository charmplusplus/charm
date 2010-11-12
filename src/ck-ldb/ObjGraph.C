/**
 *                 !!!!!!!!!!!!!!!!! DEFUNCT !!!!!!!!!!!!!!!!!!!
 *
 *  This file is not compiled anymore and its uses should be replaced by the
 *  class of the same name (ObjGraph) in ckgraph.h
 *
 *                 !!!!!!!!!!!!!!!!! DEFUNCT !!!!!!!!!!!!!!!!!!!
 */


/**
 * \addtogroup CkLdb
*/
/*@{*/

#include "ObjGraph.h"

static const double alpha = 30.e-6;
static const double beta = 3.e-9;

ObjGraph::ObjGraph(int count, BaseLB::LDStats* _stats)
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
  n_objs = stats->n_objs;
  n_edges = 0;
    // initialize node array
  int index;
  for(index = 0; index < stats->n_comm; index++) {
      const LDCommData newedgedata = stats->commData[index];

      // If this isn't an object-to-object message, ignore it
      if (!newedgedata.from_proc() && newedgedata.recv_type() == LD_OBJ_MSG)
	n_edges++;
    }
  nodelist = new Node[n_objs];
  edgelist = new Edge[n_edges];

  // Now initialize the node and the edge arrays
  int cur_node = 0;
  int cur_edge = 0;
  // initialize node array
  for(index = 0; index < stats->n_objs; index++) {
      LDObjData &odata = stats->objData[index];
      if(cur_node >= n_objs)
	CkPrintf("Error %d %d\n",cur_node,n_objs);
      Node* thisnode = nodelist + cur_node;
      thisnode->node_index = cur_node;
      thisnode->proc = stats->from_proc[index];
      thisnode->index = index;
      thisnode->n_out = 0;
      thisnode->outEdge = 0;
      thisnode->n_in = 0;
      thisnode->inEdge = 0;
      cur_node++;
      const int hashval = calc_hashval(odata.omID(),
				       odata.objID());
      thisnode->nxt_hash = node_table[hashval];
      node_table[hashval] = thisnode;
  }

    // initialize edge array
    for(index=0; index < stats->n_comm; index++) {
      LDCommData &newedgedata = stats->commData[index];

      // If this isn't an object-to-object message, ignore it
      if (newedgedata.from_proc() || newedgedata.recv_type()!=LD_OBJ_MSG)
	continue;

      if(cur_edge >= n_edges)
	CkPrintf("Error %d %d\n",cur_edge,n_edges);

      Edge* thisedge = edgelist + cur_edge;
      thisedge->edge_index = cur_edge;
      thisedge->index = index;
      thisedge->from_node = -1;
      thisedge->to_node = -1;
      thisedge->nxt_out = 0;
      thisedge->nxt_in = 0;
      cur_edge++;
  }
  if(cur_node != n_objs)
      CkPrintf("did not fill table %d %d\n",cur_node,n_objs);

  if(cur_edge != n_edges)
    CkPrintf("did not fill edge table %d %d\n",cur_edge,n_edges);

  // Now go through the comm lists
  for(cur_edge = 0; cur_edge < n_edges; cur_edge++) {
    Edge* newedge = edgelist + cur_edge;
    int index = newedge->index;
    const LDCommData newedgedata = stats->commData[index];

    Node* from_node = find_node(newedgedata.sender);
    if (from_node == 0) {
      if (!_lb_args.ignoreBgLoad() && _stats->complete_flag) 
	CkPrintf("ObjGraph::find_node: Didn't locate from node match!\n");
      continue;
    }

    Node* to_node = find_node(newedgedata.receiver.get_destObj());
    if (to_node == 0) {
      if (!_lb_args.migObjOnly() && _stats->complete_flag) 
        CkPrintf("ObjGraph::find_node: Didn't locate to node match!\n");
      continue;
    }

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
  LDCommData commData = stats->commData[e->index];
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

ObjGraph::Node* ObjGraph::find_node(const LDObjKey &edge_key)
{
  const LDOMid &edge_omid = edge_key.omID();
  const LDObjid &edge_id = edge_key.objID();
  const int from_hashval = calc_hashval(edge_omid,edge_id);
  //  CkPrintf("From = %d\n",from_hashval);
  Node* from_node = node_table[from_hashval];

  while (from_node != 0) {
    const LDOMid omid =
      stats->objData[from_node->index].omID();
    const LDObjid objid =
      stats->objData[from_node->index].objID();
    //    CkPrintf("Comparing %d to %d\n",objid.id[0],edge_id.id[0]);
    if (LDOMidEqual(omid,edge_omid) && LDObjIDEqual(objid,edge_id) )
      break;
    from_node = from_node->nxt_hash;
  }

  return from_node;
}





/*@}*/

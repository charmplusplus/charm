/** \file TeamLB.C
 *  Written by Esteban Meneses, 2010-11-24
 *  Updated by Abhinav Bhatele, 2010-12-09 to use ckgraph
 */

/**
 * \addtogroup CkLdb
*/
/*@{*/

#include "TeamLB.h"
#include "ckgraph.h"
#include <metis.h>

CreateLBFunc_Def(TeamLB, "Use Metis(tm) to partition object graph at two levels: team level and processor level")

TeamLB::TeamLB(const CkLBOptions &opt): CBase_TeamLB(opt)
{
  lbname = "TeamLB";
  if (CkMyPe() == 0)
    CkPrintf("[%d] TeamLB created\n",CkMyPe());

  // setting number of teams and team size
  teamSize = _lb_args.teamSize();
  numberTeams = CkNumPes() / teamSize;
}

/**
 * @brief METIS function that performs a balanced k-way partitioning of the
 * graph, considering the communication volume (hence the "V" in the name of
 * the function).

   extern "C" void METIS_PartGraphRecursive(int*, int*, int*, int*, int*, int*,
			      int*, int*, int*, int*, int*);
 */


/**
 * @brief Load balancing function. It uses METIS in a two step approach. The
 * first step consists in splitting the objects into teams. METIS is able to
 * minimize the communication volume across the teams while balancing the load
 * among the different teams. The second step goes deep in each team to balance
 * the load in the processors belonging to that particular team.
 */
void TeamLB::work(LDStats* stats)
{
  /** ========================== INITIALIZATION ============================= */
  ProcArray *parr = new ProcArray(stats);
  ObjGraph *ogr = new ObjGraph(stats);

  /** ============================= STRATEGY ================================ */
  if (_lb_args.debug() >= 2) {
    CkPrintf("[%d] In TeamLB Strategy...\n", CkMyPe());
  }

  // convert ObjGraph to the adjacency structure
  int numVertices = ogr->vertices.size();       // number of vertices
  int numEdges = 0;                             // number of edges

  double maxLoad = 0.0;
  int maxBytes = 0, i, j, k;

  /** both object load and number of bytes exchanged are normalized to an
   *  integer between 0 and 256 */
  for(i = 0; i < numVertices; i++) {
    if(ogr->vertices[i].getVertexLoad() > maxLoad)
      maxLoad = ogr->vertices[i].getVertexLoad();
    numEdges += ogr->vertices[i].sendToList.size();
    for(j = 0; j < ogr->vertices[i].sendToList.size(); j++) {
      if(ogr->vertices[i].sendToList[j].getNumBytes() > maxBytes)
        maxBytes = ogr->vertices[i].sendToList[j].getNumBytes();
    }
  }

  /* adjacency list */
  idx_t *xadj = new idx_t[numVertices + 1];
  /* id of the neighbors */
  idx_t *adjncy = new idx_t[numEdges];
  /* weights of the vertices */
  idx_t *vwgt = new idx_t[numVertices];
  /* weights of the edges */
  idx_t *adjwgt = new idx_t[numEdges];

  int edgeNum = 0;

  for(i = 0; i < numVertices; i++) {
    xadj[i] = edgeNum;
    vwgt[i] = (int)( (ogr->vertices[i].getVertexLoad() * 128) /maxLoad );
    for(j = 0; j < ogr->vertices[i].sendToList.size(); j++) {
      adjncy[edgeNum] = ogr->vertices[i].sendToList[j].getNeighborId();
      adjwgt[edgeNum] = (int)( (ogr->vertices[i].sendToList[j].getNumBytes() * 128) / maxBytes );
      edgeNum++;
    }
  }
  xadj[i] = edgeNum;
  CkAssert(edgeNum == numEdges);

  idx_t options[METIS_NOPTIONS];
  METIS_SetDefaultOptions(options);
  //options[METIS_OPTION_PTYPE] = METIS_PTYPE_RB;
  // C style numbering
  options[METIS_OPTION_NUMBERING] = 0;

  idx_t edgecut;          // number of edges cut by the partitioning
  // mapping of objs to partitions
  idx_t *pemap = new idx_t[numVertices];

  // number of constrains
  idx_t ncon = 1;
  real_t ubvec[ncon];
  // allow 10% imbalance tolerance
  ubvec[0] = 1.1;

  // Specifies size of vertices for computing the total communication volume
  idx_t *vsize = NULL;
  // This array of size nparts specifies the desired weight for each partition
  // and setting it to NULL indicates graph should be equally divided among
  // partitions
  real_t *tpwgts = NULL;

  if (_lb_args.debug() >= 1)
  CkPrintf("[%d] calling METIS_PartGraphRecursive.\n", CkMyPe());

  METIS_PartGraphRecursive(&numVertices, &ncon, xadj, adjncy, vwgt, vsize,
      adjwgt, &numberTeams, tpwgts, ubvec, options, &edgecut, pemap);

  int *global_pemap = new int[numVertices];

  // partitioning each team
  if(teamSize > 1) {
    idx_t *team_xadj = new idx_t[numVertices + 1];
    idx_t *team_adjncy = new idx_t[numEdges];
    idx_t *team_vwgt = new idx_t[numVertices];
    idx_t *team_adjwgt = new idx_t[numEdges];
    idx_t *team_pemap = new idx_t[numVertices];
    idx_t *team_vsize = NULL;
    real_t *team_tpwgts = NULL;

    int teamEdgecut, node;
    int *mapping = new int[numVertices];
    int *invMapping = new int[numVertices];

    // traversing the list of teams and load balancing each one
    for(i=0; i<numberTeams; i++) {
      int teamMembers = 0;	// number of vertices in a team

      // collecting all the elements of a particular team
      // mapping stores the association of local to global index
      // invMapping stores the inverse association
      for(j = 0; j < numVertices; j++) {
	if(pemap[j] == i) {
	  mapping[teamMembers] = j;
	  invMapping[j] = teamMembers;
	  team_vwgt[teamMembers] = vwgt[j];
	  teamMembers++;
	}
      }

      // building up the adjacency data structures
      int teamIndex = 0;
      for(j = 0; j < teamMembers; j++) {
	team_xadj[j] = teamIndex;
	for(k = xadj[mapping[j]]; k < xadj[mapping[j]+1]; k++) {
	  node = adjncy[k];
	  if(pemap[node] == i) {
	    team_adjncy[teamIndex] = invMapping[node];
	    team_adjwgt[teamIndex] = adjwgt[k];
	    teamIndex++;
	  }
	}
      }
      team_xadj[teamMembers] = teamIndex;

      // calling METIS library
      METIS_PartGraphRecursive(&teamMembers, &ncon, team_xadj, team_adjncy,
        team_vwgt, team_vsize, team_adjwgt, &teamSize, team_tpwgts, ubvec,
        options, &teamEdgecut, team_pemap);

      // converting local mapping into global mapping
      for(j = 0; j < teamMembers; j++) {
	global_pemap[mapping[j]] = i*teamSize + team_pemap[j];
      }
			
    } // end for

    delete[] team_xadj;
    delete[] team_adjncy;
    delete[] team_vwgt;
    delete[] team_adjwgt;
    delete[] team_pemap;
    delete[] team_vsize;
    delete[] team_tpwgts;

    delete[] mapping;
    delete[] invMapping;
  } else {
    delete[] global_pemap;
    global_pemap = pemap;
  }

  delete[] xadj;
  delete[] adjncy;
  delete[] vwgt;
  delete[] adjwgt;
  delete[] vsize;
  delete[] tpwgts;


  if (_lb_args.debug() >= 1) {
   CkPrintf("[%d] TeamLB done! \n", CkMyPe());
  }

  for(i = 0; i < numVertices; i++) {
    if(pemap[i] != ogr->vertices[i].getCurrentPe())
      ogr->vertices[i].setNewPe(pemap[i]);
  }

  delete[] pemap;

  /** ============================== CLEANUP ================================ */
  ogr->convertDecisions(stats);
}

#include "TeamLB.def.h"

/*@}*/

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
#include "metis.h"

CreateLBFunc_Def(TeamLB, "Use Metis(tm) to partition object graph at two levels: team level and processor level")

TeamLB::TeamLB(const CkLBOptions &opt): CentralLB(opt)
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
    numEdges += ogr->vertices[i].edgeList.size();
    for(j = 0; j < ogr->vertices[i].edgeList.size(); j++) {
      if(ogr->vertices[i].edgeList[j].getNumBytes() > maxBytes)
        maxBytes = ogr->vertices[i].edgeList[j].getNumBytes();
    }
  }

  /* adjacency list */
  idxtype *xadj = new idxtype[numVertices + 1];
  /* id of the neighbors */
  idxtype *adjncy = new idxtype[numEdges];
  /* weights of the vertices */
  idxtype *vwgt = new idxtype[numVertices];
  /* weights of the edges */
  idxtype *adjwgt = new idxtype[numEdges];

  int edgeNum = 0;

  for(i = 0; i < numVertices; i++) {
    xadj[i] = edgeNum;
    vwgt[i] = (int)( (ogr->vertices[i].getVertexLoad() * 128) /maxLoad );
    for(j = 0; j < ogr->vertices[i].edgeList.size(); j++) {
      adjncy[edgeNum] = ogr->vertices[i].edgeList[j].getNeighborId();
      adjwgt[edgeNum] = (int)( (ogr->vertices[i].edgeList[j].getNumBytes() * 128) / maxBytes );
      edgeNum++;
    }
  }
  xadj[i] = edgeNum;
  CkAssert(edgeNum == numEdges);

  int wgtflag = 3;      // weights both on vertices and edges
  int numflag = 0;      // C Style numbering
  int options[5];
  options[0] = 0;       // use default values
  int edgecut;          // number of edges cut by the partitioning
  idxtype *pemap = new idxtype[numVertices];

  if (_lb_args.debug() >= 1)
  CkPrintf("[%d] calling METIS_PartGraphRecursive.\n", CkMyPe());

  METIS_PartGraphRecursive(&numVertices, xadj, adjncy, vwgt, adjwgt,
	    &wgtflag, &numflag, &numberTeams, options, &edgecut, pemap);

  int *global_pemap = new int[numVertices];

  // partitioning each team
  if(teamSize > 1) {
    idxtype *team_xadj = new idxtype[numVertices + 1];
    idxtype *team_adjncy = new idxtype[numEdges];
    idxtype *team_vwgt = new idxtype[numVertices];
    idxtype *team_adjwgt = new idxtype[numEdges];
    idxtype *team_pemap = new idxtype[numVertices];

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
      METIS_PartGraphRecursive(&teamMembers, team_xadj, team_adjncy, team_vwgt,
		  team_adjwgt, &wgtflag, &numflag, &teamSize, options,
		  &teamEdgecut, team_pemap);

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

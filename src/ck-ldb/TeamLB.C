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

#include <charm++.h>

#include "cklists.h"

#include "TeamLB.h"

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

static void printStats(int count, int numobjs, double *cputimes, 
                       int **comm, int *map)
{
  int i, j;
  double *petimes = new double[count];
  for(i=0;i<count;i++) {
    petimes[i] = 0.0;
  }
  for(i=0;i<numobjs;i++) {
    petimes[map[i]] += cputimes[i];
  }
  double maxpe = petimes[0], minpe = petimes[0];
  CkPrintf("\tPE\tTimexSpeed\n");
  for(i=0;i<count;i++) {
    CkPrintf("\t%d\t%lf\n",i,petimes[i]);
    if(maxpe < petimes[i])
      maxpe = petimes[i];
    if(minpe > petimes[i])
      minpe = petimes[i];
  }
  delete[] petimes;
  CkPrintf("\tLoad Imbalance=%lf seconds\n", maxpe-minpe);
  int ncomm = 0;
  for(i=0;i<numobjs;i++) {
    for(j=0;j<numobjs;j++) {
      if(map[i] != map[j])
        ncomm += comm[i][j];
    }
  }
  CkPrintf("\tCommunication (off proc msgs) = %d\n", ncomm/2);
}

/**
 * @brief METIS function that performs a balanced k-way partitioning of the graph, considering
 * the communication volume (hence the "V" in the name of the function).
 */
extern "C" void METIS_PartGraphVKway(int*, int*, int*, int*, int*,
				     int*, int*, int*, int*,
				     int*, int*);


/**
 * @brief Load balancing function. It uses METIS in a two step approach. The first step consists in
 * splitting the objects into teams. METIS is able to minimize the communication volume across the teams
 * while balancing the load among the different teams. The second step goes deep in each team to balance
 * the load in the processors belonging to that particular team.
 */
void TeamLB::work(LDStats* stats)
{
	int i, j, m, k, node;
	int option = 0;

	if (_lb_args.debug() >= 2) {
		CkPrintf("[%d] In TeamLB Strategy...\n", CkMyPe());
	}

	// making a communication hash
	stats->makeCommHash();

	int n_pes = stats->nprocs();
	int numobjs = stats->n_objs;

	// removing non-migratable objects
	removeNonMigratable(stats, n_pes);

	// allocate space for the computing data
	double *objtime = new double[numobjs];
	int *objwt = new int[numobjs];
	int *teamObjwt = new int[numobjs];
	int *origmap = new int[numobjs];
	LDObjHandle *handles = new LDObjHandle[numobjs];

	for(i=0;i<numobjs;i++) {
		objtime[i] = 0.0;
		objwt[i] = 0;
		origmap[i] = 0;
	}

	for (i=0; i<stats->n_objs; i++) {
		LDObjData &odata = stats->objData[i];
		if (!odata.migratable) 
			CmiAbort("TeamLB does not dupport nonmigratable object.\n");
		int frompe = stats->from_proc[i];
		origmap[i] = frompe;
		objtime[i] = odata.wallTime*stats->procs[frompe].pe_speed;
		handles[i] = odata.handle;
	}

	// to convert the weights on vertices to integers
	double max_objtime = objtime[0];
	for(i=0; i<numobjs; i++) {
		if(max_objtime < objtime[i])
			max_objtime = objtime[i];
	}
	double ratio = 1000.0/max_objtime;
	for(i=0; i<numobjs; i++) {
		objwt[i] = (int)(objtime[i]*ratio);
	}
	int **comm = new int*[numobjs];
	for (i=0; i<numobjs; i++) {
		comm[i] = new int[numobjs];
		for (j=0; j<numobjs; j++)  {
			comm[i][j] = 0;
		}
	}

	const int csz = stats->n_comm;
	for(i=0; i<csz; i++) {
		LDCommData &cdata = stats->commData[i];
		if(!cdata.from_proc() && cdata.receiver.get_type() == LD_OBJ_MSG){
			int senderID = stats->getHash(cdata.sender);
			int recverID = stats->getHash(cdata.receiver.get_destObj());
			if (stats->complete_flag == 0 && recverID == -1) continue;
			CmiAssert(senderID < numobjs && senderID >= 0);
			CmiAssert(recverID < numobjs && recverID >= 0);
			comm[senderID][recverID] += cdata.messages;
			comm[recverID][senderID] += cdata.messages;
		} else if (cdata.receiver.get_type() == LD_OBJLIST_MSG) {
			int nobjs;
			LDObjKey *objs = cdata.receiver.get_destObjs(nobjs);
			int senderID = stats->getHash(cdata.sender);
			for (j=0; j<nobjs; j++) {
				int recverID = stats->getHash(objs[j]);
				if((senderID == -1)||(recverID == -1))
					if (_lb_args.migObjOnly()) continue;
					else CkAbort("Error in search\n");
				comm[senderID][recverID] += cdata.messages;
				comm[recverID][senderID] += cdata.messages;
			}
		}
	}

	// ignore messages sent from an object to itself
	for (i=0; i<numobjs; i++)
		comm[i][i] = 0;

	// construct the graph in CSR format
	int *xadj = new int[numobjs+1];
	int *teamXadj = new int[numobjs+1];
	int numedges = 0;
	for(i=0;i<numobjs;i++) {
		for(j=0;j<numobjs;j++) {
			if(comm[i][j] != 0)
				numedges++;
		}
	}
	int *adjncy = new int[numedges];
	int *teamAdjncy = new int[numedges];
	int *edgewt = new int[numedges];
	int *teamEdgewt = new int[numedges];
	xadj[0] = 0;
	int count4all = 0;
	for (i=0; i<numobjs; i++) {
		for (j=0; j<numobjs; j++) { 
			if (comm[i][j] != 0) { 
				adjncy[count4all] = j;
				edgewt[count4all++] = comm[i][j];
			}
		}
		xadj[i+1] = count4all;
	}

	if (_lb_args.debug() >= 2) {
		CkPrintf("Pre-LDB Statistics step %d\n", step());
		printStats(n_pes, numobjs, objtime, comm, origmap);
	}

	int wgtflag = 3; // Weights both on vertices and edges
	int numflag = 0; // C Style numbering
	int options[5];
	options[0] = 0;
	int edgecut, teamEdgecut;
	int *newmap, *teamMap, *mapping, *invMapping, *procMapping;
	int sameMapFlag = 1;

	if (n_pes < 1) {
		CkPrintf("error: Number of Pe less than 1!");
	} else if (n_pes == 1) {
		newmap = origmap;
		sameMapFlag = 1;
	} else {
		sameMapFlag = 0;
		newmap = new int[numobjs];
		teamMap = new int[numobjs];
		mapping = new int[numobjs];
		invMapping = new int[numobjs];
		procMapping = new int[numobjs];
		if (_lb_args.debug() >= 1)
			CkPrintf("[%d] calling METIS_PartGraphVKway.\n", CkMyPe());
		METIS_PartGraphVKway(&numobjs, xadj, adjncy, objwt, edgewt,
			&wgtflag, &numflag, &numberTeams, options,
			&edgecut, teamMap);
		if (_lb_args.debug() >= 1)
			CkPrintf("[%d] after calling METIS_PartGraphVKway.\n", CkMyPe());
	}

	// partitioning each team
	if(teamSize > 1){

		// traversing the list of teams and load balancing each one
		for(i=0; i<numberTeams; i++){
			int teamMembers = 0;

			// collecting all the elements of a particular team
			// mapping stores the association of local to global index
			// invMapping stores the inverse association
			for(j=0; j<numobjs; j++){
				if(teamMap[j] == i){
					mapping[teamMembers] = j;
					invMapping[j] = teamMembers;
					teamObjwt[teamMembers] = objwt[j];
					teamMembers++;
				}
			}

			// building up the adjacency data structures
			int teamIndex = 0;
			for(j=0; j<teamMembers; j++){
				teamXadj[j] = teamIndex;
				for(k=xadj[mapping[j]]; k<xadj[mapping[j]+1]; k++){
					node = adjncy[k];
					if(teamMap[node] == i){
						teamAdjncy[teamIndex] = invMapping[node];
						teamEdgewt[teamIndex] = edgewt[k];
						teamIndex++;
					}
				} 
			}
			teamXadj[teamMembers] = teamIndex;

			// calling METIS library
			METIS_PartGraphVKway(&teamMembers, teamXadj, teamAdjncy, teamObjwt, teamEdgewt, 
								&wgtflag, &numflag, &teamSize, options, &teamEdgecut, procMapping);

			// converting local mapping into global mapping
			for(j=0; j<teamMembers; j++){
				newmap[mapping[j]] = i*teamSize + procMapping[j];
			}
			

		}

		delete[] mapping;
		delete[] invMapping;
		delete[] procMapping;
		delete[] teamMap;

	} else {
		delete[] newmap;
		newmap = teamMap;
	}
	
	if (_lb_args.debug() >= 2) {
		CkPrintf("Post-LDB Statistics step %d\n", step());
		printStats(n_pes, numobjs, objtime, comm, newmap);
	}

	for(i=0;i<numobjs;i++)
		delete[] comm[i];
	delete[] comm;
	delete[] objtime;
	delete[] xadj;
	delete[] adjncy;
	if(objwt) delete[] objwt;
	if(edgewt) delete[] edgewt;
	
	if(!sameMapFlag) {
		for(i=0; i<numobjs; i++) {
			if(origmap[i] != newmap[i]) {
				CmiAssert(stats->from_proc[i] == origmap[i]);
				stats->to_proc[i] =  newmap[i];
				if (_lb_args.debug() >= 3)
					CkPrintf("[%d] Obj %d migrating from %d to %d\n", CkMyPe(),i,stats->from_proc[i],stats->to_proc[i]);
			}
		}
	}
	
	delete[] origmap;
	if(newmap != origmap)
		delete[] newmap;
	if (_lb_args.debug() >= 1) {
		CkPrintf("[%d] TeamLB done! \n", CkMyPe());
	}
}

#include "TeamLB.def.h"

/*@}*/

#include "AAPLearner.h"
#include "ComlibManager.h"
#include "EachToManyMulticastStrategy.h"

AAPLearner::AAPLearner() {
   init();
}

void AAPLearner::init() {
    alpha = ALPHA;
    beta = BETA;
}

Strategy *AAPLearner::optimizePattern(Strategy *strat, 
                                           ComlibGlobalStats &stats) {
    CharmStrategy *in_strat = (CharmStrategy *)strat;
    double npes;              //, *pelist;
    CharmStrategy *ostrat = NULL;

    /*
      if(in_strat->getType() == ARRAY_STRATEGY) {
      in_strat->ainfo.getCombinedPeList(pelist, npes);
      }
      
      if(in_strat->getType() == GROUP_STRATEGY) {
      CkGroupID gid;
      //Convert to combined pelist
      in_strat->ginfo.getSourceGroup(gid, pelist, npes);
      }
    */

    double degree = 0, msgsize = 0, nmsgs = 0;
    stats.getAverageStats(strat->getInstance(), msgsize, nmsgs, degree, npes);

    double dcost = computeDirect(npes, msgsize, degree);
    double mcost = computeMesh(npes, msgsize, degree);
    double gcost = computeGrid(npes, msgsize, degree);
    double hcost = computeHypercube(npes, msgsize, degree);
    double mincost = min4(dcost, mcost, gcost, hcost);

    int minstrat = -1;
    if(dcost == mincost) 
        minstrat = USE_DIRECT;
    else if(mcost == mincost)                     
        minstrat = USE_MESH;                
    else if(gcost == mincost) 
        minstrat = USE_GRID;
    else if(hcost == mincost) 
        minstrat = USE_HYPERCUBE;

    CkPrintf("Choosing router %d, %g, %g, %g, : %g,%g,%g\n", minstrat, 
             mcost, hcost, dcost, npes, msgsize, degree);
    
    if(in_strat->getType() == ARRAY_STRATEGY) {
        CkArrayID said, daid;
        CkArrayIndexMax *sidxlist, *didxlist;
        int nsrc, ndest;
        
        in_strat->ainfo.getSourceArray(said, sidxlist, nsrc);
        in_strat->ainfo.getDestinationArray(daid, didxlist, ndest);
                
        ostrat = new EachToManyMulticastStrategy
            (minstrat, said, daid,
             nsrc, sidxlist, ndest,
             didxlist);

        ostrat->setInstance(in_strat->getInstance());
    }
    
    //Group strategy implement later, foo bar !!
    if(in_strat->getType() == GROUP_STRATEGY) {
        CkGroupID gid;
        int src_npes, *src_pelist;
        int dest_npes, *dest_pelist;
        in_strat->ginfo.getSourceGroup(gid, src_pelist, src_npes);
        in_strat->ginfo.getDestinationGroup(gid, dest_pelist, dest_npes); 

        ostrat = new EachToManyMulticastStrategy
            (minstrat, src_npes, src_pelist, dest_npes, dest_pelist);
    }

    return ostrat;
}

//P = number of processors, m = msgsize, d = degree
double AAPLearner::computeDirect(double P, double m, double d) {
    double cost = 0.0;
    cost = d * alpha;
    cost += d * m * beta;
    
    return cost;
}

//P = number of processors, m = msgsize, d = degree
double AAPLearner::computeMesh(double P, double m, double d) {

    double cost = 0.0;
    cost = 2 * sqrt((double) P) * alpha;
    cost += 2 * d * m * beta;
    
    return cost;
}

//P = number of processors, m = msgsize, d = degree
double AAPLearner::computeHypercube(double P, double m, double d) {

    if(P == 0)
        return 0;

    double cost = 0.0;
    double log_2_P = log(P)/log(2.0);
    
    if(d >= P/2) {
        cost = log_2_P * alpha;
        cost += P/2 * log_2_P * m * beta;
    }
    else {
        cost = log_2_P * alpha;
        cost += log_2_P * d * m * beta;
    }

    return cost;
}

//P = number of processors, m = msgsize, d = degree
double AAPLearner::computeGrid(double P, double m, double d) {
    double cost = 0.0;
    cost = 3 * cbrt((double) P) * alpha;
    cost += 3 * d * m * beta;
    
    return cost;
}


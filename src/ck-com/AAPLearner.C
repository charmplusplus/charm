// #ifdef filippo

// #include "AAPLearner.h"
// #include "ComlibManager.h"
// #include "EachToManyMulticastStrategy.h"

// #define max(a,b) ((a > b) ? a : b)

// AAPLearner::AAPLearner() {
//     init();
// }

// void AAPLearner::init() {
//     alpha = ALPHA;
//     beta = BETA;
// }

// Strategy *AAPLearner::optimizePattern(Strategy *strat, 
//                                            ComlibGlobalStats &stats) {
//     CharmStrategy *in_strat = (CharmStrategy *)strat;
//     double npes;              //, *pelist;
//     CharmStrategy *ostrat = NULL;

//     /*
//       if(in_strat->getType() == ARRAY_STRATEGY) {
//       in_strat->ainfo.getCombinedPeList(pelist, npes);
//       }
      
//       if(in_strat->getType() == GROUP_STRATEGY) {
//       CkGroupID gid;
//       //Convert to combined pelist
//       in_strat->ginfo.getSourceGroup(gid, pelist, npes);
//       }
//     */

//     double degree = 0, msgsize = 0, nmsgs = 0;
//     stats.getAverageStats(strat->getInstance(), msgsize, nmsgs, degree, npes);

//     double dcost = computeDirect(npes, msgsize, degree);
//     double mcost = computeMesh(npes, msgsize, degree);
//     double gcost = computeGrid(npes, msgsize, degree);
//     double hcost = computeHypercube(npes, msgsize, degree);
//     double mincost = min4(dcost, mcost, gcost, hcost);

//     int minstrat = USE_MESH;
//     if(dcost == mincost) 
//         minstrat = USE_DIRECT;
//     else if(mcost == mincost)                     
//         minstrat = USE_MESH;                
//     else if(gcost == mincost) 
//         minstrat = USE_GRID;
//     else if(hcost == mincost) 
//         minstrat = USE_HYPERCUBE;

//     //CkPrintf("Choosing router %d, %g, %g, %g, %g; %g : %g,%g,%g\n", minstrat, 
//     //       mcost, hcost, gcost, dcost, mincost, npes, msgsize, degree);
    
//     if(in_strat->getType() == ARRAY_STRATEGY) {
//         CkArrayID said, daid;
//         CkArrayIndex *sidxlist, *didxlist;
//         int nsrc, ndest;
        
//         in_strat->ainfo.getSourceArray(said, sidxlist, nsrc);
//         in_strat->ainfo.getDestinationArray(daid, didxlist, ndest);
                
//         ostrat = new EachToManyMulticastStrategy
//             (minstrat, said, daid,
//              nsrc, sidxlist, ndest,
//              didxlist);

//         ostrat->setInstance(in_strat->getInstance());
//         ((EachToManyMulticastStrategy *) ostrat)->enableLearning();
//     }
    
//     //Group strategy implement later, foo bar !!
//     if(in_strat->getType() == GROUP_STRATEGY) {
//         CkGroupID gid;
//         int src_npes, *src_pelist;
//         int dest_npes, *dest_pelist;
//         in_strat->ginfo.getSourceGroup(gid, src_pelist, src_npes);
//         in_strat->ginfo.getDestinationGroup(gid, dest_pelist, dest_npes); 

//         ostrat = new EachToManyMulticastStrategy
//             (minstrat, src_npes, src_pelist, dest_npes, dest_pelist);
//         ((EachToManyMulticastStrategy *) ostrat)->enableLearning();
//         ostrat->setInstance(in_strat->getInstance());
//     }

//     return ostrat;
// }

// //P = number of processors, m = msgsize, d = degree
// double AAPLearner::computeDirect(double P, double m, double d) {
//     double cost1, cost2;

//     /*  //Old equations do not model bursts 
//       cost = d * alpha;
//       cost += d * m * beta;
//     */

//     cost1 = (d-1) * ALPHA_NIC1 + alpha + m * beta + d * m * GAMMA_NIC; 
//     cost2 = alpha + d * ALPHA_NIC2 +  d * m * beta + m * GAMMA_NIC;
    
//     return max(cost1, cost2); 
// }

// //P = number of processors, m = msgsize, d = degree
// double AAPLearner::computeMesh(double P, double m, double d) {

//     double cost1, cost2;

//     /* old equation 
//     cost = 2 * sqrt((double) P) * alpha;
//     cost += 2 * d * m * beta;
//     */

//     double sqrt_p = ceil(sqrt((double) P));

//     cost1 = 2 * (sqrt_p - 2) * ALPHA_NIC1 + 2 * alpha + 2 * m * beta
//       + 2 * d * m * GAMMA_NIC; 
//     cost2 = 2 * alpha + 2 * (sqrt_p - 2) * ALPHA_NIC2 + 2 * d * m * beta + 2 * m *GAMMA_NIC;
    
//     return max(cost1, cost2) + d * ALPHA_CHARM; 
// }

// //P = number of processors, m = msgsize, d = degree
// double AAPLearner::computeHypercube(double P, double m, double d) {

//     //Temporarily disabling hypercube
//     return 100;

//     if(P == 0)
//         return 0;

//     double cost = 0.0;
//     double log_2_P = log(P)/log(2.0);
    
//     if(d >= P/2) {
//       cost = log_2_P * alpha;
//       cost += (P/2) * log_2_P * m * (beta + GAMMA_NIC + GAMMA_MEM);
//     }
//     else {
//       cost = log_2_P * alpha;
//       cost += log_2_P * d * m * (beta + GAMMA_NIC + GAMMA_MEM);
//     }
    
//     return cost + d * ALPHA_CHARM;
// }

// //P = number of processors, m = msgsize, d = degree
// double AAPLearner::computeGrid(double P, double m, double d) {
//     double cost1, cost2 = 0.0;
//     /*
//       cost = 3 * cubeRoot((double) P) * alpha;
//       cost += 3 * d * m * beta;
//     */

//     double cbrt_p = ceil(cubeRoot((double) P));

//     cost1 = 3 * (cbrt_p - 2) * ALPHA_NIC1 + 3 * alpha + 3 * m * beta +
//       3 * d *m * GAMMA_NIC;  
//     cost2 = 3 * alpha + 3 * (cbrt_p - 2) * ALPHA_NIC2 + 3 * d * m * beta + 3 * m *GAMMA_NIC; 
    
//     return max(cost1, cost2) + d * ALPHA_CHARM;
// }

// #endif

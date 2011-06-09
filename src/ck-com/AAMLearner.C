// #ifdef filippo

// #include "AAMLearner.h"
// #include "ComlibManager.h"

// #include "EachToManyMulticastStrategy.h"
// //#include "RingMulticastStrategy.h"

// AAMLearner::AAMLearner() {
//    init();
// }

// void AAMLearner::init() {
//     alpha = ALPHA;
//     beta = BETA;
//     gamma = GAMMA;
// }

// Strategy *AAMLearner::optimizePattern(Strategy *strat, 
//                                            ComlibGlobalStats &stats) {
//     CharmStrategy *in_strat = (CharmStrategy *)strat;
//     double npes;              //, *pelist;
//     CharmStrategy *ostrat = NULL;

//     double degree = 0, msgsize = 0, nmsgs = 0;
//     stats.getAverageStats(strat->getInstance(), msgsize, nmsgs, 
//                           degree, npes);

//     double dcost = computeDirect(npes, msgsize, degree);
//     double mcost = computeMesh(npes, msgsize, degree);
//     double gcost = computeGrid(npes, msgsize, degree);
//     double hcost = computeHypercube(npes, msgsize, degree);
//     double mincost = min4(dcost, mcost, gcost, hcost);

//     int minstrat = -1;
//     if(in_strat->getType() == ARRAY_STRATEGY) {
//         CkArrayID said, daid;
//         CkArrayIndex *sidxlist, *didxlist;
//         int nsrc, ndest;
        
//         in_strat->ainfo.getSourceArray(said, sidxlist, nsrc);
//         in_strat->ainfo.getDestinationArray(daid, didxlist, ndest);
               
//         if(dcost == mincost) 
//             minstrat = USE_DIRECT;        
        
//         else if(mcost == mincost) 
//             minstrat = USE_MESH;                
//         else if(gcost == mincost) 
//             minstrat = USE_GRID;
//         else if(hcost == mincost) 
//             minstrat = USE_HYPERCUBE;               

//         //CkPrintf("Choosing router %d, %g, %g, %g\n", minstrat, 
//         //       mcost, hcost, dcost);
        
//         //if(minstrat != USE_DIRECT) {
//         ostrat = new EachToManyMulticastStrategy
//             (minstrat, said, daid,
//              nsrc, sidxlist, ndest,
//              didxlist);
        
//         ostrat->setMulticast();

//         /*
//           }        
//           else {
//           ostrat = new RingMulticastStrategy(said, daid);
          
//           }
//         */
        
//         ostrat->setInstance(in_strat->getInstance());
//         ((EachToManyMulticastStrategy *)ostrat)->enableLearning();
//     }
//     else
//         CkAbort("Groups Not Implemented Yet\n");

//     //Group strategy implement later, foo bar !!
    
//     return ostrat;
// }

// //P = number of processors, m = msgsize, d = degree
// double AAMLearner::computeDirect(double P, double m, double d) {
//     double cost = 0.0;
//     cost = d * alpha;
//     cost += d * m * beta;
    
//     return cost;
// }

// /******************* CHECK EQUATIONS FOR AAM ***********/
// //P = number of processors, m = msgsize, d = degree
// double AAMLearner::computeMesh(double P, double m, double d) {
//     double cost = 0.0;
//     cost = 2 * sqrt((double) P) * alpha;
//     cost += d * m * (beta + gamma);
    
//     return cost;
// }

// //P = number of processors, m = msgsize, d = degree
// double AAMLearner::computeHypercube(double P, double m, double d) {

//     if(P == 0)
//         return 0;

//     double cost = 0.0;
//     double log_2_P = log(P)/log(2.0);
    
//     cost = log_2_P * alpha;
//     cost += d * m * (beta + gamma);

//     return cost;
// }

// //P = number of processors, m = msgsize, d = degree
// double AAMLearner::computeGrid(double P, double m, double d) {

//     double cost = 0.0;
//     cost = 3 * cubeRoot((double) P) * alpha;
//     cost += d * m * (beta + gamma);
    
//     return cost;
// }

// #endif

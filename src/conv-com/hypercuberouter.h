/**
   @addtogroup ConvComlibRouter
   @{
   @file 
   @brief Dimensional Exchange (Hypercube) based routing strategy. 
*/

#ifndef _HYPERCUBEROUTER_H
#define _HYPERCUBEROUTER_H
#include "petable.h"

/// Dimensional Exchange (Hypercube) based router
class HypercubeRouter : public Router
{
 private:
    PeTable *PeHcube, *PeHcube1;
    int *buffer;
    int* msgnum, InitCounter;
    int *penum,*gpes;
    int **next;
    int Dim, stage, MyPe, NumPes, numDirectSteps, two_pow_ndirect;
    int procMsgCount;
    void InitVars();
    void CreateStageTable(int, int *);
    void LocalProcMsg(comID id);
    void start_hcube(comID id);
    
 public:
    
    HypercubeRouter(int, int, Strategy*, int ndirect = 0);
    ~HypercubeRouter();
    void NumDeposits(comID, int);
    void EachToAllMulticast(comID , int , void *, int);
    void EachToManyMulticast(comID , int , void *, int, int *, int);
    void EachToManyMulticastQ(comID id, CkQ<MessageHolder *> &msgq);
    
    void ProcMsg(int, msgstruct **) {;}
    void RecvManyMsg(comID, char *);
    void ProcManyMsg(comID, char *);
    void DummyEP(comID id, int);
    void SetMap(int *);
    
    //FIX this, some initialization done here
    void SetID(comID id);
};
#endif

/*@}*/

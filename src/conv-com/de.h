#ifndef _DE_H
#define _DE_H
#include <converse.h>
#include "convcomlib.h"
#include "petable.h"

//Dimensional Exchange (Hypercube) based router
class DimexRouter : public Router
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
    
    DimexRouter(int, int, int ndirect = 0);
    ~DimexRouter();
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

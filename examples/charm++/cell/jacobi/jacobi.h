#ifndef __JACOBI_H__
#define __JACOBI_H__

#include "jacobi.decl.h"
#include "jacobi_config.h"


// DMK - DEBUG
#include "main.h"  // Include temp for malloc_aligned and malloc_free calls


#define SWAP(a, b, t) { register t __tmp__ = a; a = b; b = __tmp__; }


class EastWestGhost : public CMessage_EastWestGhost {
  public:
    EastWestGhost() : CMessage_EastWestGhost() {};
    float data[NUM_ROWS];
    int iterCount;
};


class NorthSouthGhost : public CMessage_NorthSouthGhost {
  public:
    NorthSouthGhost() : CMessage_NorthSouthGhost() {};
    float data[NUM_COLS];
    int iterCount;
};


class Jacobi : public CBase_Jacobi {

  ////////////////////////////////////////////////////////////////////////////////////////////////
  //  Member Variables
  public:
    float* matrix;     // Pointer to the first buffer (the read buffer per iteration)
    float* matrixTmp;  // Pointer to the second buffer (the write buffer per iteration)
  private:
    int ghostCount;
    int ghostCountNeeded;
    int iterCount;

    EastWestGhost* eastMsgSave[2];
    EastWestGhost* westMsgSave[2];
    NorthSouthGhost* northMsgSave[2];
    NorthSouthGhost* southMsgSave[2];

    EastWestGhost* futureEastMsg;
    EastWestGhost* futureWestMsg;
    NorthSouthGhost* futureNorthMsg;
    NorthSouthGhost* futureSouthMsg;


  ////////////////////////////////////////////////////////////////////////////////////////////////
  // Constructor(s) / Destructor
  public:
    Jacobi();
    Jacobi(CkMigrateMessage *msg);
    ~Jacobi();


  ////////////////////////////////////////////////////////////////////////////////////////////////
  // Member Functions
  public:
    void startIteration();

    void northData(int size, float* ghostData, int iterRef);
    void southData(int size, float* ghostData, int iterRef);
    void eastData(int size, float* ghostData, int iterRef);
    void westData(int size, float* ghostData, int iterRef);

    void northData_msg(NorthSouthGhost*);
    void southData_msg(NorthSouthGhost*);
    void eastData_msg(EastWestGhost*);
    void westData_msg(EastWestGhost*);

    void attemptCalculation();
    void doCalculation();
    void doCalculation_post();
};


#endif //__JACOBI_H__

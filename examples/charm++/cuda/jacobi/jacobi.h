#ifndef __JACOBI_H__
#define __JACOBI_H__

#include "jacobi.decl.h"
#include "jacobi_config.h"


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

  // Declare CkIndex_Jacobi as a friend so the accelerated member functions
  //   can access the member variables of the class
  friend class CkIndex_Jacobi;

  private:

    /// Member Variables ///
    float* matrix;     // Pointer to the first buffer (the read buffer per iteration)
    float* matrixTmp;  // Pointer to the second buffer (the write buffer per iteration)

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

    /// Member Functions ///
    void attemptCalculation();

  public:

    /// Constructor(s) / Destructor ///
    Jacobi();
    Jacobi(CkMigrateMessage *msg);
    ~Jacobi();

    /// Entry Methods ///
    void startIteration();

    void northData(int size, float* ghostData, int iterRef);
    void southData(int size, float* ghostData, int iterRef);
    void eastData(int size, float* ghostData, int iterRef);
    void westData(int size, float* ghostData, int iterRef);

    void northData_msg(NorthSouthGhost*);
    void southData_msg(NorthSouthGhost*);
    void eastData_msg(EastWestGhost*);
    void westData_msg(EastWestGhost*);

    void doCalculation_post();
};


#endif //__JACOBI_H__

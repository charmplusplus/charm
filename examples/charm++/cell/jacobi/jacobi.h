#ifndef __JACOBI_H__
#define __JACOBI_H__

#include "jacobi.decl.h"
#include "jacobi_shared.h"


class Jacobi : public CBase_Jacobi {

  ////////////////////////////////////////////////////////////////////////////////////////////////
  //  Member Variables
  private:
    volatile float* matrix;     // Pointer to the first buffer (the read buffer per iteration)
    volatile float* matrixTmp;  // Pointer to the second buffer (the write buffer per iteration)
    int ghostCount;
    int ghostCountNeeded;
    int iterCount;


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

    void attemptCalculation();
    void doCalculation();
    void doCalculation_post();
};


#endif //__JACOBI_H__

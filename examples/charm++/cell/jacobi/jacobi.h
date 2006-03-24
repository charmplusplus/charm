#ifndef __JACOBI_H__
#define __JACOBI_H__

#include "jacobi.decl.h"
#include "jacobi_shared.h"


class Jacobi : public CBase_Jacobi {

  ////////////////////////////////////////////////////////////////////////////////////////////////
  //  Member Variables
  private:
    float* matrix;     // Pointer to the first buffer (the read buffer per iteration)
    float* matrixTmp;  // Pointer to the second buffer (the write buffer per iteration)
    int ghostCount;
    int ghostCountNeeded;


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

    void northData(int size, float* ghostData);
    void southData(int size, float* ghostData);
    void eastData(int size, float* ghostData);
    void westData(int size, float* ghostData);

    void attemptCalculation();
    void doCalculation();
};


#endif //__JACOBI_H__

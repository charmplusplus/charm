/**
   @addtogroup ConvComlibRouter
   @{
   @file 
   @brief Class for grid based routing 
*/


#ifndef _GRIDROUTER_H
#define _GRIDROUTER_H

#include <math.h>
//#include <converse.h>
#include "petable.h"

#define ROWLEN COLLEN
#define RowLen(pe) ColLen(pe)
#define PELISTSIZE ((ROWLEN-1)/sizeof(int)+1)

inline int ColLen(int npes)
{
  int len= (int)sqrt((double)npes);
  if (npes > (len*len)) len++;
  return(len);
}

inline int Expect(int pe, int npes)
{
  int i, len=ColLen(npes);
  for (i=len-1;i>=0;i--) {
  	int myrow=pe/len;
	int toprep=i*len;
	int offset=pe-myrow*len;
	if ((toprep+offset) <= (npes-1)) return(i+1);
  }
  return(len);
}

#include "persistent.h"

//Grid based router
class GridRouter : public Router
{
 private:
    PeTable *PeMesh, *PeMesh1, *PeMesh2;
    int *onerow, *gpes, 
        *rowVector, *colVector, 
        *growVector, *gcolVector;

    int myrow, mycol;
    int rvecSize, cvecSize;
    int MyPe, NumPes, COLLEN;
    int LPMsgCount, LPMsgExpected;
    int recvExpected, recvCount;
    void InitVars();
    void LocalProcMsg(comID id); 
    void sendRow(comID id);
    
 public:
    GridRouter(int, int, Strategy*);
    ~GridRouter();
    void NumDeposits(comID, int);
    
    void EachToAllMulticast(comID , int , void *, int);
    void EachToManyMulticast(comID , int , void *, int, int *, int);
    void EachToManyMulticastQ(comID id, CkQ<MessageHolder *> &msgq);
	
    void RecvManyMsg(comID, char *);
    void ProcManyMsg(comID, char *);
    void DummyEP(comID id, int);
    void ProcMsg(int, msgstruct **) {;}
    void SetMap(int *);
};

#endif

/*@}*/

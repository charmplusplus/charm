/*
Charm++ File: Checkpoint Library
added 01/03/2003 by Chao Huang, chuang10@uiuc.edu

CkStartCheckpoint() is a function to start the procedure
of saving the status of a Charm++ program into disk files.
A corresponding restarting mechanism can later use the
files saved to restore the execution. A callback should
be provided to continue after the checkpoint is done.

Checkpoint manager is a Group to aid the saving and
restarting of Charm++ programs. ...

--- Updated 12/14/2003 by Gengbin, gzheng@uiuc.edu
    rewrote to allow code reuse with following 5 functions, 
    these functions each handle both packing and unpacking of a system data:
	void CkPupROData(PUP::er &p);
	void CkPupMainChareData(PUP::er &p);
	void CkPupGroupData(PUP::er &p);
	void CkPupNodeGroupData(PUP::er &p);
	void CkPupArrayElementsData(PUP::er &p);
    Completely changed the data file format for array elements to become
    one file for each processor. 
    Two main checkpoint/restart subroutines are greatly simplified.
*/
#ifndef _CKCHECKPOINT_H
#define _CKCHECKPOINT_H

#include "CkCheckpoint.decl.h"
// loop over all CkLocMgr and do "code"
#define  CKLOCMGR_LOOP(code)	\
  for(i=0;i<numGroups;i++) {	\
    IrrGroup *obj = CkpvAccess(_groupTable)->find((*CkpvAccess(_groupIDTable))[i]).getObj();	\
    if(obj->isLocMgr())  {	\
      CkLocMgr *mgr = (CkLocMgr*)obj;	\
      code	\
    }	\
  }
/***
  * Location iterator that save each location
 ***/
void printIndex(const CkArrayIndex &idx,char *dest);

/**
 * There is only one Checkpoint Manager in the whole system
**/
class CkCheckpointMgr : public CBase_CkCheckpointMgr {
private:
	CkCallback restartCB;
	double chkptStartTimer;
public:
	CkCheckpointMgr() { }
	CkCheckpointMgr(CkMigrateMessage *m):CBase_CkCheckpointMgr(m) { }
	void Checkpoint(const char *dirname,CkCallback& cb);
	void SendRestartCB(CkReductionMsg *m);
	void pup(PUP::er& p){ CBase_CkCheckpointMgr::pup(p); p|restartCB; }
};

// utility functions to pup system global tables
void CkPupROData(PUP::er &p);
void CkPupMainChareData(PUP::er &p);
void CkPupGroupData(PUP::er &p);
void CkPupNodeGroupData(PUP::er &p);
void CkPupArrayElementsData(PUP::er &p);
void CkPupProcessorData(PUP::er &p);
void CkRemoveArrayElements();

void CkStartCheckpoint(char* dirname,const CkCallback& cb);
void CkRestartMain(const char* dirname);


#endif //_CKCHECKPOINT_H

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
*/
#ifndef _CKCHECKPOINT_H
#define _CKCHECKPOINT_H
#include "CkCheckpoint.decl.h"

/***
  * Location iterator that save each location
 ***/
void printIndex(const CkArrayIndex &idx,char *dest);
class ElementSaver : public CkLocIterator {
private:
	FILE *indexFile; //Output list of array indices and data files
	const char *dirName; //Output directory
	const int locMgrIdx;
public:
	ElementSaver(const char *dirName_,const int locMgrIdx_);
	~ElementSaver();
	void addLocation(CkLocation &loc);
};

/***
  *  Restore each array location listed in the index file
 ***/
class ElementRestorer {
private:
	FILE *indexFile; //Input list of array indices and data files
	const char *dirName; //Input directory
	CkLocMgr *dest; //Place to put new array elements
public:
	ElementRestorer(const char *dirName_,CkLocMgr *dest_);
	~ElementRestorer();
	bool restore(void);
};

/**
 * There is only one Checkpoint Manager in the whole system
**/
class CkCheckpointMgr : public IrrGroup {
private:
public:
	CkCheckpointMgr() { }
	CkCheckpointMgr(CkMigrateMessage *m):IrrGroup(m) { }
	void Checkpoint(int len, char dirname[],CkCallback& cb);

	void pup(PUP::er& p){ IrrGroup::pup(p); }
};

void CkStartCheckpoint(char* dirname,const CkCallback& cb = CkCallback(CkCallback::ignore));
void CkRestartMain(const char* dirname);

#endif //_CKCHECKPOINT_H

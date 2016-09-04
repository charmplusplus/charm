#ifndef _IRRGROUP_H_
#define _IRRGROUP_H_

#include "pup.h"
#include "charm.h"
#include "Chare.h"

//Superclass of all Groups that cannot participate in reductions.
//  Undocumented: should only be used inside Charm++.
/*forward*/ class Group;
class IrrGroup : public Chare {
  protected:
    CkGroupID thisgroup;
  public:
    IrrGroup(CkMigrateMessage *m): Chare(m) { }
    IrrGroup();
    virtual ~IrrGroup(); //<- needed for *any* child to have a virtual destructor

    virtual void pup(PUP::er &p);//<- pack/unpack routine
    virtual void ckJustMigrated(void);
    inline const CkGroupID &ckGetGroupID(void) const {return thisgroup;}
    inline CkGroupID CkGetGroupID(void) const {return thisgroup;}
    virtual int ckGetChareType() const;
    virtual char *ckDebugChareName();
    virtual int ckDebugChareID(char *, int);

    // Silly run-time type information
    virtual bool isNodeGroup() { return false; };
    virtual bool isLocMgr(void){ return false; }
    virtual bool isArrMgr(void){ return false; }
    virtual bool isReductionMgr(void){ return false; }
    static bool isIrreducible(){ return true;}
    virtual void flushStates() {}
		/*
			FAULT_EVAC
		*/
		virtual void evacuate(){};
		virtual void doneEvacuate(){};
    virtual void CkAddThreadListeners(CthThread tid, void *msg);
};
#endif

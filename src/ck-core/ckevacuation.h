/*
	FAULT_EVAC
	Resuing this file shamelessly for the Migrate away all
	objects project
*/

#ifndef _CKEVACUATION_H
#define _CKEVACUATION_H



struct evacMsg{
	char header[CmiMsgHeaderSizeBytes];
	int pe;	//processor that is being evacuated
	int remainingElements; // number of array elements that still exist on that processor
};

CpvCExtern(char *,_validProcessors);
extern int _ckEvacBcastIdx;
extern int _ckAckEvacIdx;
CkpvExtern(char ,startedEvac);
extern int allowMessagesOnly;
extern int evacuate; //Evacuate flag, set to 0 normally. set to 1 when the SIGUSR1 signal is received. after the startedEvac flag has been set it is set to 2

void _ckEvacBcast(struct evacMsg *);
void _ckAckEvac(struct evacMsg *);
void CkDecideEvacPe();
int getNextPE(const CkArrayIndex &i);
int getNextSerializer();
int CkNumValidPes();
void CkStopScheduler();
void CkEvacuatedElement();
void CkEmmigrateElement(void *arg);
void CkClearAllArrayElementsCPP();

class CkElementEvacuate : public CkLocIterator {
	int lastPE;
	public:
	CkElementEvacuate(){lastPE=0;};
	void addLocation(CkLocation &loc);
};

class CkElementInformHome : public CkLocIterator {
		void addLocation(CkLocation &loc);
};


#endif //_CK_CHECKPOINT_H

#ifndef  _MACHINE_LRTS_H_
#define  _MACHINE_LRTS_H_

void LrtsPrepareEnvelope(char *msg, int size);

/* The machine-specific send function */
CmiCommHandle LrtsSendFunc(int destNode, int size, char *msg, int mode);

#if CMK_PERSISTENT_COMM
void LrtsSendPersistentMsg(PersistentHandle h, int destPE, int size, void *m);
#endif

/* ### Beginning of Machine-startup Related Functions ### */
void LrtsInit(int *argc, char ***argv, int *numNodes, int *myNodeID);

void LrtsPreCommonInit(int everReturn);
void LrtsPostCommonInit(int everReturn);
/* ### End of Machine-startup Related Functions ### */

/* ### Beginning of Machine-running Related Functions ### */
void LrtsAdvanceCommunication();
void LrtsDrainResources(); /* used when exit */
void LrtsExit();
/* ### End of Machine-running Related Functions ### */
void LrtsPostNonLocal();

void* LrtsAlloc(int, int);
void  LrtsFree(void*);
#endif

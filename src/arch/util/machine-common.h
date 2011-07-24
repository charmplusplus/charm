#ifndef  _MACHINE_H_
#define  _MACHINE_H_
/* The machine-specific send function */
static CmiCommHandle LrtsSendFunc(int destNode, int size, char *msg, int mode);

/* ### Beginning of Machine-startup Related Functions ### */
static void LrtsInit(int *argc, char ***argv, int *numNodes, int *myNodeID);

static void LrtsPreCommonInit(int everReturn);
static void LrtsPostCommonInit(int everReturn);
/* ### End of Machine-startup Related Functions ### */

/* ### Beginning of Machine-running Related Functions ### */
static void LrtsAdvanceCommunication();
static void LrtsDrainResources(); /* used when exit */
static void LrtsExit();
/* ### End of Machine-running Related Functions ### */
static void LrtsPostNonLocal();

#endif

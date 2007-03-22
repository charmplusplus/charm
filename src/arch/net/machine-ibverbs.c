/** @file
 * Ibverbs (infiniband)  implementation of Converse NET version
 * @ingroup NET
 * contains only Ibverbs specific code for:
 * - CmiMachineInit()
 * - CmiNotifyStillIdle()
 * - DeliverViaNetwork()
 * - CommunicationServer()

  created by 
	Sayantan Chakravorty, sayantan@gmail.com ,21st March 2007
*/

/**
 * @addtogroup NET
 * @{
 */

typedef struct {
char none;  
} CmiIdleState;

static CmiIdleState *CmiNotifyGetState(void) { return NULL; }

static void CmiNotifyStillIdle(CmiIdleState *s);

static void CmiNotifyBeginIdle(CmiIdleState *s)
{
  CmiNotifyStillIdle(s);
}

void CmiNotifyIdle(void) {
  CmiNotifyStillIdle(NULL);
}


static void CmiMachineInit(char **argv)
{
}

static void CmiMachineExit()
{
}

static void CmiNotifyStillIdle(CmiIdleState *s) {
}


static void DeliverViaNetwork(OutgoingMsg ogm, OtherNode node, int rank, unsigned int broot, int copy){
}

static void CommunicationServer(int sleepTime, int where){
}

/** @file
 * Templated machine layer
 * @ingroup Machine
 *
 * This file explains what the machine layer has to provide (which functions
 * need to be implemented). Depending on the flags set in the files
 * conv-common.h, conv-mach.h and the possible other suboption file
 * conv-mach-suboption.h, some additional functions may be needed to be
 * implemented.

 * Throughout the file, "#if CMK_VARIABLE" means it was set to 1 in the .h
 * files, "#if ! CMK_VARIABLE" means it was set to 0.

/*@{*/

/*==========================================================*/
/*==========================================================*/
/*==========================================================*/

/** FUNCTIONS ALWAYS TO BE IMPLEMENTED

 * This first section of the file reports which methods must always be
 * implemented inside the machine layer.
 */ 

void ConverseInit(int, char**, CmiStartFn, int, int);
void ConverseExit(int exitcode);

void CmiAbort(const char *, ...);

void          CmiSyncSendFn(int, int, char *);
void          CmiFreeSendFn(int, int, char *);

void          CmiSyncBroadcastFn(int, char *);
void          CmiFreeBroadcastFn(int, char *);

void          CmiSyncBroadcastAllFn(int, char *);
void          CmiFreeBroadcastAllFn(int, char *);

/* Poll the network for messages */
//Different machine layers have different names for this function  

/* Poll the network and when a message arrives and insert this arrived
   message into the local queue. For SMP this message would have to be
   inserted into the thread's queue with the correct rank **/
//Pump messages is called when the processor goes idle
void PumpMessages();  

/* Free network resources when the messages have been sent out. Also
called when machine goes idle and at other places depending on the
implementation *********/
void CmiReleaseSentMessages(); 

//Called when the processor goes idle. Typically calls pump messages
//and releaseSentMessages. The idle handler has to be explicitly
//registered in ConverseInit through a call to CcdCallOnConditionKeep
void CmiNotifyIdle();


/*==========================================================*/
/*==========================================================*/
/*==========================================================*/

/************ Recommended routines ***********************/
/************ You dont have to implement these but they are supported
 in the converse syntax and some rare programs may crash. But most
 programs dont need them. *************/

CmiCommHandle CmiAsyncSendFn(int, int, char *);
CmiCommHandle CmiAsyncBroadcastFn(int, char *);
CmiCommHandle CmiAsyncBroadcastAllFn(int, char *);

int           CmiAsyncMsgSent(CmiCommHandle handle);
void          CmiReleaseCommHandle(CmiCommHandle handle);


/*==========================================================*/
/*==========================================================*/
/*==========================================================*/

//Optional routines which could use common code which is shared with
//other machine layer implementations.

/* MULTICAST/VECTOR SENDING FUNCTIONS

 * In relations to some flags, some other delivery functions may be needed.
 */

#if ! CMK_MULTICAST_LIST_USE_COMMON_CODE
void          CmiSyncListSendFn(int, const int *, int, char*);
CmiCommHandle CmiAsyncListSendFn(int, const int *, int, char*);
void          CmiFreeListSendFn(int, const int *, int, char*);
#endif

#if ! CMK_MULTICAST_GROUP_USE_COMMON_CODE
void          CmiSyncMulticastFn(CmiGroup, int, char*);
CmiCommHandle CmiAsyncMulticastFn(CmiGroup, int, char*);
void          CmiFreeMulticastFn(CmiGroup, int, char*);
#endif

#if ! CMK_VECTOR_SEND_USES_COMMON_CODE
void          CmiSyncVectorSend(int, int, int *, char **);
CmiCommHandle CmiAsyncVectorSend(int, int, int *, char **);
void          CmiSyncVectorSendAndFree(int, int, int *, char **);
#endif


/** NODE SENDING FUNCTIONS

 * If there is a node queue, and we consider also nodes as entity (tipically in
 * SMP versions), these functions are needed.
 */

#if CMK_NODE_QUEUE_AVAILABLE

void          CmiSyncNodeSendFn(int, int, char *);
CmiCommHandle CmiAsyncNodeSendFn(int, int, char *);
void          CmiFreeNodeSendFn(int, int, char *);

void          CmiSyncNodeBroadcastFn(int, char *);
CmiCommHandle CmiAsyncNodeBroadcastFn(int, char *);
void          CmiFreeNodeBroadcastFn(int, char *);

void          CmiSyncNodeBroadcastAllFn(int, char *);
CmiCommHandle CmiAsyncNodeBroadcastAllFn(int, char *);
void          CmiFreeNodeBroadcastAllFn(int, char *);

#endif


/** GROUPS DEFINITION

 * For groups of processors (establishing and managing) some more functions are
 * needed, they also con be found in common code (convcore.C) or here.
 */

#if ! CMK_MULTICAST_DEF_USE_COMMON_CODE
void     CmiGroupInit(void);
CmiGroup CmiEstablishGroup(int npes, int *pes);
void     CmiLookupGroup(CmiGroup grp, int *npes, int **pes);
#endif


/** MESSAGE DELIVERY FUNCTIONS

 * In order to deliver the messages to objects (either converse register
 * handlers, or charm objects), a scheduler is needed. The one implemented in
 * convcore.C can be used, or a new one can be implemented here. At present, all
 * machines use the default one, exept sim-linux.

 * If the one in convcore.C is used, still one function is needed.
 */

#if CMK_CMIDELIVERS_USE_COMMON_CODE /* use the default one */

CpvDeclare(void*, CmiLocalQueue);
void *CmiGetNonLocal(void);

#elif /* reimplement the scheduler and delivery */

void CsdSchedulerState_new(CsdSchedulerState_t *state);
void *CsdNextMessage(CsdSchedulerState_t *state);
int  CsdScheduler(int maxmsgs);

void CmiDeliversInit(void);
int  CmiDeliverMsgs(int maxmsgs);
void CmiDeliverSpecificMsg(int handler);

#endif


/** SHARED VARIABLES DEFINITIONS

 * In relation to which CMK_SHARED_VARS_ flag is set, different
 * functions/variables need to be defined and initialized correctly.
 */

#if CMK_SHARED_VARS_UNAVAILABLE /* Non-SMP version of shared vars. */

int _Cmi_mype;
int _Cmi_numpes;
int _Cmi_myrank; /* Normally zero; only 1 during SIGIO handling */

void CmiMemLock(void);
void CmiMemUnlock(void);

#endif

#if CMK_SHARED_VARS_POSIX_THREADS_SMP

int _Cmi_numpes;
int _Cmi_mynodesize;
int _Cmi_mynode;
int _Cmi_numnodes;

int CmiMyPe(void);
int CmiMyRank(void);
int CmiNodeFirst(int node);
int CmiNodeSize(int node);
int CmiNodeOf(int pe);
int CmiRankOf(int pe);

/* optional, these functions are implemented in "machine-smp.C", so including
   this file avoid the necessity to reimplement them.
 */
void CmiNodeBarrier(void);
void CmiNodeAllBarrier(void);
CmiNodeLock CmiCreateLock(void);
void CmiDestroyLock(CmiNodeLock lock);

#endif

/* NOT VERY USEFUL */
#if CMK_SHARED_VARS_NT_THREADS /*Used only by win32 versions*/

int _Cmi_numpes;
int _Cmi_mynodesize;
int _Cmi_mynode;
int _Cmi_numnodes;

int CmiMyPe(void);
int CmiMyRank(void);
int CmiNodeFirst(int node);
int CmiNodeSize(int node);
int CmiNodeOf(int pe);
int CmiRankOf(int pe);

/* optional, these functions are implemented in "machine-smp.C", so including
   this file avoid the necessity to reimplement them.
 */
void CmiNodeBarrier(void);
void CmiNodeAllBarrier(void);
CmiNodeLock CmiCreateLock(void);
void CmiDestroyLock(CmiNodeLock lock);

#endif


/** TIMERS DEFINITIONS

 * In relation to what CMK_TIMER_USE_ is selected, some * functions may need to
 * be implemented.
 */

/* If all the CMK_TIMER_USE_ are set to 0, the following timer functions are
   needed. */

void   CmiTimerInit(char **argv);
double CmiTimer(void);
double CmiWallTimer(void);
double CmiCpuTimer(void);
int    CmiTimerIsSynchronized(void);

/* If one of the following is set to 1, barriers are needed:
   CMK_TIMER_USE_GETRUSAGE
   CMK_TIMER_USE_RDTSC
*/

int CmiBarrier(void);
int CmiBarrierZero(void);


/** STDIO FUNCTIONS

 * Default code is provided in convcore.C but for particular machine layers they
 * can branch to custom reimplementations.

 */

#if CMK_USE_LRTS_STDIO

int LrtsPrintf(const char *, va_list);
int LrtsError(const char *, va_list);
int LrtsScanf(const char *, va_list);
int LrtsUsePrintf(void);
int LrtsUseError(void);
int LrtsUseScanf(void);

#endif


/** SPANNING TREE

 * During some working operations (such as quiescence detection), spanning trees
 * are used. Default code in convcore.C can be used, or a new definition can be
 * implemented here.
 */

#if ! CMK_SPANTREE_USE_COMMON_CODE

int      CmiNumSpanTreeChildren(int) ;
int      CmiSpanTreeParent(int) ;
void     CmiSpanTreeChildren(int node, int *children);

int      CmiNumNodeSpanTreeChildren(int);
int      CmiNodeSpanTreeParent(int) ;
void     CmiNodeSpanTreeChildren(int node, int *children) ;

#endif



/** IMMEDIATE MESSAGES

 * If immediate messages are supported, the following function is needed. There
 * is an exeption if the machine progress is also defined (see later for this).

 * Moreover, the file "immediate.C" should be included, otherwise all its
 * functions and variables have to be redefined.
*/

#if CMK_CCS_AVAILABLE

#include "immediate.C"

#if ! CMK_MACHINE_PROGRESS_DEFINED /* Hack for some machines */
void CmiProbeImmediateMsg(void);
#endif

#endif


/** MACHINE PROGRESS DEFINED

 * Some machines (like BlueGene/L) do not have coprocessors, and messages need
 * to be pulled out of the network manually. For this reason the following
 * functions are needed. Notice that the function "CmiProbeImmediateMsg" must
 * not be defined anymore.
 */

#if CMK_MACHINE_PROGRESS_DEFINED

CpvDeclare(unsigned, networkProgressCount);
int  networkProgressPeriod;

void CmiMachineProgressImpl(void);

#endif

/*****************************************************************************
 * A few useful built-in CCS handlers.
 *****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>

#include "converse.h"
#include "ckhashtable.h"
#include "pup.h"
#include "debug-charm.h"
#include "conv-ccs.h"
#include "sockRoutines.h"
#include "queueing.h"

#if CMK_CCS_AVAILABLE

/**********************************************
  "ccs_getinfo"-- takes no data
    Return the number of parallel nodes, and
      the number of processors per node as an array
      of 4-byte big-endian ints.
*/

static void ccs_getinfo(char *msg)
{
  int nNode=CmiNumNodes();
  int len=(1+nNode)*sizeof(ChMessageInt_t);
  ChMessageInt_t *table=(ChMessageInt_t *)malloc(len);
  int n;
  table[0]=ChMessageInt_new(nNode);
  for (n=0;n<nNode;n++)
    table[1+n]=ChMessageInt_new(CmiNodeSize(n));
  CcsSendReply(len,(const char *)table);
  free(table);
  CmiFree(msg);
}

/**********************************************
  "ccs_killport"-- takes one 4-byte big-endian port number
    Register a "client kill port".  When this program exits,
    it will connect to this TCP port and write "die\n\0" it.
*/

typedef struct killPortStruct{
  skt_ip_t ip;
  unsigned int port;
  struct killPortStruct *next;
} killPortStruct;
/*Only 1 kill list per node-- no Cpv needed*/
static killPortStruct *killList=NULL;

static void ccs_killport(char *msg)
{
  killPortStruct *oldList=killList;
  int port=ChMessageInt(*(ChMessageInt_t *)(msg+CmiMsgHeaderSizeBytes));
  skt_ip_t ip;
  unsigned int connPort;
  CcsCallerId(&ip,&connPort);
  killList=(killPortStruct *)malloc(sizeof(killPortStruct));
  killList->ip=ip;
  killList->port=port;
  killList->next=oldList;
  CmiFree(msg);
}
/*Send any registered clients kill messages before we exit*/
static int noMoreErrors(int c,const char *m) {return -1;}
extern "C" void CcsImpl_kill(void)
{
  skt_set_abort(noMoreErrors);
  while (killList!=NULL)
  {
    SOCKET fd=skt_connect(killList->ip,killList->port,20);
    if (fd!=INVALID_SOCKET) {
      skt_sendN(fd,"die\n",strlen("die\n")+1);
      skt_close(fd);
    }
    killList=killList->next;
  }
}


/*************************************************
List interface:
   This lets different parts of a Charm++ program register 
"list retrieval functions", which expose some aspect of a
running program.  Example lists include: all readonly globals, 
messages waiting in a queue, or all extant array elements.

  "ccs_list_len" (4-byte character count, null-terminated ASCII string)
Return the number of items currently in the list at the given path.

  "ccs_list_items.bin" (4-byte start, 4-byte end+1,4-byte extra
data count n, n-byte extra data, 4-byte character count c, c-byte
ASCII request string [without NULL terminator])
Return the given items (from start to end) of the given queue, 
formatted as raw network binary data.

  "ccs_list_items.fmt" [parameters as for .bin]
Return the given items, formatted as tagged network binary data.

  "ccs_list_items.txt" [parameters as for .bin]
Return the given items, formatted as ASCII text.
*/

class CpdListAccessor_c : public CpdListAccessor {
  const char *path; //Path to this item
  CpdListLengthFn_c lenFn;
  void *lenParam;
  CpdListItemsFn_c itemsFn;
  void *itemsParam;
public:
  CpdListAccessor_c(const char *path_,
            CpdListLengthFn_c lenFn_,void *lenParam_,
            CpdListItemsFn_c itemsFn_,void *itemsParam_)
  {
      path=path_;
      lenFn=lenFn_;
      lenParam=lenParam_;
      itemsFn=itemsFn_;
      itemsParam=itemsParam_;
  }
  
  virtual const char *getPath(void) const {return path;}
  virtual int getLength(void) const {return (*lenFn)(lenParam);}
  virtual void pup(PUP::er &p,CpdListItemsRequest &req) {
    (itemsFn)(itemsParam,(pup_er *)&p,&req);
  }
};

static void CpdListBoundsCheck(CpdListAccessor *l,int &lo,int &hi)
{
    int len=l->getLength();
    if (lo<0) lo=0;
    if (hi>len) hi=len;  
}

typedef CkHashtableTslow<const char *,CpdListAccessor *> CpdListTable_t;
CpvStaticDeclare(CpdListTable_t *,cpdListTable);

static CpdListAccessor *CpdListLookup(const char *path)
{
  CpdListAccessor *acc=CpvAccess(cpdListTable)->get(path);
  if (acc==NULL) {
    CmiError("CpdListAccessor> Unrecognized list path '%s'\n",path);
    return NULL;
  }
  return acc;
}

static const int CpdListMaxLen=80;
static CpdListAccessor *CpdListLookup(const ChMessageInt_t *lenAndPath)
{
  int len=ChMessageInt(lenAndPath[0]);
  const char *path=(const char *)(lenAndPath+1);
  char pathBuf[CpdListMaxLen+1]; //Temporary null-termination buffer
  if ((len<0) || (len>CpdListMaxLen)) {
    CmiError("CpdListAccessor> Invalid list path length %d!\n",len);
    return NULL; //Character count makes no sense
  }
  strncpy(pathBuf,path,len);
  pathBuf[len]=0; //Ensure string is null-terminated
  return CpdListLookup(pathBuf);
}

//CCS External access routines:

//Get the length of the given list:
static void CpdList_ccs_list_len(char *msg)
{
  const ChMessageInt_t *req=(const ChMessageInt_t *)(msg+CmiMsgHeaderSizeBytes);
  CpdListAccessor *acc=CpdListLookup(req);
  if (acc!=NULL) {
    ChMessageInt_t reply=ChMessageInt_new(acc->getLength());
    CcsSendReply(sizeof(reply),(void *)&reply);
  }
  CmiFree(msg);
}

//Read a list contents request header:
//  first item to send, 4-byte network integer
//  last item+1 to send, 4-byte network integer
//  extra data length, 4-byte network integer
//  extra data, list-defined bytes
//  list path length, 4-byte network integer (character count)
//  list path name, null-terminated ASCII
static CpdListAccessor *CpdListHeader_ccs_list_items(char *msg,
	     CpdListItemsRequest &h)
{
  int msgLen=CmiSize((void *)msg)-CmiMsgHeaderSizeBytes;
  CpdListAccessor *ret=NULL;
  const ChMessageInt_t *req=(const ChMessageInt_t *)(msg+CmiMsgHeaderSizeBytes);
  h.lo=ChMessageInt(req[0]);
  h.hi=ChMessageInt(req[1]);
  h.extraLen=ChMessageInt(req[2]);
  if (h.extraLen>=0 && ((int)(3*sizeof(ChMessageInt_t)+h.extraLen))<msgLen) {
    h.extra=(void *)(req+3);
    ret=CpdListLookup((ChMessageInt_t *)(h.extraLen+(char *)h.extra));
    if (ret!=NULL) CpdListBoundsCheck(ret,h.lo,h.hi);
  }
  return ret;
}

static void CpdList_ccs_list_items_txt(char *msg)
{
  CpdListItemsRequest req;
  CpdListAccessor *acc=CpdListHeader_ccs_list_items(msg,req);
  if (acc!=NULL) {
    int bufLen;
    { 
      PUP::sizerText p; 
      acc->pup(p,req);
      bufLen=p.size(); 
    }
    char *buf=new char[bufLen];
    { 
      PUP::toText p(buf); 
      acc->pup(p,req); 
      if (p.size()!=bufLen)
	CmiError("ERROR! Sizing/packing length mismatch for %s list pup function!\n",
		acc->getPath());
    }
    CcsSendReply(bufLen,(void *)buf);
  }
  CmiFree(msg);
}

//Introspection object
class CpdList_introspect : public CpdListAccessor {
  CpdListTable_t *tab;
public:
  CpdList_introspect(CpdListTable_t *tab_) :tab(tab_) { }
  virtual const char *getPath(void) const { return "converse/lists";}
  virtual int getLength(void) const {
    int len=0;
    CkHashtableIterator *it=tab->iterator();
    while (NULL!=it->next()) len++;
    delete it;
    return len;
  }
  virtual void pup(PUP::er &p,CpdListItemsRequest &req) {
    CkHashtableIterator *it=tab->iterator();
    void *objp;
    int curObj=0;
    while (NULL!=(objp=it->next())) {
      if (curObj>=req.lo && curObj<req.hi) {
	CpdListAccessor *acc=*(CpdListAccessor **)objp;
	char *pathName=(char *)acc->getPath();
	beginItem(p,curObj);
	p(pathName,strlen(pathName));
      }
      curObj++;
    }
  }
};

#endif /*CMK_CCS_AVAILABLE*/
/*We have to include these virtual functions, even when CCS is
disabled, to avoid bizarre link-time errors.*/

CpdListAccessor::~CpdListAccessor() { }
CpdSimpleListAccessor::~CpdSimpleListAccessor() { }
const char *CpdSimpleListAccessor::getPath(void) const {return path;}
int CpdSimpleListAccessor::getLength(void) const {return length;}
void CpdSimpleListAccessor::pup(PUP::er &p,CpdListItemsRequest &req) 
{
	for (int i=req.lo;i<req.hi;i++) {
		beginItem(p,i);
		(*pfn)(p,i);
	}
}

static void CpdListBeginItem_impl(PUP::er &p,int itemNo)
{
	p.synchronize(0x7137FACEu);
	p.comment("---------- Next list item: ----------");
	p(itemNo);
}

extern "C" void CpdListBeginItem(pup_er p,int itemNo)
{
  CpdListBeginItem_impl(*(PUP::er *)p,itemNo);
}

void CpdListAccessor::beginItem(PUP::er &p,int itemNo)
{
  CpdListBeginItem_impl(p,itemNo);
}

// C++ and C client API
void CpdListRegister(CpdListAccessor *acc)
#if CMK_CCS_AVAILABLE
{
  CpvAccess(cpdListTable)->put(acc->getPath())=acc;
}
#else
{ }
#endif

extern "C" void CpdListRegister_c(const char *path,
            CpdListLengthFn_c len,void *lenParam,
            CpdListItemsFn_c items,void *itemsParam)
#if CMK_CCS_AVAILABLE
{
  CpdListRegister(new CpdListAccessor_c(path,
	     len,lenParam,items,itemsParam));
}
#else
{ }
#endif

#if CMK_CCS_AVAILABLE
// Initialization	
static void CpdListInit(void) {
  CpvInitialize(CpdListTable_t *,cpdListTable);
  CpvAccess(cpdListTable)=new CpdListTable_t(31,0.5,
	      CkHashFunction_string,CkHashCompare_string);
  CpdListRegister(new CpdList_introspect(CpvAccess(cpdListTable)));

  CcsRegisterHandler("ccs_list_len",(CmiHandler)CpdList_ccs_list_len);
  CcsRegisterHandler("ccs_list_items.txt",(CmiHandler)CpdList_ccs_list_items_txt);
}

#if CMK_WEB_MODE
/******************************************************
Web performance monitoring interface:
	Clients will register for performance data with 
processor 0.  Every WEB_INTERVAL (few seconds), this code
calls all registered web performance functions on all processors.  
The resulting integers are sent back to the client.  The current
reply format is ASCII and rather nasty. 

The actual call sequence is:
CCS Client->CWebHandler->...  (processor 0)
  ...->CWeb_Collect->... (all processors)
...->CWeb_Reduce->CWeb_Deliver (processor 0 again)
*/

#if 0
#  define WEBDEBUG(x) CmiPrintf x
#else
#  define WEBDEBUG(x) /*empty*/
#endif

#define WEB_INTERVAL 1000 /*Time, in milliseconds, between performance snapshots*/
#define MAXFNS 20 /*Largest number of performance functions to expect*/

typedef struct {
	char hdr[CmiMsgHeaderSizeBytes];
	int fromPE;/*Source processor*/
	int perfData[MAXFNS];/*Performance numbers*/
} CWeb_CollectedData;

/*This needs to be made into a list of registered clients*/
static int hasApplet=0;
static CcsDelayedReply appletReply;

typedef int (*CWebFunction)(void);
static CWebFunction CWebPerformanceFunctionArray[MAXFNS];
static int CWebNoOfFns;
static int CWeb_ReduceIndex;
static int CWeb_CollectIndex;

/*Deliver the reduced web performance data to the waiting client:
*/
static int collectedCount;
static CWeb_CollectedData **collectedValues;

static void CWeb_Deliver(void)
{
  int i,j;

  if (hasApplet) {
    WEBDEBUG(("CWeb_Deliver to applet\n"));
    /*Send the performance data off to the applet*/
    char *reply=(char *)malloc(6+14*CmiNumPes()*CWebNoOfFns);
    sprintf(reply,"perf");
  
    for(i=0; i<CmiNumPes(); i++){
      for (j=0;j<CWebNoOfFns;j++)
      {
        char buf[20];
        sprintf(buf," %d",collectedValues[i]->perfData[j]);
        strcat(reply,buf);
      }
    }
    CcsSendDelayedReply(appletReply,strlen(reply) + 1, reply);
    free(reply);
    hasApplet=0;
  }
  else
    WEBDEBUG(("CWeb_Deliver (NO APPLET)\n"));
  
  /* Free saved performance data */
  for(i = 0; i < CmiNumPes(); i++){
    CmiFree(collectedValues[i]);
    collectedValues[i] = 0;
  }
  collectedCount = 0;
}

/*On PE 0, this handler accumulates all the performace data
*/
static void CWeb_Reduce(void *msg){
  CWeb_CollectedData *cur,*prev;
  int src;
  if(CmiMyPe() != 0){
    CmiAbort("CWeb performance data sent to wrong processor...\n");
  }
  WEBDEBUG(("CWeb_Reduce"));
  cur=(CWeb_CollectedData *)msg;
  src=cur->fromPE;
  prev = collectedValues[src]; /* Previous value, ideally 0 */
  collectedValues[src] = cur;
  if(prev == 0) collectedCount++;
  else CmiFree(prev); /*<- caused by out-of-order perf. data delivery*/

  if(collectedCount == CmiNumPes()){
    CWeb_Deliver();
  }
}

/*On each PE, this handler collects the performance data
and sends it to PE 0.
*/
static void CWeb_Collect(void)
{
  CWeb_CollectedData *msg;
  int i;

  WEBDEBUG(("CWeb_Collect on %d\n",CmiMyPe()));
  msg = (CWeb_CollectedData *)CmiAlloc(sizeof(CWeb_CollectedData));
  msg->fromPE = CmiMyPe();
  
  /* Evaluate each performance function*/
  for(i = 0; i < CWebNoOfFns; i++)
    msg->perfData[i] = CWebPerformanceFunctionArray[i] ();

  /* Send result off to node 0 */  
  CmiSetHandler(msg, CWeb_ReduceIndex);
  CmiSyncSendAndFree(0, sizeof(CWeb_CollectedData), msg);

  /* Re-call this function after a delay */
  CcdCallFnAfter((CcdVoidFn)CWeb_Collect, 0, WEB_INTERVAL);
}

extern "C" void CWebPerformanceRegisterFunction(CWebFunction fn)
{
  if (CmiMyRank()!=0) return; /* Should only register from rank 0 */
  if (CWebNoOfFns>=MAXFNS) CmiAbort("Registered too many CWebPerformance functions!");
  CWebPerformanceFunctionArray[CWebNoOfFns] = fn;
  CWebNoOfFns++;
}

/*This is called on PE 0 by clients that wish
to receive performance data.
*/
static void CWebHandler(void){
  if(CcsIsRemoteRequest()) {
    static int startedCollection=0;
    
    WEBDEBUG(("CWebHandler request on %d\n",CmiMyPe()));    
    hasApplet=1;
    appletReply=CcsDelayReply();
    
    if(startedCollection == 0){
      WEBDEBUG(("Starting data collection on every processor\n"));    
      int i;
      startedCollection=1;
      collectedCount=0;
      collectedValues = (CWeb_CollectedData **)malloc(sizeof(void *) * CmiNumPes());
      for(i = 0; i < CmiNumPes(); i++)
        collectedValues[i] = 0;
      
      /*Start collecting data on each processor*/
      for(i = 0; i < CmiNumPes(); i++){
        char *msg = (char *)CmiAlloc(CmiMsgHeaderSizeBytes);
        CmiSetHandler(msg, CWeb_CollectIndex);
        CmiSyncSendAndFree(i, CmiMsgHeaderSizeBytes,msg);
      }
    }
  }
}

/** This "usage" section keeps track of percent of wall clock time
spent actually processing messages on each processor.   
It's a simple performance measure collected by the CWeb framework.
**/

CpvStaticDeclare(double, startTime);
CpvStaticDeclare(double, beginTime);
CpvStaticDeclare(double, usedTime);
CpvStaticDeclare(int, PROCESSING);

/* Called when processor becomes busy
*/
static void usageStart(void)
{
   if(CpvAccess(PROCESSING)) return;

   CpvAccess(startTime)  = CmiWallTimer();
   CpvAccess(PROCESSING) = 1;
}

/* Called when processor becomes idle
*/
static void usageStop(void)
{
   if(!CpvAccess(PROCESSING)) return;

   CpvAccess(usedTime)   += CmiWallTimer() - CpvAccess(startTime);
   CpvAccess(PROCESSING) = 0;
}

/* Call this when the program is started
 -> Whenever traceModuleInit would be called
 -> -> see conv-core/convcore.c
*/
static void initUsage()
{
   CpvInitialize(double, startTime);
   CpvInitialize(double, beginTime);
   CpvInitialize(double, usedTime);
   CpvInitialize(int, PROCESSING);
   CpvAccess(startTime)  = CmiWallTimer();
   CpvAccess(beginTime)  = CmiWallTimer();
   CpvAccess(usedTime)   = 0.;
   CpvAccess(PROCESSING) = 1;
   CcdCallOnConditionKeep(CcdPROCESSOR_BEGIN_BUSY,(CcdVoidFn)usageStart,0);
   CcdCallOnConditionKeep(CcdPROCESSOR_BEGIN_IDLE,(CcdVoidFn)usageStop,0);      
}

static int getUsage(void)
{
   int usage = 0;
   double time      = CmiWallTimer();
   double totalTime = time - CpvAccess(beginTime);

   if(CpvAccess(PROCESSING))
   {
      CpvAccess(usedTime) += time - CpvAccess(startTime);
      CpvAccess(startTime) = time;
   }
   if(totalTime > 0.)
      usage = (int)((100 * CpvAccess(usedTime))/totalTime);
   CpvAccess(usedTime)  = 0.;
   CpvAccess(beginTime) = time;

   return usage;
}

static int getSchedQlen(void)
{
  return(CqsLength((Queue)CpvAccess(CsdSchedQueue)));
}

void CWebInit(void)
{
  CcsRegisterHandler("perf_monitor", (CmiHandler)CWebHandler);
  
  CWeb_CollectIndex=CmiRegisterHandler((CmiHandler)CWeb_Collect);
  CWeb_ReduceIndex=CmiRegisterHandler((CmiHandler)CWeb_Reduce);
  
  initUsage();
  CWebPerformanceRegisterFunction(getUsage);
  CWebPerformanceRegisterFunction(getSchedQlen);

}

#endif /*CMK_WEB_MODE*/


extern "C" void CcsBuiltinsInit(char **argv)
{
  CcsRegisterHandler("ccs_getinfo",(CmiHandler)ccs_getinfo);
  CcsRegisterHandler("ccs_killport",(CmiHandler)ccs_killport);
#if CMK_WEB_MODE
  CWebInit();
#endif
  CpdListInit();
}


#endif /*CMK_CCS_AVAILABLE*/












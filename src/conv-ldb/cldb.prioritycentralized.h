#include "cldb.h"
/* for sqrt() */
#include <math.h>


typedef struct CldNeighborData
{
  int pe, load;
} *CldNeighborData;

typedef struct loadmsg_s {
  char header[CmiMsgHeaderSizeBytes];
  int pe, load;
} loadmsg;


/* work request message when idle */
typedef struct requestmsg_s {
  char header[CmiMsgHeaderSizeBytes];
  int from_pe;
  unsigned int priority;
  int notidle;
} requestmsg;

typedef struct readytoexectoken_s{
    unsigned int priority;
    void *msg;
} readytoexectoken;
/******************* Yanhua seed load balancer */

typedef struct CldProcInfo_s {
  double lastCheck;
  int    sent;			/* flag to disable idle work request */
  int    balanceEvt;		/* user event for balancing */
  int    idleEvt;		/* user event for idle balancing */
  int    idleprocEvt;		/* user event for processing idle req */
  int   load;
} *CldProcInfo;

/* this is used by master to store the highest priority for each processor */
typedef struct CldProcPriorInfo_s {
  int   pe;
  int   priority;
} *CldProcPriorInfo;

typedef struct CldSlavePriorInfo_s {
    int pe;
    double average_priority;
    //int priority_1;
    //int priority_2;
    int load;
} CldSlavePriorInfo;


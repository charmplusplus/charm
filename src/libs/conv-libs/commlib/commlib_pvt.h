#ifndef COMLIB_PVT_H
#define COMLIB_PVT_H

#include <stdio.h>
#include <string.h>
#include <converse.h>


#define PRC 0
#define RCV 1

typedef struct { 
    char core[CmiMsgHeaderSizeBytes];
    comID id;
    int magic;
    int refno;
} DummyMsg ;

typedef struct { 
    char core[CmiMsgHeaderSizeBytes];
    comID id;
} GMsg ;

typedef struct { 
    char core[CmiMsgHeaderSizeBytes];
    comID id;
    int val;
    int impltype;
} SwitchMsg ;

void KRecvManyCombinedMsg(char *msg);
void KProcManyCombinedMsg(char *msg);
void KDummyEP(DummyMsg *m);
void KSwitchEP(SwitchMsg *m);
void KDoneEP(DummyMsg *m);
void KsendGmsg(comID id);
void KGMsgHandler(GMsg *);

#endif
	

#ifndef __CHARM_PROJECTIONS_H__
#define __CHARM_PROJECTIONS_H__

// forward declaration
class envelope;

#ifdef __cplusplus
extern "C" {
#endif

void initCharmProjections();
//int  traceRegisterUserEvent(const char*);	//TODO

void charm_creation(envelope *e, int ep, int num=1);
void charm_beginExecute(envelope *e);
void charm_beginExecuteDetailed(int event,int msgType,int ep,int srcPe,int ml);
void charm_endExecute(void);
void charm_enqueueMsg(envelope *e);
void charm_dequeueMsg(envelope *e);
void charm_beginComputation(void);
void charm_endComputation(void);
void charm_messageRecv(char *env, int pe);
void charm_userEvent(int e);
void charm_userPairEvent(int e,double bt,double et);
void charm_beginPack(void);
void charm_endPack(void);
void charm_beginUnpack(void);
void charm_endUnpack(void);

#ifdef __cplusplus
}
#endif

#endif

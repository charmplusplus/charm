
#ifndef __CHARM_PROJECTIONS_H__
#define __CHARM_PROJECTIONS_H__

// forward declaration
class envelope;

#ifdef __cplusplus
extern "C" {
#endif

void initCharmProjections();
//int  traceRegisterUserEvent(const char*);	//TODO

void creation(envelope *e, int num=1);
void beginExecute(envelope *e);
void beginExecuteDetailed(int event,int msgType,int ep,int srcPe,int ml);
void endExecute(void);
void enqueue(envelope *e);
void dequeue(envelope *e);
void beginComputation(void);
void endComputation(void);
void messageRecv(char *env, int pe);
void userEvent(int e);
void beginPack(void);
void endPack(void);
void beginUnpack(void);
void endUnpack(void);

#ifdef __cplusplus
}
#endif

#endif

#include "chare.h"

CpvExtern(int, CtrRecdTraceMsg);
CpvExtern(CthThread, cThread);
CpvExtern(int, traceOn);

void traceModuleInit(pargc, argv) int *pargc; char **argv; 
{
  CpvInitialize(CthThread, cThread);
  CpvInitialize(int, traceOn);
  CpvAccess(traceOn) = 0;
}

void program_name(s,m) char *s, *m; {}

void log_init(void) {CpvAccess(CtrRecdTraceMsg) = 1;}

void trace_user_event(int eventNum)
{}

void trace_creation(msg_type,entry,envelope)
int msg_type, entry;
void *envelope;
{}

void trace_begin_execute(envelope)
void *envelope;
{}

void trace_end_execute(id,msg_type,entry)
int id, msg_type, entry;
{}

void trace_begin_charminit(void)
{}

void trace_end_charminit(void)
{}

void trace_begin_idle(void)
{}

void trace_end_idle(void)
{}

void trace_begin_pack(void){}
void trace_end_pack(void){}
void trace_begin_unpack(void){}
void trace_end_unpack(void){}

void trace_begin_computation(void)
{}

void trace_enqueue(envelope)
void *envelope;
{}

void trace_dequeue(envelope)
void *envelope;
{}

void trace_table(type,tbl,key,pe)
int type,tbl,key,pe;
{}


void send_log(void) {}

void CollectTraceFromNodes(msg, data)
char msg, data;
{}

void close_log(void){}

void PrintStsFile(str)
char *str ;
{
}

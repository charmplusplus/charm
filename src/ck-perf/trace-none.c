#include "chare.h"

CpvExtern(int, RecdTraceMsg);

void traceModuleInit(pargc, argv) int *pargc; char **argv; {}

program_name(s,m) char *s, *m; {}

log_init() {CpvAccess(RecdTraceMsg) = 1;}

trace_user_event(int eventNum)
{}

trace_creation(msg_type,entry,envelope)
int msg_type, entry;
void *envelope;
{}

trace_begin_execute(envelope)
void *envelope;
{}

trace_end_execute(id,msg_type,entry)
int id, msg_type, entry;
{}

trace_begin_charminit()
{}

trace_end_charminit()
{}

trace_begin_idle()
{}

trace_end_idle()
{}

trace_begin_computation()
{}

trace_enqueue(envelope)
void *envelope;
{}

trace_dequeue(envelope)
void *envelope;
{}

trace_table(type,tbl,key,pe)
int type,tbl,key,pe;
{}


send_log() {}

CollectTraceFromNodes(msg, data)
char msg, data;
{}

close_log(){}

void PrintStsFile(str)
char *str ;
{
}

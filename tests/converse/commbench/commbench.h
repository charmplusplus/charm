#ifndef COMMBENCH_H
#define COMMBENCH_H
#include <converse.h>

CpvExtern(int, ack_handler);

typedef struct EmptyMsg { char core[CmiMsgHeaderSizeBytes]; } EmptyMsg;

#endif

#ifndef COMMBENCH_H
#define COMMBENCH_H

CpvExtern(int, ack_handler);

typedef struct EmptyMsg {
  char core[CmiMsgHeaderSizeBytes];
} EmptyMsg;

#endif

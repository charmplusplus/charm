#include <converse.h>
#include <stdio.h>

void Cpm_megacon_ack(CpmDestination);

void handler_init() { Cpm_megacon_ack(CpmSend(0)); }

static void handler_dummy(void* msg) { CmiFree(msg); }

void handler_moduleinit() {
  int i, dummy_idx;
  for (i = 0; i < 300; i++) dummy_idx = CmiRegisterHandler(handler_dummy);
}

#include "converse.h"

// Charm4py references these symbols unconditionally. Reconverse builds have
// CMK_CCS_AVAILABLE=0, so registration is intentionally a no-op and attempting
// to reply to a CCS request is an error.
extern "C" void CcsRegisterHandlerExt(const char *, void *) {}

extern "C" void CcsSendReply(int, const void *) {
  CmiAbort("CcsSendReply is unavailable in a Reconverse build");
}

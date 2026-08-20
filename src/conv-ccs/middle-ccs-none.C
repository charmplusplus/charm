/*
Stand-in for middle-ccs.C in builds that have no Converse debugger.

middle-ccs.C carries two unrelated things: the CCS request/reply forwarding
handlers, and the parallel debugger's freeze machinery. The debugger half is
written against Converse internals Reconverse does not have, and the forwarding
half uses CmiListReduce for multicast replies, which it also does not have.

What remains here is what a control channel needs: forward a request to the PE
it is addressed to, and send a reply straight back. Broadcast and multicast CCS
requests are refused rather than silently mishandled. Sending a request to one
PE, which is how the shrink/expand control path talks to PE 0, works exactly as
it does under Converse.
*/

#include <unistd.h>
#include "middle.h"

#include "ccs-server.h"
#include "conv-ccs.h"

#if CMK_CCS_AVAILABLE

void CcsHandleRequest(CcsImplHeader *hdr, const char *reqData);

extern int rep_fw_handler_idx;

void req_fw_handler(char *msg)
{
  const int offset = CmiReservedHeaderSize + sizeof(CcsImplHeader);
  CcsImplHeader *hdr = (CcsImplHeader *)(msg + CmiReservedHeaderSize);
  const int destPE = (int)ChMessageInt(hdr->pe);

  if (destPE < 0) {
    CmiAbort("CCS> broadcast and multicast requests need the reduction "
             "support this build does not have; address the request to a "
             "single PE\n");
  }

#if CMK_SMP
  /* The request arrived on the CCS PE and is addressed elsewhere on the node. */
  if (destPE != CmiMyPe()) {
    int len = CmiReservedHeaderSize + sizeof(CcsImplHeader) +
              ChMessageInt(hdr->len);
    CmiSyncSend(destPE, len, msg);
    CmiFree(msg);
    return;
  }
#endif

  CcsHandleRequest(hdr, msg + offset);
  CmiFree(msg);
}

int CcsReply(CcsImplHeader *rep, int repLen, const void *repData)
{
  const int repPE = (int)ChMessageInt(rep->pe);
  if (repPE <= -1) {
    CmiAbort("CCS> a reply to a broadcast or multicast request needs the "
             "reduction support this build does not have\n");
  }
  CcsImpl_reply(rep, repLen, repData);
  return 0;
}

void ccs_getinfo(char *msg)
{
  int nNode = CmiNumNodes();
  int len = (1 + nNode) * sizeof(ChMessageInt_t);
  ChMessageInt_t *table = (ChMessageInt_t *)malloc(len);
  table[0] = ChMessageInt_new(nNode);
  for (int n = 0; n < nNode; n++) table[1 + n] = ChMessageInt_new(CmiNodeSize(n));
  CcsSendReply(len, (const char *)table);
  free(table);
  CmiFree(msg);
}

#endif /* CMK_CCS_AVAILABLE */

/* The debugger's freeze state. Charm++ calls these from its abort and signal
   paths whether or not a debugger could ever attach, so they have to exist. */
CpvCExtern(void *, debugQueue);
CpvDeclare(void *, debugQueue);
CpvCExtern(int, freezeModeFlag);
CpvDeclare(int, freezeModeFlag);

void CpdFreeze(void) {}
void CpdUnFreeze(void) {}
int CpdIsFrozen(void) { return 0; }

#include <stdarg.h>
void CpdNotify(int type, ...) { (void)type; }

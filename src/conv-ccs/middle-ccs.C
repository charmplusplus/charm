#include "converse.h"

#if CMK_BLUEGENE_CHARM
#include "bgconverse.h"
#endif
#include "ccs-server.h"

extern "C" void CcsHandleRequest(CcsImplHeader *hdr,const char *reqData);

extern "C" void req_fw_handler(char *msg)
{
  int offset = CmiReservedHeaderSize + sizeof(CcsImplHeader);
  CcsImplHeader *hdr = (CcsImplHeader *)(msg+CmiReservedHeaderSize);
  int destPE = (int)ChMessageInt(hdr->pe);
  if (CmiMyPe() == 0 && destPE == -1) {
    /* Broadcast message to all other processors */
    int len=CmiReservedHeaderSize+sizeof(CcsImplHeader)+ChMessageInt(hdr->len);
    CmiSyncBroadcast(len, msg);
  }
  else if (destPE < -1) {
    /* Multicast the message to your children */
    int len=CmiReservedHeaderSize+sizeof(CcsImplHeader)+ChMessageInt(hdr->len)-destPE*sizeof(ChMessageInt_t);
    int index, child, i;
    int *pes = (int*)(msg+CmiReservedHeaderSize+sizeof(CcsImplHeader));
    ChMessageInt_t *pes_nbo = (ChMessageInt_t *)pes;
    offset -= destPE * sizeof(ChMessageInt_t);
    if (ChMessageInt(pes_nbo[0]) == CmiMyPe()) {
      for (index=0; index<-destPE; ++index) pes[index] = ChMessageInt(pes_nbo[index]);
    }
    for (index=0; index<-destPE; ++index) {
      if (pes[index] == CmiMyPe()) break;
    }
    child = (index << 2) + 1;
    for (i=0; i<4; ++i) {
      if (child+i < -destPE) {
        CmiSyncSend(pes[child+i], len, msg);
      }
    }
  }
  CcsHandleRequest(hdr, msg+offset);
  CmiFree(msg);
}

/*****************************************************************************
     Blue Gene Converse Layer
     Converse function should be implemented based on the functions here
*****************************************************************************/

#ifndef  _BLUE_CONV_H_
#define  _BLUE_CONV_H_

#include <memory.h>
#include "converse.h"

#if CMK_BIGSIM_NODE
/**
  This version Blue Gene Charm++ use a whole Blue Gene node as 
  a Charm PE.
*/
static inline void BgSyncSend(int pe, int nb, char *m) 
{
  int x,y,z;
  char *dupm = (char *)CmiAlloc(nb);

  memcpy(dupm, m, nb);
  BgGetXYZ(pe, &x, &y, &z);
  BgSendPacket(x,y,z, ANYTHREAD, CmiGetHandler(m), LARGE_WORK, nb, dupm);
}

static inline void BgSyncSendAndFree(int pe, int nb, char *m)
{
  int x,y,z;
  BgGetXYZ(pe, &x, &y, &z);
  BgSendPacket(x,y,z, ANYTHREAD, CmiGetHandler(m), LARGE_WORK, nb, m);
}

static inline void BgSyncBroadcastAndFree(int nb, char *m)
{
  BgBroadcastPacketExcept(BgMyNode(), ANYTHREAD, CmiGetHandler(m), LARGE_WORK, nb, m);
}

static inline void BgSyncBroadcast(int nb, char *m)
{
  char *dupm = (char *)CmiAlloc(nb);
  memcpy(dupm, m, nb);
  BgSyncBroadcastAndFree(nb, dupm);
}

static inline void BgSyncBroadcastAll(int nb, char *m)
{
  char *dupm = (char *)CmiAlloc(nb);
  memcpy(dupm, m, nb);
  BgBroadcastAllPacket(CmiGetHandler(m), LARGE_WORK, nb, dupm);
}

static inline void BgSyncBroadcastAllAndFree(int nb, char *m)
{
  /* broadcast to all nodes */
  BgBroadcastAllPacket(CmiGetHandler(m), LARGE_WORK, nb, m);
}

#else	 /* CMK_BIGSIM_NODE */
/**
  This version of Blue Gene Charm++ use a Blue Gene thread as 
  a Charm PE.
*/
#ifndef __cplusplus
#define inline 
#endif
static inline void BgSyncSendAndFree(int pe, int nb, char *m)
{
  int x,y,z,t;
  t = pe%BgGetNumWorkThread();
  pe = pe/BgGetNumWorkThread();
  BgGetXYZ(pe, &x, &y, &z);
  BgSendPacket(x,y,z, t, CmiGetHandler(m), LARGE_WORK, nb, m);
}

static inline void BgSyncSend(int pe, int nb, char *m) 
{
  char *dupm = (char *)CmiAlloc(nb);
  memcpy(dupm, m, nb);
  BgSyncSendAndFree(pe, nb, dupm);
}

static inline void BgSyncBroadcastAndFree(int nb, char *m)
{
  BgThreadBroadcastPacketExcept(BgMyNode(), BgGetThreadID(), CmiGetHandler(m), LARGE_WORK, nb, m);
}

static inline void BgSyncBroadcast(int nb, char *m)
{
  char *dupm = (char *)CmiAlloc(nb);
  memcpy(dupm, m, nb);
  BgSyncBroadcastAndFree(nb, dupm);
}

static inline void BgSyncBroadcastAllAndFree(int nb, char *m)
{
  /* broadcast to all nodes */
  BgThreadBroadcastAllPacket(CmiGetHandler(m), LARGE_WORK, nb, m);
}

static inline void BgSyncBroadcastAll(int nb, char *m)
{
  char *dupm = (char *)CmiAlloc(nb);
  memcpy(dupm, m, nb);
  BgSyncBroadcastAllAndFree(nb, dupm);
}

static inline void BgSyncNodeSendAndFree(int node, int nb, char *m)
{
  int x,y,z;
  BgGetXYZ(node, &x, &y, &z);
  BgSendPacket(x,y,z, ANYTHREAD, CmiGetHandler(m), LARGE_WORK, nb, m);
}

static inline void BgSyncNodeSend(int node, int nb, char *m)
{
  char *dupm = (char *)CmiAlloc(nb);
  memcpy(dupm, m, nb);
  BgSyncNodeSendAndFree(node, nb, dupm);
}

static inline void BgSyncNodeBroadcastAndFree(int nb, char *m)
{
  BgBroadcastPacketExcept(BgMyNode(), ANYTHREAD, CmiGetHandler(m), LARGE_WORK, nb, m);
}

static inline void BgSyncNodeBroadcast(int nb, char *m)
{
  char *dupm = (char *)CmiAlloc(nb);
  memcpy(dupm, m, nb);
  BgSyncNodeBroadcastAndFree(nb, dupm);
}

static inline void BgSyncNodeBroadcastAllAndFree(int nb, char *m)
{
  /* broadcast to all nodes */
  BgBroadcastAllPacket(CmiGetHandler(m), LARGE_WORK, nb, m);
}

static inline void BgSyncNodeBroadcastAll(int nb, char *m)
{
  char *dupm = (char *)CmiAlloc(nb);
  memcpy(dupm, m, nb);
  BgSyncNodeBroadcastAllAndFree(nb, dupm);
}

static inline void BgSyncListSendAndFree(int npes, int *pes, int nb, char *m)
{
  BgSyncListSend(npes, pes, CmiGetHandler(m), LARGE_WORK, nb, m);
}

static inline void BgMultipleSend(unsigned int pe, int len, int sizes[], char *msgComps[])
{
  int x,y,z,t,i;
  t = pe%BgGetNumWorkThread();
  pe = pe/BgGetNumWorkThread();
  BgGetXYZ(pe, &x, &y, &z);
  for (i=0; i<len; i++) {
    int nb = sizes[i];
    char *dupm = (char *)CmiAlloc(nb);
    memcpy(dupm, msgComps[i], nb);
    BgSendPacket(x,y,z, t, CmiGetHandler(dupm), LARGE_WORK, nb, dupm);
  }
}
#endif

#endif

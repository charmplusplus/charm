#include <stdio.h>
#include "blue.h"

typedef struct {
  char core[CmiBlueGeneMsgHeaderSizeBytes];
  int data;
} RingMsg;

const int MAXITER = 1;
int passRingID;
int iter = 0;

void passRing(char *msg);

void nextxyz(int x, int y, int z, int *nx, int *ny, int *nz)
{
  int sx, sy, sz;
  BgGetSize(&sx, &sy, &sz);
  *nz = z+1;  *ny = y; *nx = x;
  if (*nz == sz) {
    *nz = 0;
    (*ny) ++;
    if (*ny == sy) {
      *ny = 0;
      (*nx) ++;
    }
    if (*nx == sx) {
      (*nx) = 0;
    }
  }
}

void BgEmulatorInit(int argc, char **argv)
{
   if (argc < 6)     CmiAbort("Usage: <ring> <x> <y> <z> <numCommTh> <numWorkTh>\n");
   BgSetSize(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]));
   BgSetNumCommThread(atoi(argv[4]));        
   BgSetNumWorkThread(atoi(argv[5]));
}

void BgNodeStart(int argc, char **argv)  {
    int x,y,z, nx, ny, nz;  
    RingMsg *msg = (RingMsg*) CmiAlloc(sizeof(RingMsg));
    msg->data =  888;
    passRingID = BgRegisterHandler(passRing);
    BgGetMyXYZ(&x, &y, &z);           
    nextxyz(x, y, z, &nx, &ny, &nz);
    if (x == 0 && y==0 && z==0) {
      printf("%d %d %d => %d %d %d\n", x,y,z,nx,ny,nz);
      BgSendPacket(nx, ny, nz, ANYTHREAD, passRingID, LARGE_WORK, sizeof(RingMsg), (char *)msg);
    }
}

void passRing(char *msg)  {
     int x, y, z,  nx, ny, nz;
     BgGetMyXYZ(&x, &y, &z);            
     nextxyz(x, y, z, &nx, &ny, &nz);
     printf("%d %d %d => %d %d %d\n", x,y,z,nx,ny,nz);
     if (x==0 && y==0 && z==0)     if (++iter == MAXITER) { BgShutdown(); return; }
     CmiAssert(((RingMsg*)msg)->data == 888);
     BgSendPacket(nx, ny, nz, ANYTHREAD, passRingID, LARGE_WORK, sizeof(RingMsg), (char*)msg);
}


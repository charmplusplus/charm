
#include "blue.h"

#define MAXITER 2

int iter = 0;
int passRingHandler;

void passRing(char *msg);

void nextxyz(int x, int y, int z, int *nx, int *ny, int *nz)
{
  int numX, numY, numZ;

  BgGetSize(&numX, &numY, &numZ);
  *nz = z+1; *ny = y; *nx = x;
  if (*nz == numZ) {
    *nz = 0; (*ny) ++;
    if (*ny == numY) {
      *ny = 0; (*nx) ++;
      if (*nx == numX) *nx = 0;
    }
  }
}

void BgGlobalInit(int argc, char **argv)
{
  passRingHandler = BgRegisterHandler(passRing);
}

/* user defined functions for bgnode start entry */
void BgNodeStart(int argc, char **argv)
{
  int x,y,z;
  int nx, ny, nz;
  int data, id;

  BgGetXYZ(&x, &y, &z);
  nextxyz(x, y, z, &nx, &ny, &nz);
  id = BgGetWorkThreadID();
  data = 888;
  if (x == 0 && y==0 && z==0) {
    CmiPrintf("[%d:%d]: (%d, %d, %d) send msg(data %d) to (%d, %d, %d). \n", CmiMyPe(), id, x, y, z, data, nx, ny, nz);
    BgSendPacket(nx, ny, nz, passRingHandler, LARGE_WORK, sizeof(int), (char *)&data);
  }
}

/* user write code */
void passRing(char *msg)
{
  int x, y, z;
  int nx, ny, nz;
  int id;
  int data = *(int *)msg;

  BgGetXYZ(&x, &y, &z);
  nextxyz(x, y, z, &nx, &ny, &nz);
  if (x==0 && y==0 && z==0) {
    if (++iter == MAXITER) BgShutdown();
  }
  id = BgGetWorkThreadID();
  CmiPrintf("[%d:%d]: (%d, %d, %d) send msg(data %d) to (%d, %d, %d). \n", CmiMyPe(), id, x, y, z, data, nx, ny, nz);
  BgSendPacket(nx, ny, nz, passRingHandler, LARGE_WORK, sizeof(int), (char *)&data);
}


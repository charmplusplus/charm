#include <stdlib.h>
#include "blue.h"

int sendID;
#define MAX_NUM_CASES 		4	

extern "C" void sendHandler(char *) ;

const int NUM_ITERATIONS=1<<8;

class Msg
{
public:
char core[CmiBlueGeneMsgHeaderSizeBytes];
char c[100];
void *operator new(size_t s) { return CmiAlloc(s); }
void operator delete(void* ptr) { CmiFree(ptr); }
};

struct userData
{
  double st;
  int    iter;

  int    caseCount;
  double pingTime[MAX_NUM_CASES] ;
};

void BgEmulatorInit(int argc, char **argv) 
{
  if (argc < 6) { 
    CmiPrintf("Usage: <program> <x> <y> <z> <numCommTh> <numWorkTh> \n"); 
    BgShutdown();
  }

  BgSetSize(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]));
  BgSetNumCommThread(atoi(argv[4]));
  BgSetNumWorkThread(atoi(argv[5]));

}

void BgNodeStart(int argc, char **argv) 
{
  //ckout << "Initializing node " << bgNode->thisIndex.x << "," 
  //      << bgNode->thisIndex.y << "," << bgNode->thisIndex.z << endl; 
  sendID = BgRegisterHandler(sendHandler);

  int x,y,z;
  BgGetMyXYZ(&x, &y, &z);

  //declare node private data
  userData *ud = new userData ;
    ud->st   = 0.;
    ud->iter = -1;

    ud->caseCount = 1;
    for(int i=0; i<MAX_NUM_CASES; i++)
	ud->pingTime[i] = 0.;

  BgSetNodeData((char*)ud) ;

  if(x == 0 && y == 0 && z == 0)
  {
  	Msg *msg = new Msg;
	BgSendLocalPacket(ANYTHREAD,sendID, SMALL_WORK, sizeof(Msg), (char *)msg);
  }

}

void sendHandler(char *info) 
{
  int x,y,z;
  BgGetMyXYZ(&x,&y,&z);

  int numBgX, numBgY, numBgZ;
  BgGetSize(&numBgX, &numBgY, &numBgZ);

  userData* ud = (userData*)BgGetNodeData();
  Msg* msg = (Msg*)info;

  if(x==0 && y==0 && z==0)
  {
   	//ckout <<"Iteration no "<<ud->iter<<endl;
  	ud->iter++;
  	if(ud->iter==0) 
	  ud->st = BgGetTime();

	if(ud->iter==NUM_ITERATIONS)
	{
	  ud->pingTime[ud->caseCount-1] = (BgGetTime() - ud->st)/NUM_ITERATIONS;


	  if(ud->caseCount==MAX_NUM_CASES)
	  {
	  	CmiPrintf("Pingpong time averaged over %d iterations\n",NUM_ITERATIONS);
		CmiPrintf("---------------------------------------------------------\n");
		CmiPrintf("case			Time(RRT)\n");
		CmiPrintf("---------------------------------------------------------\n");
		for(int i=0; i<MAX_NUM_CASES; i++)
		switch(i+1){
		case 1:
			CmiPrintf("0,0,0 <--> 0,0,%d          %f\n", numBgZ-1,ud->pingTime[0]);
			break;
		case 2:
			CmiPrintf("0,0,0 <--> %d,0,0          %f\n",numBgX-1, ud->pingTime[1]);
			break;
		case 3:
			CmiPrintf("0,0,0 <--> 0,%d,0          %f\n",numBgY-1, ud->pingTime[2]);
			break;
		case 4:
			CmiPrintf("0,0,0 <--> %d,%d,%d          %f\n",numBgX-1,numBgY-1,numBgZ-1,ud->pingTime[3]);
			break;
		}
		CmiPrintf("---------------------------------------------------------\n");

		BgShutdown();
		return;
	  }
	  ud->caseCount++;
	  ud->iter = 0;
	  ud->st = BgGetTime();

	}


	switch(ud->caseCount) {
	case 1:
		BgSendPacket(0,0,numBgZ-1, ANYTHREAD,sendID, SMALL_WORK, sizeof(Msg), (char *)msg);
		break;
	case 2:
		BgSendPacket(numBgX-1,0,0, ANYTHREAD,sendID, SMALL_WORK, sizeof(Msg), (char *)msg);
		break;
	case 3:
		BgSendPacket(0,numBgY-1,0, ANYTHREAD,sendID, SMALL_WORK, sizeof(Msg), (char *)msg);
		break;
	case 4:
		BgSendPacket(numBgX-1,numBgY-1,numBgZ-1, ANYTHREAD,sendID, SMALL_WORK, sizeof(Msg), (char *)msg);
		break;
	}
  }
  else
  	BgSendPacket(0,0,0, ANYTHREAD,sendID, SMALL_WORK, sizeof(Msg), (char *)msg);
}



#include <stdlib.h>
#include "blue.h"

int BCastID ;
int TimeID  ;
const int X_NODE = 1;
const int Y_NODE = 2;
const int Z_NODE = 3;
const int MAX_ITERATIONS = 50;

extern "C" void BCastHandler(char*);
extern "C" void TimeHandler(char*);

class Msg {
  public:
    char core[CmiBlueGeneMsgHeaderSizeBytes];
    int    type;
    double time;

  void *operator new(size_t s) { return CmiAlloc(s); }
  void operator delete(void* ptr) { CmiFree(ptr); }
};

//for reporting completion of broadcast
struct userData {
  int    expected;
  int    received;
  double maxTime;
  int    iter;
  int    numIterations;
  double startTime[MAX_ITERATIONS];
  double endTime  [MAX_ITERATIONS];
};

void BgEmulatorInit(int argc, char **argv)
{
  if (argc < 7) { 
    CmiPrintf("Usage: line <x> <y> <z> <numCommTh> <numWorkTh> <numIter> \n"); 
    BgShutdown();
  }

  BgSetSize(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]));
  BgSetNumCommThread(atoi(argv[4]));
  BgSetNumWorkThread(atoi(argv[5]));

}

void BgNodeStart(int argc, char **argv)
{
  BCastID = BgRegisterHandler(BCastHandler);
  TimeID = BgRegisterHandler(TimeHandler);

  //Initialize NodePrivateData
  userData* ud = new userData;
    ud->received = 0;
    ud->expected = 0;
    ud->maxTime  = 0.;
    ud->iter     = 0;
    ud->numIterations = atoi(argv[6]) ;  //6th Argument is number of Iterations

    int x,y,z;
    BgGetMyXYZ(&x, &y, &z);
    int numBgX, numBgY, numBgZ;
    BgGetSize(&numBgX, &numBgY, &numBgZ);

    if(z==numBgZ-1)
    {
	if(y==numBgY-1)
	  ud->expected = 1;
	else if(y!=0 || (y==0 && x==numBgX-1))
	  ud->expected = 2;
	else
	  ud->expected = 3;
    }

  ud->startTime[0] = 0.;	//NOTE: This would cause some error in time of first iteration
  if(x==0 && y==0 && z==0) 
  {
    Msg* msg = new Msg;
      	msg->type = X_NODE;
    BgSendLocalPacket(ANYTHREAD,BCastID, LARGE_WORK, sizeof(Msg),(char*)msg);
  }
      //ckout << "BgNodeInit Done for Node" << bgNode->thisIndex.x <<" "
      //      << bgNode->thisIndex.y <<" "<< bgNode->thisIndex.z << endl ;
  
  BgSetNodeData( (char*)ud);
}


void BCastHandler(char* info) {
  int i, j, k;
  BgGetMyXYZ(&i, &j, &k);

  //ckout <<"Broadcast Message Received at Node "<<i<<" "<<j<<" "<<k<<endl;

  Msg* msg = (Msg*)info;

  int numBgX, numBgY, numBgZ;
  BgGetSize(&numBgX, &numBgY, &numBgZ);

  switch(msg->type){
  case X_NODE:	//possible only on X-axis
	if(i!=numBgX-1) 	//send message further down X with type X_NODE
	  BgSendPacket(i+1, j, k, ANYTHREAD,BCastID, LARGE_WORK, sizeof(Msg), (char*)msg);

  	//send message down Y with type Y_NODE
	msg = new Msg;
	msg->type = Y_NODE;
	BgSendPacket(i, j+1, k, ANYTHREAD,BCastID, LARGE_WORK, sizeof(Msg), (char*)msg);

  	//send message down Z with type Z_NODE
	msg = new Msg;
	msg->type = Z_NODE;
	BgSendPacket(i, j, k+1, ANYTHREAD,BCastID, LARGE_WORK, sizeof(Msg), (char*)msg);

  	break;
  case Y_NODE:

	if(j!=numBgY-1) 	//send message furtherdown Y with type Y_NODE
	  BgSendPacket(i, j+1, k, ANYTHREAD,BCastID, LARGE_WORK, sizeof(Msg), (char*)msg);

  	//send message down Z with type Z_NODE
	msg = new Msg;
	msg->type = Z_NODE;
	BgSendPacket(i, j, k+1, ANYTHREAD,BCastID, LARGE_WORK, sizeof(Msg), (char*)msg);

  	break;
  case Z_NODE:

	if(k!=numBgZ-1) 	//send message furtherdown Z with type Z_NODE
	  BgSendPacket(i, j, k+1, ANYTHREAD,BCastID, LARGE_WORK, sizeof(Msg), (char*)msg);
	else 				//send timing information
	{
	  msg->time = BgGetTime();
  	  //ckout <<"Time Message sent from Node "<<i<<" "<<j<<" "<<k
	  //      <<" at time "<<msg->time<<endl;
	  BgSendLocalPacket(ANYTHREAD,TimeID, SMALL_WORK, sizeof(Msg), (char*)msg);
	}
  	break;
  }
}

void TimeHandler(char* info) 
{
  int i, j, k;
  BgGetMyXYZ(&i, &j, &k);


  userData *ud = (userData*)BgGetNodeData();
  Msg* msg = (Msg*)info;

  ud->received++;
  //ckout <<"Time Message Received at Node "<<i<<" "<<j<<" "<<k
   //     <<" Received "<<ud->received<<" expected "<<ud->expected<< endl;
  if(ud->maxTime < msg->time)
  	ud->maxTime = msg->time;

  if(ud->expected == ud->received)
  {
  	msg->time = ud->maxTime;
   	switch(ud->expected) {
	  case 1:
		BgSendPacket(i, j-1, k, ANYTHREAD,TimeID, SMALL_WORK, sizeof(Msg), (char*)msg);
	  	break;
	  case 2:
	  	if(j == 0)
		  BgSendPacket(i-1, j, k, ANYTHREAD,TimeID, SMALL_WORK, sizeof(Msg), (char*)msg);
		else
		  BgSendPacket(i, j-1, k, ANYTHREAD,TimeID, SMALL_WORK, sizeof(Msg), (char*)msg);
	  	break;
	  case 3:
	  	if(i == 0)
		{
		  ud->endTime[ud->iter] = ud->maxTime;
		  //ckout << "Iteration No " << ud->iter << ", end time " << ud->endTime[ud->iter] << endl;

		  if(ud->iter == ud->numIterations-1)
		  {
			//print result
			CmiPrintf("-------------------------------------------------------------------------\n");
			CmiPrintf("Iter No:	StartTime	 	EndTime			TotalTime \n");
			CmiPrintf("-------------------------------------------------------------------------\n");
			double avg=0;
			for(int i=0; i<ud->numIterations; i++)
			{
			CmiPrintf("%d               %f               %f               %f\n", i, ud->startTime[i], ud->endTime[i], ud->endTime[i]-ud->startTime[i]);

			  avg += ud->endTime[i]-ud->startTime[i];
			}
			CmiPrintf("-------------------------------------------------------------------------\n");
			CmiPrintf("Average BroadCast Time:  			%f\n", avg/ud->numIterations);
			CmiPrintf("-------------------------------------------------------------------------\n");
			BgShutdown();
			return;
		  }
		  ud->iter += 1;
		  ud->startTime[ud->iter] = BgGetTime();
		  //ckout << "Iteration No " << ud->iter << ", start time " << ud->startTime[ud->iter] << endl;
		  msg->type = X_NODE;
		  BgSendPacket(0, 0, 0, ANYTHREAD,BCastID, LARGE_WORK, sizeof(Msg), (char*)msg);
		}
		else
		  BgSendPacket(i-1, j, k, ANYTHREAD,TimeID, SMALL_WORK, sizeof(Msg), (char*)msg);
	  	break;

	  default:
	  	CmiPrintf("\n\n ERROR: More than expected number of message received at node %d %d %d\n\n", i,j,k);
		break;
	}
	 //cleanup for next iteration
	 ud->received = 0;
	 ud->maxTime  = 0;	

  }
  else 
  	delete msg;
}



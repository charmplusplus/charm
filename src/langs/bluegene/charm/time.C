#include "BlueGene.h"

#define sendID   		1

void sendHandler(ThreadInfo *) ;
const int MAX_ITERATIONS=1000;
int NUM_ITERATIONS;

class Msg: public PacketMsg
{};

void BgInit(Main *main) 
{
  int num_args = main->getNumArgs(); 
   
  if (num_args < 7) { 
    CkAbort("Usage: <program> <x> <y> <z> <numCommTh> <numWorkTh> <numIterations>\n"); 
  }

  CreateBgNodeMsg *bgNodeMsg = new CreateBgNodeMsg;    
  bgNodeMsg->numBgX = atoi(main->getArgs()[1]);
  bgNodeMsg->numBgY = atoi(main->getArgs()[2]);
  bgNodeMsg->numBgZ = atoi(main->getArgs()[3]);
  
  bgNodeMsg->numCTh = atoi(main->getArgs()[4]);
  bgNodeMsg->numWTh = atoi(main->getArgs()[5]);

  NUM_ITERATIONS = atoi(main->getArgs()[6]);
  ckout << NUM_ITERATIONS << endl;

  main->CreateBlueGene(bgNodeMsg);
}

void* BgNodeInit(BgNode *bgNode) 
{
  //ckout << "Initializing node " << bgNode->thisIndex.x << "," 
  //      << bgNode->thisIndex.y << "," << bgNode->thisIndex.z << endl; 

  bgNode->registerHandler(sendID, sendHandler) ;

  if(bgNode->thisIndex.x == 0 &&
     bgNode->thisIndex.y == 0 &&
     bgNode->thisIndex.z == 0)
  {
  	Msg *msg = new Msg;
	ckout << "adding message" << endl;
  	bgNode->addMessage(msg, sendID, SMALL_WORK);
  	ckout << "after addMessage" << endl;
  }

  int *neelam;
  return (void*)neelam ;
}

void BgFinish() 
{}

void sendHandler(ThreadInfo *info) 
{
  //int x,y,z;
  //info->bgNode->getXYZ(x,y,z);
  ckout << "handler" << endl;

  double time[MAX_ITERATIONS];
  
  //if(x==0 && y==0 && z==0)
  {
	for(int i=0; i<NUM_ITERATIONS; i++)
		time[i] = info->getTime();
	for(int i=0; i<NUM_ITERATIONS; i++)
		ckout << time[i] << endl;

	info->bgNode->finish();
  }
}




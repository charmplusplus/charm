#include "BlueGene.h"

#define sendID   		1
#define MAX_NUM_CASES 		4	

void sendHandler(ThreadInfo *) ;

const int NUM_ITERATIONS=2000;

class Msg: public PacketMsg
{};

struct userData
{
  double st;
  int    iter;

  int    caseCount;
  double pingTime[MAX_NUM_CASES] ;
};

void BgInit(Main *main) 
{
  int num_args = main->getNumArgs(); 
   
  if (num_args < 6) { 
    CkAbort("Usage: <program> <x> <y> <z> <numCommTh> <numWorkTh> \n"); 
  }

  CreateBgNodeMsg *bgNodeMsg = new CreateBgNodeMsg;    
  bgNodeMsg->numBgX = atoi(main->getArgs()[1]);
  bgNodeMsg->numBgY = atoi(main->getArgs()[2]);
  bgNodeMsg->numBgZ = atoi(main->getArgs()[3]);
  
  bgNodeMsg->numCTh = atoi(main->getArgs()[4]);
  bgNodeMsg->numWTh = atoi(main->getArgs()[5]);

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
  	bgNode->addMessage(msg, sendID, SMALL_WORK);
  }

  //declare node private data
  userData *ud = new userData ;
    ud->st   = 0.;
    ud->iter = -1;

    ud->caseCount = 1;
    for(int i=0; i<MAX_NUM_CASES; i++)
	ud->pingTime[i] = 0.;

  return (void*)ud ;
}

void BgFinish() 
{}

void sendHandler(ThreadInfo *info) 
{
  int x,y,z;
  info->bgNode->getXYZ(x,y,z);

  userData* ud = (userData*)info->bgNode->nvData;
  Msg* msg = (Msg*)info->msg;

  if(x==0 && y==0 && z==0)
  {
   	//ckout <<"Iteration no "<<ud->iter<<endl;
  	ud->iter++;
  	if(ud->iter==0) 
	  ud->st = info->getTime();

	if(ud->iter==NUM_ITERATIONS)
	{
	  ud->pingTime[ud->caseCount-1] = (info->getTime() - ud->st)/NUM_ITERATIONS;


	  if(ud->caseCount==MAX_NUM_CASES)
	  {
	  	ckout <<"Pingpong time averaged over "<<NUM_ITERATIONS<<" iterations"<<endl;
		ckout <<"---------------------------------------------------------"<<endl;
		ckout <<"case			Time"<<endl;
		ckout <<"---------------------------------------------------------"<<endl;
		for(int i=0; i<MAX_NUM_CASES; i++)
		switch(i+1){
		case 1:
			ckout <<"0,0,0 <--> 0,0,1          "<<ud->pingTime[0]<<endl;
			break;
		case 2:
			ckout <<"0,0,0 <--> "<<info->bgNode->numBgX-1<<",0,0          "<<ud->pingTime[1]<<endl;
			break;
		case 3:
			ckout <<"0,0,0 <--> 0,"<<info->bgNode->numBgY-1<<",0          "<<ud->pingTime[2]<<endl;
			break;
		case 4:
			ckout <<"0,0,0 <--> 4,4,"<<4<<"          "<<ud->pingTime[3]<<endl;
			break;
		}
		ckout <<"---------------------------------------------------------"<<endl;
		info->bgNode->finish();
		return;
	  }
	  ud->caseCount++;
	  ud->iter = 0;
	  ud->st = info->getTime();

	}

	switch(ud->caseCount) {
	case 1:
		info->sendPacket(0,0,1, msg, sendID, SMALL_WORK);
		break;
	case 2:
		info->sendPacket(info->bgNode->numBgX-1,0,0, msg, sendID, SMALL_WORK);
		break;
	case 3:
		info->sendPacket(0,info->bgNode->numBgY-1,0, msg, sendID, SMALL_WORK);
		break;
	case 4:
		info->sendPacket(4,4,4, msg, sendID, SMALL_WORK);
		break;
	}
  }
  else
  	info->sendPacket(0,0,0, msg, sendID, SMALL_WORK);
}




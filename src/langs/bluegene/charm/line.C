#include "BlueGene.h"

const int BCastID = 1;
const int TimeID  = 2;
const int X_NODE = 1;
const int Y_NODE = 2;
const int Z_NODE = 3;
const int MAX_ITERATIONS = 50;

extern "C" void BCastHandler(ThreadInfo*);
extern "C" void TimeHandler(ThreadInfo*);

class Msg: public PacketMsg {
  public:
    int    type;
    double time;
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

void BgInit(Main* main) 
{
  int num_args = main->getNumArgs(); 
   
  if (num_args < 6) { 
    CkAbort("Usage: line2 <x> <y> <z> <numCommTh> <numWorkTh> \n"); 
  }

  CreateBgNodeMsg *bgNodeMsg = new CreateBgNodeMsg;    
  //bgNodeMsg->numBgX = atoi(main->getArgs()[1]);
  //bgNodeMsg->numBgY = atoi(main->getArgs()[2]);
  //bgNodeMsg->numBgZ = atoi(main->getArgs()[3]);
  
  //bgNodeMsg->numCTh = atoi(main->getArgs()[4]);
  //bgNodeMsg->numWTh = atoi(main->getArgs()[5]);

  bgNodeMsg->argc   =  main->getNumArgs() ;
  bgNodeMsg->argv   =  main->getArgs() ;

  main->CreateBlueGene(bgNodeMsg);
  ckout <<"BlueGene Initialized"<<endl;
  ckout << "Argv[0] is " << bgNodeMsg->argv[0] << endl ; 
}

void* BgNodeInit(BgNode *bgNode)
{
  bgNode->registerHandler(BCastID, BCastHandler);
  bgNode->registerHandler(TimeID, TimeHandler);

  //Initialize NodePrivateData
  userData* ud = new userData;
    ud->received = 0;
    ud->expected = 0;
    ud->maxTime  = 0.;
    ud->iter     = 0;
    ud->numIterations = atoi(bgNode->argv[6]) ;  //6th Argument is number of Iterations

    if(bgNode->thisIndex.z==bgNode->numBgZ-1)
    {
	if(bgNode->thisIndex.y==bgNode->numBgY-1)
	  ud->expected = 1;
	else if(bgNode->thisIndex.y!=0 || (bgNode->thisIndex.y==0 && bgNode->thisIndex.x==bgNode->numBgX-1))
	  ud->expected = 2;
	else
	  ud->expected = 3;
    }

  ud->startTime[0] = 0.;	//NOTE: This would cause some error in time of first iteration
  if(bgNode->thisIndex.x==0 && 
     bgNode->thisIndex.y==0 && 
     bgNode->thisIndex.z==0) 
  {
    Msg* msg = new Msg;
      	msg->type = X_NODE;
    bgNode->addMessage(msg, BCastID, LARGE_WORK);
  }
      //ckout << "BgNodeInit Done for Node" << bgNode->thisIndex.x <<" "
      //      << bgNode->thisIndex.y <<" "<< bgNode->thisIndex.z << endl ;
  
  return (void*)ud;
}

void BgFinish() {}

void BCastHandler(ThreadInfo* info) {
  int i, j, k;
  info->bgNode->getXYZ(i, j, k);

  //ckout <<"Broadcast Message Received at Node "<<i<<" "<<j<<" "<<k<<endl;

  Msg* msg = (Msg*)info->msg;

  switch(msg->type){
  case X_NODE:	//possible only on X-axis
	if(i!=info->bgNode->numBgX-1) 	//send message further down X with type X_NODE
	  info->sendPacket(i+1, j, k, msg, BCastID, LARGE_WORK);

  	//send message down Y with type Y_NODE
	msg = new Msg;
	  msg->type = Y_NODE;
	info->sendPacket(i, j+1, k, msg, BCastID, LARGE_WORK);

  	//send message down Z with type Z_NODE
	msg = new Msg;
	  msg->type = Z_NODE;
	info->sendPacket(i, j, k+1, msg, BCastID, LARGE_WORK);

  	break;
  case Y_NODE:

	if(j!=info->bgNode->numBgY-1) 	//send message furtherdown Y with type Y_NODE
	  info->sendPacket(i, j+1, k, msg, BCastID, LARGE_WORK);

  	//send message down Z with type Z_NODE
	msg = new Msg;
	  msg->type = Z_NODE;
	info->sendPacket(i, j, k+1, msg, BCastID, LARGE_WORK);

  	break;
  case Z_NODE:

	if(k!=info->bgNode->numBgZ-1) 	//send message furtherdown Z with type Z_NODE
	  info->sendPacket(i, j, k+1, msg, BCastID, LARGE_WORK);
	else 				//send timing information
	{
	  msg->time = info->getTime();
  	  //ckout <<"Time Message sent from Node "<<i<<" "<<j<<" "<<k
	  //      <<" at time "<<msg->time<<endl;
	  info->bgNode->addMessage(msg, TimeID, SMALL_WORK);
	}
  	break;
  }
}

void TimeHandler(ThreadInfo* info) 
{
  int i, j, k;
  info->bgNode->getXYZ(i, j, k);


  userData *ud = (userData*)info->bgNode->nvData;
  Msg* msg = (Msg*)info->msg;

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
		info->sendPacket(i, j-1, k, msg, TimeID, SMALL_WORK);
	  	break;
	  case 2:
	  	if(info->bgNode->thisIndex.y == 0)
		  info->sendPacket(i-1, j, k, msg, TimeID, SMALL_WORK);
		else
		  info->sendPacket(i, j-1, k, msg, TimeID, SMALL_WORK);
	  	break;
	  case 3:
	  	if(info->bgNode->thisIndex.x == 0)
		{
		  ud->endTime[ud->iter] = ud->maxTime;
		  //ckout << "Iteration No " << ud->iter << ", end time " << ud->endTime[ud->iter] << endl;

		  if(ud->iter == ud->numIterations-1)
		  {
			//print result
			ckout <<"-------------------------------------------------------------------------"<<endl;
			ckout <<"Iter No:	StartTime	 	EndTime			TotalTime "<<endl;
			ckout <<"-------------------------------------------------------------------------"<<endl;
			double avg=0;
			for(int i=0; i<ud->numIterations; i++)
			{
			ckout <<i            <<"               "
			      <<ud->startTime[i]<<"               "
			      <<ud->endTime[i]  <<"               "
			      <<ud->endTime[i]-ud->startTime[i]<<endl; 

			  avg += ud->endTime[i]-ud->startTime[i];
			}
			ckout <<"-------------------------------------------------------------------------"<<endl;
			ckout <<"Average BroadCast Time:  			"<<avg/ud->numIterations<<endl;
			ckout <<"-------------------------------------------------------------------------"<<endl;
			info->bgNode->finish();
			return;
		  }
		  ud->iter += 1;
		  ud->startTime[ud->iter] = info->getTime();
		  //ckout << "Iteration No " << ud->iter << ", start time " << ud->startTime[ud->iter] << endl;
		  msg->type = X_NODE;
		  info->sendPacket(0, 0, 0, msg, BCastID, LARGE_WORK);
		}
		else
		  info->sendPacket(i-1, j, k, msg, TimeID, SMALL_WORK);
	  	break;

	  default:
	  	ckout <<"\n\n ERROR: More than expected number of message received at node "
		      <<i<<" "<<j<<" "<<k<<"\n\n"<<endl;
		break;
	}
	 //cleanup for next iteration
	 ud->received = 0;
	 ud->maxTime  = 0;	

  }
  else 
  	delete msg;
}



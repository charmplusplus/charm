#include "BlueGene.h"
#define  computeID  1

extern "C" void compute(ThreadInfo *) ;

const int NUM_ITERATIONS = 10;	//global, readonly

class MyMsg : public PacketMsg
{
public:
  int iter ;
} ;

// for storing results of different tests
struct TimeRecord 
{
  int    test_num;
  double* max_time;
  double* start_time;

  TimeRecord() : test_num(-1) {
    max_time = new double[NUM_ITERATIONS];
    start_time = new double[NUM_ITERATIONS];
    for (int i=0; i<NUM_ITERATIONS; i++) {
      max_time[i] = start_time[i] = 0.0;
    }
  }

  ~TimeRecord() {
    delete [] max_time;
    delete [] start_time;
  }
     
  void Print(ThreadInfo* info) {
    // print result
    double average = 0.0;
    ckout <<"RESULTS FOR " 
          <<info->bgNode->numBgX <<" by "
          <<info->bgNode->numBgY <<" by "
          <<info->bgNode->numBgZ <<" with "
          <<info->bgNode->getNumCTh() <<" comm "
          <<info->bgNode->getNumWTh() <<" work" <<endl;
    ckout <<"-------------------------------------------------------------------------"<<endl;
    ckout <<"Iter No:	StartTime	 	EndTime			TotalTime "<<endl;
    ckout <<"-------------------------------------------------------------------------"<<endl;
    for (int i=0; i<NUM_ITERATIONS; i++) {
      ckout <<i 
            <<"               " <<start_time[i] 
            <<"               " <<max_time[i]   
            <<"               " <<max_time[i] - start_time[i] <<endl; 
      average += max_time[i] - start_time[i];
    }
    ckout <<"-------------------------------------------------------------------------"<<endl;
    ckout <<"Average BroadCast Time:  			"
          <<average / NUM_ITERATIONS <<endl;
    ckout <<"-------------------------------------------------------------------------"<<endl;
    info->bgNode->finish();
  }
};

void BgInit(Main *main)
{
  int num_args = main->getNumArgs(); 
   
  if (num_args < 6) { 
    CkAbort("Usage: <program> <x> <y> <z> <numCommTh> <numWorkTh> \n"); 
  }

  CreateBgNodeMsg *bgNodeMsg = new CreateBgNodeMsg;    
  //bgNodeMsg->numBgX = atoi(main->getArgs()[1]);
  //bgNodeMsg->numBgY = atoi(main->getArgs()[2]);
  //bgNodeMsg->numBgZ = atoi(main->getArgs()[3]);
  
  //bgNodeMsg->numCTh = atoi(main->getArgs()[4]);
  //bgNodeMsg->numWTh = atoi(main->getArgs()[5]);
  bgNodeMsg->argc   =  main->getNumArgs() ;
  bgNodeMsg->argv   =  main->getArgs() ;

  main->CreateBlueGene(bgNodeMsg) ;
}

void* BgNodeInit(BgNode *bgNode)
{
  // ckout << "entered BgNodeInit " << bgNode->thisIndex.x <<" "
  // << bgNode->thisIndex.y <<" "<< bgNode->thisIndex.z << endl ;

  bgNode->registerHandler(computeID, compute) ;

  if(bgNode->thisIndex.x==0 && 
     bgNode->thisIndex.y==0 && 
     bgNode->thisIndex.z==0) 
  {
    MyMsg *msg = new MyMsg;
    msg->iter = -1;
    bgNode->addMessage(msg, computeID, LARGE_WORK); 
    return new TimeRecord;
  }
  else { return NULL; }
}

void BgFinish() {}

void compute(ThreadInfo *info) 
{
  int i, j, k ;
  MyMsg *m = (MyMsg*)(info->msg);
  BgNode *bn = info->bgNode;

  bn->getXYZ(i,j,k) ; 

  if(i==0 && j==0 && k==0) {
    TimeRecord* r = (TimeRecord*)bn->nvData;
    if (r->test_num == -1) {
      r->test_num++;
      r->start_time[r->test_num] = info->getTime();
    }
    else {
      r->max_time[r->test_num] = info->getTime();
      r->test_num++;
      if (r->test_num < NUM_ITERATIONS) {
        // start bcast all over again
        r->start_time[r->test_num] = info->getTime();
      }
      else {
        // print results and quit
        r->Print(info);
        bn->finish();
        CkExit();
      }
    }
  }

  k = k + 1 ;
  if( k==bn->numBgZ )
  {
    k = 0 ;
    j = j+1 ;
    if ( j==bn->numBgY )
    {
      j = 0 ;
      i = i+1 ;
      if ( i==bn->numBgX )
      {
        i = 0 ;
      }
    }
  }
  
  //ckout << "BgNode No: " << i <<", "<< j <<", "<< k << endl ;
  info->sendPacket(i, j, k, m, computeID, LARGE_WORK) ;
}

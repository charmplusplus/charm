#include "BlueGene.h"

#define computeNumPrimeID  1
#define contributeID       2
#define reduceID	   3

void reduce(ThreadInfo *) ;
void computeNumPrime(ThreadInfo *) ;
void contribute(ThreadInfo *) ;

class Msg: public PacketMsg
{
public:
  int pc;	//number of primes
  int min, max;	//range of numbers
};

typedef struct userDataStruct
{
  int pc;
  int count;
} userData;

void BgInit(Main *main) 
{
  int num_args = main->getNumArgs(); 
   
  if (num_args < 6) { 
    CkAbort("Usage: <program> <x> <y> <z> <numCommTh> <numWorkTh>\n"); 
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
  //ckout << "Creating Blue Gene" << endl;
}

void* BgNodeInit(BgNode *bgNode) 
{
  int x = bgNode->thisIndex.x;
  int y = bgNode->thisIndex.y;
  int z = bgNode->thisIndex.z;

  //ckout << "BgNodeInit in " <<x<< ", " <<y<< ", " <<z<< endl;

  bgNode->registerHandler(reduceID, reduce);
  bgNode->registerHandler(computeNumPrimeID, computeNumPrime);
  bgNode->registerHandler(contributeID, contribute);

  //declare node private data
  userData *ud = new userData;
  ud->count = 0;
  ud->pc    = 0;
  const int RANGE = atoi(bgNode->argv[6]); 

  //Decide range of number for this node
  int min, max;
  int slot = RANGE/(bgNode->numBgX * bgNode->numBgY * bgNode->numBgZ);
  if(0==slot) { slot = 1; }

  min = slot*(x*bgNode->numBgY*bgNode->numBgZ + y*bgNode->numBgZ + z);
  max = min + slot;

  if(x==bgNode->numBgX-1 &&
     y==bgNode->numBgZ-1 &&
     z==bgNode->numBgY-1) 
       max = RANGE;	

  if(max>RANGE)	 { max = RANGE; }
  if(min>=RANGE) { min=max=0; }

  //ckout << "Range for node " <<x<< ", " <<y<< ", " <<z<< " is " << min <<", "<< max << endl;

  //Divide range to each worker thread and Fire....
  int numWTh = bgNode->getNumWTh();
  if(max-min>0)
  {
	slot = slot/numWTh;
	if(0==slot) { slot = 1; }

	for(int i=0; i<numWTh; i++)
	{	
  	  Msg *msg = new Msg;
	  msg->min = min + i*slot;
	  msg->max = msg->min + slot;

	  if(i==numWTh-1)
	  	msg->max = max;

	  if(msg->max>max) { msg->max=max; }			
	  if(msg->min>=max) { msg->min=msg->max=0; }			

  	  bgNode->addMessage(msg, computeNumPrimeID, LARGE_WORK);

	  //ckout << "	Range for thread " << i << "is " 
	  //      << msg->min  << ", " << msg->max << endl;

	}
  }
  else
  {
	for(int i=0; i<numWTh; i++)
	{
  	  Msg *msg = new Msg;
  	  msg->min=msg->max=0;
  	  bgNode->addMessage(msg, computeNumPrimeID, LARGE_WORK);
	}
  }

  return (void*)ud ;
}

void BgFinish() 
{}

/* Utility: isPrime
 * Checks whether a given number is prime or not
 * Assumption: Any number<=1 is not prime
 */
int isPrime(const int number)
{
  if(number<=1)	return 0;

  for(int i=2; i<number; i++)
  {
   	if(0 == number%i)
	  return 0;
  }

  return 1;
}

/* Utility: computeNumberOfPrimesIn
 * Computes number of prime number in a given range
 */
int computeNumberOfPrimesIn(const int min, const int max) 	
{
  int count=0;
  for(int i=min; i<max; i++)
  {
     if(isPrime(i))
     {
     	count++;
	//ckout << i << " is prime"<< endl;
     }
     else
	//ckout << i << " is not prime"<< endl;
	;
  }

  return count;
}


/* Handler: computeNumPrime
 * Compute the number of primes in the assigned range and contribute
 */
void computeNumPrime(ThreadInfo *info) 
{
  int x,y,z;
  info->bgNode->getXYZ(x,y,z);
  //ckout << "computeNumPrime in " << x << ", " << y << ", " << z << endl;

  Msg *msg = (Msg*)info->msg;

  //compute number of primes in the assigned range of numbers
  int min = msg->min;
  int max = msg->max;
  int pc = computeNumberOfPrimesIn(min, max);

  // contribute the results for reduction : reusing the same message for optimization
  msg->pc = pc;
  info->bgNode->addMessage(msg,contributeID,LARGE_WORK);
}

/* Handler: contribute
 * If the number of contributions received is equal to expected number
 *    call the reduction handler
 * Else contribute the contribution to node private data and update counter
 */
void contribute(ThreadInfo *info)
{
  int x,y,z;
  info->bgNode->getXYZ(x,y,z);
  //ckout << "contribute in " << x << ", " << y << ", " << z << endl ;

  userData* ud = (userData*)(info->bgNode->nvData);
  ud->pc += ((Msg*)(info->msg))->pc;
  ud->count++;

  //compute expected contributions
  //more: This computation need not be repeated everytime. Change it after this pgm works
  int reqCount ;
  if(z==info->bgNode->numBgZ-1)
    reqCount = 0 ;  
  else if(z>0 || (z==0 && x==info->bgNode->numBgX-1))
    reqCount = 1 ;
  else if(x>0 || (x==0 && y==info->bgNode->numBgY-1))
    reqCount = 2 ;  
  else
    reqCount = 3 ;

  reqCount += info->bgNode->getNumWTh();

  if(ud->count==reqCount)  //if data for reduction is ready
  {
    Msg *msg = (Msg*)info->msg ;
    msg->pc = ud->pc;
    info->bgNode->addMessage(msg, reduceID, LARGE_WORK) ;
    return ;
  }
  else
    delete (Msg*)info->msg;
}

/* Handler: reduce
 * If reduction has finished, print the number of primes
 * Else send the number of primes to next node in reduction sequence
 */
void reduce(ThreadInfo *info) 
{
  int pc = ((userData*)(info->bgNode->nvData))->pc;

  int x,y,z;
  info->bgNode->getXYZ(x,y,z);
  //ckout << "reduce in " << x << ", " << y << ", " << z 
  //      << " number of primes " << pc << endl;

  if(x==0 && y==0 && z==0)
  {
  delete (Msg*)info->msg;
  ckout << "Finished : The number of primes is " << pc << endl;
  ckout << "Emulation Time : " << info->getTime() << " microseconds" << endl;
  info->bgNode->finish();
  return;
  }
  //send pc to destination(decide) 
  if(z>0)
    z--;   
  else if(x>0)
    x--;
  else
    y--;

  Msg *msg = (Msg*)info->msg;
  msg->pc = pc;
  info->sendPacket(x,y,z, msg, contributeID, LARGE_WORK);
}


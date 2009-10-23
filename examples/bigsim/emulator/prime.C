#include <stdlib.h>
#include "blue.h"

int computeNumPrimeID;
int contributeID;
int reduceID;

extern "C" void reduce(char *) ;
extern "C" void computeNumPrime(char *) ;
extern "C" void contribute(char *) ;

class Msg
{
public:
  char core[CmiBlueGeneMsgHeaderSizeBytes];
  int pc;	//number of primes
  int min, max;	//range of numbers
  void *operator new(size_t s) { return CmiAlloc(s); }
  void operator delete(void* ptr) { CmiFree(ptr); }
};

BnvStaticDeclare(int, pc)
BnvStaticDeclare(int, count)

void BgEmulatorInit(int argc, char **argv)
{
  if (argc < 7) { 
    CmiPrintf("Usage: <program> <x> <y> <z> <numCommTh> <numWorkTh> <range>\n");
    BgShutdown();
  }

  BgSetSize(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]));
  BgSetNumCommThread(atoi(argv[4]));
  BgSetNumWorkThread(atoi(argv[5]));

}

void BgNodeStart(int argc, char **argv) 
{
  reduceID = BgRegisterHandler(reduce);
  computeNumPrimeID = BgRegisterHandler(computeNumPrime);
  contributeID = BgRegisterHandler(contribute);

  int x,y,z;
  BgGetMyXYZ(&x, &y, &z);

  //CmiPrintf("BgNodeStart in %d %d %d\n", x,y,z);

  //declare node private data
  BnvInitialize(int, pc);
  BnvInitialize(int, count);
  BnvAccess(count) = 0;
  BnvAccess(pc)    = 0;
  const int RANGE = atoi(argv[6]); 

  //Decide range of number for this node
  int min, max;
  int numBgX, numBgY, numBgZ;
  BgGetSize(&numBgX, &numBgY, &numBgZ);
  int slot = RANGE/(numBgX * numBgY * numBgZ);
  if(0==slot) { slot = 1; }

  min = slot*(x*numBgY*numBgZ + y*numBgZ + z);
  max = min + slot;

  if(x==numBgX-1 &&
     y==numBgZ-1 &&
     z==numBgY-1) 
       max = RANGE;	

  if(max>RANGE)	 { max = RANGE; }
  if(min>=RANGE) { min=max=0; }

  //ckout << "Range for node " <<x<< ", " <<y<< ", " <<z<< " is " << min <<", "<< max << endl;

  //Divide range to each worker thread and Fire....
  int numWTh = BgGetNumWorkThread();
  if(max-min>0)
  {
  	int numWTh = BgGetNumWorkThread();

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

  	  BgSendLocalPacket(ANYTHREAD,computeNumPrimeID, LARGE_WORK, sizeof(Msg), (char *)msg);

	}
  }
  else
  {
	for(int i=0; i<numWTh; i++)
	{
  	  Msg *msg = new Msg;
  	  msg->min=msg->max=0;
  	  BgSendLocalPacket(ANYTHREAD,computeNumPrimeID, LARGE_WORK, sizeof(Msg), (char *)msg);
	}
  }

}


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
void computeNumPrime(char *info) 
{
  int x,y,z;
  BgGetMyXYZ(&x,&y,&z);
  //CmiPrintf("computeNumPrime in %d %d %d\n", x,y,z);

  Msg *msg = (Msg*)info;

  //compute number of primes in the assigned range of numbers
  int min = msg->min;
  int max = msg->max;
  int pc = computeNumberOfPrimesIn(min, max);

  // contribute the results for reduction : reusing the same message for optimization
  msg->pc = pc;
  BgSendLocalPacket(ANYTHREAD,contributeID,LARGE_WORK, sizeof(Msg), (char *)msg);
}

/* Handler: contribute
 * If the number of contributions received is equal to expected number
 *    call the reduction handler
 * Else contribute the contribution to node private data and update counter
 */
void contribute(char *info)
{
  int x,y,z;
  BgGetMyXYZ(&x,&y,&z);
  //ckout << "contribute in " << x << ", " << y << ", " << z << endl ;

  BnvAccess(pc) += ((Msg*)info)->pc;
  BnvAccess(count)++;

  //compute expected contributions
  //more: This computation need not be repeated everytime. Change it after this pgm works
  int reqCount ;
  int numBgX, numBgY, numBgZ;
  BgGetSize(&numBgX, &numBgY, &numBgZ);
  if(z==numBgZ-1)
    reqCount = 0 ;  
  else if(z>0 || (z==0 && x==numBgX-1))
    reqCount = 1 ;
  else if(x>0 || (x==0 && y==numBgY-1))
    reqCount = 2 ;  
  else
    reqCount = 3 ;

  reqCount += BgGetNumWorkThread();

  if(BnvAccess(count)==reqCount)  //if data for reduction is ready
  {
    Msg *msg = (Msg*)info ;
    msg->pc = BnvAccess(pc);
    BgSendLocalPacket(ANYTHREAD,reduceID, LARGE_WORK, sizeof(Msg), (char *)msg);
    return ;
  }
  else
    delete (Msg*)info;
}

/* Handler: reduce
 * If reduction has finished, print the number of primes
 * Else send the number of primes to next node in reduction sequence
 */
void reduce(char *info) 
{
  int pc = BnvAccess(pc);

  int x,y,z;
  BgGetMyXYZ(&x,&y,&z);
  //ckout << "reduce in " << x << ", " << y << ", " << z 
  //      << " number of primes " << pc << endl;

  if(x==0 && y==0 && z==0)
  {
  delete (Msg*)info;
  CmiPrintf("Finished : The number of primes is %d\n",pc);
  CmiPrintf("Emulation Time : %f seconds\n", BgGetTime());
  BgShutdown();
  return;
  }
  //send pc to destination(decide) 
  if(z>0)
    z--;   
  else if(x>0)
    x--;
  else
    y--;

  Msg *msg = (Msg*)info;
  msg->pc = pc;
  BgSendPacket(x,y,z, ANYTHREAD,contributeID, LARGE_WORK, sizeof(Msg), (char *)msg);
}


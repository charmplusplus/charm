#include "parallelJacobi.h"

#if CMK_BIGSIM_CHARM
#define BgElapse BgElapse
#define BgGetTime   BgGetTime
#else
#define BgElapse(x)
#define BgGetTime   CmiWallTimer
#endif

#define ITER    3

#define DEBUG   0

#define indexof(i,j,ydim) ( ((i)*(ydim)) + (j))

CProxy_Main globalMainProxy;
CProxy_Chunk chunk_arr;

double startTime;
int numFinished;

Chunk::Chunk(int t, int x, int y){

  int xdim,ydim;
  xdim = x;
  ydim = y;
  total = t; 
  iterations =0;
  myMax = 99999.999;


  // CkPrintf("[%d] x is %d, y is %d, t is %d %f\n",CkMyPe(),x,y,t,BgGetTime());  
  myxdim = int(xdim/total);
  counter=0;

  if(thisIndex == total-1) 
    myxdim = xdim - myxdim*(total-1);    

  myydim = ydim;

  if((thisIndex != 0)&&(thisIndex != total-1)){
    A = new double[(myxdim+2)*myydim];
    B = new double[(myxdim+2)*myydim];
      //Initialize everything to zero
    for (int i=0; i<myxdim+2; i++)
      for (int j=0; j<myydim; j++) 
	A[indexof(i,j,ydim)] = B[indexof(i,j,ydim)] = 0.0;    
  }
  else {
    A = new double[(myxdim+1)*myydim];
    B = new double[(myxdim+1)*myydim];
    //Initialize everything to zero
  for (int i=0; i<myxdim+1; i++)
    for (int j=0; j<myydim; j++) 
      A[indexof(i,j,ydim)] = B[indexof(i,j,ydim)] = 0.0;  
  }

  usesAtSync = false;
  //LBDatabase *lbdb = getLBDB();
  //lbdb->SetLBPeriod(50);

  BgElapse(.5e-5);
}


Chunk::Chunk(CkMigrateMessage *m){
}


void Chunk::resetBoundary() {

  if((thisIndex !=0))
    if(thisIndex < (int)(total/2))
      for(int i=1;i<myxdim+1;i++)
	A[indexof(i,0,myydim)] = 1.0;

  if(thisIndex ==0){
    //if(thisIndex < (int)(total/2))
      for(int i=0;i<myxdim;i++)
	A[indexof(i,0,myydim)] = 1.0;
    
    for (int i = 0;2*i<myydim; i++) 
	A[indexof(0,i,myydim)] = 1.0;
  
}
}


void Chunk::print() {

  if ((myxdim>100)||(myydim>100)) return;

#if 1
  CkPrintf("thisIndex = %d,myxdim=%d,myydim=%d\n",thisIndex,myxdim,myydim);

  if(thisIndex !=0)
    for (int i=0; i<myydim; i++) {
      for (int j=1; j<myxdim+1; j++) 
	CkPrintf("%lf ", A[indexof(j,i,myydim)]) ;
      CkPrintf("\n");
    }
  else
    for (int i=0; i<myydim; i++) {
      for (int j=0; j<myxdim; j++) 
	CkPrintf("%lf ", A[indexof(j,i,myydim)]) ;
      CkPrintf("\n");
    }
#endif
}



void Chunk::testEnd(){

  if(iterations == ITER)  {
 			 
    if(CkMyPe() != 0)
      return;

    CkPrintf("Numfin=%d, total=%d, Pes = %d\n",numFinished,total,CkNumPes());
    if((CkMyPe()==0)&&(numFinished != total/CkNumPes()))
      return;

    double elapt = BgGetTime()-startTime;
    CkPrintf("Finished in %fs %fs/step and iters is %d\n", elapt, elapt/iterations,iterations);
    // BgPrint("ENDED at: %f\n");
    CkExit();
    return;
  }

}


void Chunk::startWork(){

  double* temp = (double*) malloc(sizeof(double)*myydim);

  if(iterations == ITER)  {
    //       BgPrint("FINISHED iterations at :%f\n");
    if(CkMyPe() != 0)
     return;

    CkPrintf("Numfin=%d, total=%d, Pes = %d\n",numFinished,total,CkNumPes());
    if((CkMyPe()==0)&&(numFinished != total/CkNumPes()))
      return;

    double elapt = BgGetTime()-startTime;
    CkPrintf("Finished in %fs %fs/step and iters is %d\n", elapt, elapt/iterations,iterations);
    //BgPrint0("ENDED at: %f\n");
    CkExit();
    return;
  }

     //CkPrintf("[%d]print in startWork %f\n", CkMyPe(), BgGetTime());
#if DEBUG
  print();
  CkPrintf("\n\n\n");
#endif

  if(thisIndex >0){
    for(int i=0;i<myydim;i++)
      temp[i] = A[indexof(1,i,myydim)];
    chunk_arr[thisIndex-1].getStripfromright(new (myydim,0) Msg(myydim,temp));
  } 
  else{
    //Send dummy if thisIndex==0 for dagger to work
    chunk_arr[total-1].getStripfromright(new (myydim,0) Msg(myydim,temp));
  }

  BgElapse(0.25e-5); //Time for sending to getStripfromright

  if(thisIndex < total-1){
  
      for(int i=0;i<myydim;i++)
	temp[i] = A[indexof(myxdim,i,myydim)];
      chunk_arr[thisIndex+1].getStripfromleft(new (myydim,0) Msg(myydim,temp));
  }
  else{
    //Send dummy if thisIndex==total-1:For dagger to work
    chunk_arr[0].getStripfromleft(new (myydim,0) Msg(myydim,temp));
  }

  BgElapse(0.25e-5); //Time for sending to getStripfromleft
  
#if DEBUG
    CkPrintf("\n\nA in end of startWork is \n");
  print();
#endif
}


void Chunk::doWork(){

  double maxChange = 0.0;
  double * temp;

  iterations++;
  if((iterations == ITER)&&(CkMyPe()==0))
    numFinished++;
  if (thisIndex == 0)  
    CkPrintf("Iteration: %d\n",iterations);

  resetBoundary();

    if((thisIndex !=0)&&(thisIndex != total-1))
      for (int i=1; i<myxdim+1; i++)
	for (int j=1; j<myydim-1; j++) {
	  B[indexof(i,j,myydim)] = 
	    (0.2)*(A[indexof(i,  j,  myydim)] +
		   A[indexof(i,  j+1,myydim)] +
		   A[indexof(i,  j-1,myydim)] +
		   A[indexof(i+1,j,  myydim)] +
		   A[indexof(i-1,j,  myydim)]);

	  double change =  B[indexof(i,j,myydim)] - A[indexof(i,j,myydim)];
	  if (change < 0) change = - change;
	  if (change > maxChange) maxChange = change;
	}
      
    if(thisIndex == 0)
      for (int i=1; i<myxdim; i++)
	for (int j=1; j<myydim-1; j++) {
	  B[indexof(i,j,myydim)] = 
	    (0.2)*(A[indexof(i,  j,  myydim)] +
		   A[indexof(i,  j+1,myydim)] +
		   A[indexof(i,  j-1,myydim)] +
		   A[indexof(i+1,j,  myydim)] +
		   A[indexof(i-1,j,  myydim)]);

	  double change =  B[indexof(i,j,myydim)] - A[indexof(i,j,myydim)];
	  if (change < 0) change = - change;
	  if (change > maxChange) maxChange = change;
	}
      
    if(thisIndex == total-1) {
      for (int i=1; i<myxdim; i++)
	for (int j=1; j<myydim-1; j++) {
	  B[indexof(i,j,myydim)] = 
	    (0.2)*(A[indexof(i,  j,  myydim)] +
		   A[indexof(i,  j+1,myydim)] +
		   A[indexof(i,  j-1,myydim)] +
		   A[indexof(i+1,j,  myydim)] +
		   A[indexof(i-1,j,  myydim)]);

	  double change =  B[indexof(i,j,myydim)] - A[indexof(i,j,myydim)];
	  if (change < 0) change = - change;
	  if (change > maxChange) maxChange = change;
	}
    }
  
  temp = A;
  A =B;	
  B=temp;  

  BgElapse(20e-5);

}


void Chunk::processStripfromleft(Msg* aMessage){

//Do nothing if this is 0 Pe because the message will be a dummy for the 0 Pe.

 BgElapse(2.5e-5);
 //BgPrint("%f:TEST STRING LEFT\n");
  if(thisIndex !=0) {
				
    for(int i=0;i<myydim;i++)
      A[indexof(0,i,myydim)] = aMessage->strip[i];
  }
  
}

void Chunk::processStripfromright(Msg* aMessage){

  //Do nothing if this is Pe number:(total-1) because this will be a dummy message for that Pe.

  BgElapse(2.5e-5);
 //BgPrint("%f:TEST STRING RIGHT\n");
  if(thisIndex != total -1){
    if(thisIndex != 0)
      for(int i=0;i<myydim;i++)
	A[indexof(myxdim+1,i,myydim)] = aMessage->strip[i];
    else
      for(int i=0;i<myydim;i++)
	A[indexof(myxdim,i,myydim)] = aMessage->strip[i];
  }
}


Main::Main(CkArgMsg *m)
{
  int x,y,k;

  if(m->argc != 4) CkAbort("Wrong parameters\n");
	
  x = atoi(m->argv[1]);
  y = atoi(m->argv[2]);
  k = atoi(m->argv[3]);

  if(x < k) CkAbort("Xdim must be greater than k");
  if (k < CkNumPes() || k%CkNumPes()) CkAbort("k must be multiple of numPes.");

  chunk_arr = CProxy_Chunk::ckNew(k,x,y,k);
  //chunk_arr.setReductionClient(workover, (void*)NULL);
  //   CkCallback *cb = new CkCallback(CkIndex_Chunk::workover(NULL), CkArrayIndex1D(0), chunk_arr);
  //chunk_arr.ckSetReductionClient(cb);


  BgElapse(.25e-5);
  //BgPrint0("STARTING FIRST STEP AT:%f\n");


  //  chunk_arr.stepOver(new VoidMsg);
  chunk_arr.singleStep(new VoidMsg());
  startTime = BgGetTime();
  numFinished = 0;
}




#include "parallelJacobi.def.h"




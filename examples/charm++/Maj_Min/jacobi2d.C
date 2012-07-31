#include <controlPoints.h>
#include <stdio.h>
#include <stdlib.h>
#include "TempCore.h"
#include "jacobi2d.decl.h"

#define PRIOR

int minElements;
int majElements;
int MAJLOOP = 8000000;
FILE *f;
int ldbTime = 5;
CkEntryOptions *opts, *opts1;
// See README for documentation

/*readonly*/ CProxy_Main mainProxy;
CProxy_Minor minorProxy;
CProxy_ProcFreq freqProxy;
// specify the number of worker chares in each dimension

// We want to wrap entries around, and because mod operator % sometimes misbehaves on negative values, 
// I just wrote these simple wrappers that will make the mod work as expected. -1 maps to the highest value.

double start;
//#define total_iterations 200

class Main:public CBase_Main
{
public:
  int recieve_count;
  CProxy_Jacobi array;
  CProxy_Minor arrayMin;
  int num_chares;
  int iterations;
  int total_iterations;
  double startTime;

    Main (CkArgMsg * m)
  {
    if (m->argc < 3)
      {
	CkPrintf ("%s [array_size] [block_size]\n", m->argv[0]);
	CkAbort ("Abort");
      }
    f = fopen ("temp.out", "w");
    // set iteration counter to zero
    iterations = 0;

    // store the main proxy
    mainProxy = thisProxy;
    freqProxy = CProxy_ProcFreq::ckNew ();
    majElements = minElements = 8;
    majElements = atoi (m->argv[1]);
    minElements = atoi (m->argv[2]);

    // print info

    total_iterations = 200;
    if (m->argc > 3)
      {
	total_iterations = atoi (m->argv[3]);
      }

    // Create new array of worker chares

    array = CProxy_Jacobi::ckNew (majElements, majElements);
    arrayMin = CProxy_Minor::ckNew (minElements, minElements);
    minorProxy = arrayMin;
    CkPrintf
      ("************** majorElements=%d minorElements=%d iterations=%d ********************\n",
       majElements, minElements, total_iterations);
    // save the total number of worker chares we have in this simulation
    num_chares = majElements + minElements;

    //Start the computation
    startTime = CkWallTimer ();
    start = startTime;
    recieve_count = 0;
#ifdef PRIOR
    opts = new CkEntryOptions ();
    opts1 = new CkEntryOptions ();
    opts->setPriority (-100);
    opts1->setPriority (100);

    array[0].begin_iteration (1, opts);
      for(int i=0;i<7;i++)
              arrayMin[i].begin_iteration(1,opts1);
//      arrayMin.begin_iteration(1,opts1);
#else
    array[0].begin_iteration (1);
    for (int i = 0; i < 7; i++)
      arrayMin[i].begin_iteration (1);
#endif
//      arrayMin.begin_iteration(1,opts1);
  }

  void exiting()
  {
	CkExit();
  }

  // Each worker reports back to here when it completes an iteration
  void report (CkReductionMsg *msg)
  {
    recieve_count++;
    double totaltime = CkWallTimer () - startTime;
//    printf("coming in report--------------------------------------------- rec=%d\n",recieve_count);
    if (2 == recieve_count)
//      if(minElements+1 == recieve_count)
      {
	if (iterations == total_iterations)
	  {
	    CkPrintf
	      ("Completed %d iterations; last iteration time: %.6lf total time=%f\n",
	       iterations, totaltime, CkWallTimer () - start);
//	    CkExit ();
		mainProxy.exiting();
	  }
	else
	  {
if(iterations==1) useThisCriticalPathForPriorities();
	    if (iterations % 1 == 0)
	      {
		char tempArr[50];
		for (int x = 0; x < 50; x++)
		  tempArr[x] = 0;
		float avg = 0;
		for (int pe = 0; pe < CkNumPes (); pe++)
		  {
		    sprintf (tempArr + strlen (tempArr), "%d ",
			     (int) getTemp (pe));
		    avg += getTemp (pe);
		    //if(gmaxTemp<stats->procs[pe].pe_temp) gmaxTemp=stats->procs[pe].pe_temp;
		  }
		avg /= CkNumPes ();
		CkPrintf
		  ("starting new iteration; iteration %d time: %.6lf AVG=%.6lf\n",
		   iterations, totaltime, avg);
		//printf("********* iteration=%d AVG temp=%f **********\n",iterations,avg);
//                      sprintf(tempArr,"%d %f",iterations,avg);
		// printf("temps=%s\n",tempArr);
		writeTemps (f, tempArr);
//		if (CkMyPe () == 0)
		  freqProxy.measureFreq (iterations, total_iterations);
	      }

//                CkPrintf("starting new iteration; iteration %d time: %.6lf\n", iterations, totaltime);
	    recieve_count = 0;
	    iterations++;
	    // Call begin_iteration on all worker chares in array
	    startTime = CkWallTimer ();
#ifdef PRIOR
	    opts = new CkEntryOptions ();
	    opts1 = new CkEntryOptions ();
	    opts->setPriority (-100);
	    opts1->setPriority (100);

	    array[0].begin_iteration (1, opts);
                for(int i=0;i<7;i++)
                        arrayMin[i].begin_iteration(1,opts1);
//              arrayMin.begin_iteration(1,opts1);              
#else
	    array[0].begin_iteration (1);
	    for (int i = 0; i < 7; i++)
	      arrayMin[i].begin_iteration (1);
#endif

	  }
      }
  }
};

class ProcFreq:public CBase_ProcFreq
{
public:
  int lowFreq;
  int highFreq;
    ProcFreq ()
  {
    lowFreq = highFreq = 0;
  }
  void measureFreq (int it, int total_itr)
  {
    if (cpufreq_sysfs_read (CkMyPe ()) == 1596000)
      lowFreq += 1;
    else
      highFreq += 1;
    if (it == total_itr - 1 && CkMyPe () % 4 == 0)
      printf
	("PROC#%d ddddddddddddddddddddddddddddddddddddddddddddddddddd h=%d l=%d\n",
	 CkMyPe (), highFreq, lowFreq);


  }


};


class Jacobi:public CBase_Jacobi
{
public:
//    int messages_due;
  int iterations;
  float *arr;
  int useLB;
  // Constructor, initialize values
    Jacobi (int x)
  {
    int i, j;
      majElements = x;
      useLB = 1;
      arr = new float[1000];
    for (int i = 0; i < 1000; i++)
        arr[i] = rand ();
      usesAtSync = CmiTrue;
      iterations = 0;
    // allocate two dimensional array
  }

  void pup (PUP::er & p)
  {
    CBase_Jacobi::pup (p);
//        p|messages_due;
    p | iterations;
    p | useLB;
    if (p.isUnpacking ())
      arr = new float[1000];
    for (int i = 0; i < 1000; i++)
      p | arr[i];
/* There may be some more variables used in doWork */
  }


  // a necessary function which we ignore now
  // if we were to use load balancing and migration
  // this function might become useful
  Jacobi (CkMigrateMessage * m)
  {
  }

  ~Jacobi ()
  {
    delete[]arr;
  }

  // Perform one iteration of work
  // The first step is to send the local state to the neighbors
  void begin_iteration (int i)
  {
//printf("PROC#%d coming for Jacobi[%d] iteration=%d majElements=%d\n",CkMyPe(),thisIndex,iterations,majElements);
/*
if ( iterations %10 ==0 && useLB) {
printf("calling sync\n");
	useLB = 0;
        AtSync();
}
else
*/
    {
//printf("PROC#%din elsssssssssssssssssssssss\n",CkMyPe());
      useLB = 1;
      iterations++;


      check_and_compute ();
    }
  }

  void check_and_compute ()
  {
//       if (--messages_due == 0) 
//          messages_due = 4;
      compute ();
//          mainProxy.report();
      if (thisIndex < majElements - 1)
	{
//                      printf("DONE WITH index=%d and calling for ind=%d\n",thisIndex,thisIndex+1);
#ifdef PRIOR
	  opts = new CkEntryOptions ();
	  opts1 = new CkEntryOptions ();
	  opts->setPriority (-100);
	  opts1->setPriority (100);

//printf("-------- Jacobi[%d] sending message to next one at time=%f\n",thisIndex,CkWallTimer());
	  thisProxy[thisIndex + 1].begin_iteration (1, opts);
        for(int i=(thisIndex+1)*7;i<(thisIndex+1)*7+7;i++)
                minorProxy[i].begin_iteration(1,opts1);
#else
	  thisProxy[thisIndex + 1].begin_iteration (1);
	  for (int i = (thisIndex + 1) * 7; i < (thisIndex + 1) * 7 + 7; i++)
	    minorProxy[i].begin_iteration (1);
#endif
	}
      else
	{
//                      printf("CAlling report Jacobi[%d] time=%f!!!!!!!!!!1\n",thisIndex,CkWallTimer());

//	  else
//	    mainProxy.report ();
	}
	if (iterations % ldbTime == 4) AtSync();
	else contribute(CkCallback(CkIndex_Main::report(NULL),mainProxy));
  }

  // Check to see if we have received all neighbor values yet
  // If all neighbor values have been received, we update our values and proceed
  void compute ()
  {
    // We must create a new array for these values because we don't want to update any of the
    // the values in temperature[][] array until using them first. Other schemes could be used
    // to accomplish this same problem. We just put the new values in a temporary array
    // and write them to temperature[][] after all of the new values are computed.

    for (int i = 1; i < MAJLOOP; ++i)
      {
	// update my value based on the surrounding values
	int ind = i % 1000;
	float x =
	  arr[ind] + arr[ind] + arr[ind + 1] + arr[ind + 2] + arr[ind + 3];
      }

    // Enforce the boundary conditions again

  }



  void ResumeFromSync ()
  {
//      printf("Jacobi[%d] calling resumeSync\n",thisIndex);
//    if (thisIndex == 0)
//      mainProxy.report ();
//CkPrintf("Coming in MAJ MAJ MAJ RRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR ++++++++\n");
	contribute(CkCallback(CkIndex_Main::report(NULL),mainProxy));
  }
};

class Minor:public CBase_Minor
{
public:
//    int messages_due;
  int iterations;
  float *arr;
  int useLB;
  // Constructor, initialize values
    Minor (int x)
  {
    int i, j;
      useLB = 1;
      minElements = x;
      arr = new float[1000];
    for (int i = 0; i < 1000; i++)
        arr[i] = rand ();
      usesAtSync = CmiTrue;
      iterations = 0;
    // allocate two dimensional array
  }

  void pup (PUP::er & p)
  {
    CBase_Minor::pup (p);
//        p|messages_due;
    p | iterations;
    p | useLB;
    if (p.isUnpacking ())
      arr = new float[1000];
    for (int i = 0; i < 1000; i++)
      p | arr[i];
/* There may be some more variables used in doWork */
  }


  // a necessary function which we ignore now
  // if we were to use load balancing and migration
  // this function might become useful
  Minor (CkMigrateMessage * m)
  {
  }

  ~Minor ()
  {
    delete[]arr;
  }

  // Perform one iteration of work
  // The first step is to send the local state to the neighbors
  void begin_iteration (int i)
  {
//printf("XXXXXXXXX Minor[%d] coming in begin_itertaion at time=%f\n",thisIndex,CkWallTimer());
    useLB = 1;
    iterations++;


    check_and_compute ();
  }

  void check_and_compute ()
  {
//       if (--messages_due == 0) 

//          messages_due = 4;
    compute ();
    if (iterations % ldbTime == 4/* || iterations == 100*/)
      {
//                        printf("MINOR[%d] itr=%d ----------------------------- ssssssssssssss\n",thisIndex,iterations);
	AtSync ();
      }

    else
//      mainProxy.report ();
	contribute(CkCallback(CkIndex_Main::report(NULL),mainProxy));


  }

  // Check to see if we have received all neighbor values yet
  // If all neighbor values have been received, we update our values and proceed
  void compute ()
  {
    // We must create a new array for these values because we don't want to update any of the
    // the values in temperature[][] array until using them first. Other schemes could be used
    // to accomplish this same problem. We just put the new values in a temporary array
    // and write them to temperature[][] after all of the new values are computed.

    for (int i = 1; i < MAJLOOP * .6; ++i)
      {
	// update my value based on the surrounding values
	int ind = i % 1000;
	float x =
	  arr[ind] + arr[ind] + arr[ind + 1] + arr[ind + 2] + arr[ind + 3];
      }

    // Enforce the boundary conditions again

  }


  void ResumeFromSync ()
  {
//    mainProxy.report ();;
//	CkPrintf("Coming in MIN MIN MIN RRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR \n");
	contribute(CkCallback(CkIndex_Main::report(NULL),mainProxy));
  }
};

#include "jacobi2d.def.h"

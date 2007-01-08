/*

   interpolatelog.C

   Author: Isaac Dooley
   email : idooley2@uiuc.edu
   date  : Jan 06, 2007

    This uses the gsl library for least-square fitting. It is a freely available GPL'ed library

*/



#include "blue.h"
#include "blue_impl.h"
#include "blue.h"
#include "blue_impl.h"
#include "blue_types.h"
#include "bigsim_logs.h"
#include "assert.h"
#include <sys/stat.h>
#include <sys/types.h>

#include <gsl/gsl_multifit.h>

#include <string>
#include <iostream>
#include <map>


extern BgTimeLineRec* currTline;
extern int currTlineIdx;


#define OUTPUTDIR "newtraces/"


int main()
{
  int totalProcs, numX, numY, numZ, numCth, numWth, numPes;
  BgTimeLineRec *tlinerecs;


  bool done = false;
  double newtime=1.0;
  std::string eventname;
  std::map<std::string,double> newtimes;
  std::cout << "Enter event name followed by its new time duration.\nEnter \"none -1.0\" after finishing. Don't use any spaces in names." << std::endl;
  while(!done) {
    std::cin >> eventname >> newtime;
    if(newtime < 0.0)
        done = 1;
    else
        newtimes[eventname]=newtime;
  }

  std::cout << "You entered " << newtimes.size() << " distinct events with their associated durations" << std::endl;


  // load bg trace summary file
  printf("Loading bgTrace ... \n");
  int status = BgLoadTraceSummary("bgTrace", totalProcs, numX, numY, numZ, numCth, numWth, numPes);
  if (status == -1) exit(1);
  printf("========= BgLog Version: %d ========= \n", bglog_version);
  printf("Found %d (%dx%dx%d:%dw-%dc) emulated procs on %d real procs.\n", totalProcs, numX, numY, numZ, numWth, numCth, numPes);

  int* allNodeOffsets = BgLoadOffsets(totalProcs,numPes);

  tlinerecs = new BgTimeLineRec[totalProcs];

  printf("========= Loading All Logs ========= \n");

  // load each individual trace file for each bg proc
  for (int i=0; i<totalProcs; i++)
  {
    int procNum = i;
    currTline = &tlinerecs[i];
    currTlineIdx = procNum;
    int fileNum = BgReadProc(procNum,numWth,numPes,totalProcs,allNodeOffsets,tlinerecs[i]);
    CmiAssert(fileNum != -1);
    printf("Load log of BG proc %d from bgTrace%d... \n", i, fileNum);

    BgTimeLine &timeLine = tlinerecs[i].timeline; // Really a CkQ< BgTimeLog *>

    printf("%d entries in timeLine\n", timeLine.length());

    // Scan through each event for this emulated processor
    for(int j=0;j<timeLine.length();j++){
        BgTimeLog* timeLog = timeLine[j];
        std::string name(timeLog->name);

        // If name of this event is one that needs to have its duration modified
        if( newtimes.find(name) != newtimes.end() ) {
            double oldstart = timeLog->startTime;
            double oldend   = timeLog->endTime;
            double newstart = oldstart;
            double newend   = oldstart+newtimes[name];

            timeLog->startTime = newstart;
            timeLog->endTime   = newend;

            printf("Rewriting duration of event %d name=%s from [%.10lf , %.10lf] to [%.10lf , %.10lf]\n", j, timeLog->name, oldstart,oldend,newstart,newend);

            for(int m=0;m<timeLog->msgs.length();m++){
                double oldsendtime = timeLog->msgs[m]->sendTime;
                double newsendtime;

                if(oldstart == oldend){
                    newsendtime = oldstart;
                } else {
                    // Linearly map the old range onto the new range
                    newsendtime = newstart + (oldsendtime-oldstart)/(oldend-oldstart) * (newend-newstart);
                }

                timeLog->msgs[m]->sendTime = newsendtime;
                printf("changing message %d send time from %.10lf to %.10lf\n", m, oldsendtime, newsendtime);
            }

        }

    }

  }


// Create output directory
    mkdir(OUTPUTDIR, 0777);

// We should write out the timelines to the same number of files as we started with.
// The mapping from VP to file was probably round robin. Here we cheat and make just one file
// TODO : fix this to write out in same initial pattern
    BgWriteTraceSummary(totalProcs, 1, numX, numY, numZ, numCth, numWth, OUTPUTDIR);
//  for(int i=0; i<numPes; i++)
    BgWriteTimelines(0, &tlinerecs[0], totalProcs, numWth, OUTPUTDIR);

  delete [] allNodeOffsets;
  printf("End of program\n");
}




void fitCurve(){


        int n = 5;  // number of sample input evaluations
        int cs = 2; // number of coefficients

        gsl_multifit_linear_workspace * work = gsl_multifit_linear_alloc(n,cs);

        //  Find C     where y=Xc

        gsl_matrix *X;  // Each row is a set of parameters  [1, a, a^2, b, c, a*b*c] for each input parameter set
        gsl_vector *y;  // vector of cycle accurate times for each input parameter set
        gsl_vector *c;  // coefficients which are produced by least square fit
        gsl_matrix *cov;
        double chisqr;

        X = gsl_matrix_alloc (n,cs);
        y = gsl_vector_alloc (n);
        c = gsl_vector_alloc(cs);
        cov = gsl_matrix_alloc(cs,cs);

        double val;

        for(int i=0;i<n;i++){
            for(int j=0;j<cs;j++){
                gsl_matrix_set(X,i,j,val);
            }
        }

        for(int i=0;i<n;i++){
            gsl_vector_set(y,i,val);
        }

        // Do we need to initialize c?
        for(int j=0;j<cs;j++){
            gsl_vector_set(c,j,0.0);
        }

        gsl_multifit_linear(X,y,c,cov,&chisqr,work);

        // Estimate time for a given set of parameters p
        gsl_vector *desired_params;
        double desired_time, desired_time_err;
        gsl_multifit_linear_est(desired_params,c,cov,&desired_time,&desired_time_err);

        // We now have a predicted time for the desired parameters

        gsl_multifit_linear_free(work);
}


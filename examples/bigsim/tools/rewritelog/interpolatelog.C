/*

   interpolatelog.C

   Author: Isaac Dooley
   email : idooley2@uiuc.edu
   date  : Jan 06, 2007

    This uses the gsl library for least-square fitting. Gnu Scientific Library is available under GPL

    Currently we hard code in two parameters some places. This should be fixed


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

#include <EventInterpolator.h>

#include <string>
#include <iostream>
#include <fstream>
#include <map>

extern BgTimeLineRec* currTline;
extern int currTlineIdx;

#define OUTPUTDIR "newtraces/"


int main()
{


  // Load in Mambo Times
  EventInterpolator interpolator("MamboTimes"); 


  double parray[2] = {2.0,4.0};
  std::cout << "Interpolated value chisqr=" << interpolator.get_chisqr() << " value = " << interpolator.predictTime(parray) << std::endl;
  

  // Load in Parameter File



#if 0
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
  
#endif
 
  
  std::cout << "End of program" << std::endl;
}


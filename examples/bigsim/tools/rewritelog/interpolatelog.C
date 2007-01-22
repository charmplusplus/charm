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
#include <utility> // for std::pair
#include <vector>

#define OUTPUTDIR "newtraces/"
#define CYCLE_TIMES_FILE "nopme"
#define sec_per_cycle 0.00000000025
#define cycle_per_sec 4000000000.0

#undef DEBUG

extern BgTimeLineRec* currTline;
extern int currTlineIdx;

int main()
{
    // Load in Mambo Times
    EventInterpolator interpolator(CYCLE_TIMES_FILE);

    int totalProcs, numX, numY, numZ, numCth, numWth, numPes;
 

	interpolator.printCoefficients();

    // load bg trace summary file
    printf("Loading bgTrace ... \n");
    int status = BgLoadTraceSummary("bgTrace", totalProcs, numX, numY, numZ, numCth, numWth, numPes);
    if (status == -1) exit(1);
    printf("========= BgLog Version: %d ========= \n", bglog_version);
    printf("Found %d (%dx%dx%d:%dw-%dc) emulated procs on %d real procs.\n", totalProcs, numX, numY, numZ, numWth, numCth, numPes);

    int* allNodeOffsets = BgLoadOffsets(totalProcs,numPes);

    printf("========= Loading All Logs ========= \n");

    // load each individual trace file for each bg proc

    unsigned rewritten_count=0;
    unsigned total_count=0;
    bool negative_durations_occured = false;

    printf("Loading bgTrace files ...\n");

    // Create output directory
    mkdir(OUTPUTDIR, 0777);

    for (int procNum=0; procNum<totalProcs; procNum++)
    {
	    BgTimeLineRec tlinerec;
	  	  
		currTline = &tlinerec;
		currTlineIdx = procNum;
        int fileNum = BgReadProc(procNum,numWth,numPes,totalProcs,allNodeOffsets,tlinerec);
        CmiAssert(fileNum != -1);

#ifdef DEBUG
    printf("Load log of BG proc %d from bgTrace%d... \n", procNum, fileNum);
#endif

        BgTimeLine &timeLine = tlinerec.timeline; // Really a CkQ< BgTimeLog *>

#ifdef DEBUG
        printf("%d entries in timeLine\n", timeLine.length());
#endif

        // Scan through each event for this emulated processor
        for(int j=0;j<timeLine.length();j++){
            BgTimeLog* timeLog = timeLine[j];
            std::string name(timeLog->name);
            total_count++;
            // If name of this event is one that needs to have its duration modified
            if( interpolator.haveNewTiming(procNum,timeLog->seqno) ) {

                double newduration;
                if( interpolator.haveExactTime(procNum,timeLog->seqno) )
                    newduration = interpolator.lookupExactTime(procNum,timeLog->seqno);
                else
                    newduration = interpolator.predictTime(procNum,timeLog->seqno) * sec_per_cycle;

                if(newduration >= 0.0){

                    rewritten_count++;

                    double oldstart = timeLog->startTime;
                    double oldend   = timeLog->endTime;
                    double newstart = oldstart;
                    double newend   = oldstart+newduration;

                    timeLog->startTime = newstart;
                    timeLog->endTime   = newend;

//                     printf("Rewriting duration of event %d name=%s from [%.10lf , %.10lf] to [%.10lf , %.10lf]\n", j, timeLog->name, oldstart,oldend,newstart,newend);

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
//                         printf("changing message %d send time from %.10lf to %.10lf\n", m, oldsendtime, newsendtime);
                    }
                }
                else {
                    negative_durations_occured=true;
                }
            }

        }

		// Write out the file
		BgWriteTimelines(procNum,&tlinerec,1,numWth,OUTPUTDIR);

    }
    if(negative_durations_occured){
        cerr << "======================  WARNING ======================" << endl;
        cerr << "||  One or more new durations were less than zero. \n||  This probably means your model or input times are \n||  not good enough." << endl;
        cerr << "======================================================" << endl;
    }

    interpolator.printMinInterpolatedTimes();

    printf("Writing new bgTrace files ...\n");


    // Write out the timelines to the same number of files as we started with.
    BgWriteTraceSummary(totalProcs, 1, numX, numY, numZ, numCth, numWth, OUTPUTDIR);
    
    delete [] allNodeOffsets;

    std::cout << "Of the " << total_count << " events found in the bgTrace files, " << rewritten_count << " were found in the param files" << endl;

    std::cout << "Those " << rewritten_count << " events were given new durations" << std::endl;

    interpolator.printMatches();


    std::cout << "End of program" << std::endl;

}


/*

    @file interpolatelog.C

	@author: Isaac Dooley
	email : idooley2@uiuc.edu
	date  : Jan 2007

    This uses the gsl library for least-square fitting. Gnu Scientific Library is available under GPL

    Currently we hard code in two parameters some places.

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

// Scale the duration of all unknown events by this factor
#define time_factor 0.2

// Set these for more output:
#define DEBUG
#undef PRINT_NEW_TIMES
#define WRITE_OUTPUT_FILES

// Needed to communicated with the bigsim file reading and writing routines:
extern BgTimeLineRec* currTline;
extern int currTlineIdx;


int main()
{
    // Load in Mambo Times
    EventInterpolator interpolator(CYCLE_TIMES_FILE);

    int totalProcs, numX, numY, numZ, numCth, numWth, numPes;
	double ratio_sum=0.0;
	unsigned long ratio_count= 0;

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


    for (int fileNum=0; fileNum<numPes; fileNum++){
        BgTimeLineRec *tlinerecs = new BgTimeLineRec[numPes];
        int rec_count = 0;

        for(int procNum=fileNum;procNum<totalProcs;procNum+=numPes){

            BgTimeLineRec &tlinerec = tlinerecs[rec_count];
            rec_count++;

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

                double oldstart = timeLog->startTime;
                double oldend   = timeLog->endTime;
                double oldduration = oldend-oldstart;
                double newduration;

                // If this event occurs in the paramerter file and cycle-accurate simulations, use that data to predict its runtime
                if( interpolator.haveNewTiming(procNum,timeLog->seqno) ) {
                    if( interpolator.haveExactTime(procNum,timeLog->seqno) )
                        newduration = interpolator.lookupExactTime(procNum,timeLog->seqno) * sec_per_cycle;
                    else
                        newduration = interpolator.predictTime(procNum,timeLog->seqno) * sec_per_cycle;

                }
                // If event is not in parameter file then we just scale its duration by a simple constant
                else {
                newduration = oldduration*time_factor ;
                }


                if(newduration >= 0.0) {
                    double newstart = oldstart;
                    double newend   = oldstart+newduration;

                    timeLog->startTime = newstart;
                    timeLog->endTime   = newend;
                    timeLog->execTime  = newduration;
                    rewritten_count++;



#ifdef PRINT_NEW_TIMES
                    printf("Rewriting duration of event %d name=%s from [%.10lf , %.10lf] (%.10lf) to [%.10lf , %.10lf] (%.10lf) ratio=%.10lf\n", j, timeLog->name, oldstart,oldend,oldend-oldstart,newstart,newend,newend-newstart,(oldend-oldstart)/(newend-newstart));
#endif

                    double ratio = (oldend-oldstart)/(newend-newstart);
                    if(ratio >= 0.5 && ratio <= 50.0){
                    ratio_sum += ratio;
                    ratio_count ++;
                    }

                    // Rewrite times of messages sent from this event
                    for(int m=0;m<timeLog->msgs.length();m++){
                    double oldsendtime = timeLog->msgs[m]->sendTime;
                    double newsendtime;

                    assert(oldsendtime <= oldend && oldsendtime >= oldstart );

                    if(oldstart == oldend){
                        newsendtime = oldstart;
                    } else {
                        // Linearly map the old range onto the new range
                        newsendtime = newstart + ((oldsendtime-oldstart)*(newend-newstart))/(oldend-oldstart);
                    }

                    timeLog->msgs[m]->sendTime = newsendtime;

#ifdef PRINT_NEW_TIMES
                    printf("pe=%d changing message %d send time from %.10lf to %.10lf\n", procNum, m, oldsendtime, newsendtime);
#endif


                    assert(newsendtime <= newend && newsendtime >= newstart );


                    }
                }
                else {
                negative_durations_occured=true;
                }
            }




        }

#ifdef WRITE_OUTPUT_FILES
            // Write out the file
            cout << "writing " << rec_count << " simulated processors to this bgTrace file" << endl;
            BgWriteTimelines(fileNum,tlinerecs,rec_count,numWth,OUTPUTDIR);
#endif
        delete[] tlinerecs;

    }


    if(negative_durations_occured){
	  cerr << "======================  WARNING ======================" << endl;
	  cerr << "||  One or more new durations were less than zero. \n||  This probably means your model or input times are \n||  not good enough." << endl;
	  cerr << "======================================================" << endl;
    }

    interpolator.printMinInterpolatedTimes();

    printf("Writing new bgTrace files ...\n");

	printf("average duration ratio: %.15lf", ratio_sum / (double)ratio_count);

#ifdef WRITE_OUTPUT_FILES
    // Write out the timelines to the same number of files as we started with.
    BgWriteTraceSummary(totalProcs, numPes, numX, numY, numZ, numCth, numWth, OUTPUTDIR);
#endif

    delete [] allNodeOffsets;

    std::cout << "Of the " << total_count << " events found in the bgTrace files, " << rewritten_count << " were found in the param files" << endl;

    std::cout << "Those " << rewritten_count << " events were given new durations" << std::endl;

    interpolator.printMatches();

    std::cout << "End of program" << std::endl;

}


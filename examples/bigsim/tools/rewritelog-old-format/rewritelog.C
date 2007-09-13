/*
   rewritelog.C

   Author: Isaac Dooley
   email : idooley2@uiuc.edu
   date  : Jan 06, 2007

   This program will take in a set of bgTrace files from a run
   of the Big Emulator and it will rewrite the files so that
   named events can be replaced with better approximate timings
   supplied interactively by the user of this program. For example
   a cycle-accurate simulator might be able to give the exact times
   for some Charm++ entry methods.

   The modified bgTrace files are put into a directory called
   "newtraces"

   The user supplies durations which are used to change the end
   times for each log entry with the associated name. Any messages
   sent during this duration are mapped linearly onto the new times
   range for the log entry. For example, an message sent one-third
   the way through the old time range for the event will still occur
   one-third the way through the new time range.

   Currently nothing else in the trace files is modified.

   This program was written by someone with little experience with
   the POSE based Big Simulator, so these traces might not quite
   act as the author expected. Please report any problems to
   ppl@cs.uiuc.edu


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

#include <string>
#include <iostream>
#include <map>

#define OUTPUTDIR "newtraces/"


int main()
{
    int totalProcs, numX, numY, numZ, numCth, numWth, numPes;

    // Create output directory
    mkdir(OUTPUTDIR, 0777);

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

    printf("========= Loading All Logs ========= \n");

    // load each individual trace file for each bg proc
    for (int procNum=0; procNum<totalProcs; procNum++)
    {
        BgTimeLineRec tlinerec;

        int procNum = i;
        int fileNum = BgReadProc(procNum,numWth,numPes,totalProcs,allNodeOffsets,tlinerec);
        CmiAssert(fileNum != -1);
        printf("Load log of BG proc %d from bgTrace%d... \n", procNum, fileNum);

        BgTimeLine &timeLine = tlinerec.timeline; // Really a CkQ< BgTimeLog *>

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

        BgWriteTimelines(procNum, tlinerec, 1, numWth, OUTPUTDIR);

    }

    BgWriteTraceSummary(totalProcs, 1, numX, numY, numZ, numCth, numWth, OUTPUTDIR);

    delete [] allNodeOffsets;
    printf("End of program\n");
}



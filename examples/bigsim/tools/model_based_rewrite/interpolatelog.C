/*

    @file interpolatelog.C

	@author: Isaac Dooley
	email : idooley2@uiuc.edu
	date  : June 2007

    This uses the gsl library for least-square fitting. GNU Scientific Library is available under GPL

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

#define KEEP_RATIO 1.0

char *OUTPUTDIR = "newtraces-keep1.0/";
const char *CYCLE_TIMES_FILE = "expectedTimings";
#define sec_per_cycle 0.00000000025

// Scale the duration of all unknown events by this factor
// 0.5 means the new event durations will be twice as fast as the durations in the bgTrace files
#define time_dilation_factor 1.0


// Set these for more output:
#define DEBUG
//#undef PRINT_NEW_TIMES
#define WRITE_OUTPUT_FILES

// Needed to communicated with the bigsim file reading and writing routines:
extern BgTimeLineRec* currTline;
extern int currTlineIdx;


/** linearly map a point from an interval with sepcified bounds to a new interval linearly
    @return a value between new_lower and new_upper
*/
double map_linearly_to_interval(double val, double old_lower, double old_upper, double new_lower, double new_upper){
  double new_val;
  
	double new_temp;
  assert(old_upper >= old_lower);
  assert(val <= old_upper);
  assert(val >= old_lower);


	if((old_upper-old_lower) >0.000001)
	{
  if(val == old_lower)
	new_val = new_lower;
  else if(val == old_upper)
	new_val = new_upper;
  else
	{
			new_temp = (val-old_lower)/(old_upper-old_lower);
			assert(new_temp<=1.0);
			assert(new_temp>=0.0);
			new_val = new_lower + new_temp*(new_upper-new_lower);
	}
	}
	else
	{
		new_val=(new_lower+new_upper)/2;
	}
 // printf("Val %.8f  New_val %.8f  New Upper %.8f   New Lower %.8f Old Upper %.8f  Old Lower %.8f\n",val,new_val,new_upper,new_lower, old_upper, old_lower);
	assert(new_upper >= new_lower);
  assert(new_val <= new_upper);
  assert(new_val >= new_lower);

 
  return new_val;
}

int main()
{
    // Create model from sample Predicted Parameterized Times
    // (from cycle accurate simulator or real machine)
  EventInterpolator interpolator(CYCLE_TIMES_FILE, KEEP_RATIO);
  int totalProcs, numX, numY, numZ, numCth, numWth, numPes;
	double ratio_sum=0.0;
	unsigned long ratio_count= 0;
	double ratio_bracketed_sum=0.0;
	unsigned long ratio_bracketed_count = 0;

	double old_times_sum=0.0;
	double old_bracketed_times_sum=0.0;
	double new_bracketed_times_sum=0.0;

	ofstream of_ratio_all("ratio_all");
	ofstream of_ratio_bracketed("ratio_bracketed");

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

    unsigned found_event_count=0;
    unsigned total_count=0;
    bool negative_durations_occured = false;

    printf("Loading bgTrace files ...\n");

    // Create output directory
    mkdir(OUTPUTDIR, 0777);

		int numNodes = totalProcs / numWth;

    for (int fileNum=0; fileNum<numPes; fileNum++){
        BgTimeLineRec *tlinerecs = new BgTimeLineRec[totalProcs/numPes+1];
        int rec_count = 0;

        //for(int procNum=fileNum;procNum<totalProcs;procNum+=numPes){
				 for (int nodeNum=fileNum;nodeNum<numNodes;nodeNum+=numPes) {
						for (int procNum=nodeNum*numWth; procNum<(nodeNum+1)*numWth; procNum++) {

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

                double old_start = timeLog->startTime;
                double old_end   = timeLog->endTime;
                double old_duration = old_end-old_start;

						double old_bracket_start=0.0;
						double old_bracket_end=0.0;
						int have_bracket_start=0;
						int have_bracket_end=0;
						double begin_piece=0.0;
						double middle_piece=old_duration;
						double end_piece=0.0;
						assert(old_duration >= 0.0);
						assert(old_end >= 0.0);
						if(old_end > old_start){
						  old_times_sum += old_duration;
						  //FIXME: check only the right kind of events.
						  // Look for BG_EVENT_PRINT 'events' inside this event.
						  for(int i=0;i<timeLog->evts.length();i++){
								char *data = (char*)timeLog->evts[i]->data;
								if(strncmp(data,"startTraceBigSim",16)==0){
								  old_bracket_start = old_start+timeLog->evts[i]->rTime;
					  			have_bracket_start = 1;
								}
								else if(strncmp(data,"endTraceBigSim",14)==0){
					  			old_bracket_end = old_start+timeLog->evts[i]->rTime;
									  have_bracket_end = 1;
                 }
						  }
						  // If we have bracketed timings, the middle part will be the old
						  // bracketed time region, and the begin and end pieces will be non-zero
						  if(have_bracket_end && have_bracket_start){
								begin_piece = old_bracket_start - old_start;
								middle_piece = old_bracket_end - old_bracket_start;
								end_piece = old_duration - begin_piece - middle_piece;
								old_bracketed_times_sum += middle_piece;
								assert(begin_piece >= 0.0);
								assert(middle_piece >= 0.0);
								assert(end_piece >= 0.0);
				  		}
						  else{
								old_bracket_start = old_start;
								old_bracket_end = old_end;
								assert(old_bracket_end - old_bracket_start >= 0.0);
						  }
				  		// If this event occurs in the paramerter file and cycle-accurate simulations, use that data to predict its runtime
				  		if( interpolator.haveNewTiming(procNum,timeLog->seqno) ) {
								double old_middle_piece = middle_piece;
								
								// GAGAN : Making it directly based on the model 
								//middle_piece = interpolator.getNewTiming(procNum,timeLog->seqno) * sec_per_cycle;
								
								middle_piece = interpolator.getNewTiming(procNum,timeLog->seqno);
								
								found_event_count ++;
								double ratio =  old_middle_piece / middle_piece;
								if(ratio > 1e-10 && ratio < 1e10){
								  ratio_bracketed_sum += ratio;
								  ratio_bracketed_count ++;
									of_ratio_bracketed << ratio << endl;
								}
//								cout<<" used interpolation tool"<<endl;
							}
				  		// If event is not in parameter file then we just scale its duration by a simple constant
						  else {
	//							cout<<" used time dilation for "<< procNum <<"  "<< timeLog->seqno<<endl;
								middle_piece = middle_piece*time_dilation_factor ;
							}
				  if(middle_piece < 0.0) {
						middle_piece=0.0;
						negative_durations_occured=true;
				  }
				  if(have_bracket_end && have_bracket_start){
						new_bracketed_times_sum += middle_piece;
				  }
				  // Scale the begin and end pieces by time_dilation_factor;
				  begin_piece = begin_piece*time_dilation_factor;
				  end_piece = end_piece*time_dilation_factor;
				  assert(begin_piece >= 0.0);
				  assert(middle_piece >= 0.0);
				  assert(end_piece >= 0.0);
				  double new_start    = old_start;
				  double new_duration = begin_piece + middle_piece + end_piece;
				  double new_end      = new_start + new_duration;
				  timeLog->startTime = new_start;
				  timeLog->endTime   = new_end;
				  timeLog->execTime  = new_duration;
				  double new_bracket_start = new_start+begin_piece;
				  double new_bracket_end = new_start+begin_piece+middle_piece;
#ifdef PRINT_NEW_TIMES
				  printf("Rewriting duration of event %d name=%s from [%.10lf , %.10lf] (%.10lf) to [%.10lf , %.10lf] (%.10lf) ratio=%.10lf\n", j, timeLog->name, old_start,old_end,old_end-old_start,new_start,new_end,new_end-new_start,(old_end-old_start)/(new_end-new_start));
#endif
				  double ratio = (old_duration)/(new_duration);
				  if(ratio >= 1e-10 && ratio <= 1e10){
						ratio_sum += ratio;
						ratio_count ++;
						of_ratio_all << ratio << endl;
				  }
				  // Rewrite times of messages sent from this event
				  for(int m=0;m<timeLog->msgs.length();m++){
						double old_send = timeLog->msgs[m]->sendTime;
						double new_send, new_recv;
						double old_recv;
						assert(old_send <= old_end);
						assert(old_send >= old_start);
						// We have three places where the message is coming from
						// We linearly map the value into the beginning, middle, or end piece
						if(old_send < old_bracket_start){
						  new_send = map_linearly_to_interval(old_send, old_start,old_bracket_start,new_start,new_bracket_start);
						}
						else if(old_send < old_bracket_end){
					  	new_send = map_linearly_to_interval(old_send, old_bracket_start,old_bracket_end,new_bracket_start,new_bracket_end);
						}
						else {
						  new_send = map_linearly_to_interval(old_send, old_bracket_end,old_end,new_bracket_end,new_end);
						}
					timeLog->msgs[m]->sendTime = new_send;
#ifdef PRINT_NEW_TIMES
					printf("pe=%d changing message %d send time from %.10lf to %.10lf\n", procNum, m, old_send, new_send);
					printf("pe=%d compare %d send time from %.10lf to recv time %.10lf\n", procNum, m, new_send,timeLog->msgs[m]->recvTime  );
#endif
					old_recv = timeLog->msgs[m]->recvTime ;
					new_recv = (old_recv - old_send) + new_send;
					timeLog->msgs[m]->recvTime = new_recv; 
					assert(new_send <= new_end);
					assert(new_send >= new_start);
				  assert(old_recv >= old_send);
					if(new_recv < new_send)
					{
							printf( "Bad case send: %.10f recv: %.10f \n", new_send, new_recv); 
					}
					assert(new_recv >= new_send);
				  }
				}
      }
    }
   }
#ifdef WRITE_OUTPUT_FILES
            // Write out the file
            cout << "writing " << rec_count << " simulated processors to this bgTrace file" << endl;
            BgWriteTimelines(fileNum,tlinerecs,rec_count,OUTPUTDIR);
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
	printf("average duration speedup(including unbracketed pieces): %.8lf\n", ratio_sum / (double)ratio_count);
	printf("average bracketed speedup: %.8lf\n", ratio_bracketed_sum / (double)ratio_bracketed_count);
	cout << "Sum of times of bracketed portions of input logs: " << old_bracketed_times_sum << endl;
	cout << "Sum of times of bracketed portions of output logs: " << new_bracketed_times_sum << endl;
  const char* fname = "bgTrace";
#ifdef WRITE_OUTPUT_FILES
  // Write out the timelines to the same number of files as we started with.
  BgWriteTraceSummary(numPes, numX, numY, numZ, numWth, numCth, fname , OUTPUTDIR);
#endif
  delete [] allNodeOffsets;
  cout << "Of the " << total_count << " events found in the bgTrace files, " << found_event_count << " were found in the param files" << endl;
  interpolator.printMatches();
	cout << "The sum of all positive duration events from the original bgTrace files is: " << old_times_sum << endl;
  cout << "End of program" << endl;
}


#include <stdlib.h>
#include <stdio.h>
#include <string.h>


////////////////////////////////////////////////////////////////////////////////////////////////////
// Defines

#define SPU_DEC_FREQ  79800000  // Hz

#define RECV_FUNC              0
#define PRE_FETCHING_FUNC      1
#define FETCHING_FUNC          2
#define READY_FUNC             3
#define EXECUTED_FUNC          4
#define COMMIT_FUNC            5
#define USER_FUNC_START        6
#define NUM_USER_FUNCS        25


////////////////////////////////////////////////////////////////////////////////////////////////////
// Global Data

#define PROJ_BUF_SIZE  (4 * 1024)

// NOTE : !!! This structure must match the structure definition in spert_ppu.cpp !!!
typedef struct __projections_buffer_entry {

  //unsigned long long int startTime;
  //unsigned int runTime;
  unsigned int speIndex;
  unsigned int funcIndex;

  unsigned long long int recvTimeStart;
  unsigned int recvTimeEnd;
  unsigned int preFetchingTimeStart;
  unsigned int preFetchingTimeEnd;
  unsigned int fetchingTimeStart;
  unsigned int fetchingTimeEnd;
  unsigned int readyTimeStart;
  unsigned int readyTimeEnd;
  unsigned int userTimeStart;
  unsigned int userTimeEnd;
  unsigned int executedTimeStart;
  unsigned int executedTimeEnd;
  unsigned int commitTimeStart;    
  unsigned int commitTimeEnd;

  unsigned int userTime0Start;
  unsigned int userTime0End;
  unsigned int userTime1Start;
  unsigned int userTime1End;
  unsigned int userTime2Start;
  unsigned int userTime2End;

  unsigned long long int userAccumTime0;
  unsigned long long int userAccumTime1;
  unsigned long long int userAccumTime2;
  unsigned long long int userAccumTime3;

} ProjBufEntry;

ProjBufEntry projBuf[PROJ_BUF_SIZE];


////////////////////////////////////////////////////////////////////////////////////////////////////
// Function Prototypes

int createStsFile(int numEntries, char* filePrefix);


////////////////////////////////////////////////////////////////////////////////////////////////////
// Function Bodies

int main(int argc, char* argv[]) {

  char buf[1024];

  // Introduce Self
  printf("RawToProj\n");
  printf("  SPU_DEC_FREQ = %lf MHz\n", SPU_DEC_FREQ / 1000000.0);

  // Make sure there are two and only two command line parameter
  if (argc != 3) {
    printf("Converts an 'Offload API raw timing file' into a series of log files usable by the Charm++ 'Projections' tool.\n");
    printf("USAGE: %s rawFileName projectionsFileNamePrefix\n", argv[0]);
    return EXIT_FAILURE;
  }

  // Open the timing file
  FILE* projFile = fopen(argv[1], "r");
  if (projFile == NULL) {
    printf("ERROR : Unable to open \"%s\" for reading... exiting.\n", argv[1]);
    return EXIT_FAILURE;
  }

  // Seek in the timing file to EOF - 4 (to read the number of entries)
  fseek(projFile, -4, SEEK_END);

  // Read the number of entries in the timing file
  int numEntries = 0;
  if (sizeof(int) != fread((void*)(&numEntries), 1, sizeof(int), projFile)) {
    printf("ERROR : Unable to read total number of entries... exiting.\n");
    // Close the projFile; necessary to avoid a Resource leak
    fclose(projFile);
    return EXIT_FAILURE;
  }

  // Let the user know how many entries there are to process
  printf("Number of Entries: %d\n", numEntries);

  // Seek back to the beginning of the timing file (to read the entries)
  fseek(projFile, 0, SEEK_SET);

  // Create sts file
  if (!createStsFile(numEntries, argv[2])) {
    // Close the projFile; necessary to avoid a Resource leak
    fclose(projFile);
    return EXIT_FAILURE;
  }

  // Open the log files
  // TODO : Assuming only a single Cell chip for now
  FILE* logFile[9];
  int eventCount[9] = { 0 };
  unsigned long long int maxEndTime[9] = { 0 };

  // Write the log file headers
  sprintf(buf, "PROJECTIONS-RECORD ");
  int numEntryOffset = strlen(buf);
  for (int i = 0; i < 9; i++) {

    char fileNameBuf[256];

    // Open the log file
    sprintf(fileNameBuf, "%s.%d.log", argv[2], i);
    logFile[i] = fopen(fileNameBuf, "w+");
    if (logFile[i] == NULL) {
      printf("ERROR : Unable to open \"%s\" for writing... exiting.\n", fileNameBuf);
      // Close the projFile; necessary to avoid a Resource leak
      fclose(projFile);
      return EXIT_FAILURE;
    }

    // Write the header (without the number of entries for now)
    sprintf(buf, "PROJECTIONS-RECORD ");
    fwrite(buf, 1, strlen(buf), logFile[i]);
    sprintf(buf, "                          \n");  // Give plenty of room to fill in a number later
    fwrite(buf, 1, strlen(buf), logFile[i]);
  }

  // Create the time unit conversion (from spu decrementer cycles to ms)
  register double spuDecCyclePeriod_sec = 1.0 / ((double)SPU_DEC_FREQ);
  register double spuDecCyclePeriod_2us = spuDecCyclePeriod_sec * 1000000;
  register double spuDecCyclePeriod_2ns = spuDecCyclePeriod_sec * 1000000000;
  register unsigned int eventIDCounter[9] = { 0 };

  // Keep running summations
  register long long int sumCount = 0;
  register double readySum = 0.0;
  register double userTime0Sum = 0.0;
  register double userTime1Sum = 0.0;
  register double userTime2Sum = 0.0;
  register double userAccumTime0Sum = 0.0;
  register double userAccumTime1Sum = 0.0;
  register double userAccumTime2Sum = 0.0;
  register double userAccumTime3Sum = 0.0;

  // Process the contents of the timing file
  while (numEntries > 0) {

    // Calculate the number of bytes to read from the file
    register int numEntriesToRead = ((numEntries > PROJ_BUF_SIZE) ? (PROJ_BUF_SIZE) : (numEntries));
    register int numBytesToRead = numEntriesToRead * sizeof(ProjBufEntry);

    // Read the entries (PROJ_BUF_SIZE entries at a time)
    register size_t numBytesRead = fread((void*)projBuf, 1, numBytesToRead, projFile);
    if (numBytesRead != numBytesToRead) {
      printf("ERROR : Unable to read entries from file... exiting\n");
      // Close the projFile; necessary to avoid a Resource leak
      fclose(projFile);
      return EXIT_FAILURE;
    }

    // Decrement the numEntries left
    numEntries -= numEntriesToRead;

    // Process each entry in the buffer
    for (int i = 0; i < numEntriesToRead; i++) {


      // DEBUG
      /*
      printf("[DEBUG] SPE_%d :: { funcIndex = %d,\n"
             "[DEBUG] SPE_%d      recvTimeStart = %llu, recvTimeEnd = %u,\n"
             "[DEBUG] SPE_%d      preFetchingStart = %u, preFetchingTimeEnd = %u,\n"
             "[DEBUG] SPE_%d      fetchingStart = %u, fetchingTimeEnd = %u,\n"
             "[DEBUG] SPE_%d      readyStart = %u, readyTimeEnd = %u,\n"
             "[DEBUG] SPE_%d      executedStart = %u, executedTimeEnd = %u,\n"
             "[DEBUG] SPE_%d      commitTimeStart = %u, commitTimeEnd = %u\n"
             "[DEBUG] SPE_%d    }\n",
             projBuf[i].speIndex, projBuf[i].funcIndex,
             projBuf[i].speIndex, projBuf[i].recvTimeStart, projBuf[i].recvTimeEnd,
             projBuf[i].speIndex, projBuf[i].preFetchingTimeStart, projBuf[i].preFetchingTimeEnd,
             projBuf[i].speIndex, projBuf[i].fetchingTimeStart, projBuf[i].fetchingTimeEnd,
             projBuf[i].speIndex, projBuf[i].readyTimeStart, projBuf[i].readyTimeEnd,
             projBuf[i].speIndex, projBuf[i].executedTimeStart, projBuf[i].executedTimeEnd,
             projBuf[i].speIndex, projBuf[i].commitTimeStart, projBuf[i].commitTimeEnd,
             projBuf[i].speIndex
	    );
      */


      register int logFileIndex = projBuf[i].speIndex + 1;

      //// Create the BEGIN PROCESSING (2) event
      //// FORMAT : 2 mIdx eIdx itime event pe msglen irecvtime id1 id2 id3 icputime numpapievents
      //register double tmp_startTime_us = ((double)projBuf[i].startTime) * spuDecCyclePeriod_2us;
      //sprintf(buf, "2 0 %u %llu 0 0 0 0 0 0 0 0 0\n", projBuf[i].funcIndex, (unsigned long long int)tmp_startTime_us);
      //fwrite(buf, 1, strlen(buf), logFile[logFileIndex]);
      //eventCount[logFileIndex]++;
      //
      //// Create the END PROCESSING (3) event
      //// FORMAT : 3 mIdx eIdx itime event pe msglen irecvtime numpapievents
      //register unsigned long long int tmp_endTime = projBuf[i].startTime + ((unsigned long long int)projBuf[i].runTime);
      //register double tmp_endTime_us = ((double)tmp_endTime) * spuDecCyclePeriod_2us;
      //register unsigned long long int tmp_endTime_us_llu = (unsigned long long int)tmp_endTime_us;
      //if (tmp_endTime_us_llu > maxEndTime[logFileIndex]) maxEndTime[logFileIndex] = tmp_endTime_us_llu;
      //sprintf(buf, "3 0 %u %llu 0 0 0 0 0\n", projBuf[i].funcIndex, tmp_endTime_us_llu);
      //fwrite(buf, 1, strlen(buf), logFile[logFileIndex]);
      //eventCount[logFileIndex]++;

      register unsigned long long int tmp_startTime;
      register double tmp_startTime_ns;
      register unsigned long long int tmp_endTime;
      register double tmp_endTime_ns;


      ///// RECV /////
      // Create the BEGIN PROCESSING (2) event
      tmp_startTime_ns = ((double)projBuf[i].recvTimeStart) * spuDecCyclePeriod_2ns;
      sprintf(buf, "2 0 %u %llu 0 0 0 0 0 0 0 0 0\n", RECV_FUNC, (unsigned long long int)tmp_startTime_ns);
      fwrite(buf, 1, strlen(buf), logFile[logFileIndex]);
      eventCount[logFileIndex]++;

      // Create the CREATION (1) event
      // Create the END PROCESSING (3) event
      tmp_endTime = projBuf[i].recvTimeStart + projBuf[i].recvTimeEnd;
      tmp_endTime_ns = ((double)tmp_endTime) * spuDecCyclePeriod_2ns;
      sprintf(buf, "1 0 %u %llu %u %u 0 0\n",
              PRE_FETCHING_FUNC, (unsigned long long int)tmp_endTime_ns, eventIDCounter[logFileIndex], logFileIndex
             );
      fwrite(buf, 1, strlen(buf), logFile[logFileIndex]);
      sprintf(buf, "3 0 %u %llu 0 0 0 0 0\n", RECV_FUNC, (unsigned long long int)tmp_endTime_ns);
      fwrite(buf, 1, strlen(buf), logFile[logFileIndex]);
      eventCount[logFileIndex]++;


      ///// PRE_FETCHING /////
      // Create the BEGIN PROCESSING (2) event
      tmp_startTime = projBuf[i].recvTimeStart + projBuf[i].preFetchingTimeStart;
      tmp_startTime_ns = ((double)tmp_startTime) * spuDecCyclePeriod_2ns;
      sprintf(buf, "2 0 %u %llu %u %u 0 0 0 0 0 0 0\n",
              PRE_FETCHING_FUNC, (unsigned long long int)tmp_startTime_ns, eventIDCounter[logFileIndex], logFileIndex
             );
      fwrite(buf, 1, strlen(buf), logFile[logFileIndex]);
      eventIDCounter[logFileIndex]++;
      eventCount[logFileIndex]++;

      // Create the CREATION (1) event
      // Create the END PROCESSING (3) event
      tmp_endTime = projBuf[i].recvTimeStart + projBuf[i].preFetchingTimeEnd;
      tmp_endTime_ns = ((double)tmp_endTime) * spuDecCyclePeriod_2ns;
      sprintf(buf, "1 0 %u %llu %u %u 0 0\n",
              FETCHING_FUNC, (unsigned long long int)tmp_endTime_ns, eventIDCounter[logFileIndex], logFileIndex
             );
      fwrite(buf, 1, strlen(buf), logFile[logFileIndex]);
      sprintf(buf, "3 0 %u %llu 0 0 0 0 0\n", PRE_FETCHING_FUNC, (unsigned long long int)tmp_endTime_ns);
      fwrite(buf, 1, strlen(buf), logFile[logFileIndex]);
      eventCount[logFileIndex]++;


      ///// FETCHING /////
      // Create the BEGIN PROCESSING (2) event
      tmp_startTime = projBuf[i].recvTimeStart + projBuf[i].fetchingTimeStart;
      tmp_startTime_ns = ((double)tmp_startTime) * spuDecCyclePeriod_2ns;
      sprintf(buf, "2 0 %u %llu %u %u 0 0 0 0 0 0 0\n",
              FETCHING_FUNC, (unsigned long long int)tmp_startTime_ns, eventIDCounter[logFileIndex], logFileIndex
             );
      fwrite(buf, 1, strlen(buf), logFile[logFileIndex]);
      eventIDCounter[logFileIndex]++;
      eventCount[logFileIndex]++;

      // Create the CREATION (1) event
      // Create the END PROCESSING (3) event
      tmp_endTime = projBuf[i].recvTimeStart + projBuf[i].fetchingTimeEnd;
      tmp_endTime_ns = ((double)tmp_endTime) * spuDecCyclePeriod_2ns;
      sprintf(buf, "1 0 %u %llu %u %u 0 0\n",
              READY_FUNC, (unsigned long long int)tmp_endTime_ns, eventIDCounter[logFileIndex], logFileIndex
             );
      fwrite(buf, 1, strlen(buf), logFile[logFileIndex]);
      sprintf(buf, "3 0 %u %llu 0 0 0 0 0\n", FETCHING_FUNC, (unsigned long long int)tmp_endTime_ns);
      fwrite(buf, 1, strlen(buf), logFile[logFileIndex]);
      eventCount[logFileIndex]++;


      ///// READY /////
      register int func_index = USER_FUNC_START + projBuf[i].funcIndex;
      // Create the BEGIN PROCESSING (2) event
      tmp_startTime = projBuf[i].recvTimeStart + projBuf[i].readyTimeStart;
      tmp_startTime_ns = ((double)tmp_startTime) * spuDecCyclePeriod_2ns;
      sprintf(buf, "2 0 %u %llu %u %u 0 0 0 0 0 0 0\n",
              func_index, (unsigned long long int)tmp_startTime_ns, eventIDCounter[logFileIndex], logFileIndex
             );
      fwrite(buf, 1, strlen(buf), logFile[logFileIndex]);
      eventIDCounter[logFileIndex]++;
      eventCount[logFileIndex]++;

      // Create the CREATION (1) event
      // Create the END PROCESSING (3) event
      tmp_endTime = projBuf[i].recvTimeStart + projBuf[i].readyTimeEnd;
      tmp_endTime_ns = ((double)tmp_endTime) * spuDecCyclePeriod_2ns;
      sprintf(buf, "1 0 %u %llu %u %u 0 0\n",
              EXECUTED_FUNC, (unsigned long long int)tmp_endTime_ns, eventIDCounter[logFileIndex], logFileIndex
             );
      fwrite(buf, 1, strlen(buf), logFile[logFileIndex]);
      sprintf(buf, "3 0 %u %llu 0 0 0 0 0\n", func_index, (unsigned long long int)tmp_endTime_ns);
      fwrite(buf, 1, strlen(buf), logFile[logFileIndex]);
      eventCount[logFileIndex]++;

      readySum += (tmp_endTime_ns - tmp_startTime_ns);
      sumCount++;


      ///// USER TIME 0 /////
      tmp_startTime = projBuf[i].recvTimeStart + projBuf[i].userTime0Start;
      tmp_startTime_ns = ((double)tmp_startTime) * spuDecCyclePeriod_2ns;
      tmp_endTime = projBuf[i].recvTimeStart + projBuf[i].userTime0End;
      tmp_endTime_ns = ((double)tmp_endTime) * spuDecCyclePeriod_2ns;
      userTime0Sum += (tmp_endTime_ns - tmp_startTime_ns);


      ///// USER TIME 1 /////
      tmp_startTime = projBuf[i].recvTimeStart + projBuf[i].userTime1Start;
      tmp_startTime_ns = ((double)tmp_startTime) * spuDecCyclePeriod_2ns;
      tmp_endTime = projBuf[i].recvTimeStart + projBuf[i].userTime1End;
      tmp_endTime_ns = ((double)tmp_endTime) * spuDecCyclePeriod_2ns;
      userTime1Sum += (tmp_endTime_ns - tmp_startTime_ns);


      ///// USER TIME 2 /////
      tmp_startTime = projBuf[i].recvTimeStart + projBuf[i].userTime2Start;
      tmp_startTime_ns = ((double)tmp_startTime) * spuDecCyclePeriod_2ns;
      tmp_endTime = projBuf[i].recvTimeStart + projBuf[i].userTime2End;
      tmp_endTime_ns = ((double)tmp_endTime) * spuDecCyclePeriod_2ns;
      userTime2Sum += (tmp_endTime_ns - tmp_startTime_ns);


      ///// USER ACCUM TIME 0 /////
      userAccumTime0Sum += (((double)projBuf[i].userAccumTime0) * spuDecCyclePeriod_2ns);

      ///// USER ACCUM TIME 1 /////
      userAccumTime1Sum += (((double)projBuf[i].userAccumTime1) * spuDecCyclePeriod_2ns);

      ///// USER ACCUM TIME 2 /////
      userAccumTime2Sum += (((double)projBuf[i].userAccumTime2) * spuDecCyclePeriod_2ns);

      ///// USER ACCUM TIME 3 /////
      userAccumTime3Sum += (((double)projBuf[i].userAccumTime3) * spuDecCyclePeriod_2ns);


      ///// EXECUTED /////
      // Create the BEGIN PROCESSING (2) event
      tmp_startTime = projBuf[i].recvTimeStart + projBuf[i].executedTimeStart;
      tmp_startTime_ns = ((double)tmp_startTime) * spuDecCyclePeriod_2ns;
      sprintf(buf, "2 0 %u %llu %u %u 0 0 0 0 0 0 0\n",
              EXECUTED_FUNC, (unsigned long long int)tmp_startTime_ns, eventIDCounter[logFileIndex], logFileIndex
             );
      fwrite(buf, 1, strlen(buf), logFile[logFileIndex]);
      eventIDCounter[logFileIndex]++;
      eventCount[logFileIndex]++;

      // Create the CREATION (1) event
      // Create the END PROCESSING (3) event
      tmp_endTime = projBuf[i].recvTimeStart + projBuf[i].executedTimeEnd;
      tmp_endTime_ns = ((double)tmp_endTime) * spuDecCyclePeriod_2ns;
      sprintf(buf, "1 0 %u %llu %u %u 0 0\n",
              COMMIT_FUNC, (unsigned long long int)tmp_endTime_ns, eventIDCounter[logFileIndex], logFileIndex
             );
      fwrite(buf, 1, strlen(buf), logFile[logFileIndex]);
      sprintf(buf, "3 0 %u %llu 0 0 0 0 0\n", EXECUTED_FUNC, (unsigned long long int)tmp_endTime_ns);
      fwrite(buf, 1, strlen(buf), logFile[logFileIndex]);
      eventCount[logFileIndex]++;


      ///// COMMIT /////
      // Create the BEGIN PROCESSING (2) event
      tmp_startTime = projBuf[i].recvTimeStart + projBuf[i].commitTimeStart;
      tmp_startTime_ns = ((double)tmp_startTime) * spuDecCyclePeriod_2ns;
      sprintf(buf, "2 0 %u %llu %u %u 0 0 0 0 0 0 0\n",
              COMMIT_FUNC, (unsigned long long int)tmp_startTime_ns, eventIDCounter[logFileIndex], logFileIndex
             );
      fwrite(buf, 1, strlen(buf), logFile[logFileIndex]);
      eventIDCounter[logFileIndex]++;
      eventCount[logFileIndex]++;

      // Create the END PROCESSING (3) event
      tmp_endTime = projBuf[i].recvTimeStart + projBuf[i].commitTimeEnd;
      tmp_endTime_ns = ((double)tmp_endTime) * spuDecCyclePeriod_2ns;
      sprintf(buf, "3 0 %u %llu 0 0 0 0 0\n", COMMIT_FUNC, (unsigned long long int)tmp_endTime_ns);
      fwrite(buf, 1, strlen(buf), logFile[logFileIndex]);
      eventCount[logFileIndex]++;


      // Update the max time
      // NOTE : Should only need to check the commit as it is the last state the work request moves through
      //   and thus should have the latest end time.
      register unsigned long long int tmp_endTime_ns_llu = (unsigned long long int)tmp_endTime_ns;
      if (tmp_endTime_ns_llu > maxEndTime[logFileIndex]) maxEndTime[logFileIndex] = tmp_endTime_ns_llu;
    }
  }

  // Close the log files
  for (int i = 0; i < 9; i++) {

    // Let the user know the number of events for each log file / spe
    printf("PE %d -=> # events:%d, end time:%llu\n", i, eventCount[i], maxEndTime[i]);

    // Write the END COMPUTATION (7)
    sprintf(buf, "7 %llu\n", maxEndTime[i] + 10);   // NOTE : Add a small amount for margin
    fwrite(buf, 1, strlen(buf), logFile[i]);

    // Seek to the point where number entries should be written and write the actual number of events
    fseek(logFile[i], numEntryOffset, SEEK_SET);
    sprintf(buf, "%d", eventCount[i]);
    fwrite(buf, 1, strlen(buf), logFile[i]);

    // Close the file
    fclose(logFile[i]);
  }

  // Display some avgs
  printf(":: Ready Avg: %lf ns\n", readySum / (double)sumCount);
  printf(":: User Time 0 Avg: %lf ns\n", userTime0Sum / (double)sumCount);
  printf(":: User Time 1 Avg: %lf ns\n", userTime1Sum / (double)sumCount);
  printf(":: User Time 2 Avg: %lf ns\n", userTime2Sum / (double)sumCount);
  printf(":: User Accum Time 0 Avg: %lf ns\n", userAccumTime0Sum / (double)sumCount);
  printf(":: User Accum Time 1 Avg: %lf ns\n", userAccumTime1Sum / (double)sumCount);
  printf(":: User Accum Time 2 Avg: %lf ns\n", userAccumTime2Sum / (double)sumCount);
  printf(":: User Accum Time 3 Avg: %lf ns\n", userAccumTime3Sum / (double)sumCount);
  
  // Close the projFile
  fclose(projFile);

  // All good
  return EXIT_SUCCESS;
}



int createStsFile(int numEntries, char* filePrefix) {

  char buf[1024];

  // Create the sts file
  sprintf(buf, "%s.sts", filePrefix);
  FILE* stsFile = fopen(buf, "w+");
  if (stsFile == NULL) {
    printf("ERROR: Unable to open \"%s.sts\" for writing... exiting.\n", buf);
    return 0;
  }

  sprintf(buf, "PROJECTIONS_ID\n");
  fwrite(buf, 1, strlen(buf), stsFile);

  sprintf(buf, "VERSION 6.6\n");
  fwrite(buf, 1, strlen(buf), stsFile);

  sprintf(buf, "MACHINE net-linux-cell\n");
  fwrite(buf, 1, strlen(buf), stsFile);

  // TODO : For now, assume only one Cell processor
  sprintf(buf, "PROCESSORS 9\n");
  fwrite(buf, 1, strlen(buf), stsFile);

  sprintf(buf, "TOTAL_CHARES 1\n");
  fwrite(buf, 1, strlen(buf), stsFile);

  sprintf(buf, "TOTAL_EPS %d\n", USER_FUNC_START + NUM_USER_FUNCS);
  fwrite(buf, 1, strlen(buf), stsFile);

  sprintf(buf, "TOTAL_MSGS 1\n");
  fwrite(buf, 1, strlen(buf), stsFile);

  sprintf(buf, "CHARE 0 dummy_work_request_chare\n");
  fwrite(buf, 1, strlen(buf), stsFile);

  sprintf(buf, "ENTRY CHARE %d spert_recv 0 0\n", RECV_FUNC);
  fwrite(buf, 1, strlen(buf), stsFile);

  sprintf(buf, "ENTRY CHARE %d spert_preFetching 0 0\n", PRE_FETCHING_FUNC);
  fwrite(buf, 1, strlen(buf), stsFile);

  sprintf(buf, "ENTRY CHARE %d spert_fetching 0 0\n", FETCHING_FUNC);
  fwrite(buf, 1, strlen(buf), stsFile);

  sprintf(buf, "ENTRY CHARE %d spert_ready 0 0\n", READY_FUNC);
  fwrite(buf, 1, strlen(buf), stsFile);

  sprintf(buf, "ENTRY CHARE %d spert_executed 0 0\n", EXECUTED_FUNC);
  fwrite(buf, 1, strlen(buf), stsFile);

  sprintf(buf, "ENTRY CHARE %d spert_commit 0 0\n", COMMIT_FUNC);
  fwrite(buf, 1, strlen(buf), stsFile);

  // TODO : For now, just hard code 25 function indexes
  for (int i = 0; i < 25; i++) {
    sprintf(buf, "ENTRY CHARE %d func_index_%d 0 0\n", i + USER_FUNC_START, i);
    fwrite(buf, 1, strlen(buf), stsFile);
  }

  sprintf(buf, "MESSAGE 0 0\n");
  fwrite(buf, 1, strlen(buf), stsFile);

  sprintf(buf, "END\n");
  fwrite(buf, 1, strlen(buf), stsFile);

  // Close the sts file
  fclose(stsFile);

  return 1;
}

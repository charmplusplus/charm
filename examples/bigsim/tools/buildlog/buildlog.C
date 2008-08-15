/**********************************************
 * buildlog.C
 * 
 * Manually generate bgTrace logs for 4 PEs
 * running on 2 physical processors.
 *
 * Program flow:
 *   1. PE 0 sends messages to PE 1 and PE 2
 *   2. PE 2 sends a message to PE 3
 *
 * Rewritten by Ryan Mokos in July 2008.
 *********************************************/

#include "bigsim_logs.h"

#define totalPEs 4

int main() {

  double curTime;                     // current simulation time
  BgTimeLog *newlog, *prevlog;        // time logs
  BgMsgEntry *msgEntry;               // entry for sent message
  BgTimeLineRec tlinerecs[totalPEs];  // time line record

  // Near the end of this file, the time lines will be written to
  // bgTrace files in sequential order (i.e., with 4 PEs and 2 real
  // procs, PEs 0 and 1 will be written to bgTrace0 and PEs 2 and 3
  // will be written to bgTrace1).  However, when bgTrace files are
  // read by the simulator, they are read in a cyclic order (i.e.,
  // with 4 PEs and 2 real procs, PEs [0,1,2,3] will be read from
  // bgTrace files [0,1,0,1]).  This lookup table is used to change
  // the order in which time lines are written to bgTrace files from
  // sequential to cyclic.  Examples:
  //
  //   4 PEs on 1 real proc
  //     [0,1,2,3]
  //     read order:   [0,1,2,3]
  //     write order:  [0,1,2,3]
  //     lookup table: {0,1,2,3}
  //
  //   4 PEs on 2 real procs
  //     [0,2]
  //     [1,3]
  //     read order:   [0,1,2,3]
  //     write order:  [0,2,1,3]
  //     lookup table: {0,2,1,3}
  //
  //   6 PEs on 2 real procs
  //     [0,2,4]
  //     [1,3,5]
  //     read order:   [0,1,2,3,4,5]
  //     write order:  [0,2,4,1,3,5]
  //     lookup table: {0,2,4,1,3,5}
  //
  //   8 PEs on 3 real procs
  //     [0,3,6]
  //     [1,4,7]
  //     [2,5]
  //     read order:   [0,1,2,3,4,5,6,7]
  //     write order:  [0,3,6,1,4,7,2,5]
  //     lookup table: {0,3,6,1,4,7,2,5}
  int pe_lookup[totalPEs] = {0, 2, 1, 3};

  // turn on log generation
  BgGenerateLogs();

  /*************** PE 0 ***************/
  // Entry 0
  //   - name: SendTo1And2
  //   - startTime: 0.000000
  //   - endTime:   0.000010
  //   - send message 0
  //     - send time: 0.000005
  //     - rec. time: 0.000006
  //     - destination: PE 1
  //     - message size: 128
  //     - group: 1 (unicast message)
  //   - send message 1
  //     - send time: 0.000009
  //     - rec. time: 0.000010
  //     - destination: PE 2
  //     - message size: 128
  //     - group: 1 (unicast message)
  //   - backward dependencies: none
  //   - forward dependencies: none

  curTime = 0.000000;
  newlog = new BgTimeLog();
  newlog->setName("SendTo1And2");                // set the name of the SEB
  newlog->setTime(curTime, curTime + 0.000010);  // set the start and end times
    // send message 0 to PE 1
  curTime = 0.000005;
  msgEntry = new BgMsgEntry(0, 128, curTime, curTime + 0.000001, 1, 0);
  msgEntry->group = 1;
  newlog->addMsg(msgEntry);
    // send message 1 to PE 2
  curTime += 0.000004;
  msgEntry = new BgMsgEntry(1, 128, curTime, curTime + 0.000001, 2, 0);
  msgEntry->group = 1;
  newlog->addMsg(msgEntry);
    // insert log entry into the timeline for PE 0
  tlinerecs[pe_lookup[0]].logEntryInsert(newlog);

  /*************** PE 1 ***************/
  // Entry 0 (message from PE 0)
  //   - name: msgep
  //   - entry point (ep): 4
  //   - rec. time: 0.000006
  //   - startTime: 0.000006
  //   - endTime:   0.000008
  //   - backward dependencies: none
  //   - forward dependencies: Entry 1
  // Entry 1
  //   - name: DoStuff
  //   - entry point (ep): 0
  //   - startTime: 0.000008
  //   - endTime:   0.000020
  //   - backward dependencies: Entry 0
  //   - forward dependencies: none

  // Entry 0
  // Note: forward dependency on Entry 1 is not set
  //       here--it is automatically set by assigning
  //       a backward dependency for Entry 1
  curTime = 0.000006;
  newlog = new BgTimeLog(BgMsgID(0, 0));         // receive message with ID of (0,0)
  newlog->setName("msgep");                      // human-readable name
  newlog->setEP(4);                              // function handle
  newlog->setTime(curTime, curTime + 0.000002);  // set start/end time of execution
  tlinerecs[pe_lookup[1]].logEntryInsert(newlog);           // insert log entry into PE 1 timeline

  // Entry 1
  curTime += 0.000002;
  prevlog = newlog;
  newlog = new BgTimeLog();
  newlog->setName("DoStuff");
  newlog->setEP(0);
  newlog->setTime(curTime, curTime + 0.000012);
    // sets backward dep. of Entry 1 to Entry 0; also sets forward dep. of Entry 0 to Entry 1
  newlog->addBackwardDep(prevlog);
    // insert log entry into the timeline for PE 1
  tlinerecs[pe_lookup[1]].logEntryInsert(newlog);

  /*************** PE 2 ***************/
  // Entry 0 (message from PE 0)
  //   - name: msgep
  //   - entry point (ep): 2
  //   - rec. time: 0.000010
  //   - startTime: 0.000010
  //   - endTime:   0.000012
  //   - backward dependencies: none
  //   - forward dependencies: Entry 1
  // Entry 1
  //   - name: SendTo3
  //   - startTime: 0.000012
  //   - endTime:   0.000020
  //   - send message 0
  //     - send time: 0.000018
  //     - rec. time: 0.000020
  //     - destination: PE 3
  //     - message size: 188
  //     - group: 1 (unicast message)
  //   - backward dependencies: Entry 0
  //   - forward dependencies: none

  // Entry 0
  curTime = 0.000010;
  newlog = new BgTimeLog(BgMsgID(0, 1));  // node 0, msg 1
  newlog->setName("msgep");
  newlog->setEP(2);
  newlog->setTime(curTime, curTime + 0.000002);
  tlinerecs[pe_lookup[2]].logEntryInsert(newlog);

  // Entry 1
  curTime += 0.000002;
  prevlog = newlog;
  newlog = new BgTimeLog();
  newlog->setName("SendTo3");
  newlog->setEP(0);
  newlog->setTime(curTime, curTime + 0.000008);
    // send message 0 to PE 3
  curTime += 0.000006;
  msgEntry = new BgMsgEntry(0, 188, curTime, curTime + 0.000002, 3, 0);
  msgEntry->group = 1;
  newlog->addMsg(msgEntry);
  newlog->addBackwardDep(prevlog);
  tlinerecs[pe_lookup[2]].logEntryInsert(newlog);

  /*************** PE 3 ***************/
  // Entry 0 (message from PE 2)
  //   - name: msgep
  //   - entry point (ep): 1
  //   - rec. time: 0.000020
  //   - startTime: 0.000020
  //   - endTime:   0.000050
  //   - backward dependencies: none
  //   - forward dependencies: Entry 1
  // Entry 1
  //   - name: DoStuff
  //   - entry point (ep): 0
  //   - startTime: 0.000050
  //   - endTime:   0.000100
  //   - backward dependencies: Entry 0
  //   - forward dependencies: Entry 2
  // Entry 2
  //   - name: DoMoreStuff
  //   - entry point (ep): 0
  //   - startTime: 0.000100
  //   - endTime:   0.000200
  //   - backward dependencies: Entry 1
  //   - forward dependencies: none

  // Entry 0
  curTime = 0.000020;
  newlog = new BgTimeLog(BgMsgID(2, 0));  // node 2, msg 0
  newlog->setName("msgep");
  newlog->setEP(1);
  newlog->setTime(curTime, curTime + 0.000030);
  tlinerecs[pe_lookup[3]].logEntryInsert(newlog);

  // Entry 1
  curTime += 0.000030;
  prevlog = newlog;
  newlog = new BgTimeLog();
  newlog->setName("DoStuff");
  newlog->setEP(0);
  newlog->setTime(curTime, curTime + 0.000050);
  newlog->addBackwardDep(prevlog);
  tlinerecs[pe_lookup[3]].logEntryInsert(newlog);

  // Entry 2
  curTime += 0.000050;
  prevlog = newlog;
  newlog = new BgTimeLog();
  newlog->setName("DoMoreStuff");
  newlog->setEP(0);
  newlog->setTime(curTime, curTime + 0.000100);
  newlog->addBackwardDep(prevlog);
  tlinerecs[pe_lookup[3]].logEntryInsert(newlog);

  /*************** Output ***************/
  // write summary file for totalPEs (4) processor traces, running on 
  // 2 physical processors
  BgWriteTraceSummary(2, totalPEs, 1, 1);

  // write timelines into 2 separate bgTrace files: 1 for each
  // physical processor; note that reading of the log files will be
  // cyclic (file 0, file 1, file 0, file 1, etc.), which means the
  // time line records have to be written cyclicly
  BgWriteTimelines(0, tlinerecs, 2);
  BgWriteTimelines(1, tlinerecs + 2, 2);

  // write ascii format of the files
  char fname[128];
  strcpy(fname, "bgTrace");
  for (int i = 0; i < totalPEs; i++) {
    BgWriteThreadTimeLine(fname, pe_lookup[i], 0, 0, 0, tlinerecs[i].timeline);
  }

}

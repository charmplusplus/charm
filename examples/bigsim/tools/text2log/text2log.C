#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "bigsim_logs.h"

// these should be plenty
#define MAX_MESSAGES 1000000
#define LINE_SIZE 1024

// pair of ints, used to keep track of forward dependencies
struct IntPair {
  int one, two;
  IntPair() {}
  IntPair(int unused) {}  // for remove function in CkVec
  inline void pup(PUP::er &p) {
    p|one; p|two;
  }
};


// all the info needed for a message send
struct MsgSend {
  int msgID, dstNode, tid, size, group;
  double sent, recvtime;
  MsgSend() {}
  MsgSend(int unused) {}  // for remove function in CkVec
  inline void pup(PUP::er &p) {
    p|msgID; p|dstNode; p|tid; p|size;
    p|group; p|sent; p|recvtime;
  }
};


int main(int argc, char* argv[]) {

  int numPEs, numNodes, numXNodes, numYNodes, numZNodes, numWth, numTraces;

  /************* Read in command-line parameters *************/
  if ((argc > 1) && (strcmp(argv[1], "-help") == 0)) {
    printf("\n");
    printf("Usage: ./text2log <# log files> <X dim nodes> <Y dim nodes> <Z dim nodes> <# cores/node> [<# emulating procs>]\n");
    printf("\n");
    printf("<# log files>  - Integer that specifies the number of log#.txt files to read.\n");
    printf("                 Each log file must contain exactly one time line.\n");
    printf("<X dim nodes>  - Integer that specifies the number of simulated nodes in the X dimension.\n");
    printf("<Y dim nodes>  - Integer that specifies the number of simulated nodes in the Y dimension.\n");
    printf("<Z dim nodes>  - Integer that specifies the number of simulated nodes in the Z dimension.\n");
    printf("<# cores/node> - Integer that specifies the number of cores (worker threads) on each simulated node.\n");
    printf("                 There must be a log#.txt file for every core, and <# nodes> * <# cores/node> must equal the number of log#.txt files.\n");
    printf("<# emulating procs> - Optional integer that specifies the number of bgTrace files to create.\n");
    printf("                      If not specified, this number defaults to the number of specified log.txt files.\n");
    printf("\n");
    printf("All logs must be of the following format:\n");
    printf("\n");
    printf("[<eventNum>] name:<eventName> (srcpe:<srcpe> msgID:<msgID>) ep:<ep> charm_ep:<charm_ep>\n");
    printf("startTime:<startTime> endTime:<endTime>\n");
    printf("-msgID:<msgID> sent:<sendTime> recvtime:<recvTime> dstNode:<dstNode> tid:<tid> size:<size> group:<group>\n");
    printf("forward: <optional list of ints enclosed in []s>\n");
    printf("\n");
    printf("Where the message send lines ('-msgID: ...') are optional.  See the examples.tgz tarball for examples.\n");
    printf("\n");
    printf("For messages:\n");
    printf("\n");
    printf("destNode destTID Behavior\n");
    printf("======== ======= ==============================================\n");
    printf("   -1      -1    Broadcast to ALL worker threads of ALL nodes\n");
    printf("   -2      -1    Multicast to all nodes given by the pe list in the task msg\n");
    printf("   -1       K    Invalid\n");
    printf("    N      -1    Send to ALL worker threads of node N\n");
    printf("    N       K    Send to worker thread K of node N\n");
    printf(" -100-N    -1    Broadcast to all worker threads of all nodes except for N (no worker threads of N receive)\n");
    printf(" -100-N     K    Broadcast to all worker threads of all nodes except worker K of node N\n");
    printf("\n");

    exit(0);
  }

  if ((argc < 6) || (argc > 7)) {
    printf("ERROR: there should be 6 or 7 arguments.  Usage: ./text2log <# log files> <X dim nodes> <Y dim nodes> <Z dim nodes> <# cores/node> [<# emulating procs>]\n");
    printf("Running with -help will provide detailed info on the parameters.\n");
    exit(-1);
  }

  numPEs = atoi(argv[1]);
  numXNodes = atoi(argv[2]);
  numYNodes = atoi(argv[3]);
  numZNodes = atoi(argv[4]);
  numNodes = numXNodes * numYNodes * numZNodes;
  numWth = atoi(argv[5]);
  if (argc == 7) {
    numTraces = atoi(argv[6]);
  } else {
    numTraces = numPEs;
  }

  // command-line parameter sanity checks
  if (numPEs <= 0) {
    printf("ERROR: the number of specified log files (%d) must be > 0\n", numPEs);
    exit(-1);
  }
  if (numTraces <= 0) {
    printf("ERROR: the number of emulating procs (%d) must be > 0\n", numTraces);
    exit(-1);
  }
  if ((numNodes * numWth) != numPEs) {
    printf("ERROR: the number of nodes * the number of cores per node (%d) must equal the number of specifed log files (%d)\n", 
	   numNodes * numWth, numPEs);
    exit(-1);
  }
  if (numTraces > numNodes) {
    printf("ERROR: the number of emulating procs (%d) must be <= the number of nodes (%d)\n", numTraces, numNodes);
    exit(-1);
  }

  // turn on log generation
  BgGenerateLogs();

  /************* Read in the log#.txt files and convert them to bgTrace files *************/
  BgTimeLineRec tlinerecs[numPEs];  // time lines
  char fileName[20], line[LINE_SIZE];
  FILE *filePtr;
  int lineNum, eventCount;
  BgTimeLog *newLog;
  BgMsgEntry *newMsg;
  double lastEndTime;
  int msgIDsFree[MAX_MESSAGES];

  // event fields
  int eventNum, srcpe, msgID, ep, charm_ep;
  char eventName[20];
  double startTime, endTime;

  CkVec<IntPair> dependencies;
  CkVec<MsgSend> eventMessages;

  /************* Loop through all log files *************/
  for (int logNum = 0; logNum < numPEs; logNum++) {

    sprintf(fileName, "log%d.txt", logNum);
    filePtr = fopen(fileName, "r");
    if (!filePtr) {
      printf("Cannot open file %s... exiting.\n", fileName);
      exit(-1);
    }
    printf("Reading file %s...\n", fileName);

    // make msgIDsFree ready to use
    for (int i = 0; i < MAX_MESSAGES; i++) {
      msgIDsFree[i] = 1;
    }

    // clear the dependencies
    dependencies.clear();

    // read events from the file, one line at a time
    lineNum = 0;
    eventNum = 0;
    eventCount = 0;
    lastEndTime = 0.0;
    while (fgets(line, LINE_SIZE, filePtr)) {

      // ignore empty lines
      if (strcmp(line, "\n") == 0) {
	lineNum++;
	continue;
      }

      // look for the first line of an event, then read the entire event
      if (line[0] == '[') {

	// first line of event
	if (sscanf(line, "[%d] name:%s (srcpe:%d msgID:%d) ep:%d charm_ep:%d\n", 
		   &eventNum, &eventName, &srcpe, &msgID, &ep, &charm_ep) != 6) {
	  printf("ERROR: line %d of file %s not read in properly\n", lineNum, fileName);
	  printf("line=%s\n", line);
	  printf("Looking for the first line of an event with the following format:\n");
	  printf("[<eventNum>] name:<eventName> (srcpe:<srcpe> msgID:<msgID>) ep:<ep> charm_ep:<charm_ep>\n");
	  printf("Where eventNum, srcpe, msgID, ep, and charm_ep are ints, and where eventName is a string no longer than 20 characters\n");
	  exit(-1);
	}
	if (eventNum != eventCount) {
	  printf("ERROR: event numbered incorrectly.  Event %d in file %s should be event %d.\n", eventNum, fileName, eventCount);
	  exit(-1);
	}
	lineNum++;

	// second line of event
	if (!fgets(line, LINE_SIZE, filePtr)) {
	  printf("ERROR: line %d of file %s was not read in properly\n", lineNum, fileName);
	  printf("line=%s\n", line);
	  printf("If this line doesn't exist in the file, then please add a 'starTime...' line or delete this event\n");
	  exit(-1);
	}
	if (sscanf(line, "startTime:%lf endTime:%lf\n", &startTime, &endTime) != 2) {
	  printf("ERROR: line %d of file %s not read in properly\n", lineNum, fileName);
	  printf("line=%s\n", line);
	  printf("Looking for the second line of an event with the following format:\n");
	  printf("startTime:<startTime> endTime:<endTime>\n");
	  printf("Where startTime and endTime are doubles in seconds\n");
	  exit(-1);
	}
	lineNum++;

	// read in messages; skip if they don't exist
	eventMessages.removeAll();
	while (fgets(line, LINE_SIZE, filePtr)) {
	  if (line[0] == '-') {
	    MsgSend ms;
	    if (sscanf(line, "-msgID:%d sent:%lf recvtime:%lf dstNode:%d tid:%d size:%d group:%d\n", 
		       &ms.msgID, &ms.sent, &ms.recvtime, &ms.dstNode, &ms.tid, &ms.size, &ms.group) != 7) {
	      printf("ERROR: line %d of file %s not read in properly\n", lineNum, fileName);
	      printf("line=%s\n", line);
	      printf("Looking for a message send line starting with the following format:\n");
	      printf("-msgID:<msgID> sent:<sendTime> recvtime:<recvTime> dstNode:<dstNode> tid:<tid> size:<size> group:<group>\n");
	      printf("Where msgID, dstNode, tid, size, and group are ints, and sendTime and recvTime are doubles in seconds\n");
	      exit(-1);
	    }
	    eventMessages.insertAtEnd(ms);
	    lineNum++;
	  } else {
	    // no messages left to read, so break out of loop
	    break;
	  }
	}

	// last line of the event has already been read in
	if (strncmp(line, "forward:", strlen("forward:")) != 0) {
	  printf("ERROR: line %d of file %s not read in properly\n", lineNum, fileName);
	  printf("line=%s\n", line);
	  printf("Looking for the last line of an event with the following format:\n");
	  printf("forward: <optional list of ints enclosed in []s>\n");
	  printf("For example: 'forward: [4] [13] [27]'\n");
	  exit(-1);
	}
	// extract forward dependencies
	char tempLine[LINE_SIZE];
	strcpy(tempLine, line + strlen("forward:"));
	char *token = strtok(tempLine, " []\n");
	while (token != NULL) {
	  IntPair ip;
	  ip.one = eventNum;
	  ip.two = atoi(token);
	  // ensure forward dependents aren't negative
	  if (ip.two < 0) {
	    printf("ERROR: line %d of file %s contains a negative forward dependent (%d)\n", lineNum, fileName, ip.two);
	    exit(-1);
	  }
	  // ensure forward dependents don't point to their own event
	  if (ip.two == eventNum) {
	    printf("ERROR: event %d at line %d of file %s contains a forward dependent (%d) that points to its own event\n", 
		   eventNum, lineNum, fileName, ip.two);
	    exit(-1);
	  }
	  dependencies.insertAtEnd(ip);
	  token = strtok(NULL, " []\n");
	}
	lineNum++;

      } else {
	printf("ERROR: Bad log format at line %d of file %s:\n", lineNum, fileName);
	printf("line=%s\n", line);
	printf("An event needs to begin with a line with the following format:\n");
	printf("[<eventNum>] name:<eventName> (srcpe:<srcpe> msgID:<msgID>) ep:<ep> charm_ep:<charm_ep>\n");
	printf("Where eventNum, srcpe, msgID, ep, and charm_ep are ints, and where eventName is a string no longer than 20 characters\n");
	exit(-1);
      }

      /************* Sanity checks *************/
      // check srcpe and msgID
      if ((srcpe < -1) || (srcpe > numPEs)) {
	printf("ERROR: [file %s, event %d] srcpe (%d) must be between -1 and %d, inclusive\n", fileName, eventNum, srcpe, numPEs - 1);
	exit(-1);
      }
      if (msgID < -1) {
	printf("ERROR: [file %s, event %d] msgID (%d) must be -1 or greater\n", fileName, eventNum, msgID);
	exit(-1);
      }
      if (((srcpe == -1) && (msgID != -1)) || ((msgID == -1) && (srcpe != -1))) {
	printf("ERROR: [file %s, event %d] if either srcpe (%d) or msgID (%d) is -1, the other must also be -1\n", fileName, eventNum, srcpe, msgID);
	exit(-1);
      }

      // check name
      if ((srcpe >= 0) && (strcmp(eventName, "msgep") != 0)) {
	printf("WARNING: [file %s, event %d] because the event looks like a message receive, its name should be 'msgep' instead of '%s'\n", 
	       fileName, eventNum, eventName);
      }

      // check ep and charm_ep
      if (ep < -1) {
	printf("ERROR: [file %s, event %d] ep (%d) must be -1 or greater\n", fileName, eventNum, ep);
	  exit(-1);
      }
      if (charm_ep < -1) {
	printf("ERROR: [file %s, event %d] charm_ep (%d) must be -1 or greater\n", fileName, eventNum, charm_ep);
	  exit(-1);
      }

      // check startTime and endTime
      if (startTime < lastEndTime) {
	printf("ERROR: [file %s, event %d] startTime (%lf) must be >= the end time of the last event (%lf)\n", fileName, eventNum, startTime, lastEndTime);
	  exit(-1);
      }
      if (endTime < startTime) {
	printf("ERROR: [file %s, event %d] endTime (%lf) must be >= the startTime of this event (%lf)\n", fileName, eventNum, endTime, startTime);
	  exit(-1);
      }

      // check each message
      for (int i = 0; i < eventMessages.length(); i++) {
	// check the msgID and group
	if (eventMessages[i].msgID < 0) {
	  printf("ERROR: [file %s, event %d] msgID of message send (%d) must be >= 0\n", fileName, eventNum, eventMessages[i].msgID);
	  exit(-1);
	}
	if (eventMessages[i].msgID >= MAX_MESSAGES) {
	  printf("ERROR: [file %s, event %d] msgID of message send (%d) is greater than the max message ID number (%d)\n", 
		 fileName, eventNum, eventMessages[i].msgID, MAX_MESSAGES);
	  exit(-1);
	}
	if (eventMessages[i].group < -1) {
	  printf("ERROR: [file %s, event %d] group (%d) of msgID %d must be >= -1 (and only == -1 for multicasts)\n", 
		 fileName, eventNum, eventMessages[i].group, eventMessages[i].msgID);
	  exit(-1);
	}
	// check that the msgID hasn't been used yet; account for
	// multicasts, in which it's used more than once
	if (msgIDsFree[eventMessages[i].msgID] > 0) {  // not used yet
	  if (eventMessages[i].group < 1) {  // group == -1 should only happen in successive multicast messages
	    printf("ERROR: [file %s, event %d] group (%d) of msgID %d must be >= 1 (and only > 1 for multicasts) because this doesn't look like part of a multicast\n", 
		   fileName, eventNum, eventMessages[i].group, eventMessages[i].msgID);
	    exit(-1);
	  } else if (eventMessages[i].group == 1) {  // normal unicast or broadcast
	    msgIDsFree[eventMessages[i].msgID] = 0;
	  } else {  // multicast
	    msgIDsFree[eventMessages[i].msgID] = 1 - eventMessages[i].group;
	  }
	} else if (msgIDsFree[eventMessages[i].msgID] == 0) {  // used too many times
	  printf("ERROR: [file %s, event %d] msgID (%d) has already been used (or used too many times for a multicast)\n", 
		 fileName, eventNum, eventMessages[i].msgID);
	  exit(-1);
	} else {  // part of a multicast
	  if (eventMessages[i].group != -1) {
	    printf("ERROR: [file %s, event %d] group (%d) of msgID %d must be -1 because its supposedly part of a multicast\n", 
		   fileName, eventNum, eventMessages[i].group, eventMessages[i].msgID);
	    exit(-1);
	  } else {
	    msgIDsFree[eventMessages[i].msgID]++;
	  }
	}
	// check message send and receive times
	if ((eventMessages[i].sent < startTime) || (eventMessages[i].sent > endTime)) {
	  printf("ERROR: [file %s, event %d] send time (%lf) of msgID %d must be between the startTime (%lf) and endTime (%lf) of the event, inclusive\n", 
		 fileName, eventNum, eventMessages[i].sent, eventMessages[i].msgID, startTime, endTime);
	  exit(-1);
	}
	if (eventMessages[i].recvtime < eventMessages[i].sent) {
	  printf("ERROR: [file %s, event %d] recvtime (%lf) of msgID %d must be >= the message send time (%lf)\n", 
		 fileName, eventNum, eventMessages[i].recvtime, eventMessages[i].msgID, eventMessages[i].sent);
	  exit(-1);
	}
	// check the destination node
	if ((eventMessages[i].dstNode <= -100 - numNodes) || 
	    ((eventMessages[i].dstNode < -1) && (eventMessages[i].dstNode > -100)) || 
	    (eventMessages[i].dstNode >= numNodes)) {
	  printf("ERROR: [file %s, event %d] dstNode (%d) of msgID %d must be:\n", 
		 fileName, eventNum, eventMessages[i].dstNode, eventMessages[i].msgID);
	  printf("          between -100 and %d, inclusive, for a broadcast-all-except\n", -100 - numNodes + 1);
	  printf("          -1 for a boadcast-all\n");
	  printf("          between 0 and %d, inclusive, for a normal unicast or multicast send\n", numNodes - 1);
	  exit(-1);
	}
	// check the destination thread ID
	if ((eventMessages[i].tid < -1) || (eventMessages[i].tid >= numWth)) {
	  printf("ERROR: [file %s, event %d] tid (%d) of msgID %d must be:\n", 
		 fileName, eventNum, eventMessages[i].tid, eventMessages[i].msgID);
	  printf("          -1 for a broadcast to all cores in the node\n");
	  printf("          between 0 and %d, inclusive, for a normal unicast or multicast send to one core of the node\n", numWth - 1);
	  exit(-1);
	}
	// check the message size
	if (eventMessages[i].size < 1) {
	  printf("ERROR: [file %s, event %d] the size (%d) of msgID %d must be at least 1 byte\n", 
		 fileName, eventNum, eventMessages[i].size, eventMessages[i].msgID);
	  exit(-1);
	}
      }
      // ensure the first event in time line 0 isn't a stand-alone event
      if ((logNum == 0) && (eventNum == 0) && (strcmp(eventName, "addMsg") == 0)) {
	printf("ERROR: the first simulation event (event 0 of file %s) should not be a stand-alone event (i.e., named 'addMsg')\n", fileName);
	exit(-1);
      }
      // ensure some of the stand-alone event criteria
      if ((strcmp(eventName, "addMsg") == 0) && ((srcpe != -1) || (msgID != -1))) {
	printf("ERROR: [file %s, event %d] stand-alone event (with event name 'addMsg') must have both srcpe (%d) and msgID (%d) set to -1\n", 
	       fileName, eventNum, srcpe, msgID);
	exit(-1);
      }

      /************* Create the BgTimeLog and insert into time line *************/
      newLog = new BgTimeLog(BgMsgID(srcpe, msgID));
      newLog->setName(eventName);
      newLog->setEP(ep);
      newLog->setCharmEP((short)charm_ep);
      newLog->setTime(startTime, endTime);
      // set receive time for first event of each time line to 0
      if (eventNum == 0) {
	newLog->recvTime = 0.0;
      }
      // set recvTime of stand-alone events to startTime
      if (strcmp(eventName, "addMsg") == 0) {
	newLog->recvTime = newLog->startTime;
      }

      for (int i = 0; i < eventMessages.length(); i++) {
	newMsg = new BgMsgEntry(eventMessages[i].msgID, eventMessages[i].size, eventMessages[i].sent, 
				eventMessages[i].recvtime, eventMessages[i].dstNode, eventMessages[i].tid);
	newMsg->group = eventMessages[i].group;
	newLog->addMsg(newMsg);
      }

      tlinerecs[logNum].logEntryInsert(newLog);

      lastEndTime = endTime;
      eventCount++;
    }

    /************* Connect all dependencies in this time line *************/
    for (int i = 0; i < dependencies.length(); i++) {
      if (dependencies[i].two >= eventCount) {
	printf("ERROR: [file %s, event %d] event has a forward dependent (%d) that is out of range (must be between 0 and %d, inclusive)\n", 
	       fileName, dependencies[i].one, dependencies[i].two, eventCount - 1);
	exit(-1);
      }
      tlinerecs[logNum][dependencies[i].two]->addBackwardDep(tlinerecs[logNum][dependencies[i].one]);
    }

  }

  /************* Ensure all messages are received *************/
  for (int pe = 0; pe < numPEs; pe++) {
    for (int event = 0; event < tlinerecs[pe].length(); event++) {
      for (int msg = 0; msg < tlinerecs[pe][event]->msgs.length(); msg++) {
	int destNode = tlinerecs[pe][event]->msgs[msg]->dstNode;
	int found = 0;

	if (destNode <= -100) {  // broadcast except
	  int nodeException = -100 - destNode;
	  int peException = -1;
	  if (tlinerecs[pe][event]->msgs[msg]->tID >= 0) {
	    peException = (nodeException * numWth) + tlinerecs[pe][event]->msgs[msg]->tID;
	  }
	  for (int destPE = 0; destPE < numPEs; destPE++) {
	    found = 0;
	    for (int destEvent = 0; destEvent < tlinerecs[destPE].length(); destEvent++) {
	      if ((tlinerecs[destPE][destEvent]->msgId.pe() == pe) && 
		  (tlinerecs[destPE][destEvent]->msgId.msgID() == tlinerecs[pe][event]->msgs[msg]->msgID)) {
		found++;
		// update event receive time based on message receive time
		tlinerecs[destPE][destEvent]->recvTime = tlinerecs[pe][event]->msgs[msg]->recvTime;
		// ensure the event startTime is not before the message receive time
		if (tlinerecs[destPE][destEvent]->startTime < tlinerecs[destPE][destEvent]->recvTime) {
		  printf("ERROR: event %d in time line %d has a startTime (%lf) that is before its recvTime (%lf)\n", 
			 destEvent, destPE, tlinerecs[destPE][destEvent]->startTime, tlinerecs[destPE][destEvent]->recvTime);
		  exit(-1);
		}
	      }
	    }
	    if ((destPE == peException) || ((peException == -1) && ((destPE / numWth) == nodeException))) {
	      // this is an exempt destPE, so it shouldn't have been found here
	      if (found > 0) {
		printf("ERROR: broadcast-except message with msgID %d sent from event %d in time line %d is received by PE %d (it should not be)\n", 
		       msg, event, pe, destPE);
		exit(-1);
	      }
	    } else {
	      // it should only be found once
	      if (found > 1) {
		printf("ERROR: broadcast-except message with msgID %d sent from event %d in time line %d is received %d times by PE %d (should be 1 time)\n", 
		       msg, event, pe, found, destPE);
		exit(-1);
	      } else if (found < 1) {
		printf("ERROR: broadcast-except message with msgID %d sent from event %d in time line %d is never received by PE %d\n", 
		       msg, event, pe, destPE);
		exit(-1);
	      }
	    }
	  }
	} else if (destNode == -1) {  // broadcast all
	  for (int destPE = 0; destPE < numPEs; destPE++) {
	    found = 0;
	    for (int destEvent = 0; destEvent < tlinerecs[destPE].length(); destEvent++) {
	      if ((tlinerecs[destPE][destEvent]->msgId.pe() == pe) && 
		  (tlinerecs[destPE][destEvent]->msgId.msgID() == tlinerecs[pe][event]->msgs[msg]->msgID)) {
		found++;
		// update event receive time based on message receive time
		tlinerecs[destPE][destEvent]->recvTime = tlinerecs[pe][event]->msgs[msg]->recvTime;
		// ensure the event startTime is not before the message receive time
		if (tlinerecs[destPE][destEvent]->startTime < tlinerecs[destPE][destEvent]->recvTime) {
		  printf("ERROR: event %d in time line %d has a startTime (%lf) that is before its recvTime (%lf)\n", 
			 destEvent, destPE, tlinerecs[destPE][destEvent]->startTime, tlinerecs[destPE][destEvent]->recvTime);
		  exit(-1);
		}
	      }
	    }
	    // it should only be found once
	    if (found > 1) {
	      printf("ERROR: broadcast-all message with msgID %d sent from event %d in time line %d is received %d times by PE %d (should be 1 time)\n", 
		     msg, event, pe, found, destPE);
	      exit(-1);
	    } else if (found < 1) {
	      printf("ERROR: broadcast-all message with msgID %d sent from event %d in time line %d is never received by PE %d\n", 
		     msg, event, pe, destPE);
	      exit(-1);
	    }
	  }
	} else {  // unicast
	  int destPE = (tlinerecs[pe][event]->msgs[msg]->dstNode * numWth) + tlinerecs[pe][event]->msgs[msg]->tID;
	  for (int destEvent = 0; destEvent < tlinerecs[destPE].length(); destEvent++) {
	    if ((tlinerecs[destPE][destEvent]->msgId.pe() == pe) && 
		(tlinerecs[destPE][destEvent]->msgId.msgID() == tlinerecs[pe][event]->msgs[msg]->msgID)) {
	      found++;
	      // update event receive time based on message receive time
	      tlinerecs[destPE][destEvent]->recvTime = tlinerecs[pe][event]->msgs[msg]->recvTime;
	      // ensure the event startTime is not before the message receive time
	      if (tlinerecs[destPE][destEvent]->startTime < tlinerecs[destPE][destEvent]->recvTime) {
		printf("ERROR: event %d in time line %d has a startTime (%lf) that is before its recvTime (%lf)\n", 
		       destEvent, destPE, tlinerecs[destPE][destEvent]->startTime, tlinerecs[destPE][destEvent]->recvTime);
		exit(-1);
	      }
	    }
	  }
	  // it should only be found once
	  if (found > 1) {
	    printf("ERROR: unicast message with msgID %d sent from event %d in time line %d is received %d times by PE %d (should be 1 time)\n", 
		   msg, event, pe, found, destPE);
	    exit(-1);
	  } else if (found < 1) {
	    printf("ERROR: unicast message with msgID %d sent from event %d in time line %d is never received by PE %d\n", 
		   msg, event, pe, destPE);
	    exit(-1);
	  }
	}
      }
    }
  }

  /************* Ensure all message receives have a sender and no backward dependents *************/
  for (int pe = 0; pe < numPEs; pe++) {
    for (int event = 0; event < tlinerecs[pe].length(); event++) {
      if (tlinerecs[pe][event]->msgId.pe() >= 0) {
	// ensure no backward dependents
	if (tlinerecs[pe][event]->backwardDeps.length() > 0) {
	  printf("ERROR: event %d in time line %d has %d backward dependent(s); it should have 0 because it looks like a message receive event\n", 
		 event, pe, tlinerecs[pe][event]->backwardDeps.length());
	  exit(-1);
	}
	// ensure it has a sender
	srcpe = tlinerecs[pe][event]->msgId.pe();
	msgID = tlinerecs[pe][event]->msgId.msgID();
	bool found = false;
	for (int srcEvent = 0; srcEvent < tlinerecs[srcpe].length(); srcEvent++) {
	  for (int msg = 0; msg < tlinerecs[srcpe][srcEvent]->msgs.length(); msg++) {
	    if (tlinerecs[srcpe][srcEvent]->msgs[msg]->msgID == msgID) {
	      found = true;
	      break;
	    }
	  }
	  if (found) break;
	}
	if (!found) {
	  printf("ERROR: event %d in time line %d does not have a corresponding message sent from time line %d\n", event, pe, srcpe);
	  exit(-1);
	}
      }
    }
  }

  /************* Write time lines to bgTrace files *************/
  // write trace summary file
  BgWriteTraceSummary(numTraces, numXNodes, numYNodes, numZNodes, numWth, 1, NULL, NULL);

  // Write trace files in round-robin fashion by node.
  // NOTE: The worker threads (i.e., cores) on each node must be kept
  // together.  For example:
  //
  //    8 target PEs, 2 emulating procs (2 trace files), numWth = 1
  //       tlinerecs = [0 1 2 3 4 5 6 7]
  //       bgTrace0 (tlinetemp[0]) = [0 2 4 6]
  //       bgTrace1 (tlinetemp[1]) = [1 3 5 7]
  //
  //    Same as above but with numWth = 2 -> the time lines must be
  //    grouped in 2s (01, 23, 45, 67)
  //       tlinerecs = [0 1 2 3 4 5 6 7]
  //       bgTrace0 (tlinetemp[0]) = [01 45] = [0 1 4 5]
  //       bgTrace1 (tlinetemp[1]) = [23 67] = [2 3 6 7]
  for (int trace = 0; trace < numTraces; trace++) {
    int numProcs;
    if (trace < (numPEs % numTraces)) {
      numProcs = (numPEs / numTraces) + 1;
    } else {
      numProcs = numPEs / numTraces;
    }
    BgTimeLineRec **tlinetemp = new BgTimeLineRec*[numProcs];
    for (int node = 0; node < (numProcs / numWth); node++) {
      for (int proc = 0; proc < numWth; proc++) {
	tlinetemp[(node*numWth)+proc] = &tlinerecs[(numTraces*node+trace)*numWth+proc];
      }
    }
    BgWriteTimelines(trace, tlinetemp, numProcs);
    delete [] tlinetemp;
  }
}

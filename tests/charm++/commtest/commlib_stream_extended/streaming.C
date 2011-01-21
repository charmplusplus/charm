#include "streaming.decl.h"

#include <comlib.h>
#include <cassert>

/*
 * Test of Streaming Strategy
 * by Lukasz Wesolowski
 * 04/12/07
 * This program checks the functionality of new strategy parameters
 * specifying maximum size of message and maximum buffer size. 
 * After all the tests are executed, the array elements are forced
 * to migrate and the tests are repeated a second time.
 * 
 * The program runs on 2 processors with 2 chares total
 */

CProxy_Main mainProxy;
CProxy_StreamingArray streamingArrayProxy;
int nElements;

ComlibInstanceHandle stratStreaming;


#define PERIOD_IN_MS 5000
#define NMSGS 100
#define MAX_MESSAGE_SIZE 10000
#define MAX_BUFFER_SIZE  100000
#define ENVELOPE_OVERHEAD_ESTIMATE 100

class streamingMessage : public CMessage_streamingMessage {
public:
  int length;
  char* msg;
};

// mainchare

class Main : public CBase_Main{
private:
  int nDone;
  int numArrayStreaming;

public:

  Main(CkArgMsg *m) {    
    nDone = 0;
    numArrayStreaming=0;

    // com_debug = 1;
    nElements = 2;
    //if(m->argc >1 ) nElements=atoi(m->argv[1]);
    delete m;

    mainProxy = thishandle;
	
    // create streaming strategy
    StreamingStrategy *strategy = new StreamingStrategy(PERIOD_IN_MS, NMSGS,
					       MAX_MESSAGE_SIZE, MAX_BUFFER_SIZE);
    stratStreaming = ComlibRegister(strategy);

    streamingArrayProxy = CProxy_StreamingArray::ckNew(nElements);

    streamingArrayProxy.simpleTest();
      
  }

  void finishStartup() {
    nDone++; 
    if (nDone == CkNumPes()) {
      nDone = 0; 
      streamingArrayProxy.testStreamingTimeout(); 
    }
  }

  void finishTimeoutTest() {
    nDone++;
    if (nDone == CkNumPes()) {
      nDone = 0;
      streamingArrayProxy.testStreamingMaxCount();
    }
  }

  void finishMaxCountTest() {
    nDone++;
    if (nDone==CkNumPes()) {
      nDone = 0;
      streamingArrayProxy.testStreamingMaxMsgSize();
    }
  }

  void finishMaxMsgSizeTest() {
    nDone++;
    if (nDone==CkNumPes()) {
      nDone=0;
      streamingArrayProxy.testStreamingMaxBufSize();
    }
  }

  void finishArrayStreaming() {
    nDone++;
    if (nDone == nElements) {
      nDone = 0;
      numArrayStreaming++;
      if (numArrayStreaming == 1)
	streamingArrayProxy.migrate();
      else
	CkExit();
    }
  }
  void finishMigrate() {
    nDone++;
    if (nDone == nElements) {
      nDone = 0;
      streamingArrayProxy.simpleTest();
    }
  }

};

class StreamingArray : public CBase_StreamingArray {
private:
  int msgCount;
  int msgCountAfterTimeout;
  CProxy_StreamingArray localProxy;
  double lastSavedTime;

public:

  StreamingArray() {
    msgCount=0;
    msgCountAfterTimeout=0;
  }

  StreamingArray(CkMigrateMessage *m) {}

  void simpleTest() {
    CkPrintf("[%d] ****** Test 0 - Startup phase ******\n", CkMyPe()); 
    localProxy = thisProxy; 
    ComlibAssociateProxy(stratStreaming, localProxy);
    char msg[] = "|This is a short streaming message|";
    for (int i=0; i<CkNumPes(); i++) {
      if (i==CkMyPe()) continue;
      streamingMessage* b = new(strlen(msg)+1,0) streamingMessage;	
      memcpy(b->msg, msg, strlen(msg)+1);
      b->length = strlen(msg);
      localProxy[i].simpleReceive(b);
    }    
  }

  void simpleReceive(streamingMessage *m) {
    CkPrintf("[%d] Startup message arrived: %s\n", CkMyPe(), m->msg); 
    delete m; 
    mainProxy.finishStartup(); 
  }
    
  void testStreamingTimeout() {
    CkPrintf("[%d] ****** Test 1 - Timeout ******\n", CkMyPe());

    char msg[] = "|This is a short streaming message|";
    lastSavedTime= CkWallTimer();
    for (int i=0; i<CkNumPes(); i++) {
      if (i==CkMyPe()) continue;
      streamingMessage* b = new(strlen(msg)+1,0) streamingMessage;	
      memcpy(b->msg, msg, strlen(msg)+1);
      b->length = strlen(msg);
      // note: broadcasts such as localProxy.receive(b) are not supported
      //       by streaming strategies and will cause the program to crash
      localProxy[i].receiveAfterTimeout(b);
    }
  }

  void receiveAfterTimeout(streamingMessage* m) {
    // The .75 factor is necessary because different processors will execute 
    //   this function at slightly differt times
    assert(CkWallTimer()-lastSavedTime > .75 * PERIOD_IN_MS/1000);
    CkPrintf("[%d]Message: %s arrived after %f seconds, timeout is %d seconds\n", 
	     CkMyPe(), m->msg, CkWallTimer()-lastSavedTime, PERIOD_IN_MS/1000);
    assert(strcmp(m->msg,"|This is a short streaming message|") == 0);
    delete m;
    mainProxy.finishTimeoutTest();
  }

  void testStreamingMaxCount() {
    CkPrintf("[%d] ****** Test 2 - Flush on max message count ******\n", CkMyPe());
    char msg[] = "|This is a short streaming message|";
    lastSavedTime=CkWallTimer();
    for (int i=0; i<CkNumPes(); i++) {
      if (i==CkMyPe()) continue;
      for (int j=0; j<NMSGS; j++) {
	streamingMessage* b = new(strlen(msg)+1,0) streamingMessage;	
	memcpy(b->msg, msg, strlen(msg)+1);
	b->length = strlen(msg);
	localProxy[i].receiveWithoutTimeout(b);
      }
      for (int k=0; k<NMSGS-1; k++) {
	streamingMessage* b = new(strlen(msg)+1,0) streamingMessage;	
	memcpy(b->msg, msg, strlen(msg)+1);
	b->length = strlen(msg);
	localProxy[i].receiveGroupAfterTimeout(b);
      }
    }
  }

  void receiveWithoutTimeout(streamingMessage* m) {
    msgCount++;;
    if (msgCount==NMSGS) {
      msgCount=0;
      assert(CkWallTimer()-lastSavedTime < .25 * PERIOD_IN_MS/1000);
      CkPrintf("[%d] %d messages arrived after %f seconds - timeout was not incurred\n", 
	       CkMyPe(), NMSGS, CkWallTimer()-lastSavedTime);
      assert(strcmp(m->msg,"|This is a short streaming message|") == 0);
    }
    delete m;
  }

  void receiveGroupAfterTimeout(streamingMessage* m) {
    msgCountAfterTimeout++;
    if (msgCountAfterTimeout==NMSGS-1) {
      msgCountAfterTimeout=0;
      assert(CkWallTimer()-lastSavedTime > .75 * PERIOD_IN_MS/1000);
      CkPrintf("[%d] %d messages arrived after %f seconds\n",
	       CkMyPe(), NMSGS-1, CkWallTimer()-lastSavedTime);
      mainProxy.finishMaxCountTest();
    }
    delete m;
  }

  void testStreamingMaxMsgSize() {
    CkPrintf("[%d] ****** Test 3 - Flush on max message size ******\n", CkMyPe());
    lastSavedTime=CkWallTimer();
    for (int i=0; i<CkNumPes(); i++) {
      if (i==CkMyPe()) continue;
      streamingMessage* b = new(MAX_MESSAGE_SIZE,0) streamingMessage;
      localProxy[i].receiveLargeMessage(b);
    }
  }

  void receiveLargeMessage(streamingMessage* m) {
    assert(CkWallTimer()-lastSavedTime < .25 * PERIOD_IN_MS/1000);
    CkPrintf("[%d] large message received after %f seconds with no "
	     "timeout incurred\n", CkMyPe(), CkWallTimer()-lastSavedTime);
    delete m;
    mainProxy.finishMaxMsgSizeTest();
  }

  void testStreamingMaxBufSize() {
    CkPrintf("[%d] ****** Test 4 - Flush on max buffer size reached ******\n", 
	     CkMyPe());
    int msgSize = MAX_MESSAGE_SIZE - ENVELOPE_OVERHEAD_ESTIMATE;
    int totalSize = 0;
    int firstMsg=1;
    lastSavedTime=CkWallTimer();
    for (int i=0; i<CkNumPes(); i++) {
      if (i==CkMyPe()) continue;
      while (totalSize < MAX_BUFFER_SIZE) {
	totalSize+=msgSize;
	streamingMessage *b = new(msgSize,0) streamingMessage;
	b->length=msgSize;
	if (firstMsg) {
	  localProxy[i].receiveAfterMaxBufSizeReached(b);
	  firstMsg=0;
	}
	else {
	  localProxy[i].receiveAndIgnore(b);
	}
      }
      totalSize=0;
    }
  }

  void receiveAfterMaxBufSizeReached(streamingMessage *m) {
    assert(CkWallTimer()-lastSavedTime < .25 * PERIOD_IN_MS/1000);
    CkPrintf("[%d] Received group of large messages after maximum "
	     "buffer size was reached on sending processor (%f seconds)\n", 
	     CkMyPe(), CkWallTimer()-lastSavedTime);
    delete m;
    mainProxy.finishArrayStreaming();
  }

  void receiveAndIgnore(streamingMessage *m) {
    delete m;
  }

  void migrate() {
    int migrateTo = 0;
    if (CkMyPe() != (CkNumPes() - 1))
      migrateTo = CkMyPe() + 1;
    else
      migrateTo = 0;
    CkPrintf("[%d] Migrating element %d to processor %d\n",
	     CkMyPe(), thisIndex, migrateTo);
    migrateMe(migrateTo);
    mainProxy.finishMigrate();    
  }

};


#include "streaming.def.h"

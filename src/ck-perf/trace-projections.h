/**
 * \addtogroup CkPerf
*/
/*@{*/

#ifndef _PROJECTIONS_H
#define _PROJECTIONS_H

#include <stdio.h>
#include <errno.h>
#include <unistd.h>

#include "trace.h"
#include "trace-common.h"
#include "ckhashtable.h"

#if CMK_PROJECTIONS_USE_ZLIB
#include <zlib.h>
#endif

#include "pup.h"

#define PROJECTION_VERSION  "7.0"

#define PROJ_ANALYSIS 1

// Macro to make projections check for errors before an fprintf succeeds.
#define CheckAndFPrintF(f,string,data) \
do { \
  int result = fprintf(f,string,data); \
  if (result == -1) { \
    CmiAbort("Projections I/O error!"); \
  } \
} while(false)

/// a log entry in trace projection
class LogEntry {
  public:
    double time;
    double endTime; // Should be used for all bracketed events. Currently only used for bracketed user supplied note
    double cputime;
    double recvTime;
    int event;
    int pe;
    unsigned short mIdx;
    unsigned short eIdx;
    int msglen;
    CmiObjId   id;
    int numpes;
    int *pes;
    int userSuppliedData;
    char *userSuppliedNote;
    unsigned long memUsage;
    double stat;	//Used for storing User Stats

    // this is taken out so as to provide a placeholder value for non-PAPI
    // versions (whose value is *always* zero).
    //int numPapiEvents;
#if CMK_HAS_COUNTER_PAPI
    LONG_LONG_PAPI papiValues[NUMPAPIEVENTS];
#endif
    unsigned char type; 
    char *fName;
    int flen;

  public:
    
    LogEntry() {
      fName=NULL;flen=0;pes=NULL;numpes=0;userSuppliedNote = NULL;
    }

    LogEntry(double tm, unsigned char t, unsigned short m=0, 
	     unsigned short e=0, int ev=0, int p=0, int ml=0, 
	     CmiObjId *d=NULL, double rt=0., double cputm=0., int numPe=0, double statVal=0.) {
      type = t; mIdx = m; eIdx = e; event = ev; pe = p; 
      time = tm; msglen = ml;
      if (d) id = *d; else {id.id[0]=id.id[1]=id.id[2]=id.id[3]=0; };
      recvTime = rt; cputime = cputm;
      // initialize for papi as well as non papi versions.
#if CMK_HAS_COUNTER_PAPI
      //numPapiEvents = NUMPAPIEVENTS;
#else
      //numPapiEvents = 0;
#endif
      userSuppliedNote = NULL;
      fName = NULL;
      flen=0;
      pes=NULL;
      numpes=numPe;
      stat=statVal;
    }

    LogEntry(double _time,unsigned char _type,unsigned short _funcID,
	     int _lineNum,char *_fileName){
      time = _time;
      type = _type;
      mIdx = _funcID;
      event = _lineNum;
      userSuppliedNote = NULL;      
      pes=NULL;
      numpes=0;
      setFName(_fileName);
    }

    // Constructor for User Supplied Data
    LogEntry(double _time,unsigned char _type, int value,
	     int _lineNum,char *_fileName){
      time = _time;
      type = _type;
      userSuppliedData = value;
      userSuppliedNote = NULL;
      pes=NULL;
      numpes=0;
      setFName(_fileName);
    }

    // Constructor for User Supplied Data
    LogEntry(double _time,unsigned char _type, char* note,
	     int _lineNum,char *_fileName){
      time = _time;
      type = _type;
      pes=NULL;
      numpes=0;
      setFName(_fileName);
      if(note != NULL)
	setUserSuppliedNote(note);
    }


   // Constructor for bracketed user supplied note
    LogEntry(double bt, double et, unsigned char _type, char *note, int eventID){
      time = bt;
      endTime = et;
      type = _type;
      pes=NULL;
      numpes=0;
      event = eventID;
      if(note != NULL)
	setUserSuppliedNote(note);
    }


    // Constructor for multicast data
    LogEntry(double tm, unsigned short m, unsigned short e, int ev, int p,
	     int ml, CmiObjId *d, double rt, int numPe, int *pelist){

      type = CREATION_MULTICAST; 
      mIdx = m; 
      eIdx = e; 
      event = ev; 
      pe = p; 
      time = tm; 
      msglen = ml;
      
      if (d) id = *d; else {id.id[0]=id.id[1]=id.id[2]=id.id[3]=-1; };
      recvTime = rt; 
      numpes = numPe;
      userSuppliedNote = NULL;
      if (pelist != NULL) {
	pes = new int[numPe];
	for (int i=0; i<numPe; i++) {
	  pes[i] = pelist[i];
	}
      } else {
	pes= NULL;
      }

    }


    void setFName(char *_fileName){
      if(_fileName == NULL){
	fName = NULL;
	flen = 0;
      }else{
	fName = new char[strlen(_fileName)+2];
	fName[0] = ' ';
	memcpy(fName+1,_fileName,strlen(_fileName)+1);
	flen = strlen(fName)+1;
      }	
    }

    // complementary function for adding papi data
    void addPapi(LONG_LONG_PAPI *papiVals){
#if CMK_HAS_COUNTER_PAPI
   	memcpy(papiValues, papiVals, sizeof(LONG_LONG_PAPI)*NUMPAPIEVENTS);
#endif
    }
   
    void setUserSuppliedData(int data){
      userSuppliedData = data;
    }

    void setUserSuppliedNote(char *note){

      int length = strlen(note)+1;
      userSuppliedNote = new char[length];
      memcpy(userSuppliedNote,note,length);
      for(int i=0;i<length;i++){
	if(userSuppliedNote[i] == '\n' || userSuppliedNote[i] == '\r'){
	  userSuppliedNote[i] = ' ';
	}
      }
	  
    }
	

    /// A constructor for a memory usage record
    LogEntry(unsigned char _type, double _time, long _memUsage) {
      time = _time;
      type = _type;
      memUsage = _memUsage;
      fName = NULL;
      flen = 0;
      pes=NULL;
      numpes=0;
    }


    void *operator new(size_t s) {void*ret=malloc(s);_MEMCHECK(ret);return ret;}
    void *operator new(size_t, void *ptr) { return ptr; }
    void operator delete(void *ptr) {free(ptr); }
#if defined(WIN32) || CMK_MULTIPLE_DELETE
    void operator delete(void *, void *) { }
#endif

    void setNewStartTime(double t) {
      time -= t;
      if (endTime>=t) endTime -= t;
      if (recvTime>=t) recvTime -= t;
    }

    void pup(PUP::er &p);
    ~LogEntry(){
      if (fName) delete [] fName;
      if (userSuppliedNote) delete [] userSuppliedNote;
    }
};

class TraceProjections;

/// log pool in trace projection
class LogPool {
  friend class TraceProjections;
#ifdef PROJ_ANALYSIS
  // The macro is here "just-in-case". Somehow, it seems it is not necessary
  //   to declare friend classes ahead of time in C++.
  friend class TraceProjectionsBOC;
  friend class KMeansBOC;
#endif  //PROJ_ANALYSIS
  friend class controlPointManager;
  private:
    bool writeData;
    int writeSummaryFiles;
    unsigned int poolSize;
    unsigned int numEntries;
    unsigned int lastCreationEvent;
    LogEntry *pool;
    FILE *fp;
    FILE *deltafp;
    FILE *stsfp;
    FILE *rcfp;
    FILE *topofp;
    FILE *statisfp;
    char *fname;
    char *dfname;
    char *pgmname;
    int binary;
    int nSubdirs;
#if CMK_PROJECTIONS_USE_ZLIB
    gzFile deltazfp;
    gzFile zfp;
    int compressed;
#endif
    // **CW** prevTime stores the timestamp of the last event
    // written out to log. This allows the implementation of
    // simple delta encoding and should only be used when
    // writing out logs.
    double prevTime;
    double timeErr;
    double globalStartTime; // used at the end on Pe 0 only
    double globalEndTime; // used at the end on Pe 0 only

    int numPhases;
    bool hasFlushed;
    //cppcheck-suppress unsafeClassCanLeak
    bool *keepPhase;  // one decision per phase

    int headerWritten;
    bool fileCreated;
    void writeHeader();

    // for statistics 
    double beginComputationTime;
    double endComputationTime;
    double statisLastTimer;
    double statisLastProcessTimer;
    double statisLastIdleTimer;
    double statisLastPackTimer;
    double statisLastUnpackTimer;
    double statisTotalExecutionTime;
    double statisTotalIdleTime;
    double statisTotalPackTime;
    double statisTotalUnpackTime;
    long long statisTotalCreationMsgs;
    long long statisTotalCreationBytes;
    long long statisTotalMCastMsgs;
    long long statisTotalMCastBytes;
    long long statisTotalEnqueueMsgs;
    long long statisTotalDequeueMsgs;
    long long statisTotalRecvMsgs;
    long long statisTotalRecvBytes;
    long long statisTotalMemAlloc;
    long long statisTotalMemFree;


  public:
    LogPool(char *pgm);
    ~LogPool();
    void setBinary(int b) { binary = b; }
    void setNumSubdirs(int n) { nSubdirs = n; }
    void setWriteSummaryFiles(int n) { writeSummaryFiles = n;}
#if CMK_PROJECTIONS_USE_ZLIB
    void setCompressed(int c) { compressed = c; }
#endif
    void createFile(const char *fix="");
    void createSts(const char *fix="");
    void createTopo(const char *fix="");
    void createRC();
    void openLog(const char *mode);
    void closeLog(void);
    void writeLog(void);
    void write(int writedelta);
    void writeSts(void);
    void writeSts(TraceProjections *traceProj);
    void writeRC(void);
    void writeTopo();
    void writeStatis();
    void initializePhases() {
      keepPhase = new bool[numPhases];
      for (int i=0; i<numPhases; i++) {
	keepPhase[i] = true;
      }
    }

    void setAllPhases(bool val) {
      for (int i=0; i<numPhases; i++) {
	keepPhase[i] = val;
      }
    }

    void add(unsigned char type, unsigned short mIdx, unsigned short eIdx,
	     double time, int event, int pe, int ml=0, CmiObjId* id=0, 
	     double recvT=0.0, double cpuT=0.0, int numPe=0, double statVal=0.0);

    // complementary function to set papi info to current log entry
    // must be called after an add()
    void addPapi(LONG_LONG_PAPI *papVals) {
      pool[numEntries-1].addPapi(papVals);
    }

	/** add a record for a user supplied piece of data */
	void addUserSupplied(int data);
	/* Creates LogEntry for stat. Called by Trace-projections updateStat functions*/
        void updateStat(unsigned char type,int e, double cputime,double time,double stat, int pe);
	/** add a record for a user supplied piece of data */
	void addUserSuppliedNote(char *note);


	void add(unsigned char type,double time,unsigned short funcID,int lineNum,char *fileName);
  
  	void addMemoryUsage(unsigned char type,double time,double memUsage);
	void addUserSuppliedBracketedNote(char *note, int eventID, double bt, double et);

    void addCreationMulticast(unsigned short mIdx,unsigned short eIdx,double time,int event,int pe, int ml=0, CmiObjId* id=0, double recvT=0., int numPe=0, int *pelist=NULL);
    void flushLogBuffer();
    void postProcessLog();

    void setWriteData(bool b){
      writeData = b;
    }
    void modLastEntryTimestamp(double ts);

    void setNewStartTime() {
      for(UInt i=0; i<numEntries; i++) pool[i].setNewStartTime(globalStartTime);
    }
};

/*
	class that represents a key in a CkHashtable with a string as a key
*/
class StrKey {
	std::string str;
	int len;
	unsigned int key;
	public:
	StrKey(const char *name){
		str = std::string(name);
		len = str.size();
		key = 0;
		for(int i=0;i<len;i++){
			key += str[i];
		}
	}
	static CkHashCode staticHash(const void *k,size_t){
		return ((StrKey *)k)->key;
	}
	static int staticCompare(const void *a,const void *b,size_t){
		StrKey *p,*q;
		p = (StrKey *)a;
		q = (StrKey *)b;
		if(p->len != q->len){
			return 0;
		}
		for(int i=0;i<p->len;i++){
			if(p->str[i] != q->str[i]){
				return 0;
			}
		}
		return 1;
	}
	inline CkHashCode hash() const{
		return key;
	}
	inline int compare(const StrKey &t) const {
		if(len != t.len){
			return 0;
		}
		for(int i=0;i<len;i++){
			if(str[i] != t.str[i]){
				return 0;
			}	
		}
		return 1;
	}
	inline const char *getStr(){
		return str.c_str();
	}
};

class NestedEvent {
 public:
  int event, msgType, ep, srcPe, ml;
  CmiObjId *idx;
  NestedEvent() {}
  NestedEvent(int _event, int _msgType, int _ep, int _srcPe, int _ml, CmiObjId *_idx) :
    event(_event), msgType(_msgType), ep(_ep), srcPe(_srcPe), ml(_ml), idx(_idx) { }
};

/// class for recording trace projections events 
/**
  TraceProjections will log Converse/Charm++ events and write into .log files;
  events descriptions will be written into .sts file.
*/
class TraceProjections : public Trace {
#ifdef PROJ_ANALYSIS
  // The macro is here "just-in-case". Somehow, it seems it is not necessary
  //   to declare friend classes ahead of time in C++.
  friend class TraceProjectionsBOC;
  friend class KMeansBOC;
#endif // PROJ_ANALYSIS
 private:
    LogPool* _logPool;        /**<  logpool for all events */
    int curevent;
    int execEvent;
    int execEp;
    int execPe;
    int inEntry;
    int computationStarted;

    int funcCount;
    CkHashtableT<StrKey,int> funcHashtable;

    int traceNestedEvents;
    CkQ<NestedEvent> nestedEvents;
    
    int currentPhaseID;
    LogEntry* lastPhaseEvent;

    //as user now can specify the idx, it's possible that user may specify an existing idx
    //so that we need a data structure to track idx. --added by Chao Mei
    CkVec<int> idxVec;
    int idxRegistered(int idx);    
public:
    int converseExit; // used for exits that bypass CkExit.
    double endTime;

    TraceProjections(char **argv);
    void userEvent(int e);
    void userBracketEvent(int e, double bt, double et);
    void userSuppliedBracketedNote(char*, int, double, double);

    void userSuppliedData(int e);
    void userSuppliedNote(char* note);
    void memoryUsage(double m);
    //UserStat function declartions for Trace-Projections
    int traceRegisterUserStat(const char*, int);
    void updateStatPair(int e, double stat, double time);
    void updateStat(int e, double stat);

    void creation(envelope *e, int epIdx, int num=1);
    void creation(char *m);
    void creationMulticast(envelope *e, int epIdx, int num=1, int *pelist=NULL);
    void creationDone(int num=1);
    void beginExecute(envelope *e, void *obj=NULL);
    void beginExecute(char *msg);
    void beginExecute(CmiObjId  *tid);
    void beginExecute(int event,int msgType,int ep,int srcPe,int ml,CmiObjId *idx=NULL, void *obj=NULL);
    void changeLastEntryTimestamp(double ts);
    void beginExecuteLocal(int event,int msgType,int ep,int srcPe,int ml,CmiObjId *idx=NULL);
    void endExecute(void);
    void endExecute(char *msg);
    void endExecuteLocal(void);
    void messageRecv(char *env, int pe);
    void beginIdle(double curWallTime);
    void endIdle(double curWallTime);
    void beginPack(void);
    void endPack(void);
    void beginUnpack(void);
    void endUnpack(void);
    void enqueue(envelope *e);
    void dequeue(envelope *e);
    void beginComputation(void);
    void endComputation(void);

    int traceRegisterUserEvent(const char*, int);
    void traceClearEps();
    void traceWriteSts();
    void traceClose();
    void traceBegin();
    void traceEnd();
#if CMK_SMP_TRACE_COMMTHREAD
    void traceBeginOnCommThread();
    void traceEndOnCommThread();
#endif
    void traceCommSetMsgID(char *msg);
    void traceGetMsgID(char *msg, int *pe, int *event);
    void traceSetMsgID(char *msg, int pe, int event);
    void traceFlushLog() { _logPool->flushLogBuffer(); }

    //functions that perform function tracing
    CkHashtableIterator *getfuncIterator(){return funcHashtable.iterator();};
    int getFuncNumber(){return funcHashtable.numObjects();};
    void regFunc(const char *name, int &idx, int idxSpecifiedByUser=0);
    void beginFunc(const char *name,char *file,int line);
    void beginFunc(int idx,char *file,int line);
    void endFunc(const char *name);
    void endFunc(int num);

    /* start recognizing phases in trace-projections */
    /* _TRACE_END_PHASE must be called collectively on all processors */
    /*   in order for phase numbers to match up. */
    void endPhase();

    /* This is for moving projections to being a charm++ module */
    void closeTrace(void);

    void setWriteData(bool b){
      _logPool->setWriteData(b);
    }

    /* for overiding basic thread listener support in Trace class */
    virtual void traceAddThreadListeners(CthThread tid, envelope *e);
};

using namespace PUP;

class toProjectionsFile : public toTextFile {
 protected:
  virtual void bytes(void *p,int n,size_t itemSize,dataType t);
 public:
  //Begin writing to this file, which should be opened for ascii write.
  toProjectionsFile(FILE *f_) :toTextFile(f_) {}
};
class fromProjectionsFile : public fromTextFile {
 protected:
  virtual void bytes(void *p,int n,size_t itemSize,dataType t);
 public:
  //Begin writing to this file, which should be opened for ascii read.
  fromProjectionsFile(FILE *f_) :fromTextFile(f_) {}
};

#if CMK_PROJECTIONS_USE_ZLIB
class toProjectionsGZFile : public PUP::er {
  gzFile f;
 protected:
  virtual void bytes(void *p,int n,size_t itemSize,dataType t);
 public:
  //Begin writing to this gz file, which should be opened for gz write.
  toProjectionsGZFile(gzFile f_) :er(IS_PACKING), f(f_) {}
};
#endif




#if CMK_TRACE_ENABLED
/// Disable the outputting of the trace logs
void disableTraceLogOutput();

/// Enable the outputting of the trace logs
void enableTraceLogOutput();

/// Force the log file to be flushed
void flushTraceLog();
#else
static inline void disableTraceLogOutput() { }
static inline void enableTraceLogOutput() { }
static inline void flushTraceLog() { }
#endif

#endif

/*@}*/

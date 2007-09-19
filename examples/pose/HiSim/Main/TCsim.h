#include "bigsim.h"
#include "blue_impl.h"
#include "bigsim_logs.h"
#include "trace-projections.h"

#define  yourUniqueTaskIDtype TaskID*
#define factor (1e8)

class TaskID{
 public:
  int srcNode;
  int msgID;
  int index;
  TaskID(){srcNode = -100; msgID=-100;index=-100;}
  TaskID(int s, int m, int i){
    srcNode = s;
    msgID = m;
    index = i;
  }
  TaskID(TaskID *other){
    srcNode = other->srcNode;
    msgID = other->msgID;
    index = other->index;
  }
  bool operator==(const TaskID other) const {
    //   printf("== called with my:(s:%d m:%d i:%d) and other:(s:%d m:%d i:%d) \n",srcNode,msgID,index,other.srcNode,other.msgID,other.index);
    if(srcNode ==-1 && msgID == -1)
      return (index == other.index);
    else
      return ((srcNode==other.srcNode)&&(msgID==other.msgID));
  }
  TaskID& operator=(const TaskID other){
    srcNode = other.srcNode;
    msgID = other.msgID;
    index = other.index;
    return *this;
  }
  void pup(PUP::er &p){
    p|srcNode; p|msgID; p|index;
  }
  inline int isInvalid() {
    return srcNode == -100 && msgID == -100 && index == -100;
  }
  // for hash table lookup
  inline unsigned int hash(void) const {
    return  msgID*1000 + srcNode;
  }
  inline int compare(const TaskID &v) const {
    if (srcNode == v.srcNode && msgID == v.msgID) return 1;
    return 0;
  }
  static unsigned int staticHash(const void *key,size_t keyLen) {
    return ((const TaskID *)key)->hash();
  }
  static int staticCompare(const void *a,const void *b,size_t keyLen) {
    return ((const TaskID *)a)->compare(*(const TaskID *)b);
  }
}; 

class MsgKey {
  public: 
    int srcNode;
    int msgID;
    MsgKey(int s, int m): srcNode(s), msgID(m) {}
    inline unsigned int hash(void) const {
         return  msgID*1000 + srcNode;
    }
    inline int compare(const MsgKey &v) const {
        if (srcNode == v.srcNode && msgID == v.msgID) return 1;
	return 0;
    }
    static unsigned int staticHash(const void *key,size_t keyLen) {
        return ((const MsgKey *)key)->hash();
    }
    static int staticCompare(const void *a,const void *b,size_t keyLen) {
                return ((const MsgKey *)a)->compare(*(const MsgKey *)b);
    }
};

class BGprocMsg {
 public:
  int nodePID;
 public:
  BGprocMsg(int node) : nodePID(node) {}
  // insert all log data for a BG processor here
  BGprocMsg& operator=(const BGprocMsg& obj) {
    eventMsg::operator=(obj);
    nodePID=obj.nodePID;
    // must provide assignment operator for POSE message classes
    return *this;
  }
};

class BGnodeMsg {
 public:
  int nodeIdx, procsPerNode, switchID;
 public:
  BGnodeMsg(int nodeid, int nwth, int switchid): nodeIdx(nodeid), procsPerNode(nwth), switchID(switchid)  {}
  BGnodeMsg& operator=(const BGnodeMsg& obj) {
    eventMsg::operator=(obj);
    nodeIdx=obj.nodeIdx;
    procsPerNode=obj.procsPerNode;
    switchID=obj.switchID;
    return *this;
  }
};


class TaskMsg {
 public:

  TaskID taskID;
  int destNode;
  int destNodeCode;
  int desttID;   //dest threadID
  POSE_TimeType receiveTime;
  int msgsize;
  
  TaskMsg(){destNode=desttID=-1;receiveTime=0;msgsize=0;taskID.srcNode=taskID.msgID=taskID.index=-1;}
  TaskMsg(int s, int m, int i){
    taskID.srcNode=s;
    taskID.msgID=m;
    taskID.index=i;
    destNode = -1;
    desttID = -1;
    receiveTime = msgsize = 0;
  }
  TaskMsg(int s, int m, int i, POSE_TimeType rt, int ms, int dn, int dnc, int dt){
    taskID.srcNode=s;
    taskID.msgID=m;
    taskID.index=i;
    receiveTime = rt;
    msgsize = ms;
    destNode = dn;
    destNodeCode = dnc;
    desttID = dt;
  }
  TaskMsg(TaskID& t, POSE_TimeType rt, int ms, int dn, int dnc, int dt){
    taskID.srcNode=t.srcNode;
    taskID.msgID=t.msgID;
    taskID.index=t.index;
    receiveTime = rt;
    msgsize = ms;
    destNode = dn;
    destNodeCode = dnc;
    desttID = dt;
  }
  TaskMsg& operator=(const TaskMsg& obj) {
    eventMsg::operator=(obj);
    taskID = obj.taskID;
    destNode = obj.destNode;
    desttID = obj.desttID;
    receiveTime = obj.receiveTime;
    msgsize = obj.msgsize;
    destNodeCode=obj.destNodeCode;
    return *this;
  }
  
  TaskID getTaskID (){
    return taskID;
  }
  int isNull(){return ((destNode==-1)&&(desttID==-1)&&(receiveTime==0));}
  int getDestNode() {return destNode;}  
  void pup(PUP::er &p){
    taskID.pup(p);
    p|destNode;p|destNodeCode;p|desttID;p|receiveTime;p|msgsize;
  }
  
};


class Task
{
  //implement a taskList type to store a list of generated tasks
  // datafields: 
 private:
  POSE_TimeType convertToInt(double inp);
  struct ProjEvent {
    int index;
    POSE_TimeType startTime;
    void pup(PUP::er &p) { p|index; p|startTime; }
  };
  struct bgPrintEvent {
    char* data;
    POSE_TimeType sTime;
    void pup(PUP::er &p) { p|sTime;
   			   int slen = 0;
	                   if (p.isPacking()) slen = strlen((char *)data)+1;
	                   p|slen;
	                   if (p.isUnpacking()) data=(char *)malloc(sizeof(char)*slen);
	                   p((char *)data,slen); }
  };  
 public:
  TaskID taskID;
  POSE_TimeType receiveTime, execTime, startTime, newStartTime;
  char name[20];
  TaskID* backwardDeps;
  TaskID* forwardDeps;
  TaskMsg* taskmsgs;
  ProjEvent* projevts;
  bgPrintEvent* printevts;
  int printevtLen, projevtLen;
  int bDepsLen, fDepsLen,taskMsgsLen;
  int done;
  void convertFrom(BgTimeLog*, int index, int numWth);
  // implement these events
  POSE_TimeType getRecvTime(){return receiveTime;}
  yourUniqueTaskIDtype getTaskID();
  void pup(PUP::er &p);
};


class BGproc {

 private:
  int numX,numY,numZ,numCth,numWth,numPes,totalProcs,procNum;
  int nodePID;		// node pose ID it belongs to
  Task* taskList;
  int numTasks;
  int *done;

  LogEntry  *logs;
  int numLogs;
  FILE *proj;
  int binary;
  FILE *lf;

  CkHashtableT<TaskID, int> msgTable;
  void buildHash();

  int locateTask(TaskID* taskID);
  inline int locateTask(TaskID &taskID)
		{ return locateTask(&taskID); }
  void markTask(TaskID* taskID);
  void loadProjections();

 public:
  BGproc();
  BGproc(BGprocMsg *m); 
  ~BGproc();
  inline BGproc& operator=(const BGproc& obj) 
	{ rep::operator=(obj); 
	  // one need to store done array, others are const
	//????EJB that comment assumes BGproc assignment only occurs during
	//????EJB post initialization checkpoint recovery.  Hopefully true.
	  numTasks = obj.numTasks;
          if (!done) done = new int[numTasks];
	  for (int i=0; i<numTasks; i++) done[i] = obj.done[i];
	  return *this; }
  void pup(PUP::er &p);
//  void cpPup(PUP::er &p) { }
  // Event methods
  void executeTask(TaskMsg *m);
  void executeTask_anti(TaskMsg *m);
  void executeTask_commit(TaskMsg *m);
  //local methods (maybe make private)
  void updateReceiveTime(TaskMsg* m);
  int dependenciesMet(yourUniqueTaskIDtype taskID);
  void updateStartTime(yourUniqueTaskIDtype taskID, POSE_TimeType newTime);
  void enableGenTasks(TaskID* taskID,POSE_TimeType oldStartTime, POSE_TimeType newStartTime);
  void SendMessage(TaskMsg *inMsg, int myNode, int srcSwitch,
		   int destNodeCode, int destTID, int taskOffset);
  void GeneralSend(TaskMsg *inMsg, int myNode, int srcSwitch, 
		  int destSwitch, int destNodeCode, int destTID, int destNode, 
		  int taskOffset);
  void DirectSend(TaskMsg *inMsg, int myNode, int srcSwitch,
	       int destSwitch, int destNodeCode, int destTID, int destNode, 
	       int taskOffset);
  void NetSend(TaskMsg *inMsg, int myNode, int srcSwitch,
	       int destSwitch, int destNodeCode, int destTID, int destNode, 
	       int taskOffset);
  void enableDependents(yourUniqueTaskIDtype taskID);
  TaskMsg* getGeneratedTasks(yourUniqueTaskIDtype taskID);
  int getNumGeneratedTasks(yourUniqueTaskIDtype taskID);
  POSE_TimeType getDuration(yourUniqueTaskIDtype taskID);
  POSE_TimeType getDuration(TaskID taskID);
  POSE_TimeType getReceiveTime(TaskID taskID);
  void terminus();
};

class BGnode {
 private:
  int nodePID;		// index to pose
  int myNodeIndex;	// index in node array
  int firstProcPID;	// first BGproc
  int procsPerNode; 
  int switchPID;	// swiytch PID
 public:
  BGnode() { }
  BGnode(BGnodeMsg *);
  ~BGnode() { }
  void pup(PUP::er &p);
  void cpPup(PUP::er &p) { }
  inline BGnode& operator=(const BGnode& obj) 
    { rep::operator=(obj); 
    nodePID=obj.nodePID;
    myNodeIndex=obj.myNodeIndex;
    firstProcPID=obj.firstProcPID;
    procsPerNode=obj.procsPerNode;
    switchPID=obj.switchPID;
    return *this; }
  // Event methods
  void recvOutgoingMsg(TaskMsg *rm);
  inline void recvOutgoingMsg_anti(TaskMsg *m)  {restore(this);}
  inline void recvOutgoingMsg_commit(TaskMsg *m)  {}
  void recvIncomingMsg(TaskMsg *rm);
  inline void recvIncomingMsg_anti(TaskMsg *rm) {restore(this);}
  inline void recvIncomingMsg_commit(TaskMsg *rm)  {}
};

class TransMsg {
public:
int id;
int origSrc;
int dstNode;
int msgId;

TransMsg(){}
TransMsg(int s,int mid,int d):origSrc(s),dstNode(d),msgId(mid){}
TransMsg & operator=(const TransMsg & obj) {
eventMsg::operator=(obj);
id = obj.id;
origSrc = obj.origSrc;
dstNode = obj.dstNode;
msgId = obj.msgId;
return *this;
}
};

class Transceiver {
public:
int id;
Transceiver(){}
Transceiver(TransMsg *);
~Transceiver(){}

void sendMessage(TransMsg *);
void sendMessage_anti(TransMsg *){restore(this);}
void sendMessage_commit(TransMsg *){}
void recvMessage(TransMsg *);
void recvMessage_anti(TransMsg *){restore(this);}
void recvMessage_commit(TransMsg *){}

Transceiver& operator=(const Transceiver& obj) {
        rep::operator=(obj);
        id = obj.id;
        return *this;
}
bool operator==(const Transceiver & obj) const {
    return (id==obj.id);
}

};

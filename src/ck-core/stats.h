#ifndef _STATS_H
#define _STATS_H

class Stats {
  private:
    UInt pe;
    UInt charesCreated; // # of new chare msgs sent
    UInt charesProcessed; // # of new chare msgs processed
    UInt forCharesCreated; // # of for chare msgs created
    UInt forCharesProcessed; // # of for chare msgs processed
    UInt groupInitsCreated; // # of group init msgs created
    UInt groupInitsProcessed; // # of groupinits processed
    UInt groupMsgsCreated; // # of for group msgs created
    UInt groupMsgsProcessed; // # of for group msgs processed
    UInt nodeGroupInitsCreated; // # of node group init msgs created
    UInt nodeGroupInitsProcessed; // # of node group inits processed
    UInt nodeGroupMsgsCreated; // # of for nodegroup msgs created
    UInt nodeGroupMsgsProcessed; // # of for nodegroup msgs processed
  public:
    void *operator new(size_t size) { return _allocMsg(StatMsg, size); }
    void operator delete(void *ptr) { CkFreeMsg(ptr); }
    Stats() {
      pe = CkMyPe();
      charesCreated = 0; 
      charesProcessed = 0; 
      forCharesCreated = 0; 
      forCharesProcessed = 0; 
      groupInitsCreated = 0; 
      groupInitsProcessed = 0; 
      groupMsgsCreated = 0; 
      groupMsgsProcessed = 0; 
      nodeGroupInitsCreated = 0; 
      nodeGroupInitsProcessed = 0; 
      nodeGroupMsgsCreated = 0; 
      nodeGroupMsgsProcessed = 0; 
    }
    void combine(const Stats* const other) {
      charesCreated += other->charesCreated; 
      charesProcessed += other->charesProcessed; 
      forCharesCreated += other->forCharesCreated; 
      forCharesProcessed += other->forCharesProcessed; 
      groupInitsCreated += other->groupInitsCreated; 
      groupInitsProcessed += other->groupInitsProcessed; 
      groupMsgsCreated += other->groupMsgsCreated; 
      groupMsgsProcessed += other->groupMsgsProcessed; 
      nodeGroupInitsCreated += other->nodeGroupInitsCreated; 
      nodeGroupInitsProcessed += other->nodeGroupInitsProcessed; 
      nodeGroupMsgsCreated += other->nodeGroupMsgsCreated; 
      nodeGroupMsgsProcessed += other->nodeGroupMsgsProcessed; 
    }
    void recordCreateChare(int x=1) { charesCreated += x; }
    void recordProcessChare(int x=1) { charesProcessed += x; }
    void recordSendMsg(int x=1) { forCharesCreated += x; }
    void recordProcessMsg(int x=1) { forCharesProcessed += x; }
    void recordCreateGroup(int x=1) { groupInitsCreated += x; }
    void recordProcessGroup(int x=1) { groupInitsProcessed += x; }
    void recordSendBranch(int x=1) { groupMsgsCreated += x; }
    void recordProcessBranch(int x=1) { groupMsgsProcessed += x; }
    void recordCreateNodeGroup(int x=1) { nodeGroupInitsCreated += x; }
    void recordProcessNodeGroup(int x=1) { nodeGroupInitsProcessed += x; }
    void recordSendNodeBranch(int x=1) { nodeGroupMsgsCreated += x; }
    void recordProcessNodeBranch(int x=1) { nodeGroupMsgsProcessed += x; }
    UInt getPe(void) const { return pe; }
    UInt getCharesCreated(void) const { return charesCreated; }
    UInt getCharesProcessed(void) const { return charesProcessed; }
    UInt getForCharesCreated(void) const { return forCharesCreated; }
    UInt getForCharesProcessed(void) const { return forCharesProcessed; }
    UInt getGroupsCreated(void) const { return groupInitsCreated; }
    UInt getGroupsProcessed(void) const { return groupInitsProcessed; }
    UInt getGroupMsgsCreated(void) const { return groupMsgsCreated; }
    UInt getGroupMsgsProcessed(void) const { return groupMsgsProcessed; }
    UInt getNodeGroupsCreated(void) const { return nodeGroupInitsCreated; }
    UInt getNodeGroupsProcessed(void) const { return nodeGroupInitsProcessed; }
    UInt getNodeGroupMsgsCreated(void) const { return nodeGroupMsgsCreated; }
    UInt getNodeGroupMsgsProcessed(void) const {return nodeGroupMsgsProcessed;}
};

CkpvExtern(Stats*, _myStats);

#if CMK_WITH_STATS
#define _STATS_RECORD_CREATE_CHARE_1() \
          CkpvAccess(_myStats)->recordCreateChare()
#define _STATS_RECORD_CREATE_CHARE_N(x) \
          CkpvAccess(_myStats)->recordCreateChare(x)
#define _STATS_RECORD_PROCESS_CHARE_1() \
          CkpvAccess(_myStats)->recordProcessChare()
#define _STATS_RECORD_PROCESS_CHARE_N(x) \
          CkpvAccess(_myStats)->recordProcessChare(x)
#define _STATS_RECORD_SEND_MSG_1() \
          CkpvAccess(_myStats)->recordSendMsg()
#define _STATS_RECORD_SEND_MSG_N(x) \
          CkpvAccess(_myStats)->recordSendMsg(x)
#define _STATS_RECORD_PROCESS_MSG_1() \
          CkpvAccess(_myStats)->recordProcessMsg()
#define _STATS_RECORD_PROCESS_MSG_N(x) \
          CkpvAccess(_myStats)->recordProcessMsg(x)
#define _STATS_RECORD_CREATE_GROUP_1() \
          CkpvAccess(_myStats)->recordCreateGroup()
#define _STATS_RECORD_CREATE_GROUP_N(x) \
          CkpvAccess(_myStats)->recordCreateGroup(x)
#define _STATS_RECORD_PROCESS_GROUP_1() \
          CkpvAccess(_myStats)->recordProcessGroup()
#define _STATS_RECORD_PROCESS_GROUP_N(x) \
          CkpvAccess(_myStats)->recordProcessGroup(x)
#define _STATS_RECORD_SEND_BRANCH_1() \
          CkpvAccess(_myStats)->recordSendBranch()
#define _STATS_RECORD_SEND_BRANCH_N(x) \
          CkpvAccess(_myStats)->recordSendBranch(x)
#define _STATS_RECORD_PROCESS_BRANCH_1() \
          CkpvAccess(_myStats)->recordProcessBranch()
#define _STATS_RECORD_PROCESS_BRANCH_N(x) \
          CkpvAccess(_myStats)->recordProcessBranch(x)
#define _STATS_RECORD_CREATE_NODE_GROUP_1() \
          CkpvAccess(_myStats)->recordCreateNodeGroup()
#define _STATS_RECORD_CREATE_NODE_GROUP_N(x) \
          CkpvAccess(_myStats)->recordCreateNodeGroup(x)
#define _STATS_RECORD_PROCESS_NODE_GROUP_1() \
          CkpvAccess(_myStats)->recordProcessNodeGroup()
#define _STATS_RECORD_PROCESS_NODE_GROUP_N(x) \
          CkpvAccess(_myStats)->recordProcessNodeGroup(x)
#define _STATS_RECORD_SEND_NODE_BRANCH_1() \
          CkpvAccess(_myStats)->recordSendNodeBranch()
#define _STATS_RECORD_SEND_NODE_BRANCH_N(x) \
          CkpvAccess(_myStats)->recordSendNodeBranch(x)
#define _STATS_RECORD_PROCESS_NODE_BRANCH_1() \
          CkpvAccess(_myStats)->recordProcessNodeBranch()
#define _STATS_RECORD_PROCESS_NODE_BRANCH_N(x) \
          CkpvAccess(_myStats)->recordProcessNodeBranch(x)
#else
#define _STATS_RECORD_CREATE_CHARE_1() do{}while(0)
#define _STATS_RECORD_CREATE_CHARE_N(x) do{}while(0)
#define _STATS_RECORD_PROCESS_CHARE_1() do{}while(0)
#define _STATS_RECORD_PROCESS_CHARE_N(x) do{}while(0)
#define _STATS_RECORD_SEND_MSG_1() do{}while(0)
#define _STATS_RECORD_SEND_MSG_N(x) do{}while(0)
#define _STATS_RECORD_PROCESS_MSG_1() do{}while(0)
#define _STATS_RECORD_PROCESS_MSG_N(x) do{}while(0)
#define _STATS_RECORD_CREATE_GROUP_1() do{}while(0)
#define _STATS_RECORD_CREATE_GROUP_N(x) do{}while(0)
#define _STATS_RECORD_PROCESS_GROUP_1() do{}while(0)
#define _STATS_RECORD_PROCESS_GROUP_N(x) do{}while(0)
#define _STATS_RECORD_SEND_BRANCH_1() do{}while(0)
#define _STATS_RECORD_SEND_BRANCH_N(x) do{}while(0)
#define _STATS_RECORD_PROCESS_BRANCH_1() do{}while(0)
#define _STATS_RECORD_PROCESS_BRANCH_N(x) do{}while(0)
#define _STATS_RECORD_CREATE_NODE_GROUP_1() do{}while(0)
#define _STATS_RECORD_CREATE_NODE_GROUP_N(x) do{}while(0)
#define _STATS_RECORD_PROCESS_NODE_GROUP_1() do{}while(0)
#define _STATS_RECORD_PROCESS_NODE_GROUP_N(x) do{}while(0)
#define _STATS_RECORD_SEND_NODE_BRANCH_1() do{}while(0)
#define _STATS_RECORD_SEND_NODE_BRANCH_N(x) do{}while(0)
#define _STATS_RECORD_PROCESS_NODE_BRANCH_1() do{}while(0)
#define _STATS_RECORD_PROCESS_NODE_BRANCH_N(x) do{}while(0)
#endif
#endif

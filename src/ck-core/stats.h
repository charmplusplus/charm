#ifndef _STATS_H
#define _STATS_H

class Stats {
  private:
    UInt charesCreated; // # of new chare msgs sent
    UInt charesProcessed; // # of new chare msgs processed
    UInt forCharesCreated; // # of for chare msgs created
    UInt forCharesProcessed; // # of for chare msgs processed
    UInt groupInitsCreated; // # of group init msgs created
    UInt groupInitsProcessed; // # of groupinits processed
    UInt groupMsgsCreated; // # of for group msgs created
    UInt groupMsgsProcessed; // # of for group msgs processed
  public:
    void *operator new(size_t size) { return _allocMsg(StatMsg, size); }
    void operator delete(void *ptr) { CkFreeMsg(ptr); }
    Stats() {
      charesCreated = 0; 
      charesProcessed = 0; 
      forCharesCreated = 0; 
      forCharesProcessed = 0; 
      groupInitsCreated = 0; 
      groupInitsProcessed = 0; 
      groupMsgsCreated = 0; 
      groupMsgsProcessed = 0; 
    }
    void combine(Stats* other) {
      charesCreated += other->charesCreated; 
      charesProcessed += other->charesProcessed; 
      forCharesCreated += other->forCharesCreated; 
      forCharesProcessed += other->forCharesProcessed; 
      groupInitsCreated += other->groupInitsCreated; 
      groupInitsProcessed += other->groupInitsProcessed; 
      groupMsgsCreated += other->groupMsgsCreated; 
      groupMsgsProcessed += other->groupMsgsProcessed; 
    }
    void recordCreateChare(int x=1) { charesCreated += x; }
    void recordProcessChare(int x=1) { charesProcessed += x; }
    void recordSendMsg(int x=1) { forCharesCreated += x; }
    void recordProcessMsg(int x=1) { forCharesProcessed += x; }
    void recordCreateGroup(int x=1) { groupInitsCreated += x; }
    void recordProcessGroup(int x=1) { groupInitsProcessed += x; }
    void recordSendBranch(int x=1) { groupMsgsCreated += x; }
    void recordProcessBranch(int x=1) { groupMsgsProcessed += x; }
    UInt getCharesCreated(void) { return charesCreated; }
    UInt getCharesProcessed(void) { return charesProcessed; }
    UInt getForCharesCreated(void) { return forCharesCreated; }
    UInt getForCharesProcessed(void) { return forCharesProcessed; }
    UInt getGroupsCreated(void) { return groupInitsCreated; }
    UInt getGroupsProcessed(void) { return groupInitsProcessed; }
    UInt getGroupMsgsCreated(void) { return groupMsgsCreated; }
    UInt getGroupMsgsProcessed(void) { return groupMsgsProcessed; }
};

CpvExtern(Stats*, _myStats);

#endif

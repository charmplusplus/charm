/**
 \defgroup CkLdb  Charm++ Load Balancing Framework 
*/
/*@{*/

#ifndef BASELB_H
#define BASELB_H

#include "LBDatabase.h"

/// Base class for all LB strategies.
/**
  BaseLB is the base class for all LB strategy class.
  it does some tracking about how many lb strategies are created.
  it also defines some common functions.
*/
class BaseLB: public Group
{
protected:
  LBDatabase *theLbdb;
  LDBarrierReceiver receiver;
public:
  BaseLB() ;
  void unregister(); 
};

/// migration decision for an obj.
struct MigrateInfo {  
    LDObjHandle obj;
    int from_pe;
    int to_pe;
};

/**
  message contains the migration decision from LB strategies.
*/
class LBMigrateMsg : public CMessage_LBMigrateMsg {
public:
  int n_moves;
  MigrateInfo* moves;

  char * avail_vector;
  int next_lb;

  // Other methods & data members

  static void* alloc(int msgnum, size_t size, int* array, int priobits);
  static void* pack(LBMigrateMsg* in);
  static LBMigrateMsg* unpack(void* in);
};

#endif

/*@}*/

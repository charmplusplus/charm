#ifndef _CKARRAY_H
#define _CKARRAY_H

#include "charm++.h"

#if CMK_LBDB_ON
#include "LBDatabase.h"
#endif

extern void _registerCkArray(void);

class PtrQ;
class PtrVec;

#define ALIGN8(x)       (int)(8*(((x)+7)/8))

#define MessageIndex(mt)        CMessage_##mt##::__idx
#define ChareIndex(ct)          CProxy_##ct##::__idx
#define EntryIndex(ct,ep,mt)    CProxy_##ct##::ckIdx_##ep##((##mt##*)0)
#define ConstructorIndex(ct,mt) EntryIndex(ct,ct,mt)

typedef int MessageIndexType;
typedef int ChareIndexType;
typedef int EntryIndexType;

extern CkGroupID _RRMapID;

#if CMK_LBDB_ON
class LBDatabase;
#endif

class Array1D;
class ArrayMapRegisterMessage;
class ArrayElementCreateMessage;
class ArrayElementMigrateMessage;
class ArrayElementExitMessage;

class ArrayMap : public Group
{
public:
  virtual int procNum(int arrayHdl, int element) = 0;
  virtual void registerArray(ArrayMapRegisterMessage *) = 0;
};


class ArrayElement : public Chare
{
friend class Array1D;
public:
  ArrayElement(ArrayElementCreateMessage *msg);
  ArrayElement(ArrayElementMigrateMessage *msg);

private:
  ArrayElement(void) {};

protected:

  // For Backward compatibility:
  void finishConstruction(void) { finishConstruction(false); };

  void finishConstruction(bool use_local_barrier);
  void finishMigration(void);

  virtual int packsize(void) { return 0; }
  virtual void pack(void *) { return; }
  void AtSync();
  virtual void ResumeFromSync(void) {
    CkPrintf("No ResumeFromSync() defined for this element!\n");
  };

  int thisIndex;
  CkAID thisAID;
  int numElements;

public:
  void migrate(int where);
  void exit(ArrayElementExitMessage *msg);
  int getIndex(void) { return thisIndex; }
  int getSize(void)  { return numElements; }

private:
  CkChareID arrayChareID;
  CkGroupID arrayGroupID;
  Array1D *thisArray;
};

enum {unknownPe = -1};

class ArrayCreateMessage;
class ArrayMessage;
class ArrayMigrateMessage;
class ArrayElementAckMessage;

class Array1D : public Group {
friend class ArrayElement;

public:
  static  CkGroupID CreateArray(int numElements,
				CkGroupID mapID,
				ChareIndexType elementChare,
				EntryIndexType elementConstructor,
				EntryIndexType elementMigrator);

  Array1D(ArrayCreateMessage *);
  void send(ArrayMessage *msg, int index, EntryIndexType ei);
  void broadcast(ArrayMessage *msg, EntryIndexType ei);
  void RecvMapID(ArrayMap *mapPtr,int mapHandle);
  void RecvElementID(int index, ArrayElement *elem, CkChareID handle,
		     bool uses_barrier);
  void RecvForElement(ArrayMessage *msg);
  void RecvMigratedElement(ArrayMigrateMessage *msg);
  void RecvMigratedElementID(int index, ArrayElement *elem, CkChareID handle);
  void AckMigratedElement(ArrayElementAckMessage *msg);
  int array_size(void) { return numElements; };
  int num_local(void) { return numLocalElements; };
  int ckGetGroupId(void) { return thisgroup; }
  ArrayElement *getElement(int idx) { return elementIDs[idx].element; }

#if CMK_LBDB_ON
  static void staticMigrate(LDObjHandle _h, int _dest);
  static void staticSetStats(LDOMHandle _h, int _state);
  static void staticQueryLoad(LDOMHandle _h);
  static void staticResumeFromSync(void* data);
  static void staticRecvAtSync(void* data);
#endif

  typedef enum {creating, here, moving_to, arriving, at} ElementState;

private:
  void migrateMe(int index, int where);

#if CMK_LBDB_ON
  void Migrate(LDObjHandle _h, int _dest);
  void SetStats(LDOMHandle _h, int _state);
  void QueryLoad(LDOMHandle _h);

  void RegisterElementForSync(int index);
  void AtSync(int index);
  void ResumeFromSync(int index);
  void RecvAtSync();
#endif

  struct ElementIDs {
    struct BarrierClientData {
      Array1D *me;            
      int index;
    };

    ElementState state;
    int originalPE;
    int pe;
    ArrayElement *element;
    CkChareID elementHandle;
    int cameFrom;
    int curHop;
    ArrayMigrateMessage *migrateMsg;
#if CMK_LBDB_ON
    LDObjHandle ldHandle;
    bool uses_barrier;
    LDBarrierClient barrierHandle;
    BarrierClientData barrierData;
#endif
  };

  int numElements;
  int mapHandle;
  CkGroupID mapGroup;
  ArrayMap *map;
  ChareIndexType elementChareType;
  EntryIndexType elementConstType;
  EntryIndexType elementMigrateType;
  ElementIDs *elementIDs;
  int elementIDsReported;
  int numLocalElements;

#if CMK_LBDB_ON
  LDOMHandle myHandle;
  LBDatabase *the_lbdb;
#endif
  PtrQ *bufferedForElement;
  PtrQ *bufferedMigrated;
};

#include "CkArray.decl.h"

class ArrayCreateMessage : public CMessage_ArrayCreateMessage
{
public:
  int numElements;
  CkGroupID mapID;
  ChareIndexType elementChareType;
  EntryIndexType elementConstType;
  EntryIndexType elementMigrateType;
  CkGroupID loadbalancer;
};

class ArrayMessage
{
public:
  int destIndex;
  EntryIndexType entryIndex;
  int hopCount;
  int serial_num;
};

class ArrayElementAckMessage : public CMessage_ArrayElementAckMessage
{
public:
  int index;
  int arrivedAt;
  int deleteElement;
  CkChareID handle;
  int hopCount;
};

class ArrayMigrateMessage : public CMessage_ArrayMigrateMessage
{
public:
  int from;
  int index;
  int elementSize;
  void *elementData;
  int hopCount;
  bool uses_barrier;

  static void *alloc(int msgnum, int size, int *array, int priobits);
  static void *pack(ArrayMigrateMessage *);
  static ArrayMigrateMessage *unpack(void *in);
};

class RRMap : public ArrayMap
{
private:
  PtrVec *arrayVec;
public:
  RRMap(void);
  void registerArray(ArrayMapRegisterMessage *);
  int procNum(int arrayHdl, int element);
};

class ArrayInit : public Chare 
{
public:
  ArrayInit(CkArgMsg *msg) {
    _RRMapID = CProxy_RRMap::ckNew();
    delete msg;
  }
};

class ArrayMapRegisterMessage : public CMessage_ArrayMapRegisterMessage
{
public:
  int numElements;
  CkChareID arrayID;
  CkGroupID groupID;
};

class ArrayElementCreateMessage : public CMessage_ArrayElementCreateMessage {
public:
  int numElements;
  CkChareID arrayID;
  CkGroupID groupID;
  Array1D *arrayPtr;
  int index;
};

class ArrayElementMigrateMessage : public CMessage_ArrayElementMigrateMessage {
public:
  int numElements;
  CkChareID arrayID;
  CkGroupID groupID;
  Array1D *arrayPtr;
  int index;
  void* packData;
};

class ArrayElementExitMessage : public CMessage_ArrayElementExitMessage
{
public:
  int dummy;
};

#endif

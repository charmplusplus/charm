#ifndef _CKARRAY_H
#define _CKARRAY_H

#include "charm++.h"
#include "CkArray.decl.h"

extern void _registerCkArray(void);

typedef enum {false, true} boolean;

#define ALIGN8(x)       (int)(8*(((x)+7)/8))

#define MessageIndex(mt)        CMessage_##mt##::__idx
#define ChareIndex(ct)          CProxy_##ct##::__idx
#define EntryIndex(ct,ep,mt)    CProxy_##ct##::ckIdx_##ep##((##mt##*)0)
#define ConstructorIndex(ct,mt) EntryIndex(ct,ct,mt)

typedef int CkGroupID;
typedef int MessageIndexType;
typedef int ChareIndexType;
typedef int EntryIndexType;

class Array1D;

class ArrayMapCreateMessage : public CMessage_ArrayMapCreateMessage
{
public:
  int numElements;
  CkChareID arrayID;
  CkGroupID groupID;
};

class ArrayMap : public Group
{
public:
  virtual int procNum(int element) = 0;

protected:
  ArrayMap(ArrayMapCreateMessage *msg);
  void finishConstruction(void);

  CkChareID arrayChareID;
  CkGroupID arrayGroupID;
  Array1D *array;
  int numElements;
};

class Array1D;
class ArrayElementCreateMessage;
class ArrayElementMigrateMessage;
class ArrayElementExitMessage;

class ArrayElement : public Group
{
friend class Array1D;
public:
  ArrayElement(ArrayElementCreateMessage *msg);
  ArrayElement(ArrayElementMigrateMessage *msg);
  void finishConstruction(void);
  void migrate(int where);
  void finishMigration(void);
  void exit(ArrayElementExitMessage *msg);

protected:
  virtual int packsize(void);
  virtual void pack(void *pack);

  CkChareID arrayChareID;
  CkGroupID arrayGroupID;
  Array1D *thisArray;
  int numElements;
  int thisIndex;
};

class ArrayElementCreateMessage : public CMessage_ArrayElementCreateMessage
{
public:
  int numElements;
  CkChareID arrayID;
  CkGroupID groupID;
  Array1D *arrayPtr;
  int index;
};

class ArrayElementMigrateMessage : public CMessage_ArrayElementMigrateMessage
{
public:
  int numElements;
  CkChareID arrayID;
  CkGroupID groupID;
  Array1D *arrayPtr;
  int index;
};

class ArrayElementExitMessage : public CMessage_ArrayElementExitMessage
{
public:
  int dummy;
};

enum {unknownPe = -1};

class ArrayCreateMessage;
class ArrayMessage;
class ArrayMigrateMessage;
class ArrayElementAckMessage;

class Array1D : public Group
{
public:
  static  Array1D::CreateArray(int numElements,
                               ChareIndexType mapChare,
                               EntryIndexType mapConstructor,
                               ChareIndexType elementChare,
                               EntryIndexType elementConstructor,
                               EntryIndexType elementMigrator);

  Array1D(ArrayCreateMessage *);
  void send(ArrayMessage *msg, int index, EntryIndexType ei);
  void broadcast(ArrayMessage *msg, EntryIndexType ei);
  void RecvMapID(ArrayMap *mapPtr,CkChareID mapHandle, CkGroupID mapGroup);
  void RecvElementID(int index, ArrayElement *elem, CkChareID handle);
  void RecvForElement(ArrayMessage *msg);
  void RecvMigratedElement(ArrayMigrateMessage *msg);
  void RecvMigratedElementID(int index, ArrayElement *elem, CkChareID handle);
  void AckMigratedElement(ArrayElementAckMessage *msg);
  void migrateMe(int index, int where);
  int array_size(void) { return numElements; };
  int num_local(void) { return numLocalElements; };

  typedef enum {creating, here, moving_to, arriving, at} ElementState;

private:

  struct ElementIDs {
    ElementState state;
    int originalPE;
    int pe;
    ArrayElement *element;
    CkChareID elementHandle;
    int cameFrom;
    int curHop;
  };

  int numElements;
  CkChareID mapHandle;
  CkGroupID mapGroup;
  ArrayMap *map;
  ChareIndexType elementChareType;
  EntryIndexType elementConstType;
  EntryIndexType elementMigrateType;
  ElementIDs *elementIDs;
  int elementIDsReported;
  int numLocalElements;
};

class ArrayCreateMessage : public CMessage_ArrayCreateMessage
{
public:
  int numElements;
  ChareIndexType mapChareType;
  EntryIndexType mapConstType;
  ChareIndexType elementChareType;
  EntryIndexType elementConstType;
  EntryIndexType elementMigrateType;
};

class ArrayMessage : public CMessage_ArrayMessage
{
public:
  int destIndex;
  EntryIndexType entryIndex;
};

class ArrayElementAckMessage : public CMessage_ArrayElementAckMessage
{
public:
  int index;
  int arrivedAt;
  boolean deleteElement;
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

  static void *alloc(int msgnum, int size, int *array, int priobits);
  static void *pack(ArrayMigrateMessage *);
  static ArrayMigrateMessage *unpack(void *in);
};

class RRMap : public ArrayMap
{
public:
  RRMap(ArrayMapCreateMessage *msg);
  ~RRMap(void);

  int procNum(int element);
};

#endif

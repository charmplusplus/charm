#ifndef ARRAY1D_H
#define ARRAY1D_H

#include "arraydefs.h"
#include "Array1D.top.h"
#include "ArrayMap.h"
#include "ArrayElement.h"

enum {unknownPe = -1};

class ArrayCreateMessage;
class ArrayMessage;
class ArrayMigrateMessage;
class ArrayElementAckMessage;

class Array1D : public groupmember
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

  void RecvMapID(ArrayMap *mapPtr,ChareIDType mapHandle,
		 GroupIDType mapGroup);

  void RecvElementID(int index, ArrayElement *elem, ChareIDType handle);

  void RecvForElement(ArrayMessage *msg);

  void RecvMigratedElement(ArrayMigrateMessage *msg);

  void RecvMigratedElementID(int index, ArrayElement *elem,
			     ChareIDType handle);

  void AckMigratedElement(ArrayElementAckMessage *msg);

  void migrateMe(int index, int where);
  
  int array_size(void) { return numElements; };
  int num_local(void) { return numLocalElements; };

private:
  typedef enum {creating, here, moving_to, arriving, at} ElementState;

  struct ElementIDs
  {
    ElementState state;
    int originalPE;
    int pe;
    ArrayElement *element;
    ChareIDType elementHandle;
    int cameFrom;
    int curHop;
  };

  int numElements;
  ChareIDType mapHandle;
  GroupIDType mapGroup;
  ArrayMap *map;
  ChareIndexType elementChareType;
  EntryIndexType elementConstType;
  EntryIndexType elementMigrateType;
  ElementIDs *elementIDs;
  int elementIDsReported;
  int numLocalElements;

};

class ArrayCreateMessage : public comm_object
{
public:
  int numElements;
  ChareIndexType mapChareType;
  EntryIndexType mapConstType;
  ChareIndexType elementChareType;
  EntryIndexType elementConstType;
  EntryIndexType elementMigrateType;
};

class ArrayMessage : public comm_object
{
public:
  int destIndex;
  EntryIndexType entryIndex;
};

class ArrayElementAckMessage : public comm_object
{
public:
  int index;
  int arrivedAt;
  boolean deleteElement;
  ChareIDType handle;
  int hopCount;
};

class ArrayMigrateMessage : public comm_object
{
public:
  int from;
  int index;
  int elementSize;
  void *elementData;
  int hopCount;
  
  static void *alloc(int msgnum, int size, int *array, int priobits);
#if 0
  {
    int totalsize;
    totalsize = size + array[0]*sizeof(char) + 8;
    CPrintf("Allocating %d %d %d\n",msgnum,totalsize,priobits);
    ArrayMigrateMessage *newMsg = (ArrayMigrateMessage *)
      GenericCkAlloc(msgnum,totalsize,priobits);
    CPrintf("Allocated %d\n",newMsg);
    newMsg->elementData = (char *)newMsg + ALIGN8(size);

    return (void *) newMsg;
  }
#endif
  
  void *pack(int *length);
#if 0
  {
    CPrintf("Packing %d %d %d\n",from,index,elementSize);
    elementData = (void *)((char *)elementData - (char *)&elementData);
    return this;
  }
#endif

  void unpack(void *in);
#if 0
  {
    CPrintf("Unpacking %d %d %d\n",from,index,elementSize);
    elementData = (char *)&elementData + (int)elementData;
  }
#endif
};
#endif






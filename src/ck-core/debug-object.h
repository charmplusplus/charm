#ifndef _OBJECTDATA_H
#define _OBJECTDATA_H

#include "charm++.h"

#if CMK_DEBUG_MODE

#define PRIME 17  // prime number for hash table

extern char* getObjectList(void);
extern char* getObjectContents(void);

extern "C" void  CpdInitializeObjectTable(void);

typedef struct HashTableElement {
  Chare* charePtr;
  int chareIndex;
  struct HashTableElement *next;
};

class HashTable{
 public:
  HashTable();
  ~HashTable();
  void putObject(Chare* charePtr);
  void removeObject(Chare* charePtr);
  char* getObjectList(void);
  char* getObjectContents(int chareIndex);
 private:
  HashTableElement *array[PRIME];
};

#endif
#endif

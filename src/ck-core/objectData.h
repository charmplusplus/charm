#ifndef PROGRAM_DATA_H
#define PROGRAM_DATA_H

#include <charm++.h>

#define PRIME 17  // prime number for hash table

char* getObjectList();
char* getObjectContents();
void putObject(_CK_Object* charePtr);
void CpdInitializeObjectTable();

typedef struct HashTableElement {
  _CK_Object* charePtr;
  int chareIndex;
  struct HashTableElement *next;
};

class HashTable{
 public:
  HashTable();
  ~HashTable();
  void putObject(_CK_Object* charePtr);
  void removeObject(_CK_Object *charePtr);
  char* getObjectList(void);
  char* getObjectContents(int chareIndex);

 private:
  HashTableElement *array[PRIME];
  int chareIndex;
};

#endif

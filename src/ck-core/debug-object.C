#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "charm++.h"
#include "ck.h"

#include "debug-object.h"

#if CMK_DEBUG_MODE

HashTable* objectTable;

void putObject(Chare* charePtr)
{
  objectTable->putObject(charePtr);
}

void removeObject(Chare* charePtr)
{
  objectTable->removeObject(charePtr);
}

char* getObjectList(void)
{
  return(objectTable->getObjectList());
}

char* getObjectContents(int chareIndex)
{
  return(objectTable->getObjectContents(chareIndex));
}

HashTable::HashTable()
{
  for(int i = 0; i < PRIME; i++)
    array[i] = 0;
}

HashTable::~HashTable()
{
  struct HashTableElement *node, *temp;

  for(int i = 0; i < PRIME; i++){
    node = array[i];
    while(node != 0){
      temp = node;
      node = node -> next;
      free(temp);
    }
  }
}

void HashTable::putObject(Chare* charePtr)
{
  struct HashTableElement *node;
  int pigeonHole;
  int chareIndex;
  
  chareIndex = (size_t)charePtr;
  node = (struct HashTableElement *)malloc(sizeof(struct HashTableElement));
  node->charePtr = charePtr;
  node->chareIndex = chareIndex;
  pigeonHole = chareIndex % PRIME;
  node->next = array[pigeonHole];
  array[pigeonHole] = node;
}

void HashTable::removeObject(Chare* charePtr)
{
  int pigeonHole;
  struct HashTableElement *node, *prev;
  int chareIndex;
  
  chareIndex = (size_t)charePtr;
  pigeonHole = chareIndex % PRIME;

  prev = 0;
  node = array[pigeonHole];
  while(node != 0){
    if(node -> chareIndex == chareIndex){
      if(prev == 0){
	array[pigeonHole] = node->next;
      } else {
	prev -> next = node -> next;
      }
      free(node);
      return;
    }
    prev = node;
    node = node -> next;
  }
  CkError("Erroneous chareIndex supplied in removeObject()\n"); 
}

char* HashTable::getObjectList(void)
{
  struct HashTableElement *node; 
  char *temp;
  char *list, *oldlist;
  char t[10];
  int maxLength = PRIME * 20 * sizeof(char);
  
  list = (char *)malloc(maxLength);
  strcpy(list, "");
  for(int i = 0; i < PRIME; i++){
    node = array[i];
    while(node != 0){
      if ((node -> chareIndex != 0) && (node -> charePtr != 0)){
	temp = (node -> charePtr) -> showHeader();
	if((strlen(list) + strlen(temp) + 10) > maxLength){
	  maxLength *= 2;
	  oldlist = list;
	  list = (char *)malloc(maxLength);
	  strcpy(list, oldlist);
	  free(oldlist);
	}
	strcat(list, temp);
	strcat(list, "#");
	sprintf(t, "%d", (node -> chareIndex));
	strcat(list, t);
	strcat(list, "#");
	free(temp);
      }
      node = node -> next;
    }
  }
  return(list);
}

char* HashTable::getObjectContents(int chareIndex)
{
  struct HashTableElement *node;
  
  node = array[chareIndex % PRIME];
  while(node != 0){
    if(node -> chareIndex == chareIndex)
      return((node -> charePtr) -> showContents());
    node = node -> next;
  }
  
  CkError("Erroneous chareIndex supplied in getObjectCOntents()\n");
}

extern "C"
void CpdInitializeObjectTable(void)
{
  objectTable = new HashTable();
}

#endif

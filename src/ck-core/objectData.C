
#include <iostream.h>
#include <strstream.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <charm++.h>

#if CMK_DEBUG_MODE

#include "objectData.h"

HashTable* objectTable;

void putObject(_CK_Object* charePtr){
  objectTable -> putObject(charePtr);
}

void removeObject(_CK_Object* charePtr){
  objectTable -> removeObject(charePtr);
}

char* getObjectList(){
  return(objectTable -> getObjectList());
}

char* getObjectContents(int chareIndex){
  return(objectTable -> getObjectContents(chareIndex));
}

HashTable::HashTable(){
  chareIndex = 0;
  for(int i = 0; i < PRIME; i++)
    array[i] = 0;
}

HashTable::~HashTable(){
  struct HashTableElement *node, *temp;

  for(int i = 0; i < PRIME; i++){
    node = array[i];
    while(node != NULL){
      temp = node;
      node = node -> next;
      free(temp);
    }
  }
}

void HashTable::putObject(_CK_Object* charePtr){
  struct HashTableElement *node;
  int pigeonHole;
  
  chareIndex = (int)charePtr;
  node = (struct HashTableElement *)malloc(sizeof(struct HashTableElement));
  if(node == NULL){
    CmiAbort("Memory overflow\n");
  }
  node -> charePtr = charePtr;
  node -> chareIndex = chareIndex;
  pigeonHole = chareIndex % PRIME;
  node -> next = array[pigeonHole];
  array[pigeonHole] = node;
}

void HashTable::removeObject(_CK_Object* charePtr){
  int pigeonHole;
  struct HashTableElement *node, *prev;
  
  pigeonHole = (int)charePtr % PRIME;
  prev = NULL;
  node = array[pigeonHole];
  while(node != NULL){
    if(node -> chareIndex == chareIndex){
      if(prev == NULL){
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
  cerr << "Erroneous chareIndex supplied in getObjectCOntents()" << endl; 
}

char* HashTable::getObjectList(void){
  strstream str;
  struct HashTableElement *node; 
  char *temp;
  char *list, *oldlist;
  char t[10];
  int maxLength = PRIME * 20 * sizeof(char);
  
  list = (char *)malloc(maxLength);
  strcpy(list, "");
  for(int i = 0; i < PRIME; i++){
    node = array[i];
    while(node != NULL){
      if ((node -> chareIndex != 0) && (node -> charePtr != NULL)){
	//str << (node -> charePtr) -> showHeader() << "#" << (node -> chareIndex) << "#";
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

  //str << '\0';
  //return str.str();
  return(list);
}

char* HashTable::getObjectContents(int chareIndex){
  struct HashTableElement *node;
  
  node = array[chareIndex % PRIME];
  while(node != NULL){
    if(node -> chareIndex == chareIndex)
      return((node -> charePtr) -> showContents());
    node = node -> next;
  }
  
  cerr << "Erroneous chareIndex supplied in getObjectCOntents()" << endl;
}

void CpdInitializeObjectTable(){
  objectTable = new HashTable();
}

#endif

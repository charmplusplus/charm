/**
 * \addtogroup CkLdb
*/
/*@{*/

#ifndef SET_DEFS_H
#define SET_DEFS_H

class InfoRecord;

class listNode {
public:
listNode *next;
InfoRecord *info;
};

class Iterator{
public:
  int id; // for debugging
  listNode* next;
};

class Set {

private:
 listNode *head;

public:
 Set();
 ~Set();
 void insert(InfoRecord *);
 int find(InfoRecord *) ;
 void remove(InfoRecord *);
 void myRemove(listNode **n, InfoRecord *r);
 InfoRecord *iterator(Iterator *);
 InfoRecord *next(Iterator *);
 int numElements();
 void print();
};

#endif

/*@}*/

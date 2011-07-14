/**
 * \addtogroup CkLdb
*/
/*@{*/

#include "elements.h"

Set::Set() 
{
  head = (listNode *) 0;
}

Set::~Set()
{
  listNode *p = head;
  while (p){
    listNode *cur = p;
    p = p->next;
    delete cur;
  }
}

void Set::insert(InfoRecord *info) 
{
  if (!find(info))
  {
    listNode *node = new listNode();
    node->info = info;
    node->next = head;
    head = node;
  }
   
}


void Set::myRemove(listNode **n, InfoRecord *r)
{
  if ((*n)->info == r)
    *n = (*n)->next;
  else 
    myRemove(&((*n)->next), r);
}

void Set::remove(InfoRecord * r) 
{
  listNode *p = head;
  if (!head)
    return;

  listNode *q = head->next;

  if (p->info == r){
    head = head->next;
    return;
  }
     
  while (q){
    if (q->info == r){
      p->next = q->next;
      delete q;
      return;
    }
    else {
      p = q;
      q = q->next;
    }
  }
}

int Set::find(InfoRecord * r) 
{
  listNode *p = head;
  while (p) {
    if (p->info == r) return 1;
    else p = p->next;
  }
  return 0;
}

InfoRecord * Set::iterator(Iterator *iter)
{
  if (head){
    iter->next = head->next;
    return head->info;
  }
  return 0;
}

InfoRecord * Set::next(Iterator *iter)
{
  //  ckout << "set::next: " << iter->next << "\n";
  if (!iter->next)
    { return 0;
    }
  //  ckout << "set::next: iter->next->info=" << iter->next->info << "\n";
  InfoRecord *temp = iter->next->info;
  iter->next = iter->next->next;
  return temp;
}


int Set::numElements()
{
  int n;
  n = 0;
  listNode *p = head;
  while (p){
    n++;
    p = p->next;
  }
  return n;
}

void Set::print() 
{
  listNode *p = head;
  while (p){
    printf("%d ",p->info->Id);
    p = p->next;
  }
}


/*@}*/

// File: cancel.C
// CancelList stores cancellation of event notices processed by a strat
// Last Modified: 5.29.01 by Terry L. Wilmarth

#include "eventID.h"
#include "pose.h"

// Inserts an event at beginning of list; increments count and sets earliest
// if applicable; sets current if list was previously empty
void CancelList::Insert(int ts, eventID e) 
{
  CancelNode *newnode = new CancelNode(ts, e);
  count++;
  if ((ts < earliest) || (earliest < 0)) // new event has earliest timestamp
    earliest = ts;
  newnode->next = cancellations; // place at front of list
  cancellations = newnode;
  if (!current) current = newnode; // set current if list was empty
}

// Returns a pointer to a node in list; uses current to cycle through nodes
// so as to return a new item each time GetItem is called; only returns items
// with timestamp <= eGVT + MAX_FUTURE_OFFSET
CancelNode *CancelList::GetItem(int eGVT) 
{
  CancelNode *result, *start = current;
  if (!current) CkPrintf("ERROR: CancelList::GetItem: CancelList is empty\n");
  while (current->timestamp > eGVT + MAX_FUTURE_OFFSET) {
    if (current->next) current = current->next;
    else current = cancellations;
    if (current == start)
      return NULL;
  }
  result = current;
  if (current->next) current = current->next;
  else current = cancellations;
  return result;
}

// Remove a node from the list (presumably on that was cancelled)
void CancelList::RemoveItem(CancelNode *item)
{
  int isEarliest = (item->timestamp == earliest);
  CancelNode *tmp = cancellations;
  if (item == tmp) { // item is at front
    cancellations = cancellations->next;
    if (item == current) current = cancellations;
    delete item;
  }
  else { // search for the item
    while (tmp && (tmp->next != item))
      tmp = tmp->next;
    if (!tmp) CkPrintf("ERROR: CancelList::RemoveItem: item not found\n");
    tmp->next = item->next;
    if (item == current) current = tmp->next;
    if (!current) current = cancellations;
    delete item;
  }
  count--;
  if (isEarliest) { // item had earliest timestamp; recalculate earliest
    earliest = -1;
    tmp = cancellations;
    while (tmp) {
      if ((tmp->timestamp < earliest) || (earliest < 0))
	earliest = tmp->timestamp;
      tmp = tmp->next;
    }
  }
}

// Tests if CancelList is empty, returning 1 if it is, 0 otherwise.
int CancelList::IsEmpty() 
{ 
  if ((count == 0) && (cancellations == NULL))
    return 1;
  else if ((count == 0) || (cancellations == NULL)) {
    CkPrintf("ERROR: cancelList::IsEmpty: inconsistency between count=%d and cancellations=%x\n", 
	     count, cancellations);
    CkExit();
  }
  return 0;
}

// Print the list contents
void CancelList::dump(int pdb_level)
{
  int i=count;
  CancelNode *tmp = cancellations;
  
  if (!tmp) CkPrintf("[CANCELS: NULL]\n");
  else {
    CkPrintf("[CANCELS: ");
    while (tmp) {
      i--;
      tmp->dump();
      tmp = tmp->next;
    }
    CkPrintf("]\n");
  }
}

// Pup the entire list contents
void CancelList::pup(PUP::er &p) 
{ 
  int i;
  CancelNode *tmp = NULL;
  p(count); p(earliest);
  if (p.isUnpacking()) {
    i = count;
    if (i == 0) cancellations = NULL;
    while (i > 0) {
      if (i == count) {
	tmp = new CancelNode();
	tmp->pup(p);
	cancellations = tmp;
      }
      else {
	tmp->next = new CancelNode();
	tmp = tmp->next;
	tmp->pup(p);
      }
      i--;
    }
    current = cancellations;
  }
  else {
    i = count; 
    tmp = cancellations;
    while (i > 0) {
      tmp->pup(p);
      tmp = tmp->next;
      i--;
    }
  }
}


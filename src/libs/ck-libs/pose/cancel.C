/// List to store event cancellations
#include "eventID.h"
#include "pose.h"

/// Insert an event at beginning of cancellations list
void CancelList::Insert(POSE_TimeType ts, eventID e) 
{
  CancelNode *newnode = new CancelNode(ts, e);
  count++;
  if (count%1000 == 0) CkPrintf("WARNING: CancelList has %d events!\n", count);
  if ((ts < earliest) || (earliest < 0)) // new event has earliest timestamp
    earliest = ts;
  newnode->next = cancellations; // place at front of list
  cancellations = newnode;
  if (!current) current = newnode; // set current if list was empty
}

/// Return a pointer to a node in list
CancelNode *CancelList::GetItem() 
{
  CancelNode *result;
  if (!current) CkPrintf("ERROR: CancelList::GetItem: CancelList is empty\n");
  result = current;
  if (current->next) current = current->next;
  else current = cancellations;
  return result;
}

/// Remove a specific cancellation from the list
void CancelList::RemoveItem(CancelNode *item)
{
  POSE_TimeType isEarliest = (item->timestamp == earliest);
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

/// Test if cancellations is empty
int CancelList::IsEmpty() 
{ 
  CmiAssert(((count == 0) && (cancellations == NULL)) ||
	    ((count > 0) && (cancellations != NULL)));
  return (count == 0);
}

/// Dump all data fields
void CancelList::dump()
{
  CancelNode *tmp = cancellations;
#if USE_LONG_TIMESTAMPS
  if (!tmp) CkPrintf("[[Earliest=%lld of %d CANCELS: NULL]\n", earliest, count);
#else
  if (!tmp) CkPrintf("[[Earliest=%d of %d CANCELS: NULL]\n", earliest, count);
#endif
  else {
#if USE_LONG_TIMESTAMPS
    CkPrintf("[Earliest=%lld of %d CANCELS: ", earliest, count);
#else
    CkPrintf("[Earliest=%d of %d CANCELS: ", earliest, count);
#endif
    while (tmp) {
      tmp->dump();
      tmp = tmp->next;
    }
    CkPrintf("]\n");
  }
}

/// Pack/unpack/sizing operator
void CancelList::pup(PUP::er &p) 
{ 
  int i;
  CancelNode *tmp = NULL;
  p(count); p(earliest);
  if (p.isUnpacking()) {
    i = count;
    if (i == 0) {
      cancellations = NULL;
    }
    else {
      while (i > 0) {
	if (i == count) {
	  tmp = new CancelNode;
	  tmp->pup(p);
	  cancellations = tmp;
	}
	else {
	  tmp->next = new CancelNode;
	  tmp = tmp->next;
	  tmp->pup(p);
	}
	i--;
      }
      tmp->next = NULL;
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


// File: srtable.C
// SRtable is a table that stores timestamped send/recv events of all
// events and cancellations.

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "charm++.h"  // Needed for Charm++ output
#include "srtable.h"

// Basic constructor
SRtable::SRtable() 
{ 
  for (int i=0; i<GVT_WINDOW; i++)
    sends[i] = recvs[i] = 0;
  residuals = NULL;
  offset = 0;
}

// Destructor: needed to free linked lists
SRtable::~SRtable()
{
  SRentry *next, *current=residuals;
  while (current) {
    next = current->next;
    delete current;
    current = next;
  }
}

// real timestamps are offset from the array indices by gvt
void SRtable::SetOffset(int gvt) { offset = gvt; }

// Makes an SRentry out of parameters and sends to other Insert
void SRtable::Insert(int timestamp, int srSt)
{
  SRentry *entry;

  if (timestamp >= offset+GVT_WINDOW) {
    entry = new SRentry(timestamp, srSt, residuals);
    residuals = entry;
  }
  else {
    if (srSt == SEND) sends[timestamp-offset]++;
    else recvs[timestamp-offset]++;
  }
}

int SRtable::FindDifferenceTimestamp(SRtable *t)
{
  for (int i=0; i<GVT_WINDOW; i++)
    if ((sends[i] != recvs[i]) || (t->sends[i] != sends[i]) 
	|| (t->recvs[i] != recvs[i])) {
      //CkPrintf("Offset=%d Row %d: s=%d r=%d Old: s=%d r=%d\n", offset, i, 
      //       sends[i], recvs[i], t->sends[i], t->recvs[i]);
      return i + offset;
    }
  return GVT_WINDOW + offset;
}
  

// purge tables below timestamp ts
void SRtable::PurgeBelow(int ts)
{
  int start = ts - offset, i;

  if (ts == offset) return;
  else if (ts >= offset + GVT_WINDOW)
    for (i=0; i<GVT_WINDOW; i++) sends[i] = recvs[i] = 0;
  else 
    for (i=start; i<GVT_WINDOW; i++) {
      sends[i-start] = sends[i];
      recvs[i-start] = recvs[i];
      sends[i] = recvs[i] = 0;
    }
  offset = ts;
}

// try to file each residual event in table
void SRtable::FileResiduals()
{
  SRentry *tmp = residuals, *current;

  residuals = NULL;
  while (tmp) {
    current = tmp;
    tmp = tmp->next;
    current->next = NULL;
#if 1
    // memory optimization, avoid unncessary memory allocation.
    if (current->timestamp >= offset+GVT_WINDOW) {
      current->next = residuals;
      residuals = current;
    }
    else {
      if (current->sr == SEND) sends[current->timestamp-offset]++;
      else recvs[current->timestamp-offset]++;
      delete current;
    }
#else
    Insert(current->timestamp, current->sr);
    delete current;
#endif
  }
}

// Clears all data from the table
void SRtable::FreeTable()
{
  for (int i=0; i<GVT_WINDOW; i++)
    sends[i] = recvs[i] = 0;
  offset = 0;
  SRentry *tmp = residuals, *cur;
  while (tmp) {
    cur = tmp;
    tmp = tmp->next;
    delete cur;
  }
}

void SRtable::dump()
{
  CkPrintf("SRtable offset=%d\n", offset);
  for (int i=0; i<GVT_WINDOW; i++)
    CkPrintf("At timestamp %d we have %d sends and %d recvs\n", i+offset, 
	     sends[i], recvs[i]);
}


// File: srtable.C
// SRtable is a table that stores timestamped send/recv events of all
// events and cancellations.

// NOTE TO SELF: run w/ +memory_checkfreq=1
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "charm++.h"  // Needed for Charm++ output
#include "srtable.h"
#include "gvt.h"

// Basic constructor
SRtable::SRtable() 
{ 
  residuals = NULL;
  offset = 0;
  for (int i=0; i<GVT_WINDOW; i++)
    sends[i] = recvs[i] = 0;
}

// Destructor: needed to free linked lists
SRtable::~SRtable()
{ // implement me
  CkPrintf("WARNING: SRtable::~SRtable(): not implemented.\n");
}

// Makes an SRentry out of parameters and sends to other Insert
void SRtable::Insert(int timestamp, int srSt)
{
  CmiAssert(timestamp >= offset);
  //sanitize();
  if (timestamp >= offset + GVT_WINDOW) // insert in residuals
    listInsert(timestamp, srSt);
  else { // insert in table
    if (srSt == SEND)
      sends[timestamp - offset]++;
    else
      recvs[timestamp - offset]++;
  }
  //sanitize();
}

void SRtable::listInsert(int timestamp, int srSt)
{
  //sanitize();
  SRentry *tmp;
  if (!residuals)
    residuals = new SRentry(timestamp, NULL);
  else {
    if (timestamp < residuals->timestamp) {
      residuals = new SRentry(timestamp, residuals);
      if (srSt == SEND) residuals->incSends();
      else residuals->incRecvs();
    }
    else if (timestamp == residuals->timestamp) {
      if (srSt == SEND) residuals->incSends();
      else residuals->incRecvs();
    }
    else {
      tmp = residuals;
      while (tmp->next && (timestamp > tmp->next->timestamp))
	tmp = tmp->next;
      if (!tmp->next) {
	tmp->next = new SRentry(timestamp, NULL);
	if (srSt == SEND) tmp->next->incSends();
	else tmp->next->incRecvs();
      }
      else if (timestamp == tmp->next->timestamp) {
	if (srSt == SEND) tmp->next->incSends();
	else tmp->next->incRecvs();
      }
      else { //timestamp < tmp->next->timestamp
	tmp->next = new SRentry(timestamp, tmp->next);
	if (srSt == SEND) tmp->next->incSends();
	else tmp->next->incRecvs();
      }
    }
  }
  //sanitize();
}

// purge tables below timestamp ts
void SRtable::PurgeBelow(int ts)
{
  //sanitize();
  int start = ts - offset, i;
  if (ts >= offset + GVT_WINDOW) { // purge everything 
    offset = ts;
    for (i=0; i<GVT_WINDOW; i++)
      sends[i] = recvs[i] = 0;
  }
  else { // purge a range and move high entries down
    offset = ts;
    int offIdx;
    for (i=start; i<GVT_WINDOW; i++) {
      offIdx = i-start;
      sends[offIdx] = sends[i];
      sends[i] = 0;
      recvs[offIdx] = recvs[i];
      recvs[i] = 0;
    }
    for (i=(GVT_WINDOW-start); i<start; i++) // purge in-betweens
      sends[i] = recvs[i] = 0;
  }
  //sanitize();
}

// try to file each residual event in table
void SRtable::FileResiduals()
{
  //sanitize();
  SRentry *tmp, *current;
  int end = offset+GVT_WINDOW;

  tmp = residuals;
  while (tmp && (tmp->timestamp < end)) {
    current = tmp;
    tmp = tmp->next;
    sends[current->timestamp - offset] += current->sendCount;
    recvs[current->timestamp - offset] += current->recvCount;
    delete current;
  }
  residuals = tmp;
  //sanitize();
}

UpdateMsg *SRtable::packTable()
{ // packs entries with two earliest timestamps from buckets into an UpdateMsg
  UpdateMsg *um = new UpdateMsg;

  //sanitize();
  um->earlyTS = um->nextTS = -1;
  um->earlySends = um->earlyRecvs = um->nextSends = um->nextRecvs = 0;
  FindEarliest(&(um->earlyTS), &(um->earlySends), &(um->earlyRecvs), 
	       &(um->nextTS), &(um->nextSends), &(um->nextRecvs));
  return um;
}

void SRtable::FindEarliest(int *eTS, int *eS, int *eR, int *nTS, int *nS, int *nR) 
{
  int found1=0, found2=0;
  for (int i=0; i<GVT_WINDOW; i++)
    if ((sends[i] > 0) || (recvs[i] > 0)) {
      found1 = 1;  *eTS = offset+i;  *eS = sends[i];  *eR = recvs[i];
      for (int j=i+1; j<GVT_WINDOW; j++)
	if ((sends[j] > 0) || (recvs[j] > 0)) {
	  found2 = 1;  *nTS = offset+j; *nS = sends[j]; *nR = recvs[j];
	  return;
	}
      return;
    }
  // 0 or 1 earliest timestamps found; look at residuals
  if (found1 && !found2) {
    if (residuals) {
      *nTS = residuals->timestamp;
      *nS = residuals->sendCount;
      *nR = residuals->recvCount;
    }
    return;
  }
  else {
    if (residuals) {
      *eTS = residuals->timestamp;
      *eS = residuals->sendCount;
      *eR = residuals->recvCount;
      if (residuals->next) {
	*nTS = residuals->next->timestamp;
	*nS = residuals->next->sendCount;
	*nR = residuals->next->recvCount;
      }
    }
  }
}

void SRtable::dump()
{
  CkPrintf("WARNING: SRtable::dump(): not implemented.\n");
}

void SRtable::sanitize()
{
  CkPrintf("WARNING: SRtable::sanitize(): not implemented.\n");
}

/// SendRecvTable for POSE GVT calculations
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "charm++.h"  // Needed for Charm++ output
#include "srtable.h"
#include "gvt.h"

/// Basic constructor
SRtable::SRtable() 
{ 
  for (int i=0; i<GVT_WINDOW; i++) sends[i] = recvs[i] = 0;
  offset = 0;
  residuals = NULL;
}

/// Insert send/recv record at timestamp
void SRtable::Insert(int timestamp, int srSt)
{
  //sanitize();
  CmiAssert(timestamp >= offset);
  CmiAssert((srSt == 0) || (srSt == 1));
  if (timestamp >= offset + GVT_WINDOW) // insert in residuals
    listInsert(timestamp, srSt); // call helper
  else { // insert in table
    if (srSt == SEND) sends[timestamp - offset]++;
    else recvs[timestamp - offset]++;
  }
  //sanitize();
}

/// Helper function to Insert
void SRtable::listInsert(int timestamp, int srSt)
{
  //sanitize();
  SRentry *tmp;
  if (!residuals) // no residuals yet
    residuals = new SRentry(timestamp, srSt, NULL);
  else {
    if (timestamp < residuals->timestamp()) // goes before first residual
      residuals = new SRentry(timestamp, srSt, residuals);
    else if (timestamp == residuals->timestamp()) { // goes in first residual
      if (srSt == SEND) residuals->incSends();
      else residuals->incRecvs();
    }
    else { // search for position
      tmp = residuals;
      while (tmp->next() && (timestamp > tmp->next()->timestamp()))
	tmp = tmp->next();
      if (!tmp->next())  // goes at end of residual list
	tmp->setNext(new SRentry(timestamp, srSt, NULL));
      else if (timestamp == tmp->next()->timestamp()) { // goes in tmp->next
	if (srSt == SEND) tmp->next()->incSends();
	else tmp->next()->incRecvs();
      }
      else // goes after tmp but before tmp->next
	tmp->setNext(new SRentry(timestamp, srSt, tmp->next()));
    }
  }
  //sanitize();
}

/// Purge entries from table with timestamp below ts
void SRtable::PurgeBelow(int ts)
{
  //sanitize();
  CmiAssert(ts >= offset);
  int i;
  if (ts >= offset + GVT_WINDOW) { // purge everything 
    offset = ts;
    for (i=0; i<GVT_WINDOW; i++) sends[i] = recvs[i] = 0;
  }
  else { // move high entries down
    int start = ts - offset;
    offset = ts;
    for (i=start; i<GVT_WINDOW; i++) {
      sends[i-start] = sends[i];
      recvs[i-start] = recvs[i];
    }
    for (i=(GVT_WINDOW-start); i<GVT_WINDOW; i++) // purge high entries
      sends[i] = recvs[i] = 0;
  }
  //sanitize();
}

/// Move entries to table from residuals if timestamp < offset+GVT_WINDOW
void SRtable::FileResiduals()
{
  //sanitize();
  SRentry *tmp = residuals, *current;
  int end = offset+GVT_WINDOW;
  while (tmp && (tmp->timestamp() < end)) {
    current = tmp;
    tmp = tmp->next();
    sends[current->timestamp() - offset] += current->sends();
    recvs[current->timestamp() - offset] += current->recvs();
    delete current;
  }
  residuals = tmp;
  //sanitize();
}

/// Copy table to cp
void SRtable::CopyTable(SRtable *cp)
{
  //  sanitize();
  cp->offset = offset;
  for (int i=0; i<GVT_WINDOW; i++) {
    cp->sends[i] = sends[i];
    cp->recvs[i] = recvs[i];
  }
}

/// Free residual entries, reset counters and pointers
void SRtable::FreeTable() {
  //  sanitize();
  SRentry *tmp = residuals;
  while (tmp) { 
    residuals = tmp->next();
    delete(tmp);
    tmp = residuals;
  }
  for (int i=0; i<GVT_WINDOW; i++) sends[i] = recvs[i] = 0;
  offset = 0;
}

/// Dump data fields
void SRtable::dump()
{
  SRentry *tmp = residuals;
  CkPrintf("Offset:%d\nSENDS: ");
  for (int i=0; i<GVT_WINDOW; i++) CkPrintf("%d:%d ", i, sends[i]);
  CkPrintf("\nRECVS: ");
  for (int i=0; i<GVT_WINDOW; i++) CkPrintf("%d:%d ", i, recvs[i]);
  CkPrintf("\nRESIDUALS: ");
  while (tmp) {
    tmp->dump();
    tmp = tmp->next();
  }
}

/// Check validity of data fields
void SRtable::sanitize()
{
  SRentry *tmp = residuals;
  CmiAssert(offset >= 0);
  for (int i=0; i<GVT_WINDOW; i++) {
    CmiAssert(sends[i] >= 0);
    CmiAssert(recvs[i] >= 0);
  }
  while (tmp) {
    tmp->sanitize();
    tmp = tmp->next();
  }
}

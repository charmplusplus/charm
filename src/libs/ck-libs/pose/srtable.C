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
  offset = 0;
  srs = NULL;
}

/// Insert send/recv record at timestamp
void SRtable::Insert(int timestamp, int srSt)
{
  //sanitize();
  CmiAssert(timestamp >= offset);
  CmiAssert((srSt == 0) || (srSt == 1));
  SRentry *tmp;
  if (!srs) // no srs yet
    srs = new SRentry(timestamp, srSt, NULL);
  else {
    if (timestamp < srs->timestamp()) { // goes before first residual
      tmp = new SRentry(timestamp, srSt, srs);
      srs = tmp;
    }
    else if (timestamp == srs->timestamp()) { // goes in first residual
      if (srSt == SEND) srs->incSends();
      else srs->incRecvs();
    }
    else { // search for position
      tmp = srs;
      while (tmp->next() && (timestamp > tmp->next()->timestamp()))
	tmp = tmp->next();
      if (!tmp->next())  // goes at end of residual list
	tmp->setNext(new SRentry(timestamp, srSt, NULL));
      else if (timestamp == tmp->next()->timestamp()) { // goes in tmp->next
	if (srSt == SEND) tmp->next()->incSends();
	else tmp->next()->incRecvs();
      }
      else { // goes after tmp but before tmp->next
	SRentry *newEntry = new SRentry(timestamp, srSt, tmp->next());
	tmp->setNext(newEntry);
      }
    }
  }
  //sanitize();
}

/// Purge entries from table with timestamp below ts
void SRtable::PurgeBelow(int ts)
{
  //sanitize();
  CmiAssert(ts >= offset);
  SRentry *tmp = srs, *cur; 
  while (tmp && (tmp->timestamp() < ts)) {
    cur = tmp;
    tmp = tmp->next(); 
    delete cur;
  }
  srs = tmp;
  //sanitize();
}

/// Free srs entries, reset counters and pointers
void SRtable::FreeTable() {
  //sanitize();
  SRentry *tmp = srs;
  while (tmp) { 
    srs = tmp->next();
    delete(tmp);
    tmp = srs;
  }
  offset = 0;
}

/// Dump data fields
void SRtable::dump()
{
  SRentry *tmp = srs;
  CkPrintf("\nSRtable: ");
  while (tmp) {
    tmp->dump();
    tmp = tmp->next();
  }
}

/// Check validity of data fields
void SRtable::sanitize()
{
  SRentry *tmp = srs;
  CmiAssert(offset > -1);
  while (tmp) {
    tmp->sanitize();
    tmp = tmp->next();
  }
}

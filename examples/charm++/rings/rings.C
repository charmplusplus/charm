///////////////////////////////////////////////
//
//  rings.C
//
//  Definition of chares in rings
//
//  Author: Michael Lang
//  Date: 6/15/99
//
///////////////////////////////////////////////

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#include "charm++.h"
#include "rings.h"
#include "rings.def.h"

CkChareID mainhandle;

#define NUMBEROFRINGS 10

main::main(CkArgMsg *m) {
  
  int i = 0;
  count = NUMBEROFRINGS;
  //  Make several rings of varying hop sizes and repetions

  for (i=0;i<NUMBEROFRINGS;i++) {
    Token *t = new Token;
    t->value = i;
    t->hopSize = (i+1) * (int) pow(-1.0,i);
    t->loops = NUMBEROFRINGS - i + 1;
    CProxy_ring::ckNew(t);
  }

  mainhandle = thishandle;
  delete m;
}

void main::ringDone(NotifyDone *m) {
  count--;
  CkPrintf("Ring %d is finished.\n", m->value);
  if (count == 0)
    CkExit();
}

ring::ring(Token *t) {

  //  Calculate next hop

  nextHop = (CkMyPe() + (t->hopSize)) % CkNumPes();
  
  if (nextHop < 0) nextHop += CkNumPes();  // For negative hops

  //  Start with the token at Processor 0, and pass it around the ring

  if (CkMyPe() == 0) {
    CProxy_ring ring(thisgroup);
    ring[nextHop].passToken(t);
  } else {
    delete t;
  }
}

void ring::passToken(Token *t) {
  
  //  When token returns to processor 0, decrement the count and notify
  //  main if done

  if (CkMyPe() == 0) {
    CkPrintf("Loop completed on ring %d\n", t->value);
    t->loops--;
    if (t->loops == 0) {
      NotifyDone *m = new NotifyDone;
      m->value = t->value;
      CProxy_main mainchare(mainhandle);
      mainchare.ringDone(m);
    } else {
      CProxy_ring ring(thisgroup);
      ring[nextHop].passToken(t);
    }
  } else {
    CProxy_ring ring(thisgroup);
    ring[nextHop].passToken(t);
  }
}


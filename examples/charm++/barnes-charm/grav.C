/*************************************************************************/
/*                                                                       */
/*  Copyright (c) 1994 Stanford University                               */
/*                                                                       */
/*  All rights reserved.                                                 */
/*                                                                       */
/*  Permission is given to use, copy, and modify this software for any   */
/*  non-commercial purpose as long as this copyright notice is not       */
/*  removed.  All other uses, including redistribution in whole or in    */
/*  part, are forbidden without prior written permission.                */
/*                                                                       */
/*  This software is provided with absolutely no warranty and no         */
/*  support.                                                             */
/*                                                                       */
/*************************************************************************/

/*
 * GRAV.C: 
 */


#include "barnes.h"

/*
 * HACKGRAV: evaluate grav field at a given particle.
 */
  
void ParticleChunk::hackgrav(bodyptr p, unsigned ProcessId)
{

   pskip = p;
   SETV(pos0, Pos(p));
   phi0 = 0.0;
   CLRV(acc0);
   myn2bterm = 0;
   mynbcterm = 0;
   skipself = FALSE;
   hackwalk(ProcessId);
   Phi(p) = phi0;
   SETV(Acc(p), acc0);
#ifdef QUADPOLE
   Cost(p) = myn2bterm + NDIM * mynbcterm;
#else
   Cost(p) = myn2bterm + mynbcterm;
#endif
}

/*
 * GRAVSUB: compute a single body-body or body-cell interaction.
 */

void ParticleChunk::gravsub(nodeptr p, unsigned ProcessId)
{
    real drabs, phii, mor3;
    vector ai, quaddr;
    real dr5inv, phiquad, drquaddr;

    if (p != pmem) {
        SUBV(dr, Pos(p), pos0);
        DOTVP(drsq, dr, dr);
    }
    
    drsq += epssq;
    drabs = sqrt((double) drsq);
    phii = Mass(p) / drabs;
    phi0 -= phii;
    mor3 = phii / drsq;
    MULVS(ai, dr, mor3);
    ADDV(acc0, acc0, ai); 
    if(Type(p) != BODY) {                  /* a body-cell/leaf interaction? */
       mynbcterm++;
       //CkPrintf("interaction with cell %ld\n", p->key);
#ifdef QUADPOLE
       dr5inv = 1.0/(drsq * drsq * drabs);
       MULMV(quaddr, Quad(p), dr);
       DOTVP(drquaddr, dr, quaddr);
       phiquad = -0.5 * dr5inv * drquaddr;
       phi0 += phiquad;
       phiquad = 5.0 * phiquad / drsq;
       MULVS(ai, dr, phiquad);
       SUBV(acc0, acc0, ai);
       MULVS(quaddr, quaddr, dr5inv);   
       SUBV(acc0, acc0, quaddr);
#endif
    }
    else {                                      /* a body-body interaction  */
       //CkPrintf("interaction with body %ld\n", ((bodyptr)p)->num);
       myn2bterm++;
    }
}

/*
 * HACKWALK: walk the tree opening cells too close to a given point.
 */

void ParticleChunk::hackwalk(unsigned ProcessId)
{
    walksub((nodeptr)G_root, rsize * rsize, ProcessId);
}

/*
 * WALKSUB: recursive routine to do hackwalk operation.
 */

void ParticleChunk::walksub(nodeptr n, real dsq, unsigned ProcessId)
//   nodeptr n;                        /* pointer into body-tree    */
//   real dsq;                         /* size of box squared       */
{
   nodeptr* nn;
   leafptr l;
   bodyptr p;
   int i;
    
   if (subdivp(n, dsq, ProcessId)) {
      if (Type(n) == CELL) {
	 for (nn = Subp(n); nn < Subp(n) + NSUB; nn++) {
	    if (*nn != NULL) {
	       walksub(*nn, dsq / 4.0, ProcessId);
	    }
	 }
      }
      else {
	 l = (leafptr) n;
	 for (i = 0; i < l->num_bodies; i++) {
	    p = Bodyp(l)[i];
	    if (p != pskip) {
	       gravsub((nodeptr)p, ProcessId);
	    }
	    else {
	       skipself = TRUE;
	    }
	 }
      }
   }
   else {
      gravsub((nodeptr)n, ProcessId);
   }
}

/*
 * SUBDIVP: decide if a node should be opened.
 * Side effects: sets  pmem,dr, and drsq.
 */

bool ParticleChunk::subdivp(nodeptr p, real dsq, unsigned ProcessId)
//   register nodeptr p;                      /* body/cell to be tested    */
//   real dsq;                                /* size of cell squared      */
{
   SUBV(dr, Pos(p), pos0);
   DOTVP(drsq, dr, dr);
   pmem = p;
   return (tolsq * drsq < dsq);
}

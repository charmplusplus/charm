/*
 * mutil.c 
 *
 * This file contains various utility functions for the MOC portion of the
 * code
 *
 * Started 2/15/98
 * George
 *
 * $Id$
 *
 */

#include <metis.h>


/*************************************************************************
* This function checks if the vertex weights of two vertices are below 
* a given set of values
**************************************************************************/
int AreAllVwgtsBelow(int ncon, floattype alpha, floattype *vwgt1, floattype beta, floattype *vwgt2, floattype limit)
{
  int i;

  for (i=0; i<ncon; i++)
    if (alpha*vwgt1[i] + beta*vwgt2[i] > limit)
      return 0;

  return 1;
}


/*************************************************************************
* This function checks if the vertex weights of two vertices are below 
* a given set of values
**************************************************************************/
int AreAnyVwgtsBelow(int ncon, floattype alpha, floattype *vwgt1, floattype beta, floattype *vwgt2, floattype limit)
{
  int i;

  for (i=0; i<ncon; i++)
    if (alpha*vwgt1[i] + beta*vwgt2[i] < limit)
      return 1;

  return 0;
}



/*************************************************************************
* This function checks if the vertex weights of two vertices are above 
* a given set of values
**************************************************************************/
int AreAllVwgtsAbove(int ncon, floattype alpha, floattype *vwgt1, floattype beta, floattype *vwgt2, floattype limit)
{
  int i;

  for (i=0; i<ncon; i++)
    if (alpha*vwgt1[i] + beta*vwgt2[i] < limit)
      return 0;

  return 1;
}


/*************************************************************************
* This function computes the load imbalance over all the constrains
* For now assume that we just want balanced partitionings
**************************************************************************/ 
floattype ComputeLoadImbalance(int ncon, int nparts, floattype *npwgts, floattype *tpwgts)
{
  int i, j;
  floattype max, lb=0.0;

  for (i=0; i<ncon; i++) {
    max = 0.0;
    for (j=0; j<nparts; j++) {
      if (npwgts[j*ncon+i] > max)
        max = npwgts[j*ncon+i];
    }
    if (max*nparts > lb)
      lb = max*nparts;
  }

  return lb;
}

/*************************************************************************
* This function checks if the vertex weights of two vertices are below 
* a given set of values
**************************************************************************/
int AreAllBelow(int ncon, floattype *v1, floattype *v2)
{
  int i;

  for (i=0; i<ncon; i++)
    if (v1[i] > v2[i])
      return 0;

  return 1;
}

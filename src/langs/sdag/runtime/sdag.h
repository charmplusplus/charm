/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#ifndef _sdag_H_
#define _sdag_H_

#include "charm++.h"

#include "CDep.h"
#include "CCounter.h"

// returns count of values from starting and ending value
// considering stride.

static int __getCount(int start, int end, int stride)
{
  return (((end-start)/stride)+1);
}

static void __swap(int *x, int *y)
{
  int tmp = *x;
  *x = *y;
  *y = tmp;
}

#endif

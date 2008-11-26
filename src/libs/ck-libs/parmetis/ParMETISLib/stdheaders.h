/*
 * Copyright 1997, Regents of the University of Minnesota
 *
 * stdheaders.h
 *
 * This file includes all necessary header files
 *
 * Started 8/27/94
 * George
 *
 * $Id$
 */


#include <stdio.h>
#include <stdlib.h>

/* OSX does not use malloc.h, but provides malloc functionality in stdlib.h */
#if CMK_HAS_MALLOC_H
#include <malloc.h>
#endif

#include <string.h>
#include <ctype.h>
#include <math.h>
#include <stdarg.h>
#include <limits.h>
#include <time.h>
#include <ampi.h>


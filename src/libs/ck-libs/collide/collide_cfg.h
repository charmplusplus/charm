/*
Configuration flags for Collision detection.

Orion Sky Lawlor, olawlor@acm.org, 2003/3/19
*/
#ifndef __UIUC_CHARM_COLLIDE_CFG_H
#define __UIUC_CHARM_COLLIDE_CFG_H

/// Statistics collection:
#ifndef COLLIDE_STATS 
#define COLLIDE_STATS 0
#endif

/// Use bitwise trick for fast floating-point to integer conversion.
#ifndef COLLIDE_USE_FLOAT_HACK
#define COLLIDE_USE_FLOAT_HACK 0
#endif

/// Use recursive coordinate bisection within a voxel.
#ifndef COLLIDE_IS_RECURSIVE
#define COLLIDE_IS_RECURSIVE 1
#endif

/// Threshold for using recursive coordinate bisection.
#ifndef COLLIDE_RECURSIVE_THRESH
#define COLLIDE_RECURSIVE_THRESH 10 //More than this many objects -> recurse
#endif



#endif

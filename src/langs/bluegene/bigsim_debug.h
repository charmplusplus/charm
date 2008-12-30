#ifndef _BIGSIM_DEBUG_H_
#define _BIGSIM_DEBUG_H_

#ifdef TURNONDEBUG
    #define DEBUGF(x) CmiPrintf x;
#else
    #define DEBUGF(x) //CmiPrintf x;
#endif

#define MAXDEBUGLEVEL 10

//The higher the value is, the lower level the debug info is
#define DEBUGLEVEL MAXDEBUGLEVEL

#define DEBUGM(level, x) \
    { if (level == DEBUGLEVEL) CmiPrintf x; }
    
#endif

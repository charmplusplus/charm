#ifndef _GLOBALS_H_
#define _GLOBALS_H_

#include <stdio.h>
#include <stdarg.h>

namespace xi {

extern void Indent(int indent);

extern int numSdagEntries;
extern int numSlists;
extern int numOverlaps;
extern int numWhens;
extern int numFors;
extern int numIfs;
extern int numElses;
extern int numEntries;
extern int numOlists;
extern int numWhiles;
extern int numForwards;
extern int numAtomics;
extern int numForalls;
extern int numConnects;
extern int numCases;
extern int numCaseLists;

extern FILE *fh;

extern void pH(int, const char *, ...);
extern void resetNumbers(void);

}

#endif

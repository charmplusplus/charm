
#ifndef COMLIB_H
#define COMLIB_H

#include <converse.h>
#include <stdlib.h>
#include "charm++.h"

#if CMK_BLUEGENE_CHARM
#define CmiReservedHeaderSize   CmiBlueGeneMsgHeaderSizeBytes
#else
#define CmiReservedHeaderSize   CmiExtHeaderSizeBytes
#endif

extern int comm_debug;
#if CMK_OPTIMIZE
inline void ComlibPrintf(...) {}
#else
#define ComlibPrintf if(comm_debug) CmiPrintf
#endif

enum{BCAST=0,TREE, GRID, HCUBE};  

#define USE_TREE 1            //Organizes the all to all as a tree
#define USE_MESH 2            //Virtual topology is a mesh here
#define USE_HYPERCUBE 3       //Virtual topology is a hypercube
#define USE_DIRECT 4          //A dummy strategy that directly forwards 
                              //messages without any processing.
#define USE_GRID 5            //Virtual topology is a 3d grid
#define USE_LINEAR 6          //Virtual topology is a linear array

#define IS_BROADCAST -1
#define IS_SECTION_MULTICAST -2

#define MAXNUMMSGS 1000

#define PERSISTENT_BUFSIZE 65536

typedef struct {
    int refno;
    int instanceID;  
    char isAllToAll;
} comID;

typedef struct {
  int msgsize;
  void *msg;
} msgstruct ;

typedef struct { 
    char core[CmiReservedHeaderSize];
    comID id;
    int magic;
    int refno;
} DummyMsg ;

//The handler to invoke the RecvManyMsg method of router
CkpvExtern(int, RecvHandle);
//The handler to invoke the ProcManyMsg method of router
CkpvExtern(int, ProcHandle);
//The handler to invoke the DoneEP method of router
CkpvExtern(int, DummyHandle);

//Dummy msg handle.
//Just deletes and ignores the message
CkpvExtern(int, RecvdummyHandle);

inline double cubeRoot(double d) {
  return pow(d,1.0/3.0);
}


#include "router.h"

#endif
	


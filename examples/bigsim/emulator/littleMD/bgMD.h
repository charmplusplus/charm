/* File			: bgMD.h 
 * Version		: 1.0; March 24, 2001
 *
 * Description	: Sample application for Blue Gene emulator (converse version)
 *				  Prototype Molecular Dynamics: LittleMD
 * Note			: The program is converse-bluegene version of charm-bluegene version written 
 *				  previously.
 ******************************************************************************/

#ifndef __BG_MD_h__
#define __BG_MD_h__

#include "SimParams.h"
#include "Cell.h"
#include "Atom.h"

/* global constants */
#define DISTANCE			3   /* 3-away neighbors */
#define MAX_ATOMS_PER_CELL 	50 

/* Handler function Identifiers */
extern int sendCoordID;
extern int storeCoordID;
extern int retrieveForcesID;
extern int reduceID;

extern double startTime;

//TODO describe each handler function
/* Handler function prototypes */
extern "C" void sendCoord(char *);
extern "C" void storeCoord(char *);
extern "C" void retrieveForces(char *);
extern "C" void reduce(char *);

/* forward declarations */
class  CellCoord;

//TODO describe each utility function
/* Utilities functions */
extern "C" void   init_cell(Cell* cell, int xi, int yi, int zi, const SimParams* params);
extern "C" double calc_self_interactions(Cell* this_cell, const SimParams* params);
extern "C" double calc_pair_interactions(Cell* cell1, Cell* cell2, const SimParams* params);
extern "C" double calc_force(Atom *atom1 , Atom* atom2, const SimParams* params);
extern "C" double update_cell(Cell* this_cell, const SimParams* params, bool firstStep);
extern "C" void   cellPairToNode(const CellCoord* cellCoord1, const CellCoord* cellCoord2, const SimParams* params, int &x, int &y, int &z);
extern "C" void   cellToNode(const CellCoord* cellCoord, const SimParams* params, int &x, int &y, int &z);


/* forward declarations */
struct CellPairDataStruct;

typedef unsigned int Byte;

/* Node Private Data and internal structures */
class CellCoord
{
 public:
  Byte x,y,z;
  CellCoord(void) {}
  CellCoord(Byte Nx,Byte Ny,Byte Nz)
  {
  	x=Nx; y=Ny; z=Nz;
  }
  CellCoord(const CellCoord& cell)
  {
  	this->x = cell.x; this->y = cell.y; this->z = cell.z;
  }
  /*
  CellCoord& operator=(const CellCoord& cell)
  {
	this->x = cell.x; this->y = cell.y; this->z = cell.z;
  }
  */
  friend CellCoord operator+(const CellCoord &a,const CellCoord &b) 
  {
    return CellCoord(a.x+b.x,a.y+b.y,a.z+b.z);
  }
  bool operator==(const CellCoord &a) 
  {
    return (x==a.x && y==a.y && z==a.z);
  }
};

typedef struct
{
  Cell 	     *myCell;
  CellCoord  myCoord;
  int 	     neighborCount;	//basically a subset of neighbors for optimization
  CellCoord* neighborCoord;
  double     myPotEnergy;
  double     myKinEnergy;
  int        countForceReceived;
  bool	     firstStep;
} CellData;


typedef struct CellPairDataStruct
{
  Cell 			     *cell1, *cell2;
  CellCoord		     cellCoord1;
  CellCoord		     cellCoord2;
  struct CellPairDataStruct* next;
} CellPairData;

typedef struct 
{
  double systemPotEnergy;
  double systemKinEnergy;
  int    numberOfCellsDone;
  int    systemStepsCompleted;	/* Number of steps completed */
  int           nreported;
} LittleMDData;

/* Node Private Data */
typedef struct
{
  LittleMDData* lmdData;
  int		    cellCount;
  CellData*     cellData;
  CellPairData* cellPairData;
  SimParams*    spData;
  int           step;
} UserData;



/* Message Definitions */
class StepMsg
{
public:
  char core[CmiBlueGeneMsgHeaderSizeBytes];
  int step;
  StepMsg(int s) : step(s) {}
  void *operator new(size_t s) { return CmiAlloc(s); }
  void operator delete(void* ptr) { CmiFree(ptr); }
};

class CoordinateMsg	// Message to send the data from Cells to Cellpairs
{
public:
  char core[CmiBlueGeneMsgHeaderSizeBytes];
  CellCoord cellCoord1, cellCoord2;		// Coordinates of Cells
  int       nElem;				// Number of atoms in Cell1
  double    dCordBuff[MAX_ATOMS_PER_CELL*4];	// Coordinates and charge of all the cells in Cell1
  int       step;
  void *operator new(size_t s) { return CmiAlloc(s); }
  void operator delete(void* ptr) { CmiFree(ptr); }
};

class ForceMsg
{
public:
  char core[CmiBlueGeneMsgHeaderSizeBytes];
  CellCoord cellCoord;
  int       nElem;
  double    potEnergy;
  double    dForceBuff[MAX_ATOMS_PER_CELL*3];
  void *operator new(size_t s) { return CmiAlloc(s); }
  void operator delete(void* ptr) { CmiFree(ptr); }
};

class EnergyMsg
{
public:
  char core[CmiBlueGeneMsgHeaderSizeBytes];
  double potEnergy;
  double kinEnergy;
  void *operator new(size_t s) { return CmiAlloc(s); }
  void operator delete(void* ptr) { CmiFree(ptr); }
};

#endif

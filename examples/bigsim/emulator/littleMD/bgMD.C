/* File			: bgMD.C 
 * Author               : Arun Singla, Neelam Saboo, Joshua Unger,
 *                        Gengbin Zheng
 * Version		: 1.0: 3/24/2001
 *                        2.0: 3/27/2001 - make this an actual working version 
 *                             on converse bluegene by Gengbin Zheng
 *
 * Description	: Sample application for Blue Gene emulator (converse version)
 *				  Prototype Molecular Dynamics: LittleMD
 * Note			: The program is converse-bluegene version of 
 *                        charm-bluegene version written previously.
*******************************************************************************/

#include "blue.h"
#include "bgMD.h"

int sendCoordID;
int storeCoordID;
int retrieveForcesID;
int reduceID;

double startTime;

void BgEmulatorInit(int argc, char **argv) 
{
  CmiPrintf("Initializing littleMD\n");

  if (argc < 6) {
    CmiPrintf("Usage: bgMD <x> <y> <z> <numCommTh> <numWorkTh>\n");
    BgShutdown();
  }
  
  /* set machine configuration */
  BgSetSize(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]));
  BgSetNumCommThread(atoi(argv[4]));
  BgSetNumWorkThread(atoi(argv[5]));

}

void BgNodeStart(int argc, char **argv)
{
  /* register handlers */
  sendCoordID      = BgRegisterHandler(sendCoord);
  storeCoordID     = BgRegisterHandler(storeCoord);
  retrieveForcesID = BgRegisterHandler(retrieveForces);
  reduceID         = BgRegisterHandler(reduce);

  int x, y, z;
  BgGetMyXYZ(&x, &y, &z);
  //CmiPrintf("BgNodeStart [%d,%d,%d]\n", x, y, z);

  int sX, sY, sZ;
  BgGetSize(&sX, &sY, &sZ);

  /* Initialize simulation space  */
  SimParams *spData = new SimParams;
  // spData->setParams270K();
  // spData->setParams40K();
   spData->setParamsDebug();

  double pdx = spData->max_x - spData->min_x;
  double pdy = spData->max_y - spData->min_y;
  double pdz = spData->max_z - spData->min_z;

  spData->cell_dim_x = 1 + (int)(pdx / (spData->cutoff + spData->margin));
  spData->cell_dim_y = 1 + (int)(pdy / (spData->cutoff + spData->margin));
  spData->cell_dim_z = 1 + (int)(pdz / (spData->cutoff + spData->margin));
  spData->total_cells = spData->cell_dim_x * spData->cell_dim_y * spData->cell_dim_z;

  // CmiPrintf("BgNodeStart [%d,%d,%d] SimParams intialized %d %d %d\n", x, y, z, spData->cell_dim_x, spData->cell_dim_y, spData->cell_dim_z);

  // Initialize cells per Blue Gene Node
  // Find initial(x1, y1, z1) and final(x2, y2, z2) coordinates of cells mapped
  // 	to each node
  // If the number of nodes exceeds the number of cells in any dimension, 
  // 	make a one-to-one mapping (Some Blue Gene nodes might be ununsed if 
  //    simulation space is too small)
  // Find number of cells mapped (x2-x1+1)*(y2-y1+1)*(z2-z1+1) to each node
  
  int cellsNodeX = spData->cell_dim_x <= sX ? 1 : spData->cell_dim_x / sX;
  int cellsNodeY = spData->cell_dim_y <= sY ? 1 : spData->cell_dim_y / sY;
  int cellsNodeZ = spData->cell_dim_z <= sZ ? 1 : spData->cell_dim_z / sZ;

//CmiPrintf("%d %d %d cellsNodeX:%d cellsNodeY:%d cellsNodeZ:%d\n", spData->cell_dim_x, spData->cell_dim_y, spData->cell_dim_z,cellsNodeX,cellsNodeY,cellsNodeZ);

  int numCellMapped = 0;
  int x1, y1, z1, x2, y2, z2;
  if(x<spData->cell_dim_x && y<spData->cell_dim_y && z<spData->cell_dim_z)
  {
    x1 = cellsNodeX * x;
    y1 = cellsNodeY * y;
    z1 = cellsNodeZ * z;
    
    x2 = x==sX-1 ? (spData->cell_dim_x - 1) : (cellsNodeX*(x+1)-1);
    y2 = y==sY-1 ? (spData->cell_dim_y - 1) : (cellsNodeY*(y+1)-1);
    z2 = z==sZ-1 ? (spData->cell_dim_z - 1) : (cellsNodeZ*(z+1)-1);

    numCellMapped = (x2-x1+1)*(y2-y1+1)*(z2-z1+1);
    if(numCellMapped<0) {
  	CmiPrintf("Error> Number of cells mapped to Node [%d,%d,%d] is less than 0\n", x, y, z);
	CmiAbort("\n");
    }
  }

  // CmiPrintf("Mapping to Node [%d,%d,%d], [%d,%d,%d] to [%d,%d,%d] cells %d\n", x, y, z, x1, y1, z1, x2, y2, z2, numCellMapped);

  CellData *cellData = new CellData[numCellMapped];

  /* Store Neighbour Information for each Blue Gene Node, 
   * If no cells are mapped to this node, don't use it.
   */
  if(numCellMapped>0)
  {
    int cellIndex = 0;
    int neighborCount;
    for(int i=x1; i<=x2; i++)
     for(int j=y1; j<=y2; j++)
      for(int k=z1; k<=z2; k++)
      {
    	neighborCount = 0;

	cellData[cellIndex].myCoord = CellCoord(i,j,k);
  	cellData[cellIndex].firstStep = true;
	
	//Initialize the cell
	cellData[cellIndex].myCell = new Cell;
	init_cell(cellData[cellIndex].myCell, i, j, k, spData);

	//find the neighbors
	//more: scope of optimization
	cellData[cellIndex].neighborCoord = new CellCoord[(2*DISTANCE+1)*(2*DISTANCE+1)*(2*DISTANCE+1)];
	int nx, ny, nz;
     	for(int l = -DISTANCE; l <= DISTANCE; l++)
         for(int m = -DISTANCE; m <= DISTANCE; m++)
          for(int n = -DISTANCE; n <= DISTANCE; n++)
	  {
	   nx = i + l ;
	   ny = j + m ;
	   nz = k + n ;

	   if ((nx < spData->cell_dim_x && nx >= 0) && 
	       (ny < spData->cell_dim_y && ny >= 0) &&
	       (nz < spData->cell_dim_z && nz >= 0) &&
	       !(l==0 && m==0 && n==0)) //<- you aren't your own neighbor
	   {
		CellCoord* cd = &cellData[cellIndex].neighborCoord[neighborCount];
		cd->x = nx;
		cd->y = ny;
		cd->z = nz;
	        neighborCount++; 
	   }
         }

	 cellData[cellIndex].neighborCount = neighborCount;
	 cellData[cellIndex].myPotEnergy = 0.0;
	 cellData[cellIndex].myKinEnergy = 0.0;
	 cellData[cellIndex].countForceReceived = 0;

	 cellIndex++;
    }
  }
//  CmiPrintf("BgNodeStart [%d,%d,%d] cells mapped\n", x, y, z);

  /* declare and set node private data */
  LittleMDData *lmdData = new LittleMDData;
  lmdData->systemPotEnergy      = 0.0;
  lmdData->systemKinEnergy      = 0.0;
  lmdData->numberOfCellsDone    = 0;
  lmdData->systemStepsCompleted = 0;
  lmdData->nreported = 0;

  UserData *ud  = new UserData;
  ud->lmdData   = lmdData;
  ud->cellData  = cellData;
  ud->cellCount = numCellMapped;
  ud->cellPairData = NULL;
  ud->spData    = spData;
  ud->step      = 0;

  BgSetNodeData((char*)ud);

  //CmiPrintf("BgNodeStart [%d,%d,%d] creating micro-task\n", x, y, z);
  /* trigger computation at each node */
  StepMsg *msg = new StepMsg(0);
  BgSendLocalPacket(ANYTHREAD, sendCoordID, LARGE_WORK, sizeof(StepMsg), (char*)msg);

//  CmiPrintf("BgNodeStart Done [%d,%d,%d]\n", x, y, z);
  if (x==0 && y==0 && z==0) startTime = BgGetTime();
}

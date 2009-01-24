/* File			: Handlers.C 
 *
 * Description	: Handlers and Utility functions
 * Note			: The program is converse-bluegene version of charm-bluegene version written 
 *				  previously.
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "blue.h"
#include "bgMD.h"

#ifdef _MSC_VER
#define erand48(x) CrnDrand()
#endif

/*******************************************************************************
		UTILITY FUNCTIONS
 *****************************************************************************/

void PrintStep(int steps, double pe, double ke)
{
  CmiPrintf("Step %d PE: %.4e  KE: %.4e  Total: %.4e\n", steps,pe,ke,pe+ke);
}

void cellPairToNode(const CellCoord* cellCoord1, const CellCoord* cellCoord2, const SimParams* params, int &x, int &y, int &z)
{
  const CellCoord* cellCoord = cellCoord2;
  
  if(cellCoord1->x<cellCoord2->x)
    cellCoord =  cellCoord1;
  else if(cellCoord1->x==cellCoord2->x)
  {
    if(cellCoord1->y<cellCoord2->y)
      cellCoord =  cellCoord1;
    else if(cellCoord1->y==cellCoord2->y)
    {
      if(cellCoord1->z<cellCoord2->z)
 	cellCoord =  cellCoord1;
      else if(cellCoord1->z==cellCoord2->z)
 	CmiAbort("Error: Two cells have same coordinates\n");
    }
  }

  /* find the cellPair node; cellCoord to node conversion */
  //more: We can store where the pairs are stored.
  //more: Write why we are not storing
  cellToNode(cellCoord, params, x, y, z);
}

void cellToNode(const CellCoord* cellCoord, const SimParams* params, int &x, int &y, int &z)
{
  int sX, sY, sZ; BgGetSize(&sX, &sY, &sZ);

  int cellsNodeX = params->cell_dim_x <= sX ? 1 : params->cell_dim_x / sX;
  int cellsNodeY = params->cell_dim_y <= sY ? 1 : params->cell_dim_y / sY;
  int cellsNodeZ = params->cell_dim_z <= sZ ? 1 : params->cell_dim_z / sZ;

  x = cellCoord->x / cellsNodeX;
  if (x>= sX) x = sX-1;
  y = cellCoord->y / cellsNodeY;
  if (y>= sY) y = sY-1;
  z = cellCoord->z / cellsNodeZ;
  if (z>= sZ) z = sZ-1;
}

/* Utility Function: init_cell
 * Initializes atoms in a cell
 */
void init_cell(Cell* this_cell, int xi, int yi, int zi, const SimParams* params)
{
  // Set some cell values
  this_cell->x = xi;
  this_cell->y = yi;
  this_cell->z = zi;
    
  const double pdx = params->max_x - params->min_x;
  const double pdy = params->max_y - params->min_y;
  const double pdz = params->max_z - params->min_z;

  this_cell->min_x = params->min_x + (pdx * xi) / params->cell_dim_x;
  this_cell->max_x = params->min_x + (pdx * (xi+1)) / params->cell_dim_x;
  this_cell->min_y = params->min_y + (pdy * yi) / params->cell_dim_y;
  this_cell->max_y = params->min_y + (pdy * (yi+1)) / params->cell_dim_y;
  this_cell->min_z = params->min_z + (pdz * zi) / params->cell_dim_z;
  this_cell->max_z = params->min_z + (pdz * (zi+1)) / params->cell_dim_z;

  // Build up the list of atoms, using a uniform random distribution
  // Since we are doing each cell separately, we don't know exactly
  // how many atoms to put in it.  Therefore, we will allocate a
  // random number such that the total number of atoms on average
  // would come out right.  We'll do this by pretending to assign
  // all the atoms, but only actually assigning a 1/# of cells fraction
  // of them.
  double* x = new double[params->n_atoms];
  double* y = new double[params->n_atoms];
  double* z = new double[params->n_atoms];
  int atom_count = 0;

  const double prob = 1.0 / (params->cell_dim_x * params->cell_dim_y 
			    * params->cell_dim_z);
  const double cdx = this_cell->max_x - this_cell->min_x;
  const double cdy = this_cell->max_y - this_cell->min_y;
  const double cdz = this_cell->max_z - this_cell->min_z;

  // Give some seed that is unique to this cell
  unsigned short int seed[3];
  seed[0] = (unsigned short)xi;
  seed[1] = (unsigned short)yi;
  seed[2] = (unsigned short)zi;

  int i;
  for(i=0; i < params->n_atoms; i++) {
    if (erand48(seed) < prob) {
      x[atom_count] = this_cell->min_x + (cdx * erand48(seed));
      y[atom_count] = this_cell->min_y + (cdy * erand48(seed));
      z[atom_count] = this_cell->min_z + (cdz * erand48(seed));
      atom_count++;
    }
  }
  if(atom_count > 40) atom_count = 40 ; 

  // Allocate the atom array for the cell
  this_cell->atoms = new Atom[atom_count];
  this_cell->n_atoms = atom_count;

  // Store the positions into the cells.  Also randomly determine
  // a mass and charge, and zero out the velocity and force
  for(i=0;i<atom_count;i++) {
    Atom* this_atom = &(this_cell->atoms[i]);
    
    this_atom->m = 10.0 * erand48(seed);
    this_atom->q = 5.0 * erand48(seed) - 2.5;
    this_atom->x = x[i];
    this_atom->y = y[i];
    this_atom->z = z[i];
    this_atom->vx = this_atom->vy = this_atom->vz = 0;
    this_atom->vhx = this_atom->vhy = this_atom->vhz = 0;
    this_atom->fx = this_atom->fy = this_atom->fz = 0;
//CmiPrintf("m:%f, q:%f, x:%f y:%f z:%f\n", this_atom->m, this_atom->q, this_atom->x, this_atom->y, this_atom->z);
  }

  delete [] x;
  delete [] y;
  delete [] z;
}

/* Utility Function: cell_neighbor
 * Calculate forces between all atoms of cell1 and cell2
 * and return potential energy
 */
double calc_pair_interactions(Cell* cell1, Cell* cell2, const SimParams* params)
{
  double potentialEnergy = 0.0;
  int i,j;
  for(i=0;i<cell1->n_atoms;i++)
    for(j=0;j<cell2->n_atoms;j++) {
      potentialEnergy += calc_force(&(cell1->atoms[i]), &(cell2->atoms[j]), params);
    }
  return potentialEnergy;
}

/* Utility Function: calc_force
 * calculate force between two atom1 and atom2
 */
double calc_force(Atom *atom1 , Atom* atom2, const SimParams* params)
{
  double potentialEnergy = 0.0;
  // Convert values to SI units
  const double x1 = atom1->x * 1E-10;
  const double y1 = atom1->y * 1E-10;
  const double z1 = atom1->z * 1E-10;
  const double q1 = atom1->q * 1.602E-19;

  const double x2 = atom2->x * 1E-10;
  const double y2 = atom2->y * 1E-10;
  const double z2 = atom2->z * 1E-10;
  const double q2 = atom2->q * 1.602E-19;

  const double dx = x2-x1;
  const double dy = y2-y1;
  const double dz = z2-z1;
  const double r2 = dx*dx + dy*dy + dz * dz;

  const double csi = params->cutoff * 1E-10;
  const double cutoff2 = csi * csi;

  if (r2 > cutoff2)  // Outside cutoff, ignore
    return 0.0;

  const double r = sqrt(r2);
  
  const double kq1q2 = 9e9*q1*q2;
  const double f = kq1q2 / (r2*r);
  const double fx = f*dx;
  const double fy = f*dy;
  const double fz = f*dz;

  atom1->fx += fx;
  atom2->fx -= fx;
  atom1->fy += fy;
  atom2->fy -= fy;
  atom1->fz += fz;
  atom2->fz -= fz;

  potentialEnergy -= kq1q2 / r;
  return potentialEnergy;
}

/*Utility Function: calc_self_interactions
 */
double  calc_self_interactions(Cell* this_cell, const SimParams* params)
{
  double potentialEnergy = 0.0;
  int i,j;
  for(i=0;i<this_cell->n_atoms;i++)
    for(j=i+1;j<this_cell->n_atoms;j++) {
      potentialEnergy += calc_force(&(this_cell->atoms[i]),
				    &(this_cell->atoms[j]), params);
    }
  return potentialEnergy;
}

/* Utility Function: update_cell
 */
double update_cell(Cell* this_cell, const SimParams* params, bool firstStep)
{
  double kineticEnergy = 0.0;
  // This is a Verlet integrator, which uses the half-step velocity
  // to calculate new positions.  This is a time-reversible integration
  // scheme that has some nice numerical properties
  // The equations are:
  // v(t) = v(t-1/2) + dt/2 * a(t)
  // v(t+1/2) = v(t) + dt/2 * a(t)
  // x(t+1) = x(t) + dt * v(t+1/2)
  // Notice that at the end of this function, you end up with
  // v(t) and x(t+1), so to get x and v for the same point in time,
  // you must save x before it gets updated.

  const double dt = params->step_sz * 1E-15;
  const double dt2 = dt/2;

  int i;
  for(i=0;i<this_cell->n_atoms; i++) {
    Atom* this_atom = &(this_cell->atoms[i]);
    const double mass = this_atom->m * 1.66e-27;
    const double dt2m = dt2/mass;

    const double tax = dt2m * this_atom->fx;
    const double tay = dt2m * this_atom->fy;
    const double taz = dt2m * this_atom->fz;

    //    convert things to SI
    double vx = this_atom->vx * 1E-10/1E-15;
    double vy = this_atom->vy * 1E-10/1E-15;
    double vz = this_atom->vz * 1E-10/1E-15;
    double vhx = this_atom->vhx * 1E-10/1E-15;
    double vhy = this_atom->vhy * 1E-10/1E-15;
    double vhz = this_atom->vhz * 1E-10/1E-15;

    // For the first step, we are given an initial velocity.
    // After that, we must calculate it.
    if (!firstStep) {
      vx = vhx + tax;
      vy = vhy + tay;
      vz = vhz + taz;
    }
    //    cout << "F = " << this_atom->fx << ", " << this_atom->fy << endl;
    //    cout << "A = " << tax << ", " << tay << endl;
    //    cout << "V = " << vx << ", " << vy << endl;
    
    kineticEnergy += 0.5 * mass * (vx*vx+vy*vy+vz*vz);

    vhx = vx + tax;
    vhy = vy + tay;
    vhz = vz + taz;
    
    this_atom->x += dt * vhx / 1E-10; 
    this_atom->y += dt * vhy / 1E-10;
    this_atom->z += dt * vhz / 1E-10;

    // Convert other values back from SI and store
    this_atom->vx = vx * 1E-15 / 1E-10;
    this_atom->vy = vy * 1E-15 / 1E-10;
    this_atom->vz = vz * 1E-15 / 1E-10;
    this_atom->vhx = vhx * 1E-15 / 1E-10;
    this_atom->vhy = vhy * 1E-15 / 1E-10;
    this_atom->vhz = vhz * 1E-15 / 1E-10;
  }

  for(i=0;i<this_cell->n_atoms; i++)
  {
    	this_cell->atoms[i].fx = 0;
      	this_cell->atoms[i].fy = 0;
      	this_cell->atoms[i].fz = 0;
  }

  return kineticEnergy;
}

/*******************************************************************************
		HANDLER FUNCTIONS
 ******************************************************************************/

/* Handler: sendCoord
 * Sends Coordinate data of each cell on the node to the node which has 
 * corresponding cell pair
 */
void sendCoord(char* info)
{
  int x, y, z;
  BgGetMyXYZ(&x, &y, &z);
  // CmiPrintf("sendCoord in node [%d,%d,%d]\n", x, y, z);

  UserData* ud = (UserData*)BgGetNodeData();

  StepMsg* msg = (StepMsg*)info;
  if (msg->step < ud->step) {CmiAbort("error!\n");}
  if (msg->step > ud->step) {	//TODO is this a bug?: shouldn't it be msg->step == ud->step+1
    ud->step = msg->step;
    /* clear up remains of previous step, if there are any */
    /*
    CellPairData* clear = NULL;
    while(ud->cellPairData) {
      clear = ud->cellPairData;
      ud->cellPairData = clear->next;
      delete [] clear->cell1->atoms;
      delete clear->cell1;
      delete [] clear->cell2->atoms;
      delete clear->cell2;
      delete clear;
    }
    */
  }

  /* For each cell and it's each neighbour on this node, create a CoordinateMsg
   * containing it's coordinates, it's neighbour's coordinates, and coordinate &
   * charge info of all it's atoms.
   */
  // msg only new once and can be re-used !!!
  for(int i=0; i<ud->cellCount; i++)
  {
    Cell *MyCell =  ud->cellData[i].myCell ;

    ud->cellData[i].myPotEnergy = 0.0 ;
    ud->cellData[i].myKinEnergy = 0.0 ;
    ud->cellData[i].countForceReceived = 0 ;

    for(int num = 0; num < ud->cellData[i].neighborCount; num++)
    {
      //TODO:scope of optimization: store it and copy later.
      CoordinateMsg* cmsg = new CoordinateMsg ;
      cmsg->step = ud->step;
      cmsg->cellCoord1 = ud->cellData[i].myCoord ;
      cmsg->cellCoord2 = ud->cellData[i].neighborCoord[num] ;
      //CmiPrintf("  Analysing cellpair %d %d %d - %d %d %d\n",  cmsg->cellCoord1.x , cmsg->cellCoord1.y , cmsg->cellCoord1.z ,  cmsg->cellCoord2.x , cmsg->cellCoord2.y ,  cmsg->cellCoord2.z);

      cmsg->nElem= MyCell->n_atoms ;
      for(int l = 0,m=0; l < MyCell->n_atoms; l++)
      {
	    cmsg->dCordBuff[m++] = MyCell->atoms[l].x ;
	    cmsg->dCordBuff[m++] = MyCell->atoms[l].y ;
	    cmsg->dCordBuff[m++] = MyCell->atoms[l].z ;
	    cmsg->dCordBuff[m++] = MyCell->atoms[l].q ;	//charge of atom
      }

      /* The cellPair is mapped to either node of cell1 or cell2 depending 
       * on which is closer to 0,0,0 */
      int a=0, b=0, c=0;
      cellPairToNode(&cmsg->cellCoord1, &cmsg->cellCoord2, ud->spData, a, b, c);
      //TODO:scope of optimization: if same node then use addMessage
      BgSendPacket(a, b, c, ANYTHREAD, storeCoordID, SMALL_WORK, sizeof(CoordinateMsg), (char*)cmsg);
    }
  }
  delete msg;
}

/* Handler: storeCoord
 * The cellPair, on receiving CoordinateMsg, stores the cell data to node private data.
 * When both cells of cellPair have arrived, the forces and energy is computed and sent 
 * back to both the cells.
 */
void storeCoord(char* info)
{
  int i,j;
  int x, y, z;
  BgGetMyXYZ(&x, &y, &z);
  // CmiPrintf("storeCoord in node [%d,%d,%d]\n", x, y, z);

  UserData*  ud = (UserData*)BgGetNodeData();
  CoordinateMsg* msg = (CoordinateMsg*)info;
  CellCoord cellCoord1= msg->cellCoord1;
  CellCoord cellCoord2= msg->cellCoord2;

  // search if this pair is already arrived one leg
  CellPairData* temp = ud->cellPairData;
  CellPairData* prev;
  bool found = false;
  while(temp != NULL && !found)
  {
  	prev = temp;
	if(temp->cellCoord1 == msg->cellCoord2 && temp->cellCoord2 == msg->cellCoord1) 	 
	    found = true;
	else
	    temp = temp->next;
  }

  CellPairData* cellPairData = NULL;
  Cell* cell1 = NULL;
  Cell *cell2 = NULL;
  if (!found) {
	  /* Extract Cell from the message */
	  cellPairData = new CellPairData ;
	  cellPairData->cellCoord1 = msg->cellCoord1 ;
	  cellPairData->cellCoord2 = msg->cellCoord2 ;
	  cell1    = new Cell ;
	  cell1->n_atoms = msg->nElem ;
	  cell1->atoms   = new Atom[cell1->n_atoms] ;
	  for(i =0, j=0; i < cell1->n_atoms; i++)
	  {
		cell1->atoms[i].x = msg->dCordBuff[j++] ;
		cell1->atoms[i].y = msg->dCordBuff[j++] ;
		cell1->atoms[i].z = msg->dCordBuff[j++] ;
		cell1->atoms[i].q = msg->dCordBuff[j++] ;
		cell1->atoms[i].fx = 0 ; 
		cell1->atoms[i].fy = 0 ; 
		cell1->atoms[i].fz = 0 ; 
	  }
  	  cellPairData->cell1 = cell1 ;
  	  cellPairData->cell2= NULL ;
  	  cellPairData->next = ud->cellPairData ;  // cellPairData will be non null since not found1
  	  ud->cellPairData   = cellPairData ;
	  return;
  }
  else 
    cell2 = prev->cell1;

  if (cell2 == NULL) CmiAbort("god!\n");
  if (prev->cell2!= NULL) {
    CmiPrintf("cellCoord1(step:%d): %d %d %d - %d %d %d\n", msg->step, cellCoord1.x, cellCoord1.y, cellCoord1.z, cellCoord2.x, cellCoord2.y, cellCoord2.z);
    CmiAbort("unexpected error\n");
  }

  {
	  /* Extract Cell from the message */
	  cell1    = new Cell ;
	  cell1->n_atoms = msg->nElem ;
	  cell1->atoms   = new Atom[cell1->n_atoms] ;
	  for(i =0, j=0; i < cell1->n_atoms; i++)
	  {
		cell1->atoms[i].x = msg->dCordBuff[j++] ;
		cell1->atoms[i].y = msg->dCordBuff[j++] ;
		cell1->atoms[i].z = msg->dCordBuff[j++] ;
		cell1->atoms[i].q = msg->dCordBuff[j++] ;
		cell1->atoms[i].fx = 0 ; 
		cell1->atoms[i].fy = 0 ; 
		cell1->atoms[i].fz = 0 ; 
	  }
    //CmiPrintf("set cellCoord1(step:%d): %d %d %d - %d %d %d\n", msg->step, cellCoord1.x, cellCoord1.y, cellCoord1.z, cellCoord2.x, cellCoord2.y, cellCoord2.z);
	  prev->cell2 = cell1;
  }

  
  {
   	double potEnergy = calc_pair_interactions(cell1, cell2, ud->spData) ;

   	ForceMsg* forcemsg ;
	int a, b, c ;
   
   	forcemsg = new ForceMsg ;
	forcemsg->cellCoord = cellCoord1 ;
    	forcemsg->nElem=cell1->n_atoms ;
    	forcemsg->potEnergy = potEnergy ;	//<- only count potential once
    	for(i =0, j=0; i < cell1->n_atoms ; i++)
    	{
      	  forcemsg->dForceBuff[j++] =  cell1->atoms[i].fx ;
      	  forcemsg->dForceBuff[j++] =  cell1->atoms[i].fy ;
      	  forcemsg->dForceBuff[j++] =  cell1->atoms[i].fz ;
    	}
  	cellToNode(&forcemsg->cellCoord, ud->spData, a, b, c);
     	BgSendPacket(a, b, c, ANYTHREAD, retrieveForcesID, SMALL_WORK, sizeof(ForceMsg), (char*)forcemsg);
    
   	forcemsg = new ForceMsg ;
	forcemsg->cellCoord = cellCoord2 ;
    	forcemsg->nElem=cell2->n_atoms ;
    	forcemsg->potEnergy = 0.0 ;		//<- only count potential once
    	for(i =0, j=0; i < cell2->n_atoms ; i++)
    	{
      	  forcemsg->dForceBuff[j++] =  cell2->atoms[i].fx ;
      	  forcemsg->dForceBuff[j++] =  cell2->atoms[i].fy ;
      	  forcemsg->dForceBuff[j++] =  cell2->atoms[i].fz ;
    	}
  	cellToNode(&forcemsg->cellCoord, ud->spData, a, b, c);
     	BgSendPacket(a, b, c, ANYTHREAD, retrieveForcesID, SMALL_WORK, sizeof(ForceMsg), (char*)forcemsg);

  }
  delete msg;
}

/* Handler: retrieveForces
 * Extracts forces and potential energy from message sent from cellPair and
 * update atom coordinates in each cell
 */
void retrieveForces(char *info)
{
  int x, y, z;
  BgGetMyXYZ(&x, &y, &z);
  // CmiPrintf("retrieveForces in node [%d,%d,%d]\n", x, y, z);

  UserData*  ud = (UserData*)BgGetNodeData();
  ForceMsg*  forcemsg = (ForceMsg*)info;

  /* search for which cell on this node has the forces arrived */
  CellData* cellDataArray = ud->cellData ;
  CellCoord* cellCoord = &forcemsg->cellCoord; 

  int cellIndex = 0;
  bool Found = false;
  while(cellIndex < ud->cellCount && !(Found))
  {
	if(cellDataArray[cellIndex].myCoord == *cellCoord)
	   Found = true;
	else
	   cellIndex++;
  }
  
  if(!Found)
  	CmiAbort("Error> Cell not found in NodePrivateData\n");

  CellData* cellData = &cellDataArray[cellIndex];

  cellData->countForceReceived++ ;

  /* update data of the cell */
  cellData->myPotEnergy += forcemsg->potEnergy ;
  Cell *cell = cellData->myCell ;

  for(int j=0, i=0; i<cell->n_atoms; i++)
  {
    	cell->atoms[i].fx += forcemsg->dForceBuff[j++] ; 
    	cell->atoms[i].fy += forcemsg->dForceBuff[j++] ; 
    	cell->atoms[i].fz += forcemsg->dForceBuff[j++] ;	
  }

  /* check if all pair computations are done, update coordinates of atoms */
  if(cellData->countForceReceived==cellData->neighborCount)
  {
  	/* calculate self interactions */
	cellData->myPotEnergy += calc_self_interactions(cellData->myCell, ud->spData);
	cellData->myKinEnergy = update_cell(cell, ud->spData, cellData->firstStep);
	cellData->firstStep = false;

	/* start reduction */
	EnergyMsg* energyMsg = new EnergyMsg;
	energyMsg->potEnergy = cellData->myPotEnergy;
	energyMsg->kinEnergy = cellData->myKinEnergy;

	ud->lmdData->nreported++;
	if (ud->lmdData->nreported == ud->cellCount) {
          CellPairData* clear = NULL;
          while(ud->cellPairData) {
            clear = ud->cellPairData;
            ud->cellPairData = clear->next;
            delete [] clear->cell1->atoms;
            delete clear->cell1;
            delete [] clear->cell2->atoms;
            delete clear->cell2;
            delete clear;
          }
          ud->lmdData->nreported=0;
	  /*
          int x,y,z; BgGetMyXYZ(&x, &y, &z);
          CmiPrintf("report %d %d %d\n", x, y, z);
          */
	}

//    CmiPrintf("report %d %d %d\n", cellCoord->x, cellCoord->y, cellCoord->z);
	BgSendPacket(0, 0, 0, ANYTHREAD, reduceID, LARGE_WORK, sizeof(EnergyMsg), (char*)energyMsg);

  }
  delete forcemsg;
}

/* Handler: reduce
 */
void reduce(char *info)
{
  UserData*  ud = (UserData*)BgGetNodeData();
  EnergyMsg*  energyMsg = (EnergyMsg*)info;

  ud->lmdData->systemKinEnergy += energyMsg->kinEnergy ;
  ud->lmdData->systemPotEnergy += energyMsg->potEnergy ;
  ud->lmdData->numberOfCellsDone++ ;

// CmiPrintf("done: %d %d\n", ud->lmdData->numberOfCellsDone, ud->spData->total_cells);
  if(ud->lmdData->numberOfCellsDone == ud->spData->total_cells)
  {
    ud->lmdData->numberOfCellsDone = 0;
    ud->lmdData->systemStepsCompleted++ ;
    PrintStep(ud->lmdData->systemStepsCompleted, 
              ud->lmdData->systemPotEnergy,
              ud->lmdData->systemKinEnergy);
    ud->lmdData->systemPotEnergy = ud->lmdData->systemKinEnergy = 0.0;

    if (ud->lmdData->systemStepsCompleted >= ud->spData->steps)  {
      CmiPrintf("TIMING: %fs/step\n", (BgGetTime()-startTime)/ud->spData->steps);
      BgShutdown();
    }

    /* start next step */
    int sX, sY, sZ; BgGetSize(&sX, &sY, &sZ);
    // new StepMsg once and reuse it
    for (int i=0; i<sX; i++)
      for (int j=0; j<sY; j++)
        for (int k=0; k<sZ; k++) {
          StepMsg *msg = new StepMsg(ud->lmdData->systemStepsCompleted);
          BgSendPacket(i, j, k, ANYTHREAD, sendCoordID, LARGE_WORK, sizeof(StepMsg), (char*)msg);
        }
  }
  delete energyMsg;
}


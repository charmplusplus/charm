#include "cfdAMR.h"
#include "cfdAMR.decl.h"

/*
********************************************
User Code for 2D
********************************************
*/
void AmrUserDataCfdProblemInit(void)
{
  PUPable_reg(AmrUserData);
  PUPable_reg(CfdProblem);
}

void CfdProblem::pup(PUP::er &p)
{
  AmrUserData::pup(p);

  p|gridH;
  p|gridW;
  p|stepCount;
  if(p.isUnpacking()) {
     grid = new CfdGrid();
  }
  grid->pup(p);
  
}


AmrUserData* AmrUserData :: createData()
{
  CfdProblem *instance = new CfdProblem;
  return (AmrUserData *)instance;
}
  
AmrUserData* AmrUserData :: createData(void *data, int dataSize)
{
   CfdProblem *instance = new CfdProblem(data, dataSize);
   return (AmrUserData *) instance;
}


void AmrUserData :: deleteNborData(void* data)
{
  delete [](CfdLoc *) data;
}

void AmrUserData :: deleteChildData(void* data)
{
  delete [](CfdLoc *) data;
}

void CfdProblem :: doComputation(void)
{
  startStep();
  stepCount++;
  finishStep(false);
}

void ** CfdProblem ::getNborMsgArray(int* sizePtr)
{
  CfdLoc ** dataArray = new CfdLoc*[4];
  int x,y;
  //allocate space for the +x and -x nbor data
  for(int i=0; i<2; i++) {
    dataArray[i] = new CfdLoc[gridH];
    sizePtr[i] = gridH * sizeof(CfdLoc);
  }
  //allocate space for the +y and -y nbor data
  for(int i=2; i<4; i++) {
    dataArray[i] = new CfdLoc[gridW];
    sizePtr[i] = gridW * sizeof(CfdLoc);
  }

  CfdLoc** mygrid = grid->getPrimary();
  //For +ve X nbor & For -ve X nbor
  //  x = gridW; x = 1;
  for(int y=1; y< gridH+GHOSTWIDTH-1; y++) {
    dataArray[0][y-1].copy(&mygrid[gridW][y]);
    dataArray[1][y-1].copy(&mygrid[1][y]);
  }
  //For +ve Y Nbor & For -ve Y Nbor
  //  y = gridH;  y = 1;
  for(int x =1; x<gridW+GHOSTWIDTH-1;x++) {
    dataArray[2][x-1].copy(&mygrid[x][gridH]);
    dataArray[3][x-1].copy(&mygrid[x][1]);
  }
  
  return (void **) dataArray;
}


void CfdProblem :: store(void* data, int dataSize, int neighborSide)
{
  CfdLoc* dataArray = (CfdLoc*) data;
  int x,y;
  switch(neighborSide) {
  case 0:
    if(dataSize/sizeof(CfdLoc) == gridH) {
      CfdLoc** mygrid = grid->getPrimary();
      x=0;
      for(int y=1;y<gridH+GHOSTWIDTH-1;y++)
	mygrid[x][y].copy(&(dataArray[y-1]));
    }
    break;
  case 1:
    if(dataSize/sizeof(CfdLoc) == gridH) {
      CfdLoc** mygrid = grid->getPrimary();
      x=gridW+1;
      for(int y=1;y<gridH+GHOSTWIDTH-1;y++)
	mygrid[x][y].copy(&(dataArray[y-1]));
    }
    break;
  case 2:
    if(dataSize/sizeof(CfdLoc) == gridW) {
      CfdLoc** mygrid = grid->getPrimary();
      y=0;
      for(int x=1;x<gridW+GHOSTWIDTH-1;x++)
	mygrid[x][y].copy(&(dataArray[x-1]));
    }
    break;
  case 3:
    if(dataSize/sizeof(CfdLoc) == gridW) {
      CfdLoc** mygrid = grid->getPrimary();
      y=gridH+1;
      for(int x=1;x<gridW+GHOSTWIDTH-1;x++)
	mygrid[x][y].copy(&(dataArray[x-1]));
    }
    break;
  }
}

bool CfdProblem :: refineCriterion() 
{
  if(grid->averageP() > 10000.0)
    return true;
  else
    return false;
}

void **CfdProblem :: fragmentNborData(void *data, int* sizePtr)
{
  int elements = (*sizePtr)/sizeof(CfdLoc);
  int newElements = elements/2;
  CfdLoc **fragmentedArray = new CfdLoc* [2];
  CfdLoc *indata = (CfdLoc *)data;
  if(elements %2 == 0){
    *sizePtr = newElements * sizeof(CfdLoc);
    for(int i=0; i<2; i++) {
      fragmentedArray[i] = new CfdLoc[newElements];
      for(int j=0; j<newElements;j++)
	fragmentedArray[i][j] = indata[i*newElements + j];
    }
  }
  else {
    *sizePtr =( ++newElements)*sizeof(CfdLoc);
    for(int i=0; i<2; i++) {
      fragmentedArray[i] = new CfdLoc[newElements];
      for(int j=0; j<newElements-1;j++)
	fragmentedArray[i][j] = indata[i*newElements + j];
    }
    fragmentedArray[1][newElements-1] = indata[elements -1];
    fragmentedArray[0][newElements-1] = (fragmentedArray[0][newElements -2].average(fragmentedArray[1][0]));
  }
  return (void **)fragmentedArray;

}

void CfdProblem :: combineAndStore(void **dataArray, int dataSize,int neighborSide) 
{
  int size = dataSize /sizeof(CfdLoc);
  CfdLoc * buf = new CfdLoc[size];
  CfdLoc ** data = (CfdLoc**)dataArray;
  // memcpy((void *)buf, dataArray[0], dataSize);
  //memcpy((void *)tmpbuf, dataArray[1], dataSize);
  for(int i=0;i<size;i++)
    buf[i] = data[0][i].average(data[1][i]);
  DEBUGJ(("Calling store from combine and store msg size %d\n",dataSize));
  store((void *)buf,(dataSize),neighborSide);
  delete []buf;
}

void** CfdProblem :: fragmentForRefine(int *sizePtr)
{
  int newXSize = gridW/2;
  int newYSize = gridH/2;
  //  sizePtr = newXSize*newYSize**sizeof(CfdLoc);

  *sizePtr = ((gridW+2)*(gridH+2)+1)*sizeof(CfdLoc);
  CfdLoc ** dataArray = new CfdLoc* [4];
  CfdLoc** myGrid = grid->getPrimary();

  for(int i=0;i<4;i++) {
    /*dataArray[i] = new CfdLoc[newXSize*newYSize+1];
      for(int x=1;x<=newXSize;x++){
      for(int y=1;y<=newYSize;y++)
      dataArray[i][(x-1)*newYSize+(y-1)] = dataGrid[((i/2)%2)*newXSize+x][(i%2)*newYSize+y];
      }*/
    dataArray[i] = new CfdLoc[(gridW+2)*(gridH+2)+1];

    for(int y=0; y<gridH+2; y++)
      for(int x=0; x<gridW+2; x++)
	dataArray[i][y*(gridW+2)+x].copy(&myGrid[x][y]);
    //Hack to get the step count to the children
    dataArray[i][(gridW+2)*(gridH+2)] = CfdLoc((double)stepCount,0,0,0);
  }
  return  (void **)dataArray;
}

int CfdProblem :: readParentGrid(void* data, int dataSize)
{
  CfdLoc *dataArray = (CfdLoc*) data;
  CfdLoc **myGrid = grid->getPrimary();
  int size = ((gridH+2)*(gridW+2)+1)*sizeof(CfdLoc);
  if(size == dataSize) {
    for(int y=0; y<gridH+2; y++)
      for(int x=0; x<gridW+2; x++) 
	myGrid[x][y].copy(&dataArray[y*(gridW+2)+x]);
    //return the step count
    return (int)(dataArray[(gridW+2)*(gridH+2)].P);
  }
  else {
    CkError("readParentGrid: Error in the size of the parent grid\n");
    return 0;
  }
}

Properties* CfdProblem :: initParam() {
  gridW = gridH =100;
  double timeStep  = 0.00005;
  //Physical and simulation constants
  double meshSize=1.00;//Size of one side of a CFD cell, (m)
  double diffuse=1.0e-2*meshSize/timeStep;//Diffusion constant, (m/s)
  double density=1.0e3;//Mass density (Kg/m^3)
  double modulus=2.0e9;//Bulk modulus (N/m^2)
  double thik=0.01; //Plate thickness (m)
  
  return (new Properties(timeStep,meshSize,diffuse,density,modulus,thik));
}



void CfdProblem :: initGrid () {
  //Initialize the interior
  double varVel=1.0e-6;//Initial velocity variance (m/s)
  double avgPres=100.0e3;//Initial pressure (N/m^2)
  double varPres=  0.1e3;//Initial pressure variation (N/m^2)
  double  backgroundT=300.0;//Normal temperature (K)

  double Vx=varVel*0.05;//rand.nextGaussian();
  double Vy=varVel*0.05;//rand.nextGaussian();
  double P=avgPres+varPres*0.05;//rand.nextGaussian();
  double T=backgroundT;

  for (int y=0;y<gridH;y++)
    for (int x=0;x<gridW;x++) {
      CfdLoc* temp = new CfdLoc(P,Vx,Vy,T);
      //  temp->init(P,Vx,Vy,T);
      grid->set(x,y,temp);
      //delete temp;
    }
 
}

void CfdProblem :: initBoundary()
{
  double avgPres=100.0e3;//Initial pressure (N/m^2)
  double  backgroundT=300.0;//Normal temperature (K)
  //initialize the boundary
  if(isOnNegXBoundary()) {
    for (int y=0;y<gridH+GHOSTWIDTH;y++) 
      grid->set(0,y,new CfdLoc(avgPres,0,0,backgroundT));
  }
  
  if(isOnPosXBoundary()) {
    for (int y=1;y<gridH+GHOSTWIDTH;y++)
      grid->set(gridW+1,y,new CfdLoc(avgPres,0,0,backgroundT));
  }
  
  if(isOnNegYBoundary()) {
    for (int x=1;x<gridW+GHOSTWIDTH;x++) 
      grid->set(x,0,new CfdLoc(avgPres,0,0,backgroundT));
  }
  
  if(isOnPosYBoundary()) {
    for (int x=1;x<gridW+GHOSTWIDTH;x++)
      grid->set(x,gridH+1,new CfdLoc(avgPres,0,0,backgroundT));
  }
}

void CfdProblem :: startStep()
{
  //stepCount++;
  //Impose the boundary conditions
  setBoundaries();
  //Do pressure calculations
  grid->update(grid->getPrimary(),grid->getSecondary());
  grid->setSecondary();
}

//Resample (and switch buffers)
void CfdProblem :: finishStep(bool resample)
{
  int presSteps=4;//Number of pressure-only steps before resampling
  if (resample && stepCount%presSteps==presSteps-1)
    //Resample new values by new velocities
    grid->resample(grid->getSecondary(),grid->getPrimary(),
		   presSteps);
  else
    grid->swap();
  grid->setPrimary();
}

void CfdProblem ::setBoundaries()
{
  int x,y;
  if(isOnNegYBoundary()) {
    y=0;for (x=0;x<gridW+GHOSTWIDTH;x++) {//Top
      grid->at(x,y)->Vy=0;
      grid->at(x,y)->P=grid->at(x,y+1)->P;
    }
  }
  
  if(isOnPosYBoundary()) {
    y=gridH-1;for (x=0;x<gridW+GHOSTWIDTH;x++) {//Bottom (out of bounds)
      grid->at(x,y)->Vy=0;
      grid->at(x,y)->P=grid->at(x,y-1)->P;
    }
  }

  int xL=0,xR=gridW-1;

  if(isOnNegXBoundary()) {
    for (y=0;y<gridH+GHOSTWIDTH;y++) { 
      grid->at(xL,y)->Vx=0;
      grid->at(xL,y)->P=grid->at(xL+1,y)->P;
    }
  }

  if(isOnPosYBoundary()) { 
    for (y=0;y<gridH+GHOSTWIDTH;y++) { 
      grid->at(xR,y)->Vx=0;
      grid->at(xR,y)->P=grid->at(xR-1,y)->P;
    }
  }
}

void CfdGrid :: update(CfdLoc **src,CfdLoc **dest)
{
  for(int x=1; x<gridW+GHOSTWIDTH-1; x++)
    for(int y=1; y<gridH+GHOSTWIDTH-1; y++) {
      dest[x][y].update(properties,&src[x][y-1],
			&src[x-1][y],&src[x][y],
			&src[x+1][y],&src[x][y+1]);
    }

}

void CfdGrid :: resample(CfdLoc **src,CfdLoc **dest,double velScale)
{
  double scale=0.5*velScale*properties.AGlob;

  for(int destX=1; destX<gridW+GHOSTWIDTH-1; destX++)
    for(int destY=1; destY<gridH+GHOSTWIDTH-1; destY++) {
      CfdLoc *srcCur=&(src[destX][destY]);
      double srcX=destX-scale*(srcCur->Vx+src[destX+1][destY].Vx);
      double srcY=destY-scale*(srcCur->Vy+src[destX][destY+1].Vy);
      int ix=(int)srcX;
      int iy=(int)srcY;
      if (ix>=0 && iy>=0 && ix<gridW+GHOSTWIDTH-1 && iy<gridH+GHOSTWIDTH-1)
	dest[destX][destY].interpolate(srcX-ix,srcY-iy,&(src[destX][destY]),
				       &(src[ix][iy+1]),&(src[ix+1][iy+1]),
				       &(src[ix][iy]),&(src[ix+1][iy]));
      else
	dest[destX][destY].copy(srcCur);
    }
}

void CfdLoc::update(Properties prop,
		    CfdLoc* t,
		    CfdLoc* l,CfdLoc* c,CfdLoc* r,
		    CfdLoc* b) 
{
  //Diffuse out the pressure
  double neighborP=t->P+b->P+l->P+r->P;
  double Pd=prop.diffuseGlobL*c->P+prop.diffuseGlobR*neighborP;
  
  //Pressure -> velocity
  Vx=c->Vx+prop.VGlob*(l->P-r->P);
  Vy=c->Vy+prop.VGlob*(t->P-b->P);
  
  //Velocity -> pressure
  P=Pd+prop.PGlob*(l->Vx-r->Vx  +  t->Vy-b->Vy);		
}

double CfdLoc::interpolate(double dx,double dy,
			   double tl,double tr,
			   double bl,double br)
{
  double t=tl+dx*(tr-tl);
  double b=bl+dx*(br-bl);
  return b+dy*(t-b);
}

void CfdLoc::interpolate(double dx,double dy,
			 CfdLoc* src,
			 CfdLoc* tl,CfdLoc* tr,
			 CfdLoc* bl,CfdLoc* br) 
{
  Vx=interpolate(dx,dy,tl->Vx,tr->Vx,bl->Vx,br->Vx);
  Vy=interpolate(dx,dy,tl->Vy,tr->Vy,bl->Vy,br->Vy);
  P =interpolate(dx,dy,tl->P ,tr->P ,bl->P ,br->P );
  T =interpolate(dx,dy,tl->T ,tr->T ,bl->T ,br->T );
}

PUPable_def(AmrUserData);
PUPable_def(CfdProblem);


#include "cfdAMR.def.h"

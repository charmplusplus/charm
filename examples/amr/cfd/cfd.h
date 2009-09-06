#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <charm++.h>
#define GHOSTWIDTH 2

class Properties {
 public:
  double timeStep; //Delta-t, (s)
  double meshSize;//Size of one side of a CFD cell, (m)
  double diffuse;//Diffusion constant, (m/s)
  double density;//Mass density (Kg/m^3)
  double modulus;//Bulk modulus (N/m^2)
  double thickness; //Plate thickness (m)
  
  //These are the constant terms in the various state equations
  double diffuseGlobL,diffuseGlobR; //diffusion equation
  double VGlob; //velocity update
  double PGlob; //pressure update
  double AGlob; //advection equation (see CfdGrid)
  Properties(){}
  Properties(double dt,double s,double df,
	     double d,double m,double thik) 
    {
      timeStep=dt;
      meshSize=s;
      diffuse=df;
      density=d;
      modulus=m;
      
      double alpha=timeStep/meshSize*diffuse;//Diffusion rate, cells/step
      diffuseGlobL=1-alpha;
      diffuseGlobR=0.25*alpha;
      
      thickness=thik;
      double A=meshSize*thickness;//Area of cell-cell interface (m)
      double V=A*meshSize;//Volume of cell (m^3)
      VGlob=0.5*timeStep*(1.0/density)*A/V;
      PGlob=0.5*timeStep*(modulus)*A/V;
      AGlob=timeStep/meshSize;
    }
  void pup(PUP::er& p) {
    p|timeStep; 
    p|meshSize;
    p|diffuse;
    p|density;
    p|modulus;
    p|thickness; 
    p|diffuseGlobL;
    p|diffuseGlobR; 
    p|VGlob; 
    p|PGlob; 
    p|AGlob;
  }
};

class CfdLoc {
 public:
  //Everything's measured at the center of the cell
  double Vx,Vy;//Velocities (m/s)
  double P;//Pressure (N/m^2)
  double T;//Temperature (K)
  CfdLoc(double P0,double Vx0, double Vy0,double T0)
    {
      P=P0;
      Vx=Vx0;
      Vy=Vy0;
      T=T0;
    }
  
  CfdLoc() {
    P=0.0;
    Vx=0.0;
    Vy=0.0;
    T=0.0;
  }

  /*Perform one physics step, updating our values based on
    our old values and those of our neighbors.
  */
  void update(Properties prop,
	      CfdLoc* t,
	      CfdLoc* l,CfdLoc* c,CfdLoc* r,
	      CfdLoc* b);
  
  //Utility: interpolate given values to find (dx,dy)
  double interpolate(double dx,double dy,double tl,double tr,
		     double bl,double br);
  /*Interpolate all quantities from the given corners*/
  void interpolate(double dx,double dy,
		   CfdLoc* src,
		   CfdLoc* tl,CfdLoc* tr,
		   CfdLoc* bl,CfdLoc* br);
  
  /*Copy all quantities from the given*/
  void copy(CfdLoc* src) {
    Vx=src->Vx;
    Vy=src->Vy;
    P =src->P;
    T =src->T;
    //	q =src.q;
  }

  CfdLoc average(CfdLoc src) {
    double vx,vy,p,t;
    vx = (Vx+src.Vx)/2;
    vy = (Vy+src.Vy)/2;
    p = (P+src.P)/2;
    t = (T+src.T)/2;
    return CfdLoc(p,vx,vy,t);
  }

  void pup(PUP::er& p) {
    p|Vx;
    p|Vy;
    p|P;
    p|T;
  }
  //Print out debugging info.
  void printLoc() {
    printf("P=%d  kpa  Vx=%lf  m/s  Vy=%lf m/s    T=%d K per n \n",
	   (int)(P/1000),Vx,Vy,(int)T);
  }
};

class CfdGrid
{
 private:
  //The grid size and storage
  int gridW,gridH;
  CfdLoc **cfdNow,**cfdNext, **cur;
  Properties properties;
 public:
  int getWidth() {return gridW;}
  int getHeight() {return gridH;}

  CfdGrid(){}

  CfdGrid(int w,int h,Properties* prop) {
    gridW=w;gridH=h;
    properties=*prop;
    cfdNow= new CfdLoc* [gridW+GHOSTWIDTH];
    cfdNext= new CfdLoc* [gridW+GHOSTWIDTH];
    for(int x=0; x<gridW+GHOSTWIDTH;x++) {
      cfdNow[x] = new CfdLoc [gridH +GHOSTWIDTH];
      cfdNext[x] = new CfdLoc [gridH +GHOSTWIDTH];
    }

    cur=cfdNow;
    //    delete prop;
  }

  Properties getProperties() {return properties;}
  
  //Get the primary grid
  CfdLoc** getPrimary() {return cfdNow;}
  CfdLoc** getSecondary() {return cfdNext;}

  void setPrimary() {cur=cfdNow;}
  void setSecondary() {cur=cfdNext;}
  
  //Swap the primary and secondary grids
  void swap(){
    CfdLoc** tmp=cfdNow;
    cfdNow=cfdNext;
    cfdNext=tmp; 
  }

  //Get location x,y from the primary grid
  CfdLoc* at(int x,int y){
    return &(cur[x][y]);
  }
  
  //Set for initial conditions
  void set(int x,int y,CfdLoc* n) {
    cfdNow[x][y].copy(n);
    cfdNext[x][y].copy(n);
    delete n;
  }

  double averageP(){
    /*double sumP =0.0;
    int count;
    for(int x=1;x<gridW+1;x++)
      for(int y=1;y<gridH+1;y++)
      sumP += cur[x][y].P;*/
    int sumP =0;
    int count;
   
    for(int y=1;y<gridH+1;y++)
      for(int x=1;x<gridW+1;x++)
	sumP += (int)cur[x][y].P;
    count = gridH*gridW;
    return (double)(sumP/count);
  }

  //Determine grid values for the next timestep
  void update(CfdLoc **src,CfdLoc **dest);

  //Resample src into dest based on src velocities
  void resample(CfdLoc **src,CfdLoc **dest,double velScale);

  void printGrid() {
    for(int y=0; y<gridH; y++)
      for(int x=0;x<gridW;x++)
    	cur[x][y].printLoc();
  }
  
  void pup(PUP::er& p) {
    p|gridW;
    p|gridH;
    properties.pup(p);
    if(p.isUnpacking()) { 
      cfdNow= new CfdLoc* [gridW+GHOSTWIDTH];
      cfdNext= new CfdLoc* [gridW+GHOSTWIDTH];
      for(int x=0; x<gridW+GHOSTWIDTH;x++) {
	cfdNow[x] = new CfdLoc [gridH +GHOSTWIDTH];
	cfdNext[x] = new CfdLoc [gridH +GHOSTWIDTH];
      }
      cur = cfdNow;
    }
   
    for(int y=0; y<gridH+GHOSTWIDTH; y++)
      for(int x=0; x<gridW+GHOSTWIDTH;x++){
	cfdNow[x][y].pup(p);
	cfdNext[x][y].pup(p);
      }
  }

  ~CfdGrid() {
     for(int x=0;x<gridW+GHOSTWIDTH;x++) {
      delete []cfdNow[x];
      delete []cfdNext[x];
      }
    delete []cfdNow;
    delete []cfdNext;
  }
	
};

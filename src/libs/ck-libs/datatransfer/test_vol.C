/*  
Test out volume computation routines.
Orion Sky Lawlor, olawlor@acm.org, 2004/7/23
*/
#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include "cg3d.h"
using namespace cg3d;

double test_vol_mgc(const PointSet3d *ps,const Tet3d &A,const Tet3d &B);
double test_vol_planes(const PointSet3d *ps,const Tet3d &A,const Tet3d &B);
double test_vol_nop(const PointSet3d *ps,const Tet3d &A,const Tet3d &B)
{
	return 0.0; /* doesn't do anything */
}
typedef double (*test_vol_fn)(const PointSet3d *ps,const Tet3d &A,const Tet3d &B);


/// Spit out a random floating point number, uniform on [0,1]
double randFloat(void) {
	return (rand()&0xffFF)*(1.0/(float)0xffFF);
}

/// Spit out a random 3-vector, uniform over the unit cube
CkVector3d randPoint(void) { 
	return CkVector3d(randFloat(),randFloat(),randFloat()); 
}

/// Spit out a random 3-vector, uniform over a skew plane.
///  This ensures a large number of degeneracies.
CkVector3d randPlane(void) { 
	CkVector3d planeOrigin(0,0,0);
#if 0 // Roundoff-friendly perfectly flat plane
	CkVector3d planeX(1,0,0); 
	CkVector3d planeY(0,1,0);
#else //Roundoff-unfriendly skew plane
	CkVector3d planeX(0.7,0.2,0.1);
	CkVector3d planeY(0.3,0.8,-0.2);
#endif
	return planeOrigin
		+randFloat()*planeX +randFloat()*planeY; 
}

const int tetTypes=6; //Number of different tet types to generate

// Create a random tetrahedron using this random source
inline Tet3d randTet(int tetType,PointSet3d *ps) {
	CkVector3d A,B,C,D;
	do {
		A=randPoint(); B=randPoint(); C=randPoint();
		switch (tetType) 
		{ 
		case 0://Totally random tet
			break;
		case 1://Put the entire base in the plane
			A=randPlane(); B=randPlane(); C=randPlane(); break;
		case 2://Put two points in the plane
			A=randPlane(); B=randPlane();break;
		case 3://Fix three points
			A=CkVector3d(0.123,0.234,0.456); /* no break; fallthrough */
		case 4://Fix two points
			B=CkVector3d(0.987,0.876,0.765); /* no break; fallthrough */
		case 5://Fix one point
			C=CkVector3d(0.654,0.543,0.432); break;
		};
		D=randPoint();
	} while (tetVolume(A,B,C,D)<1.0e-3);
	Tet3d t(ps,A,B,C,D); testShape(t); return t;
	// return Tet3d(ps,A,B,C,D);
}


// Compare intersection volumes for a bunch of random tets:
void matchTest(int firstTest,int lastTest) {
	printf("Running random tests from %d to %d\n",firstTest,lastTest);
	int nZero=0, nBig=0;
	double diff=0;
	for (int testNo=firstTest;testNo<lastTest;testNo++) {
		if (testNo%(16*1024)==0) {
			printf("."); 
			fflush(stdout);
		}
		srand(testNo); //Random seed is number of test
		
		int tetType=(int)(0.999*randFloat()*tetTypes); //Encourage degeneracies
		PointSet3d ps;
		Tet3d A(randTet(tetType,&ps));
		Tet3d B(randTet(tetType,&ps));
		
		double mvol=test_vol_mgc(&ps,A,B);
		double pvol=test_vol_planes(&ps,A,B);
		double err=fabs(mvol-pvol);
		if (err>1.0e-9)
			CmiAbort("Mgc and planes volumes differ!");
		if (mvol<1.0e-6) nZero++;
		if (mvol>1.0e-2) nBig++;
		if (err>diff) diff=err;
	}
	printf("To %d: (%d zero, %d big) %.3g max error\n",
		lastTest,nZero,nBig,diff);
	printf("All tests from %d to %d passed\n",firstTest,lastTest);
}

/// Return the time per intersection for nTest iterations
///  of this volume function.
double timePer(test_vol_fn fn,int nTest,int degen) {
	double start=MPI_Wtime();
	for (int testNo=0;testNo<nTest;testNo++) {
		srand(testNo); //Random seed is number of test
		
		int tetType=(int)(0.999*randFloat()*degen); //Encourage degeneracies
		PointSet3d ps;
		Tet3d A(randTet(tetType,&ps));
		Tet3d B(randTet(tetType,&ps));
		
		double vol=fn(&ps,A,B);
	}
	return (MPI_Wtime()-start)/nTest;
}


// Time intersections for a bunch of random tets:
void timeTest(const char *desc,test_vol_fn fn,int nTest,int degen) {
	double zper=timePer(test_vol_nop,nTest/100,degen);
	double per=timePer(fn,nTest,degen)-zper;
	printf("%20s: %.3f us / intersection (%d)\n",desc,1.0e6*per,nTest);
}

int main(int argc,char *argv[])
{
	int firstTest=0, lastTest=100*1000;
	if (argc>1) firstTest=atoi(argv[1]);
	if (argc>2) lastTest=atoi(argv[2]);
	matchTest(firstTest,lastTest);
	int nTest=100*1000, degen=0;
	timeTest("Mgc",test_vol_mgc,nTest,degen);
	timeTest("Planes",test_vol_planes,nTest,degen);
	return 0;
}


/*  
Volume calculations for complicated shapes, that
are the intersection of simpler shapes.

Orion Sky Lawlor, olawlor@acm.org, 2003/2/24
*/
#include <stdio.h>
#include <stdlib.h>
#include "cg3d.h"

using namespace cg3d;

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

// Test intersections for a bunch of random tets:
void doTest(int firstTest,int lastTest) {
	printf("Running random tests from %d to %d\n",firstTest,lastTest);
	int nZero=0, nBig=0;
	for (int testNo=firstTest;testNo<lastTest;testNo++) {
		if (testNo%1024==0) {
			printf("."); 
			printf("to %d: (%d zero, %d big)\n",testNo,nZero,nBig);
			nZero=0; nBig=0; 
			fflush(stdout);
		}
		srand(testNo); //Random seed is number of test
		
		int tetType=(int)(0.999*randFloat()*tetTypes); //Encourage degeneracies
		PointSet3d ps;
		Tet3d A(randTet(tetType,&ps));
		Tet3d B(randTet(tetType,&ps));
		
		double volume=intersectDebug(&ps,A,B);
		if (volume==-1) abort(); //Non-manifold error
		if (volume<=0) nZero++;
		if (volume>1.0e-3) nBig++;
	}
	printf("All tests from %d to %d passed\n",firstTest,lastTest);
}


int main(int argc,char *argv[])
{
	int firstTest=0, lastTest=2000000000;
	if (argc>1) firstTest=atoi(argv[1]);
	if (argc>2) lastTest=atoi(argv[2]);
	doTest(firstTest,lastTest);

	return 0;
}



/**
 * Read a TetMesh ".noboite" mesh description file.
 * 
 * Orion Sky Lawlor, olawlor@acm.org, 3/11/2003
 */
#include <stdio.h>
#include <stdlib.h>
#include "tetmesh.h"
#include "charm.h" //For CkAbort

/// Read n integers into dest.  Subtract off base.
bool readInts(FILE *f,int *dest,int n,int base) {
	for (int i=0;i<n;i++) {
		int v;
		if (1!=fscanf(f,"%d",&v)) return false;
		dest[i]=v-base;
	}
	return true;
}
	
/// Read n 3D coordinates into dest.
bool readPts(FILE *f,CkVector3d *dest,int n) {
	for (int i=0;i<n;i++) {
		double x,y,z;
		if (3!=fscanf(f,"%lf%lf%lf",&x,&y,&z)) return false;
		dest[i]=CkVector3d(x,y,z);
	}
	return true;
}

void bad(const char *why) {
	printf("Fatal error> %s\n",why);
	abort();
}

/// Read a TetMesh (ghs3d) ".noboite" mesh description file.
void readNoboite(FILE *f,TetMesh &t) {
	if (f==NULL)
		CkAbort("Error opening TetMesh file");
	
	// The file header consists of one line with
	//  <nTets> <nPts> ...total of 17 random ints...
	const int headerLen=17;
	int header[headerLen];
	if (!readInts(f,header,headerLen,0))
		CkAbort("Error reading TetMesh file header");
	t.allocate(header[0],header[1]); 
	
	// Now come 1-based node indices
	if (!readInts(f,t.getTetConn(),4*t.getTets(),1))
		CkAbort("Error reading TetMesh file's tets");
	
	// Now come the coordinates, as floating-point numbers
	if (!readPts(f,t.getPointArray(),t.getPoints()))
		CkAbort("Error reading TetMesh file's points");
}

/// Write a TetMesh (ghs3d) ".noboite" mesh description.
void writeNoboite(FILE *f,TetMesh &t) {
	if (f==NULL)
		CkAbort("Error opening TetMesh file for write");
	int i, e=t.getTets(), n=t.getPoints();
	fprintf(f,"%d %d ", e,n);
	for (i=0;i<15;i++) fprintf(f,"-1 ");
	fprintf(f,"\n");
	for (i=0;i<e;i++) {
		for (int j=0;j<TetMesh::nodePer;j++)
			fprintf(f,"%d ",t.getTet(i)[j]+1);
		fprintf(f,"\n");
	}
	for (i=0;i<n;i++) {
		CkVector3d v=t.getPoint(i);
		fprintf(f,"%f %f %f \n",v.x,v.y,v.z);
	}
}



/**
A serial, and incredibly slow, data transfer test.
The slowness comes because it's a crude quadratic
all-pairs algorithm.

However, it's very easy to use.
*/
#include <stdio.h>
#include <stdlib.h>
#include "transfer.h"
#include "charm.h" //For CkAbort

int main(int argc,char *argv[]) {
	if (argc<2) {printf("Usage: test_transfer <src .noboite file> <dest .noboite file>\n"); exit(1);}
	TetMesh srcMesh, destMesh;
	readNoboite(fopen(argv[1],"r"),srcMesh);
	printf("Source "); printSize(srcMesh);
	readNoboite(fopen(argv[2],"r"),destMesh);
	printf("Destination "); printSize(destMesh);
	
	//Fabricate data to be transferred:
	int s,nSrc=srcMesh.getTets();
	double *src=new double[nSrc];
	double totalSrc=0; //Area-weighted source integral
	for (s=0;s<nSrc;s++) {
		double v=0.1*s; //Generate some fake data
		src[s]=v;
		totalSrc+=v*srcMesh.getTetVolume(s);
	}
	int d,nDest=destMesh.getTets();
	double *dest=new double[nDest];
	for (d=0;d<nDest;d++) dest[d]=-1;
	
	transferCells(1,src,srcMesh,dest,destMesh);
	
	double totalDest=0; //Area-weighted destination integral
	for (d=0;d<nDest;d++) 
		totalDest+=dest[d]*destMesh.getTetVolume(d);
	double err=totalSrc-totalDest;
	if (fabs(err)>1.0e-10) CkAbort("Total data transferred doesn't match!");
	
	printf("Transfer successful: total error %.3g\n",err);
	
	return 0;
}

/**
 * Conservative, accurate serial cell-centered data transfer.
 * Orion Sky Lawlor, olawlor@acm.org, 2003/2/26
 */
#include <stdio.h>
#include <string.h> // For memmove
#include <stdlib.h> // for abort
#include <vector> //for std::vector 
#include "cg3d.h" // for intersection and volume calculation
#include "tetmesh.h" 


// Compute the volume shared by cell s of srcMesh
//   and cell d of destMesh.
double getSharedVolume(int s,const TetMesh &srcMesh,
	int d,const TetMesh &destMesh,cg3d::Planar3dDest *dest=NULL) 
{
	cg3d::PointSet3d ps;
	const int *sc=srcMesh.getTet(s);
	const int *ds=destMesh.getTet(d);
	cg3d::Tet3d S(&ps,srcMesh.getPoint(sc[0]),srcMesh.getPoint(sc[1]),
	           srcMesh.getPoint(sc[2]),srcMesh.getPoint(sc[3]));
	cg3d::Tet3d D(&ps,destMesh.getPoint(ds[0]),destMesh.getPoint(ds[1]),
	           destMesh.getPoint(ds[2]),destMesh.getPoint(ds[3]));
	
	if (dest==NULL)
		return intersectDebug(&ps,S,D);
	else /* user provided a destination */ {
		cg3d::intersect(&ps,S,D,*dest);
		return -1;
	}
}


/**
 * Debugging class:
 * Write incoming face fragments to a TecPlot output file.
 */
class debugDest3d : public cg3d::Planar3dDest {
	FILE *f; //Tecplot output file
	int nTets; //Number of tets we've seen so far
	std::vector<CkVector3d> pts; //4*nTets points
public:
	debugDest3d(FILE *f_,const char *title="cg3d_debug") :f(f_) { 
		nTets=0;
		// Tecplot file header:
		fprintf(f,"TITLE = \"%s\"\n"
		          "VARIABLES = \"X\", \"Y\", \"Z\" \n",title);
	}
	
	void addTet(const CkVector3d &a,const CkVector3d &b,
	            const CkVector3d &c,const CkVector3d &d) 
	{
		nTets++;
		pts.push_back(a); pts.push_back(b); 
		pts.push_back(c); pts.push_back(d); 
	}
	void addTet(const cg3d::Tet3d &t) {
		addTet(t.getPoint(0),t.getPoint(1),t.getPoint(2),t.getPoint(3));
	}
	void addTet(const int *conn,const CkVector3d *pts) {
		addTet(pts[conn[0]],pts[conn[1]],pts[conn[2]],pts[conn[3]]);
        }
	void addTet(const TetMesh &destMesh,int d) {
		addTet(destMesh.getTet(d),&destMesh.getPoint(0));
	}
	
	// Add this face:
	void addFace(const cg3d::Planar3d &face,int src)
	{
		// Triangulate the convex planar face, and sweep triangles into tets
		for (int f=1;f<face.getPoints()-1;f++) {
			CkVector3d a=face.getPoint(0), b=face.getPoint(f),c=face.getPoint(f+1);
			addTet(1.0/3.0*(a+b+c), a,b,c);
		}
	}
	
	/// Write all accumulated tets under this zone name
	void write(const char *zone) {
		if (nTets>0) {
			fprintf(f,"ZONE T=\"%s\", N= %d, E= %d, F=FEPOINT, ET=TETRAHEDRON\n",
				  zone,pts.size(), nTets);
			for (int p=0;p<pts.size();p++)
				fprintf(f,"%f %f %f\n",pts[p].x,pts[p].y,pts[p].z);
			fprintf(f,"\n");
			for (int i=0;i<nTets;i++)
				fprintf(f,"%d %d %d %d\n", 4*i+1, 4*i+2, 4*i+3, 4*i+4);
			nTets=0; pts.erase(pts.begin(),pts.end());
		}
	}
	
	~debugDest3d() {
		write("last");
		fclose(f);
	}
};

/**
 * Debugging version of "intersect".  Writes out a tecplot file
 *  if the intersection is non-manifold.
 */
double cg3d::intersectDebug(cg3d::PointSet3d *ps,const cg3d::Tet3d &S,const cg3d::Tet3d &D) {
	try {
		cg3d::Volume3dDest dest;
		cg3d::intersect(ps,S,D,dest);
		return dest.getVolume();
	} 
	catch (const NonManifoldException &e) 
	{ /* Shapes are not manifolds: write out a debugging file*/
		printf("cg3d error> Shape intersection is not a manifold\n"
			"Area 1: %.20g (from a vertex)\n"
			"Area 2: %.20g (from random point)\n",e.a,e.b);
#if OSL_CG3D_DEBUG
		{
			const char *fName="tecManifold.plt";
			printf("Writing mismatch cells to %s\n",fName);
			debugDest3d debug(fopen(fName,"w")); 
			debug.addTet(D); 
			debug.write("dest");
			
			debug.addTet(S);
			debug.write("src");
			
			cg3d::intersect(ps,S,D,debug);
			debug.write("intersection");
		}
		abort();
#endif
	}
	return -1;
}

/**
 * Debugging routine: called when mesh volumes don't match
 */
void volumeMismatch(const TetMesh &srcMesh,int d,const TetMesh &destMesh)
{
	int s,ns=srcMesh.getTets(); //Source cells
	const char *fName="tecMismatch.plt";
	printf("cg3d error> Volumes don't match\n"
		"Writing mismatching cells to tecPlot file %s\n",fName);
	printTet(destMesh,d);
	debugDest3d debug(fopen(fName,"w"));
	debug.addTet(destMesh,d);
	debug.write("dest");
	
	/// Write out the offending src tets:
	for (s=0;s<ns;s++) {
		if (getSharedVolume(s,srcMesh,d,destMesh)>0) {
			debug.addTet(srcMesh,s);
			printTet(srcMesh,s);
		}
	}
	debug.write("src");
	
	// Write out the intersection:
	for (s=0;s<ns;s++) {
		// Compute the volume shared by s and d:
		getSharedVolume(s,srcMesh,d,destMesh,&debug);
	}
	debug.write("intersection");
}

/**
 * Conservatively, accurately transfer 
 *   srcVals, tet-centered values on srcMesh
 * to
 *   destVals, tet-centered values on destMesh
 */
void transferCells(int valsPerTet,
	double *srcVals,const TetMesh &srcMesh,
	double *destVals,const TetMesh &destMesh)
{
	int d,nd=destMesh.getTets(); //Destination cells
	int s,ns=srcMesh.getTets(); //Source cells
	int v,nv=valsPerTet; //Values (for one cell)
	const int maxV=30;
	// volumeMismatch(srcMesh,nd/2,destMesh);
	// testVolumes(9,srcMesh,246,destMesh);
	
	/* For each dest cell: */
	for (d=0;d<nd;d++) {
		
		//Accumulate volume-weighted-average destination values
		double destAccum[maxV]; 
		for (int v=0;v<nv;v++) destAccum[v]=0.0;
		double destVolume=0; // Volume accumulator
		
		/* For each source cell: */
		for (s=0;s<ns;s++) {
			// Compute the volume shared by s and d:
			double shared=getSharedVolume(s,srcMesh,d,destMesh);
			if (shared<-1.0e-10) CkAbort("Negative volume shared region!");
			if (shared>0) {
				for (int v=0;v<nv;v++) 
					destAccum[v]+=shared*srcVals[s*nv+v];
				destVolume+=shared;
				
			}
		}
		
		/* Check the relative volume error, to make sure we've 
		   totally covered each destination cell. Checking precision
		   is low, since meshing tools often use single precision. */
		double trueVolume=destMesh.getTetVolume(d);
		double volErr=destVolume-trueVolume;
		double accumScale=1.0/destVolume; //Reverse volume weighting
		if (fabs(volErr*accumScale)>1.0e-10) {
			printf("WARNING: ------------- volume mismatch: dest tet %d -------------\n"
				" True volume %g, but total is only %g (err %g)\n",
				d,trueVolume,destVolume,volErr);
#if OSL_CG3D_DEBUG
			volumeMismatch(srcMesh,d,destMesh);
			abort();
#endif
		}
		
		/* Copy the accumulated values into dest */
		for (int v=0;v<nv;v++) 
			destVals[d*nv+v]=destAccum[v]*accumScale;
	}
}



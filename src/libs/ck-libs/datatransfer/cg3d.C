/*  
Volume calculations for complicated shapes, that
are the intersection of simpler shapes.

Orion Sky Lawlor, olawlor@acm.org, 2003/2/26
*/
#include <stdio.h>
#include <string.h> // For memmove
#include <stdlib.h> // for abort
#include "cg3d.h"

using namespace cg3d;

/**
 * Roundoff control: value to compare halfspace dotproducts against.
 * Safe range for small-valued (unit cube) inputs appears to be
 * between 1.0e-16 and 1.0e-8.  Luckily, the degeneracy self-test 
 * seems to catch CkAbort epsilons, rather than being silently wrong.
 */
double cg3d::epsilon=1.0e-10; //12;

// ----------------- PointSet ----------------
/// Create an empty pointset. 
PointSet3d::PointSet3d() {
	nPts=0; nHalf=0;
	setPts=0;
}

void PointSet3d::addHalf(const CkVector3d &pt,int p) {
	halfset_t bit=(1u<<p);
	for (int h=0;h<nHalf;h++) {
		double side=half[h].side(pt);
		if (side>epsilon)  inHalf[h]|=bit;
		if (side<-epsilon) outHalf[h]|=bit;
	}
}

/// Calculate halfspace insideness for old points.
void PointSet3d::calculateHalfspaces(void) {
	for (int h=0;h<nHalf;h++) inHalf[h]=outHalf[h]=0;
	for (int p=0;p<nPts;p++) addHalf(pts[p],p);
	setPts=nPts+firstPt;
}

/// Add a halfspace to our list
int PointSet3d::addHalfspace(const CkHalfspace3d &h) {
	int r=nHalf++;
#if OSL_CG3D_DEBUG
	if (r>=maxHalfs) CkAbort("PointSet3d added too many halfspaces");
#endif
	half[r]=h;
	return firstHalf+r;
}

/// Add a point to our list
int PointSet3d::addPoint(const CkVector3d &p) {
	int r=nPts++;
#if OSL_CG3D_DEBUG
	if (r>=maxPts) CkAbort("PointSet3d added too many points");
#endif
	pts[r]=p;
	return firstPt+r;
}

// ------------------ Planar ----------------------
Planar3d::Planar3d(PointSet3d *ps_) {
	ps=ps_;
	nPts=0;
}

/// Clip this shape to lie within this halfspace.
///  Returns true if we are made empty by the intersection.
bool Planar3d::addConstraint(int halfspace)
{
	//The in array marks if the Point is in-bounds
	bool in[maxPts];
	
	int i,outDex=-1,inDex=-1;
	for (i=0;i<nPts;i++) {
		/* count anything close as being inside */
		bool isIn=!ps->isOutside(pts[i],halfspace); 
		in[i]=isIn;
		if (isIn) inDex=i;
		else outDex=i;
	}
	if (outDex==-1) return true;//<- no Points were out of bounds-- nothing to do
	if (inDex==-1) {nPts=0;return false;}//<- all Points were out of bounds-- we're emptied

	const CkHalfspace3d &h=ps->getHalfspace(halfspace);
	
	//Find the transition from out-of-bounds to in-bounds (inStart/inStart+1)
	int inStart=outDex;
	while (!in[(inStart+1)%nPts]) inStart++;   inStart%=nPts;
	int startPt=ps->addPoint(h.intersectPt(
		getPoint(inStart),getPoint((inStart+1)%nPts)
	));
	
	//Find the transition from in-bounds to out-of-bounds (inEnd/inEnd+1)
	int inEnd=inStart;
	while (in[(inEnd+1)%nPts]) inEnd++;   inEnd%=nPts;	
	int endPt=ps->addPoint(h.intersectPt(
		getPoint(inEnd),getPoint((inEnd+1)%nPts)
	));
	
	//Now we make room in our vector for the new Points (preserving order)
	if (inEnd>inStart)
	{ //start moves up to begining of array
		memmove((void *)&pts[1],(void *)&pts[inStart+1],sizeof(pts[0])*(inEnd-inStart));
		pts[0]=startPt;
		nPts=inEnd-inStart+1;
		pts[nPts++]=endPt;
	} else 
	{//Items before end do not move
		memmove((void *)&pts[inEnd+3],(void *)&pts[inStart+1],sizeof(pts[0])*(nPts-inStart-1));
		pts[inEnd+1]=endPt;
		pts[inEnd+2]=startPt;
		nPts+=2-(inStart-inEnd);
	}
#if OSL_CG3D_DEBUG
	if (nPts>maxPts) CkAbort("Planar3d added too many points");
#endif
	return true;
}

// ---------------------- Shape --------------------

Shape3d::~Shape3d() {}

/// Return true if this shape fully contains this point.
/// Works by enumerating the shape's halfspaces.
bool Shape3d::contains(const CkVector3d &pt) const {
	for (int f=0;f<getFaces();f++) 
		if (!ps->isInside(pt,getHalfspaceIndex(f))) 
			return false;
	return true;
}

/// Test this shape for validity.  Aborts if the shape doesn't
///  satisfy its invariants.
void cg3d::testShape(const Shape3d &s) {
	//Find an interior point, by averaging the shape's points
	CkVector3d sum(0);
	int nSum=0;
	int h,f,i;
	for (i=0;i<s.getPoints();i++) {
		sum+=s.getPoint(i);
		nSum++;
	}
	CkVector3d interior=sum/nSum;
	
	//Check all the halfspaces to make sure they point the right way
	for (h=0;h<s.getFaces();h++) {
		const CkHalfspace3d &half=s.getHalfspace(h);
		if (half.side(interior)<=0)
			CkAbort("'interior' point doesn't satisfy halfspace!");
		
		f=h;
		Planar3d face(s.getSet()); s.getFace(f,face);
		for (i=0;i<face.getPoints();i++) {
			double err=half.side(face.getPoint(i));
			if (fabs(err)>1.0e-10)
				CkAbort("Face doesn't lie in its own halfspace!");
		}
	}
}

// ------------------- Tet -------------------

Tet3d::Tet3d(PointSet3d *ps_,
	const CkVector3d &A,const CkVector3d &B_,const CkVector3d &C,const CkVector3d &D_)
	:Shape3d(ps_,4,4,h,p)
{
	CkHalfspace3d half;
	half.init(A,B_,C);
	CkVector3d B=B_, D=D_;
	if (half.side(D)<0) 
	{ /* This tet is inside out: swap B and D */
		B=D_; D=B_;
		half.init(A,B,C);
	}
	h[0]=ps->addHalfspace(half);
	half.init(A,D,B); h[1]=ps->addHalfspace(half);
	half.init(B,D,C); h[2]=ps->addHalfspace(half);
	half.init(A,C,D); h[3]=ps->addHalfspace(half);
	p[0]=ps->addPoint(A);
	p[1]=ps->addPoint(B);
	p[2]=ps->addPoint(C);
	p[3]=ps->addPoint(D);
}

void Tet3d::getFace(int f, Planar3d &face) const {
	switch(f) {
	case 0: face.addPoint(p[0],p[1],p[2]); break; //ABC
	case 1: face.addPoint(p[0],p[3],p[1]); break; //ADB
	case 2: face.addPoint(p[1],p[3],p[2]); break; //BDC
	case 3: face.addPoint(p[0],p[2],p[3]); break; //ACD
	};
}


// ------------------ Intersection -----------------

/* Do early-exit halfspace testing to see if these two shapes could
    possibly intersect.
 */
bool earlyExit(PointSet3d *pset,const Shape3d &shape0, const Shape3d &shape1) {
	for (int srcShape=0;srcShape<2;srcShape++) 
	{
		const Shape3d *ps=srcShape?&shape1:&shape0; // Points come from here
		const Shape3d *hs=srcShape?&shape0:&shape1; // Halfspaces from here
		int h, nh=hs->getFaces();
		int p, np=ps->getPoints();
		for (h=0;h<nh;h++) { 
			int half=hs->getHalfspaceIndex(h);
			for (p=0;p<np;p++)
				if (pset->isInside(ps->getPointIndex(p),half))
				{ //This point is really inside this halfspace--skip this halfspace
					break;
				}
			if (p==np) return true; //*All* points are outside this halfspace
		}
	}
	return false; //No halfspace outs all the points
}

/*
void printSet(PointSet3d *ps,const char *desc) {
	printf("%s: %d points\n",desc,ps->getPoints());
}
*/

void cg3d::intersect(PointSet3d *ps,const Shape3d &shape0, const Shape3d &shape1, Planar3dDest &faceDest) 
{
	ps->calculateHalfspaces(); //Cache halfspaces for existing points
	if (earlyExit(ps,shape0,shape1)) return;
	
	/* Actually intersect each surface */
	for (int srcShape=0;srcShape<2;srcShape++) 
	{
		const Shape3d *fs=srcShape?&shape1:&shape0; // Faces come from here (clip these faces)
		const Shape3d *hs=srcShape?&shape0:&shape1; // Halfspaces from here (clip against these)
		int f,h, nf=fs->getFaces(), nh=hs->getFaces();
		
		for (f=0;f<nf;f++) {
			ps->pushPoints();
			Planar3d face(ps);
			fs->getFace(f,face); // Begin with a face
		//Clip by each opposite halfspace
			for (h=0;h<nh;h++) { 
				if (!face.addConstraint(hs->getHalfspaceIndex(h))) 
					goto nextFace; //This face was clipped to nothing
			}
			//Some part of this face survived clipping:
		// Make sure we don't double-count degenerate faces:
			if (srcShape==1)
				for (h=0;h<nh;h++) {
					int half=hs->getHalfspaceIndex(h);
					int p;
					// double maxDel=0;
					for (p=0;p<face.getPoints();p++) {
						int pIdx=face.getPointIndex(p);
						if (ps->isInside(pIdx,half)||ps->isOutside(pIdx,half))
							break; //Point isn't along halfspace--this isn't it
					}
					if (p==face.getPoints()) 
					  //This face lies completely along h, so it's already been counted
						goto nextFace;
				}
			
		//This is a good face, so add it
			faceDest.addFace(face,srcShape);
		nextFace:
			ps->popPoints(); //Throw away points on this face
		}
	}
}

/************* Volume3dDest *************/
Volume3dDest::Volume3dDest() {
	hasOrigin=false;
	volume=0;
#if OSL_CG3D_DEBUG
	subVolume=new Volume3dDest(CkVector3d(-1,0,0));
#endif
}
/// Debugging-only constructor: specify sweep origin
Volume3dDest::Volume3dDest(const CkVector3d &origin_) {
	hasOrigin=true;
	origin=origin_;
	volume=0;
#if OSL_CG3D_DEBUG
	subVolume=NULL;
#endif
}
#if OSL_CG3D_DEBUG
Volume3dDest::~Volume3dDest() { //Check volume at second origin
	if (subVolume!=NULL) {
		double err=volume-subVolume->getVolume();
		if (fabs(err/(volume+1.0))>1.0e-7) /* swept volume is not a manifold */
			throw NonManifoldException(volume,subVolume->getVolume());
		delete subVolume;
	}
}
#endif

double cg3d::tetVolume(const CkVector3d &A,const CkVector3d &B,
		const CkVector3d &C,const CkVector3d &D)
{
	const static double oneSixth=1.0/6.0;
	return oneSixth*(B-A).dot((D-A).cross(C-A));
}

void Volume3dDest::addFace(const Planar3d &face,int src)
{
	if (!hasOrigin) 
	{ //Pick arbitrary point on first face as origin (prevents roundoff, provides speedup)
		hasOrigin=true;
		origin=face.getPoint(0);
	} else /* (hasOrigin) */ {
		// Triangulate the convex planar face, and sweep triangles into tets
		double faceVol=0.0;
		for (int f=1;f<face.getPoints()-1;f++) {
			faceVol+=tetVolume(origin,
				face.getPoint(0),
				face.getPoint(f),
				face.getPoint(f+1));
		}
		volume+=faceVol;
		// printf("shape %d, %d-point face: %f\n",src,
		//	face.getPoints(),faceVol);
	}
#if OSL_CG3D_DEBUG
	if (subVolume) subVolume->addFace(face,src);
#endif
}




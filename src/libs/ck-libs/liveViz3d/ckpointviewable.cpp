/*
A method for turning points into textures.

Orion Sky Lawlor, olawlor@acm.org, 2003/9/22
*/
#include "ckvector3d.h"
#include "ckviewable.h"
#include "converse.h"

#define TIME_DRAW 0 /* print out timing information for each frame */

/************************* CkPointViewable ***************/

/*
Allocate SSE-aligned space.  To work with regular
 malloc, we allocate the buffer like:
    | padding | pointer to alloc | user data |
*/
void *_ck_SSE_malloc(int size) {
	const unsigned long align=16;
	const unsigned long alignMask=align-1;
	char *alloc, *ret;
	alloc=(char *)malloc(size+alignMask+sizeof(void *));
	// Make room for the allocated pointer
	ret=alloc+sizeof(char *);
	// Add padding for alignment
	ret=(char *)(((unsigned long)(ret+alignMask))&(~alignMask));
	// Write in the pointer just before user data
	((char **)ret)[-1]=alloc;
	return ret;
}
void _ck_SSE_free(void *v) {
	char *alloc=((char **)v)[-1];
	free(alloc);
}

CkPointViewable::CkPointViewable(int nPts_) 
	:nPts(nPts_)
{
	pts=(float *)_ck_SSE_malloc(nPts*4*sizeof(float));
	box.empty();
	nPts=0; /* for "add" calls */
}
CkPointViewable::~CkPointViewable() {
	_ck_SSE_free(pts);
}
void CkPointViewable::add(const CkVector3d &p) {
	pts[4*nPts+0]=(float)p.x;
	pts[4*nPts+1]=(float)p.y;
	pts[4*nPts+2]=(float)p.z;
	pts[4*nPts+3]=(float)1.0;
	box.add(p);
	nPts++;
}
void CkPointViewable::done(void) {
	setUnivPoints(CkInterestSet(box));
}

static unsigned char *clipArr=CkImage::newClip();
void CkPointViewable::renderImage(const CkViewpoint &vp,CkImage &dest)
{
	dest.clear();
	const unsigned char brt=25;
	const unsigned char src[4]={brt,brt,brt,brt};
	// CkRect r(dest.getRect());
	unsigned int w=dest.getWidth(), h=dest.getHeight();
#if TIME_DRAW
	double startTime=CmiWallTimer();
#endif
#if CMK_USE_INTEL_SSE
	for (int i=0;i<nPts;i++) {
		// Subtle: if int sx is negative, (unsigned int)sx is large
		unsigned int sx,sy;
		vp.project_noz(&pts[4*i],(int *)&sx,(int *)&sy);
		if (sx<w && sy<h)
			dest.addPixelClip(src,dest.getPixel(sx,sy),clipArr);
	}
#else /* no SSE: regular floating point */
	for (int i=0;i<nPts;i++) {
		CkVector3d p(&pts[4*i]);
		CkVector3d s(vp.project_noz(p));
		// Subtle: if int sx is negative, (unsigned int)sx is large
		unsigned int sx=(unsigned int)s.x, sy=(unsigned int)s.y;
		if (sx<w && sy<h)
			dest.addPixelClip(src,dest.getPixel(sx,sy),clipArr);
	}
#endif
#if TIME_DRAW
	double elapsed=CmiWallTimer()-startTime;
	CkPrintf("Generated (%dx%d) view (%.3fms for %d particles)\n",
		dest.getWidth(),dest.getHeight(),1.0e3*elapsed, nPts);
#endif
	
}


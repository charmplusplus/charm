/*
Describes a 3d object that can be "viewed"-- that is,
converted into a texture map.

Orion Sky Lawlor, olawlor@acm.org, 8/28/2002
*/
#include "ckvector3d.h"
#include "ckviewable.h"
#include "charm++.h"

CkReferenceCounted::~CkReferenceCounted() {}
CkViewConsumer::~CkViewConsumer() {}
CkViewable::~CkViewable() {}

/** CkView **/
void CkTexStyle::pup(PUP::er &p) {
	p.comment("Texture corners");
	for (int i=0;i<4;i++) p|corners[i];
}

void pup_pack(PUP::er &p,CkView &v) {
	v.id.pup(p);
	v.style.pup(p);
	CkImage &img=*v.tex;
	img.pup(p);
	int len=img.getRect().area()*img.getColors();
	p(img.getData(),len);
}

void CkAllocView::pup(PUP::er &p) {
	id.pup(p);
	style.pup(p);
	p.comment("View image");
	p|img;
}
CkAllocView *pup_unpack(PUP::er &p) {
	CkAllocView *ret=new CkAllocView;
	ret->pup(p);
	return ret;
}

/***************** InterestSet *******************/
CkInterestSet::CkInterestSet(int nInterest_,const CkVector3d *loc_)
	:nInterest(nInterest_) 
{
	for (int i=0;i<nInterest;i++) loc[i]=loc_[i];
}

//Set our points as the 8 corners of this bounding box:
CkInterestSet::CkInterestSet(const CkBbox3d &box) 
	:nInterest(8)
{
	for (int i=0;i<8;i++) {
		loc[i]=CkVector3d(
			(i&1)?box.max.x:box.min.x,
			(i&2)?box.max.y:box.min.y,
			(i&4)?box.max.z:box.min.z
		);
	}
}

CkVector3d CkInterestSet::getMean(void) const {
	CkVector3d sum(0,0,0);
	for (int i=0;i<nInterest;i++) sum+=loc[i];
	return sum*(1.0/nInterest);
}

void CkInterestSet::pup(PUP::er &p) {
	p|nInterest;
	for (int i=0;i<nInterest;i++)
		p|loc[i];
}

/******************** InterestView *********************/

/// Initialize this view, from this viewpoint:
CkInterestView::CkInterestView(int wid,int ht,const CkInterestSet &univ_,
	const CkViewpoint &univ2texture)
	:CkAllocView(wid,ht), univ(univ_)
{
	style.corners[0]=univ2texture.viewplane(CkVector3d(0  , 0, 0));
	style.corners[1]=univ2texture.viewplane(CkVector3d(wid, 0, 0));
	style.corners[2]=univ2texture.viewplane(CkVector3d(0  ,ht, 0));
	style.corners[3]=univ2texture.viewplane(CkVector3d(wid,ht, 0));
	
	proj.setPoints(univ.size());
	for (int i=0;i<univ.size();i++) 
	{ //Back-project my interest points into the texture plane:
		proj[i]=univ2texture.projectViewplane(univ[i]);
	}
}



//Evaluate the root-mean-square error, in pixels, between the
// projection of our texture and the projection of the true object.
double CkInterestView::rmsError(const CkViewpoint &univ2screen) const
{
	double sum=0;
	for (int i=0;i<univ.size();i++)
		sum+=sqrError(i,univ2screen);
	return sqrt(sum/univ.size());
}
double CkInterestView::maxError(const CkViewpoint &univ2screen) const
{
	double max=0;
	for (int i=0;i<univ.size();i++) {
		double cur=sqrError(i,univ2screen);
		if (cur>max) max=cur;
	}
	return sqrt(max);
}

void CkInterestView::pup(PUP::er &p) {
	CkAllocView::pup(p);
	p.comment("Universe points");
	univ.pup(p);
	p.comment("Projected points");
	proj.pup(p);
}

/********************* InterestViewable *************/
CkInterestViewable::CkInterestViewable() {
	last=NULL;
}
CkInterestViewable::~CkInterestViewable() {
	flushLast();
}

/**
 * Search for an appropriate existing view for this viewpoint.
 * If none exists, render a new view.
 */
void CkInterestViewable::view(const CkViewpoint &univ2screen,CkViewConsumer &dest) {
	CkInterestView *ret=NULL;
	if (last && goodView(univ2screen,last)) 
	{ //Just re-use previous image:
		ret=last;
	}
	else { // Create new image from scratch
		CkViewpoint univ2texture(newViewpoint(univ2screen));
		ret=new CkInterestView(
			univ2texture.getWidth(),univ2texture.getHeight(),
			interest,univ2texture);
		ret->id=id;
		render(univ2texture,ret->getImage());
		
		flushLast(); last=ret;
	}
	dest.add(ret);
}

/**
 * Return true if this view is appropriate for use from this viewpoint.
 * Default implementation checks the rms reprojection error.
 */
bool CkInterestViewable::goodView(const CkViewpoint &univ2screen,CkInterestView *testView) {
	double rmsTol=2.0;
	return testView->rmsError(univ2screen)<rmsTol;
}
	
/**
 * Create a new CkInterestView for this (novel) viewpoint.
 * Default implementation creates a flat-on viewpoint and 
 * calls render() to create a new texture.
 */
CkViewpoint CkInterestViewable::newViewpoint(const CkViewpoint &univ2screen) {
//Find our destination rectangle onscreen, to estimate our resolution needs:
	CkRect r; r.empty();
	for (int i=0;i<interest.size();i++) { //Screen-project this interest point
		CkVector2d s(univ2screen.project_noz(interest[i]));
		univ2screen.clip(s);
		r.add((int)s.x,(int)s.y);
	}
	if (r.area()>4096*4096)
		CkAbort("liveViz3d> Absurdly large projected screen rectangle!");
	
	if (r.l==r.r) r.r++; /* enlarge vertical sliver */
	if (r.t==r.b) r.b++; /* enlarge horizontal sliver */
	
//Determine the texture size based on the onscreen size
//   (Note: OpenGL textures *MUST* be a power of two in both directions)
	const int start_sz=4, max_sz=256;
	int wid=start_sz; while (wid<r.wid()/2) wid*=2;
	int ht =start_sz; while (ht<r.ht()/2) ht*=2;
	if (wid>max_sz) wid=max_sz;
	if (ht>max_sz) ht=max_sz;
	
//Create our new view in the plane of our center, 
//  using a perspective-scaled version of the old axes:
	const CkVector3d &E=univ2screen.getEye();
	double eyeCenter=(center-E).mag();
	double eyeViewplane=(univ2screen.projectViewplane(center)-E).mag();
	double perspectiveScale=eyeCenter/eyeViewplane;
	// printf("PerspectiveScale=%f\n",perspectiveScale);
	
	double inset=1; //Pixels to expand output region by (to ensure a clean border)
	// New axes are just scaled versions of old axes:
	double Xscale=perspectiveScale*r.wid()/(wid-2*inset);
	double Yscale=perspectiveScale*r.ht()/(ht-2*inset);
	if (Xscale ==0 || Yscale==0) 
		CkAbort("liveViz3d> Illegal axis scaling");
	
	CkVector3d X=univ2screen.getX()*Xscale;
	CkVector3d Y=univ2screen.getY()*Yscale;
	// New origin is just shifted version of old origin:
	CkVector3d R=perspectiveScale*(univ2screen.viewplane(CkVector3d(r.l,r.t,0))-E)+E
		-inset*X-inset*Y;
	return CkViewpoint(E,R,X,Y,wid,ht);
}


#ifdef STANDALONE
/*(obsolete) Unit test for CkView:*/

int main(int argc,char *argv[]) {
	PUP::toTextFile p(stdout);
	
	double xSz=2.3, ySz=0.7, zSz=1.6;
	//xSz=1, ySz=1, zSz=1;
	
	int wid=200,ht=100;
	CkAxes3d axes;
	axes.setPixelSize(CkVector3d(2.3,0.7,1.6));
	CkVector3d Origin(3,2,1);
	axes.nudge(0.2,0.1);
	CkViewpoint vp(axes.makeView(wid,ht,Origin));
	const int nInterest=3;
	const CkVector3d univ_p[nInterest]={
		CkVector3d(100,50,10),
		CkVector3d(110,60,20),
		CkVector3d(120,55,30)
	};
	CkInterestSet univ(nInterest,univ_p);
	
	CkViewpoint tvp;
	CkView view(univ,vp,tvp);
	
	printf("RMS error in original view: %.3g pixels\n",view.rmsError(vp));
	
	axes.setPixelSize(axes.getPixelSize()*0.5);
	printf("RMS error in translated/scaled view: %.3g pixels\n",
	  view.rmsError(axes.makeView(wid,ht,Origin+CkVector3d(1000,2000,100)))
	);
	
	for (int i=0;i<3;i++) {
		axes.nudge(0.14,-0.09);
		printf("RMS error in increasingly rotated view: %.3g pixels\n",
		  view.rmsError(axes.makeView(wid,ht,Origin))
		);
	}
	
	printf("Original viewpoint:\n");
	p|vp;
	printf("New view:\n");
	p|tvp;
	return 0;
}

#endif


/*
Describes a 3d object that can be "viewed"-- that is,
converted into a texture map.

Orion Sky Lawlor, olawlor@acm.org, 8/28/2002
*/
#include "ckvector3d.h"
#include "viewable.h"
#include "charm++.h"


/***************** Viewpoint *****************/
void CkViewpoint::pup(PUP::er &p) {
	p.comment("CkViewpoint {");
	p.comment("axes");
	for (int i=0;i<3;i++) p|axes[i];
	p.comment("origin");
	p|origin;
	p.comment("width and height");
	p|wid; p|ht;
	p.comment("} CkViewpoint");
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

void CkInterestSet::pup(PUP::er &p) {
	p|nInterest;
	for (int i=0;i<nInterest;i++)
		p|loc[i];
}

/******************** View *********************/
void CkView::pup(PUP::er &p) {
	p.comment("CkView {");
	if (p.isUnpacking()) img=new CkAllocImage();
	p.comment("CkView image data");
	img->pup(p);
	
	p.comment("corners (universe coords)");
	for (int c=0;c<4;c++) p|corners[c];
	p.comment("true interest points");
	p|univ;
	p.comment("projected interest points");
	p|proj;
	p.comment("source view");
	p|viewable;
	p.comment("} CkView");
}


CkView::CkView() :img(NULL)
	{}

CkView::CkView(const CkInterestSet &univ_,const CkInterestSet &proj_,
		const CkVector3d *corners_,CkAllocImage *img_)
	:img(img_), univ(univ_), proj(proj_)
{
	for (int c=0;c<4;c++) corners[c]=corners_[c];
}

/*
Choose a texture viewpoint for rendering these interest points
as seen from this viewpoint.  This certainly isn't the only way
to write this--you can pick the texture viewpoint a number of ways.
*/
static CkViewpoint chooseViewpoint(const CkInterestSet &univ,
	const CkViewpoint &univ2screen, int &wid,int &ht)
{
//Pick a destination rectangle onscreen:
	CkRect r; r.empty();
	double sum_z;
	for (int i=0;i<univ.size();i++) { //Screen-project the point univ[i];
		CkVector3d s(univ2screen.project(univ[i]));
		r.add((int)s.x,(int)s.y);
		sum_z+=s.z;
	}
	double ave_z=sum_z/univ.size();
	r.enlarge(1,1); //Add a few screen pixels (for a transparent border)
	
	if (r.area()>1024*1024)
		CkAbort("liveViz3d> Absurdly large projected screen rectangle!\n");
	
//Determine the texture size (OpenGL textures *MUST* be a power of two in both directions)
	const int start_sz=4, max_sz=256;
	wid=4; while (wid<r.wid()) wid*=2;
	ht =4; while (ht<r.ht()) ht*=2;
	if (wid>max_sz) wid=max_sz;
	if (ht>max_sz) ht=max_sz;

//Determine the new view:
	double xScl=(double)wid/r.wid();
	double yScl=(double)ht/r.ht();
	CkVector3d origin=univ2screen.backProject(CkVector3d(r.l,r.t,ave_z));
	
	return CkViewpoint(wid,ht,
		univ2screen.getAxis(0)*xScl,univ2screen.getAxis(1)*yScl,
		univ2screen.getAxis(2),
		origin);
}


//Back-project our 4 corners and projected points from this viewpoint:
void CkView::setTexView(const CkViewpoint &univ2texture)
{
	int wid=img->getWidth(), ht=img->getHeight();
	CkVector3d origin=univ2texture.backProject(CkVector3d(0  , 0, 0));
	corners[0]=origin;
	corners[1]=univ2texture.backProject(CkVector3d(wid, 0, 0));
	corners[2]=univ2texture.backProject(CkVector3d(0  ,ht, 0));
	corners[3]=univ2texture.backProject(CkVector3d(wid,ht, 0));
	
	CkHalfspace3d texturePlane(univ2texture.getAxis(2),origin);
	proj.setPoints(univ.size());
	for (int i=0;i<univ.size();i++) 
	{ //Back-project my interest points to the texture plane:
		CkRay r=univ2texture.getRay(univ[i]);
		proj[i]=texturePlane.intersectPt(r.pos,r.dir);
	}
}


/* More useful constructor:
	Choose a texture viewpoint for rendering these interest points
	as seen from this viewpoint.
*/
CkView::CkView(const CkInterestSet &univ_,
	const CkViewpoint &univ2screen, //Source viewpoint 
	CkViewpoint &univ2texture) //Viewpoint to render into
	:img(NULL), univ(univ_)
{
	int wid=0, ht=0;
	univ2texture=chooseViewpoint(univ,univ2screen,wid,ht);
	img=new CkAllocImage(wid,ht,4);
	setTexView(univ2texture);
}

/*Most useful constructor: view this object from this viewpoint*/
CkView::CkView(const CkViewpoint &univ2screen, //Source viewpoint 
	CkViewable *viewThis) //Object to render
	:img(NULL), univ(viewThis->getInterestPoints()), viewable(viewThis->getViewableID())
{
	int wid=0, ht=0;
	CkViewpoint univ2texture(chooseViewpoint(univ,univ2screen,wid,ht));
	img=new CkAllocImage(wid,ht,4);
	setTexView(univ2texture);
	viewThis->view(univ2texture,*img);
}

//Evaluate the root-mean-square error, in pixels, between the
// projection of our texture and the projection of the true object.
double CkView::rmsError(const CkViewpoint &univ2screen) const
{
	double sum=0;
	for (int i=0;i<univ.size();i++)
		sum+=sqrError(i,univ2screen);
	return sqrt(sum/univ.size());
}
double CkView::maxError(const CkViewpoint &univ2screen) const
{
	double max=0;
	for (int i=0;i<univ.size();i++) {
		double cur=sqrError(i,univ2screen);
		if (cur>max) max=cur;
	}
	return sqrt(max);
}

CkViewable::~CkViewable() {}

#ifdef STANDALONE
/*Unit test for CkView:*/

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





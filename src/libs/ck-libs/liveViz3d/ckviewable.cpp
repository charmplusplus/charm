/*
Describes a 3d object that can be "viewed"-- that is,
converted into a texture map.

Orion Sky Lawlor, olawlor@acm.org, 8/28/2002
*/
#include "ckvector3d.h"
#include "ckviewable.h"
#include "converse.h"
#include "lv3d0.h"
#include <string.h>

// Only include OpenGL utility routines if we're on the client.
#ifdef CMK_LIVEVIZ3D_CLIENT
#  include "ogl/main.h"
#  include "ogl/util.h"
#endif

CkReferenceCounted::~CkReferenceCounted() {}
CkViewConsumer::~CkViewConsumer() {}
CkViewable::~CkViewable() {}

/***************** CkView *****************/
void CkView::pup(PUP::er &p) {
	p.comment("View ID:");
	id.pup(p);
	p.comment("View Priority:");
	p|prio;
}

/***************** CkImageCompressor *****************/

void CkImageCompressor::pup(PUP::er &p) {
	img->CkImage::pup(p);
	int wid=img->getWidth(), ht=img->getHeight();
	int colors=img->getColors();
	if (p.isUnpacking()) 
	{ /* client side: decompression */
		img->allocate();
		for (int y=0;y<ht;y++) {
			int first_x, last_x;
			p|first_x; p|last_x;
			memset(img->getPixel(0,y),0,first_x*colors);
			p(img->getPixel(first_x,y),(last_x-first_x)*colors);
			memset(img->getPixel(last_x,y),0,(wid-last_x)*colors);
		}
	} else /* server side: compress image */ {
		int alpha_channel;
		if (img->getColors()==3) alpha_channel=-1; /* no alpha: just RGB */
		else if (img->getLayout()==CkImage::layout_reversed)
			alpha_channel=img->getColors()-1; /* alpha in last channel */
		else /* img->getLayout()==CkImage::layout_default */
			alpha_channel=0; /* alpha in first channel */
		
		for (int y=0;y<ht;y++) {
			int first_x=0, last_x=wid-1;
			if (alpha_channel!=-1) {
				CkImage::channel_t *row=img->getPixel(0,y);
				row+=alpha_channel; /* jump to alpha channel */
				// Advance over black pixels on the left border
				while (first_x<wid && row[colors*first_x]==0) 
					first_x++;
				// Advance over black pixels on the right border
				while (last_x>=first_x && row[colors*last_x]==0) 
					last_x--;
				last_x++;
			}
			// Copy out span of good data:
			p|first_x; p|last_x;
			p(img->getPixel(first_x,y),(last_x-first_x)*colors);
		}
	}
}


/***************** CkQuadView *****************/

PUPable_def(CkQuadView);

/// Build a new image: normally only on server
CkQuadView::CkQuadView(int w,int h,int n_colors) 
	:s_tex(w,h,n_colors), x_tex(&s_tex)
{
	c_tex=NULL;
}

/// Migration constructor-- prepare for pup.
/// (CLIENT ONLY)
CkQuadView::CkQuadView(CkMigrateMessage *m) 
	:s_tex(), x_tex(&s_tex)
{
#ifndef CMK_LIVEVIZ3D_CLIENT
	CkAbort("CkQuadView migration constructor should never be called on server!\n");
#else
	c_tex=NULL;
#endif
}

void CkQuadView::pup(PUP::er &p) {
	CkView::pup(p);
	p.comment("Texture corners");
	for (int i=0;i<4;i++) p|corners[i];
	
// Pup the image:
	x_tex.pup(p);
}
CkQuadView::~CkQuadView() {
#ifdef CMK_LIVEVIZ3D_CLIENT
	if (c_tex) delete c_tex;
#endif
}

#ifdef CMK_LIVEVIZ3D_CLIENT
int oglImageFormat(const CkImage &img)
{
	if (img.getLayout()==CkImage::layout_default)
	{
		switch (img.getColors()) {
		case 1: return GL_LUMINANCE;
		case 3: return GL_RGB;
		case 4: return oglFormatARGB; /* special */
		};
	} else if (img.getLayout()==CkImage::layout_reversed)
	{
		switch (img.getColors()) {
		case 1: return GL_LUMINANCE;
		case 3: return GL_BGR;
		case 4: return GL_BGRA;
		};
	}
	/* Woa-- I don't recognize this format! */
	CkAbort("Unrecognized CkImage image format");
	return -1;
}
#endif


void CkQuadView::render(double alpha) {
#ifndef CMK_LIVEVIZ3D_CLIENT
	CkAbort("CkQuadView::render should never be called on server!\n");
#else
	if (c_tex==NULL) 
	{ // Upload server texture to OpenGL:
		int format=oglImageFormat(s_tex);
		c_tex=new oglTexture(s_tex.getData(),
			s_tex.getWidth(),s_tex.getHeight(),
			oglTexture_linear, format);
		
		//Now that we've copied the view into GL, 
		// flush the old in-memory copy:
		s_tex.deallocate();
	}
	
	glColor4f(1.0,1.0,1.0,alpha);
	CkVector3d &bl=corners[0];
	CkVector3d &br=corners[1];
	CkVector3d &tl=corners[2];
	CkVector3d &tr=corners[3];
	oglTextureQuad(*c_tex,tl,tr,bl,br);
	if (oglToggles['f']) 
	{ // Draw a little frame around this texture
		oglLine(tl,tr); oglLine(tr,br);
		oglLine(br,bl); oglLine(bl,tl);
	}
#endif
}

#if 0 /* not used */
void CkInterestView::render(double alpha) {
#ifndef CMK_LIVEVIZ3D_CLIENT
	CkAbort("CkInterestView::render should never be called on server!\n");
#else
	CkQuadView::render(alpha);
	if (oglToggles['b']) 
	{ // Draw a little box around this texture's interest points
		oglLine(univ[0],univ[1]); 
		oglLine(univ[1],univ[2]); 
		oglLine(univ[2],univ[3]); 
		oglLine(univ[3],univ[0]); 
		
		oglLine(univ[0],univ[4]); 
		oglLine(univ[1],univ[5]); 
		oglLine(univ[2],univ[6]); 
		oglLine(univ[3],univ[7]); 
		
		oglLine(univ[4],univ[5]); 
		oglLine(univ[5],univ[6]); 
		oglLine(univ[6],univ[7]); 
		oglLine(univ[7],univ[4]); 
	}
#endif
}
#endif

void CkViewNodeInit(void) {
	PUPable_reg(CkQuadView);
	PUPable_reg(LV3D_Universe);
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
CkInterestView::CkInterestView(int wid,int ht,int n_colors,
	const CkInterestSet &univ_,
	const CkViewpoint &univ2texture)
	:CkQuadView(wid,ht,n_colors), univ(univ_)
{
	corners[0]=univ2texture.viewplane(CkVector3d(0  , 0, 0));
	corners[1]=univ2texture.viewplane(CkVector3d(wid, 0, 0));
	corners[2]=univ2texture.viewplane(CkVector3d(0  ,ht, 0));
	corners[3]=univ2texture.viewplane(CkVector3d(wid,ht, 0));
	
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

/********************* InterestViewable *************/
/**
 * Return true if this view is appropriate for use from this viewpoint.
 * Default implementation checks the rms reprojection error.
 */
bool CkInterestViewable::shouldRender(const CkViewpoint &univ2screen,const CkView &oldView)
{
	double rmsTol=0.5; // RMS reprojection error tolerance, in pixels
	double rmsErr=((CkInterestView *)&oldView)->rmsError(univ2screen);
	// printf("RMS error is %.1f pixels\n",rmsErr);
	return rmsErr>rmsTol;
}

/**
 * Draw this object from this point of view, and return
 * the resulting image.  
 * The output CkView will be unref()'d exactly once after
 * each call to render.
 */
CkView *CkInterestViewable::renderView(const CkViewpoint &univ2screen) {
	CkViewpoint univ2texture;
	if (!newViewpoint(univ2screen,univ2texture))
		return NULL;
	CkInterestView *ret=new CkInterestView(
		univ2texture.getWidth(),univ2texture.getHeight(),4,
		interest,univ2texture);
	renderImage(univ2texture,ret->getImage());
	return ret;
}

/**
 * Create a new CkInterestView for this (novel) viewpoint.
 * Default implementation creates a flat-on viewpoint and 
 * calls render() to create a new texture.
 */
bool CkInterestViewable::newViewpoint(const CkViewpoint &univ2screen,CkViewpoint &univ2texture) 
{
//Find our destination rectangle onscreen, to estimate our resolution needs:
	CkRect r; r.empty();
	CkRect clipR; clipR.empty();
	for (int i=0;i<interest.size();i++) { //Screen-project this interest point
		CkVector3d s(univ2screen.project(interest[i]));
		if (s.z<=0) continue; // Just ignore behind-the-viewer points
		// return false;
		r.add((int)s.x,(int)s.y);
		univ2screen.clip(s);
		clipR.add((int)s.x,(int)s.y);
	}
	/*
	if (r.area()>4096*4096)
		CmiAbort("liveViz3d> Absurdly large projected screen rectangle!");
	*/
	if (clipR.area()<2) return false;
	
	if (r.l==r.r) r.r++; /* enlarge vertical sliver */
	if (r.t==r.b) r.b++; /* enlarge horizontal sliver */

//Round up the texture size based on the onscreen size
//   (Note: OpenGL textures *MUST* be a power of two in both directions)
//   ( for mipmapping, the textures must also be square )
	const int start_sz=4, max_sz=512;
	int wid=start_sz, ht =start_sz;  // Proposed size
	// Scale up size until both width and height are acceptable.
	while ((wid<r.wid()) || (ht<r.ht())) {
		ht*=2; wid*=2;
	}
	
//Create our new view in the plane of our center, 
//  using a perspective-scaled version of the old axes:
	const CkVector3d &E=univ2screen.getEye();
	double eyeCenter=(center-E).mag();
	double eyeViewplane=(univ2screen.projectViewplane(center)-E).mag();
	double perspectiveScale=eyeCenter/eyeViewplane;
	// printf("PerspectiveScale=%f\n",perspectiveScale);
	
	double inset=1; //Pixels to expand output region by (to ensure a clean border)
	// New axes are just scaled versions of old axes:
	double Xscale=perspectiveScale; // *r.wid()/(wid-2*inset);
	double Yscale=perspectiveScale; // *r.ht()/(ht-2*inset);
	
	// If the resulting texture is too big, scale it down:
	if (wid>max_sz) { wid=max_sz; Xscale*=r.wid()/(wid-2*inset); }
	if (ht>max_sz) { ht=max_sz; Yscale*=r.ht()/(ht-2*inset); }
	
	/*
	if (Xscale ==0 || Yscale==0) 
		CmiAbort("liveViz3d> Illegal axis scaling");
	*/
	
	CkVector3d X=univ2screen.getX()*Xscale;
	CkVector3d Y=univ2screen.getY()*Yscale;
	// New origin is just shifted version of old origin:
	CkVector3d R=perspectiveScale*(univ2screen.viewplane(CkVector3d(r.l,r.t,0))-E)+E
		-inset*X-inset*Y;
	univ2texture=CkViewpoint(E,R,X,Y,wid,ht);
	return true;
}


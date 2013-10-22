/**
Describes a 3d object that can be "viewed"-- that is,
converted into a texture map.

Orion Sky Lawlor, olawlor@acm.org, 8/28/2002
*/
#include "ckvector3d.h"
#include "ckviewable.h"
#include "converse.h"
#include "lv3d0.h"
#include <string.h>
#include <algorithm> /* for std::max and min */
#include "stats.h"

// Only include OpenGL utility routines if we're on the client.
#ifdef CMK_LIVEVIZ3D_CLIENT
#  include "ogl/ext.h"
#  include "ogl/main.h"
#  include "ogl/util.h"
#  include "ogl/liltex.h"
#  include "lv3dclient/stats.h"
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

double CkViewable::getSize(const CkViewpoint &univ2screen)
{
	return 0.1; /* dunno.  guess. */
}

void CkInterestViewable::setUnivPoints(const CkInterestSet &univPoints_) {
	interest=univPoints_;
	center=interest.getMean();
	CkBbox3d b=interest.getBox();
	boundCenter=0.5*(b.min+b.max);
	boundRadius=boundCenter.dist(b.max);
}
double CkInterestViewable::getSize(const CkViewpoint &univ2screen) {
	double camDist=univ2screen.getEye().dist(boundCenter);
	return boundRadius/camDist;
}

/***************** CkImageCompressor *****************/

/* Round up to nearest power of 2 */
int roundTo2(int x) {
	int ret=1;
	while (ret<x) ret*=2;
	return ret;
}

static stats::op_t op_pack_impostors=stats::time_op("image.pup","CkImageCompressor::pup");

/**
  Compress an image by encoding away pixels with alpha==0
  on the start and end of each row.
  
  FIXME: doesn't work in binary across platforms *unless*
    the PUP::er is network (or xlating.)
*/
void CkImageCompressor::pup(PUP::er &p) {
	stats::op_sentry stats_sentry(op_pack_impostors);
	if (p.isUnpacking()) 
	{ /* client side: decompression */
		int colors=0,layout=0;
		p|w; p|h; p|colors; p|layout;
		gl_w=roundTo2(w); gl_h=roundTo2(h); 
		img=new CkAllocImage(gl_w,gl_h,colors); 
		img->setLayout((CkImage::layout_t)layout);
		for (int y=0;y<gl_h;y++) {
			int first_x=0, last_x=0;
			if (y<h) {p|first_x; p|last_x;}
			else {first_x=0; last_x=0;}
			memset(img->getPixel(0,y),0,first_x*colors);
			p(img->getPixel(first_x,y),(last_x-first_x)*colors);
			memset(img->getPixel(last_x,y),0,(gl_w-last_x)*colors);
		}
	} else /* server side: compress image */ {
		int colors=img->getColors();
		int layout=img->getLayout();
		p|w; p|h; p|colors; p|layout;
		int alpha_channel;
		if (colors==3) alpha_channel=-1; /* no alpha: just RGB */
		else if (layout==CkImage::layout_reversed)
			alpha_channel=colors-1; /* alpha in last channel */
		else /* layout==CkImage::layout_default */
			alpha_channel=0; /* alpha in first channel */
		
		for (int y=0;y<h;y++) {
			int first_x=0, last_x=w-1;
			if (alpha_channel!=-1) {
				CkImage::channel_t *row=img->getPixel(0,y);
				row+=alpha_channel; /* jump to alpha channel */
				// Advance over black pixels on the left border
				while (first_x<w && row[colors*first_x]==0) 
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

PUPable_def(CkQuadView)

static stats::op_t op_render_pixels=stats::count_op("render.pixels","CkQuadView pixels","pixels");

/// Build a new image: normally only on server
CkQuadView::CkQuadView(int w,int h,int n_colors) 
	:s_tex(w,h,n_colors), x_tex(&s_tex)
{
	c_tex=NULL;
	pixels=w*h;
	stats::get()->add(pixels,op_render_pixels);
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

#ifdef CMK_LIVEVIZ3D_CLIENT
oglTextureFormat_t oglImageFormat(const CkImage &img)
{
	oglTextureFormat_t fmt; fmt.type=GL_UNSIGNED_BYTE;
	if (img.getLayout()==CkImage::layout_default)
	{
		switch (img.getColors()) {
		case 1: fmt.format=GL_LUMINANCE; break;
		case 3: fmt.format=GL_RGB; break;
		case 4: fmt.format=GL_BGRA; fmt.type=GL_UNSIGNED_INT_8_8_8_8_REV; break;
		default: CkAbort("Unrecognized CkImage image format");
		};
	} else if (img.getLayout()==CkImage::layout_reversed)
	{
		switch (img.getColors()) {
		case 1: fmt.format=GL_LUMINANCE; break;
		case 3: fmt.format=GL_BGR; break;
		case 4: fmt.format=GL_BGRA; break;
		default: CkAbort("Unrecognized CkImage image format");
		};
	}
	return fmt;
}

static stats::op_t op_uploadpad_pixels=stats::count_op("net.in_pad","Uploaded padded texture pixels","pixels");
static stats::op_t op_upload_pixels=stats::count_op("net.in","Uploaded valid texture pixels","pixels");
#endif


void CkQuadView::pup(PUP::er &p) {
	CkView::pup(p);
	p.comment("Texture corners");
	p|nCorners;
	for (int i=0;i<nCorners;i++) {
		p|corners[i];
		p|texCoord[i];
	}
	
// Pup the image:
	x_tex.pup(p);

#ifdef CMK_LIVEVIZ3D_CLIENT
	if (p.isUnpacking()) { /* immediately upload image to OpenGL */
		CkAllocImage *img=x_tex.getImage();
		oglTextureFormat_t fmt=oglImageFormat(*img);
		c_tex=new oglLilTex(img->getData(),x_tex.gl_w,x_tex.gl_h,
			fmt.format,fmt.type);
		stats::get()->add(x_tex.w*x_tex.h,op_upload_pixels);
		stats::get()->add(x_tex.gl_w*x_tex.gl_h,op_uploadpad_pixels);
		/// Scale texture coordinates from the partial image to OpenGL fractions:
		double tx=x_tex.w/(double)x_tex.gl_w;
		double ty=x_tex.h/(double)x_tex.gl_h;
		for (int i=0;i<nCorners;i++) 
			texCoord[i]=c_tex->texCoord(
				CkVector3d(tx*texCoord[i].x,ty*texCoord[i].y,texCoord[i].z)
			);
		
		//Now that we've copied the view into GL, 
		// flush the old in-memory copy:
		delete img;
	}
#endif

}
CkQuadView::~CkQuadView() {
#ifdef CMK_LIVEVIZ3D_CLIENT
	if (c_tex) delete c_tex;
#endif
}

/* Render us using the current color and texture state */
void CkQuadView::render(void) {
#ifdef CMK_LIVEVIZ3D_CLIENT
	c_tex->bind();
	glBegin (nCorners==4?GL_QUADS:GL_TRIANGLE_STRIP);
	for (int i=0;i<nCorners;i++) {
		glTexCoord2dv(texCoord[i]); glVertex3dv(corners[i]); 
	}
	glEnd();
#endif
}

/** "Ease" function */
static double smooth(double lin) {
	if (lin<0) return 0;
	if (lin>1) return 1;
	// return lin; // linear ease
	if (lin<0.5) return lin*lin;
	else return 1-(1-lin)*(1-lin);
}

void CkQuadView::render(double alpha,CkView *v_old) {
#ifndef CMK_LIVEVIZ3D_CLIENT
	CkAbort("CkQuadView::render should never be called on server!\n");
#else
	glEnable(GL_TEXTURE_2D);
	CkQuadView *old=(CkQuadView *)v_old; // FIXME: what if he's not a quadview?
	
	if (old==NULL || old->nCorners!=nCorners || alpha>0.99) 
	{ /* just draw us */
		glColor4f(1.0,1.0,1.0,1.0);
		render();
	} 
	else 
	{ /* have old, and need to blend with him. */
#if 0 /* use multitexture: needs to have same corners (which sucks) */
		c_tex->bind(); // we're in texture unit 0
		glActiveTextureARB(GL_TEXTURE1_ARB);
		old->c_tex->bind(); // he's in texture unit 1
		glEnable(GL_TEXTURE_2D);
		oglTextureCombineLinear(1.0-alpha);
		glBegin (nCorners==4?GL_QUADS:GL_TRIANGLE_STRIP);
		for (int i=0;i<nCorners;i++) {
			glMultiTexCoord2dvARB(GL_TEXTURE0_ARB,texCoord[i]); 
			glMultiTexCoord2dvARB(GL_TEXTURE1_ARB,old->texCoord[i]);
			glVertex3dv(alpha*corners[i]+(1-alpha)*old->corners[i]); 
		}
		glEnd();
		glDisable(GL_TEXTURE_2D);
		glActiveTextureARB(GL_TEXTURE0_ARB);
#else /* draw two copies: first him, then me. */
		double separation=1.3;
		float of=smooth(separation*(1-alpha));
		glColor4f(of,of,of,of);
		old->render();
		
		float nf=smooth(separation*alpha);
		glColor4f(nf,nf,nf,nf);
		render();
		
		glColor4f(1.0,1.0,1.0,1.0);
#endif
	}
	
	
	if (oglToggles['f']) 
	{ // Draw a little frame around this texture
		glDisable(GL_TEXTURE_2D);
		for (int i=0;i<nCorners;i++)
			oglLine(corners[i],corners[(i+1)%nCorners]);
		glEnable(GL_TEXTURE_2D);
	}
#endif
}

#if CMK_LIVEVIZ3D_INTERESTVIEWRENDER
void CkInterestView::render(double alpha) {
	CkQuadView::render(alpha);
#ifdef CMK_LIVEVIZ3D_CLIENT
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
	nCorners=4;
	corners[0]=univ2texture.viewplane(CkVector3d(0  , 0, 0));
	corners[1]=univ2texture.viewplane(CkVector3d(wid, 0, 0));
	corners[2]=univ2texture.viewplane(CkVector3d(wid,ht, 0));
	corners[3]=univ2texture.viewplane(CkVector3d(0  ,ht, 0));
	int cx=wid/2, cy=ht/2;
	projC=univ2texture.viewplane(CkVector3d(cx  ,cy  , 0));
	projX=univ2texture.viewplane(CkVector3d(cx+1,cy  , 0));
	projY=univ2texture.viewplane(CkVector3d(cx  ,cy+1, 0));
	texCoord[0]=CkVector3d(0,0,0);
	texCoord[1]=CkVector3d(1,0,0);
	texCoord[2]=CkVector3d(1,1,0);
	texCoord[3]=CkVector3d(0,1,0);
	
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
/// Return the ratio between our projected screen resolution
///   and our current texture resolution.
double CkInterestView::resRatio(const CkViewpoint &univ2screen) const
{
	CkVector2d c=univ2screen.project_noz(projC);
	CkVector2d x=univ2screen.project_noz(projX);
	CkVector2d y=univ2screen.project_noz(projY);
	double lx=(x-c).mag(), ly=(y-c).mag(); // new screen projections of our texture axes
	if (lx<1.0 && ly<1.0) return std::min(lx,ly);
	else return std::max(lx,ly);
}

/********************* InterestViewable *************/
/**
 * Return true if this view is appropriate for use from this viewpoint.
 * Default implementation checks the rms reprojection error.
 */
bool CkInterestViewable::shouldRender(const CkViewpoint &univ2screen,const CkView &oldView)
{
	CkInterestView *v=(CkInterestView *)&oldView;
	// Check if old resolution is now inadequate
	double resMax=2.0; // under-resolution tolerance, as a resolution ratio
	double resMin=0.3; // over-resolution tolerance
	double res=v->resRatio(univ2screen);
	if (res>resMax) return true;
	if (res<resMin) return true;
	
	// Check geometric reprojection error
	double rmsTol=1.0; // RMS reprojection error tolerance, in screen pixels
	double rmsErr=v->rmsError(univ2screen);
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
	priorityAdjust=0; /* no priority */
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
		if (!(s.z>0)) continue; // Just ignore behind-the-viewer points
		// return false;
		r.add((int)s.x,(int)s.y);
		univ2screen.clip(s);
		clipR.add((int)s.x,(int)s.y);
	}
	if (clipR.area()<2) return false;
	
	if (r.l==r.r) r.r++; /* enlarge vertical sliver */
	if (r.t==r.b) r.b++; /* enlarge horizontal sliver */

//Round up the texture size based on the onscreen size
//   (Note: OpenGL textures *MUST* be a power of two in both directions)
//   ( for mipmapping, the textures must also be square )
	double inset=2; //Pixels to expand output region by (to ensure a clean border)
	
	 int target_w=(int)(r.wid()+2*inset), target_h=(int)(r.ht()+2*inset);
#if 0 /* for OpenGL: power-of-two textures */
	const int start_sz=4, max_sz=512;
	int wid=start_sz, ht =start_sz;  // Proposed size
	// Scale up size until both width and height are acceptable.
	while ((wid<target_w) || (ht<target_h)) {
		ht*=2; wid*=2;
	}
#else /* non-power-of-two textures (sensible default) */
	int max_sz=512;
	int wid=target_w, ht=target_h;
#endif
	
//Create our new view in the plane of our center, 
//  using a perspective-scaled version of the old axes:
	const CkVector3d &E=univ2screen.getEye();
	double eyeCenter=(center-E).mag();
	double eyeViewplane=(univ2screen.projectViewplane(center)-E).mag();
	double perspectiveScale=eyeCenter/(1.0e-10+eyeViewplane);
	// printf("PerspectiveScale=%f\n",perspectiveScale);
	
	// New axes are just scaled versions of old axes:
	double Xscale=perspectiveScale;//*r.wid()/(wid-2*inset);
	double Yscale=perspectiveScale;//*r.ht()/(ht-2*inset);
	
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
	
	if (1) {
	/// Set priority:
	// const static double distCutoff=10.0;
	priorityAdjust=(int)(
		std::max(wid,ht)*(1.0/1.0) /* area boost */
		// +1.0/std::max(eyeCenter,distCutoff)*(50.0/distCutoff) /* nearness boost */
		// +(r==clipR)*20.0 /* fully onscreen boost */
	);
	}
	
	return true;
}


/*
Describes a 3d object that can be "viewed"-- that is,
converted into a texture map.

Orion Sky Lawlor, olawlor@acm.org, 8/28/2002
*/
#ifndef __UIUC_PPL_CHARM_VIEWABLE_H
#define __UIUC_PPL_CHARM_VIEWABLE_H

#include "charm.h"
#include "ckvector3d.h"
#include "ckimage.h"
#include "ckviewpoint.h"
#include "ckhashtable.h"

/// A CkViewableID uniquely identifies a CkViewable across processors.
///  The first 3 ints of the id are the array index; the last int
///   is the viewable inside that array index.
///   This object can be used in a CkHashtable.
class CkViewableID {
public:
	enum {nId=4};
	int id[nId];
	
	CkViewableID() {for (int i=0;i<nId;i++) id[i]=0;}
	CkViewableID(int len,const int *src) {
		int i;
		for (i=0;i<len;i++) id[i]=src[i];
		for (i=len;i<nId;i++) id[i]=0;
	}
	
	//Allows ViewableID's to be used in a hashtable:
	inline unsigned int hash(void) const {
		return id[0]+(id[1]<<8)+(id[2]<<16)+(id[3]<<24);
	}
	inline int compare(const CkViewableID &v) const {
		for (int i=0;i<nId;i++) if (id[i]!=v.id[i]) return 0;
		return 1;
	}
	static unsigned int staticHash(const void *key,size_t keyLen) {
		return ((const CkViewableID *)key)->hash();
	}
	static int staticCompare(const void *a,const void *b,size_t keyLen) {
		return ((const CkViewableID *)a)->compare(*(const CkViewableID *)b);
	}
	
	void pup(PUP::er &p) {
		p(id,nId);
	}
};
PUPmarshall(CkViewableID)

/**
 * A reference-counted object.  This is used for simple object sharing,
 * where one class creates objects that may be used by other classes
 * indefinitely, but copying is expensive.  CkReferenceCounted ojbects
 * must be allocated on the heap, not inside another object or on the stack.
 */
class CkReferenceCounted {
	int refCount;
protected:
	/// Do not explicitly call the destructor for 
	///   CkReferenceCounted objects--use unref instead.
	virtual ~CkReferenceCounted();
public:
	/// On creation, the reference count is 1, 
	///  so one call to unref will destroy the object.
	CkReferenceCounted() { refCount=1; }
	
	/// Increment the reference count.  Must be matched
	///   by some later call to unref.
	inline void ref(void) {refCount++;}
	
	/// Decrement the reference count.  If there are no
	///   more references, delete this object.
	inline void unref(void) {
		refCount--;
		if (refCount<0) CkAbort("Trying to unref deleted object!\n");
		if (refCount==0) delete this;
	}
};


/***************** Abstract Interface **************/

/**
 * A CkView is a texture image that can stand in for a CkViewable.
 * Subclasses contain some texture representation (e.g., one RGBA
 * texture).  This class is sent from the server, which generates
 * CkViews, to the client, which displays them.  CkViews travel
 * around in the parallel machine inside liveViz3dViewMsg objects.
 * 
 * CkView's are reference counted-- use ref() and unref() to save
 * copying, and always allocate them with a plain "new".
 */
class CkView : public PUP::able, public CkReferenceCounted {
public:
	/// This is the viewable we describe.
	///  This field is set by LiveViz3dArray, not CkViewable.
	CkViewableID id;
	const CkViewableID &getViewable(void) const {return id;}
	
	/// This is our approximate network priority. 
	///  Lower numbers mean higher priority.  Prio==0 is highest.
	///  This field is set by LiveViz3dArray, not CkViewable.
	unsigned int prio;
	
	/// The number of pixels we represent.
	int pixels;
	
	/// CkView subclasses are PUP::able's.
	///  This means they should have a migration constructor,
	///  working pup routine, and appropriate PUPable declarations.
	/// (CLIENT AND SERVER).
	PUPable_abstract(CkView);
	
	/// PUP this CkView and its image or geometry data.
	/// (CLIENT AND SERVER).
	virtual void pup(PUP::er &p);
	
	/**
	  Render our image to the current OpenGL context.
	  Use "alpha" fraction of your own pixels; "1-alpha" 
	  fraction of the "old" pixels.  "old" may be NULL.
	  
	  (CLIENT ONLY)
	  So the server can link without OpenGL, all OpenGL 
	   calls made by this routine should be protected 
	   by an ifdef CMK_LIVEVIZ3D_CLIENT
	*/
	virtual void render(double alpha,CkView *old) =0;
};

/// Call this routine once per node at startup--
///  it registers all the PUPable CkView's.
void CkViewNodeInit(void);

/**
 * An object that can be "viewed"-- turned into an image.
 */
class CkViewable {
public:
	CkViewable() :priorityAdjust(0) {}
	virtual ~CkViewable();

	/// Priority adjustment for re-rendering.
	///   This starts out at 0, but should be made more and more
	///   positive for more and more important views.
	int priorityAdjust;
	
	/**
	 * Return true if this object needs to be redrawn,
	 *   false if oldView is acceptable from the new viewpoint.
	 * oldView is always a CkView returned from a previous call to 
	 *   "render".
	 */
	virtual bool shouldRender(const CkViewpoint &univ2screen,const CkView &oldView) =0;
	
	/**
	 * Draw this object from this point of view, and return
	 * the resulting image (or NULL if none is appropriate).  
	 * The output CkView will be unref()'d exactly once after
	 * each call to render.
	 */
	virtual CkView *renderView(const CkViewpoint &univ2screen) =0;
	
	/// Return our onscreen size, in radians measured from the camera.
	///  This routine is *only* called after a successful shouldRender
	///   or a call to renderView, to set the priority of the resulting message.
	/// The default implementation always returns 0.1.
	virtual double getSize(const CkViewpoint &univ2screen);
};

/**
 * A CkViewConsumer accepts CkView's from viewables.
 * ViewConsumers often keep these views in a 
 * hashtable indexed by ViewableID.
 */
class CkViewConsumer {
public:
	virtual ~CkViewConsumer();
	
	/**
	 * Add this view to our list.  
	 * The incoming CkView will be unref()'d exactly once
	 * as a result of this call-- this is an ownership transfer.
	 */
	virtual void add(CkView *view) =0;
};

/***************** Basic Implementations **************/

class oglLilTex; // Forward declaration-- defined in ogl/liltex.h on client

/**
 Performs a simple zero-encoding of blank image pixels
 at each end of the image scanlines.  This happens during
 the pup routine.
*/
class CkImageCompressor {
	// source/destination image
	CkAllocImage *img;
public:
	int w,h; ///< Size of original image.
	int gl_w, gl_h; ///< Size of expanded image (to power of 2 for openGL)
	CkImageCompressor(CkAllocImage *img_) 
		:img(img_),w(img->getWidth()), h(img->getHeight()), gl_w(0), gl_h(0) { }
	void pup(PUP::er &p);
	CkAllocImage *getImage(void) {
		CkAllocImage *i=img;
		img=NULL;
		return i;
	}
};

/**
 * The simplest kind of CkView: a single, flat 
 * quadrilateral OpenGL texture.  You specify the 
 * texture, corners, and draw style.
 */
class CkQuadView : public CkView {
public:
	/// These are the XYZ universe locations of the 
	///  vertices of our texture image.  These go in a 
	///   fan, like:
	///    corners[0]==image.getPixel(0,0)
	///    corners[1]==image.getPixel(w-1,0)
	///    corners[2]==image.getPixel(w-1,h-1)
	///    corners[3]==image.getPixel(0,h-1)
	/// Subclasses MUST set these fields on the server.
	enum {maxCorners=8};
	int nCorners;
	CkVector3d corners[maxCorners]; // xyz vertices
	CkVector3d texCoord[maxCorners]; // texture coordinates

// (CLIENT AND SERVER)
	virtual void pup(PUP::er &p);
	PUPable_decl(CkQuadView);
	virtual ~CkQuadView();
	
// (SERVER ONLY)
private:
	/// Our texture image-- only valid on server side.
	CkAllocImage s_tex;
	CkImageCompressor x_tex;
public:
	/**
	  Create a new quad view to display a new texture.
	   n_colors should be 1 (greyscale, luminance),
	   3 (rgb, no alpha) or 4 (premultiplied rgba).
	   
	   Be sure to set the "corners" and "id" fields after making this call.
	   (SERVER ONLY)
	 */
	CkQuadView(int w,int h,int n_colors);
	
	CkAllocImage &getImage(void) {return s_tex;}
	const CkAllocImage &getImage(void) const {return s_tex;}

// (CLIENT ONLY)
private:
	oglLilTex *c_tex;
	void render(void);
public:
	/// Migration constructor-- prepare for pup.
	/// (CLIENT ONLY)
	CkQuadView(CkMigrateMessage *m);
	
	inline const oglLilTex *getTexture(void) const {return c_tex;}
	virtual void render(double alpha,CkView *old);
};


/**
 * This trivial viewable always shows a fixed quadview, regardless of 
 *  the viewpoint--that is, it's just a single, fixed-texture polygon.
 */
class CkFixedViewable : public CkViewable {
	CkView *v;
public:
	/**
	 * Create a new CkFixedViewable showing this texture.
	 */
	CkFixedViewable(CkView *v_) :v(v_) {v->ref();}
	~CkFixedViewable() {v->unref();}
	
	// We never need to redraw:
	virtual bool shouldRender(const CkViewpoint &univ2screen,const CkView &oldView)
	{	
		return false;
	}
	// "rendering" just means returning our fixed CkView.
	virtual CkView *renderView(const CkViewpoint &univ2screen) {
		v->ref(); // semantics of return operation
		return v;
	}
};

/***************** Implementations based on Interest Points **************/

/**
 * Describes a set of "interest points": 3d locations in space
 * that lie on important object features.
 * These might be the corners of a bounding box, or the vertices
 * of our polygon.
 */
class CkInterestSet {
	enum {maxInterest=8};
	int nInterest; //Number of points below:
	CkVector3d loc[maxInterest]; //Locations of interest points

public:
	CkInterestSet(void) {nInterest=0;}
	CkInterestSet(int nInterest_,const CkVector3d *loc_);
	CkInterestSet(const CkBbox3d &box);
	
	int size(void) const {return nInterest;}
	const CkVector3d &operator[] (int i) const {return loc[i];}
	
	void setPoints(int n) {nInterest=n;}
	CkVector3d &operator[] (int i) {return loc[i];}
	
	CkBbox3d getBox(void) const {
		CkBbox3d box; box.empty();
		for (int i=0;i<nInterest;i++) box.add(loc[i]);
		return box;
	}
	
	/// Get our average location in space
	CkVector3d getMean(void) const;
	
	void pup(PUP::er &p);
};
PUPmarshall(CkInterestSet)


/// An interestView is a QuadView that keeps track of 
///  the 3d location and projection of 
///  some of its source object's points. 
///  This lets it evaluate reprojection error on the server.
/// 
///  It gets shippped across the network as a regular CkQuadView--
///  so it has no pup routine.
class CkInterestView : public CkQuadView {
	//These are the true universe locations of our interest points:
	CkInterestSet univ;
	//These are the texture-projected locations of the interest points:
	CkInterestSet proj;
	/// World location of texture center, X, and Y axes. 
	CkVector3d projC,projX,projY;

	//Return the squared projection error (pixels) of point i:
	double sqrError(int i,const CkViewpoint &univ2screen) const
	{
		return univ2screen.project_noz(univ[i]) .distSqr(
		       univ2screen.project_noz(proj[i]) );
	}
public:
	/// Initialize this view for these interest points,
	///   from this viewpoint.  Sets the corners initially
	///   as the z=0 projection of our texture's corners.
	CkInterestView(int w,int h,int n_colors,
		const CkInterestSet &univ_,
		const CkViewpoint &univ2texture);
	
	/// Return the ratio between our projected screen resolution
	///   and our current texture resolution.
	double resRatio(const CkViewpoint &univ2screen) const;
	
	/// Evaluate the root-mean-square error, in pixels, between the
	/// projection of our texture and the projection of the true object.
	double rmsError(const CkViewpoint &univ2screen) const;
	
	/// Evaluate the maximum error, in pixels, between the
	/// projection of our texture and the projection of the true object.
	double maxError(const CkViewpoint &univ2screen) const;

#define CMK_LIVEVIZ3D_INTERESTVIEWRENDER 0 /* need to be pup::able to render... */
#if CMK_LIVEVIZ3D_INTERESTVIEWRENDER
	virtual void render(double alpha);
#endif
};



/**
 * A CkInterestViewable is a viewable whose reprojection 
 * (view coherence) is characterized by a set of "interest points".
 * The interest points are used to determine which view to use,
 * and when views are out-of-date.
 */
class CkInterestViewable : public CkViewable {
	CkInterestSet interest; ///< Our 3D interest points
	CkVector3d center; ///< Our 3D "center point", through which our impostor plane must pass
	CkVector3d boundCenter; double boundRadius;
protected:
	/// Subclasses MUST call this from their constructors or pup routines.
	void setUnivPoints(const CkInterestSet &univPoints_);
	void setCenter(const CkVector3d &center_) {center=center_;}
public:
	/// Be sure to set interest points from your constructor.
	CkInterestViewable() {}
	
	/// Default implementation checks the rms reprojection error.
	virtual bool shouldRender(const CkViewpoint &univ2screen,const CkView &oldView);
	
	/**
	 * Uses newViewpoint and renderImage to build a new CkInterestView.
	 */
	virtual CkView *renderView(const CkViewpoint &univ2screen);
	
	/**
	 * Create a new texture viewpoint (univ2texture)
	 * for this new universe viewpoint (univ2screen).
	 * Default implementation just windows the universe viewpoint
	 * to fit the new texture.  Returns false if no viewpoint
	 * is possible or appropriate.
	 */
	virtual bool newViewpoint(const CkViewpoint &univ2screen,CkViewpoint &univ2texture);
	
	/**
	 * Subclass-overridden routine:
	 * Render yourself into this new, *garbage-filled* ARGB image.
	 * To change the image format, call dest.setLayout.
	 */
	virtual void renderImage(const CkViewpoint &vp,CkImage &dest) =0;
	
	/** Returns size of bounding sphere */
	virtual double getSize(const CkViewpoint &univ2screen);
};

#endif /*def(thisHeader)*/

/*
Describes a 3d object that can be "viewed"-- that is,
converted into a texture map.

Orion Sky Lawlor, olawlor@acm.org, 8/28/2002
*/

#ifndef __UIUC_PPL_CHARM_VIEWABLE_H
#define __UIUC_PPL_CHARM_VIEWABLE_H

#include "ckvector3d.h"
#include "ckimage.h"
#include "ckviewpoint.h"
#include "ckhashtable.h"



/// A CkViewableID uniquely identifies a CkViewable across processors.
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
	static unsigned int staticHash(const void *key,unsigned int keyLen) {
		return ((const CkViewableID *)key)->hash();
	}
	static int staticCompare(const void *a,const void *b,unsigned int keyLen) {
		return ((const CkViewableID *)a)->compare(*(const CkViewableID *)b);
	}
	
	void pup(PUP::er &p) {
		p(id,nId);
	}
};
PUPmarshall(CkViewableID);

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
		if (refCount<=0) delete this;
	}
};


/***************** Abstract Interface **************/

/**
 * A style for drawing this texture.
 */
class CkTexStyle {
public:
	/// These are the universe locations of the four 
	///  corners of our texture image.
	CkVector3d corners[4];
	
	void pup(PUP::er &p);
};

/**
 * A CkView is a texture image that can stand in for a CkViewable.
 * CkView's must always be allocated on the heap, and are often 
 * actually subclasses containing Viewable-specific data.
 */
class CkView : public CkReferenceCounted {
public:
	/// This is the viewable we describe
	CkViewableID id;
	
	/// This is the texture image of our CkViewable.
	CkImage *tex;
	
	/// This is how and where to draw the texture.
	CkTexStyle style;
	
	CkImage &getImage(void) {return *tex;}
	const CkImage &getImage(void) const {return *tex;}
	
	const CkVector3d &getCorner(int i) const {return style.corners[i];}
	
	const CkViewableID &getViewable(void) const {return id;}
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
	 * Add this (new?) view to our list.
	 * The incoming CkView will be ref()'d if it is kept.
	 */
	virtual void add(CkView *view) =0;
};

/**
 * An object that can be "viewed"-- turned into an image.
 */
class CkViewable {
public:
	virtual ~CkViewable();
	
	/**
	 * Pass an appropriate view of this object, from this viewpoint,
	 * to this consumer.  Normally calls dest.add at least once.
	 */
	virtual void view(const CkViewpoint &univ2screen,CkViewConsumer &dest) =0;
};

/***************** Tiny Implementations **************/
/**
 * This trivial viewable always shows a fixed image, regardless of 
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
	
	virtual void view(const CkViewpoint &univ2screen,CkViewConsumer &dest) {
		return dest.add(v);
	}
};

/**
 * A heap-allocated texture image.
 */
class CkAllocView : public CkView {
	CkAllocImage img;
public:
	/// Make a new texture image of this size.
	///  Be sure to set corners[0..3] after this call.
	CkAllocView(int w,int h)
		:img(w,h,4) { tex=&img; }
	CkAllocImage &getImage(void) {return img;}
	const CkAllocImage &getImage(void) const {return img;}
	
	CkAllocView() {tex=&img;}
	void pup(PUP::er &p);
};

void pup_pack(PUP::er &p,CkView &v);
CkAllocView *pup_unpack(PUP::er &p);

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
	
	/// Get our average location in space
	CkVector3d getMean(void) const;
	
	void pup(PUP::er &p);
};
PUPmarshall(CkInterestSet);


/// An interestView keeps track of the 3d location and projection of 
///  some of its source object's points.  This lets it evaluate reprojection 
///  error.
class CkInterestView : public CkAllocView {
	//These are the true universe locations of our interest points:
	CkInterestSet univ;
	//These are the texture-projected locations of the interest points:
	CkInterestSet proj;

	//Return the squared projection error (pixels) of point i:
	double sqrError(int i,const CkViewpoint &univ2screen) const
	{
		return univ2screen.project_noz(univ[i]) .distSqr(
		       univ2screen.project_noz(proj[i]) );
	}
public:
	/// Initialize this view for these interest points,
	///   from this viewpoint.
	CkInterestView(int w,int h,const CkInterestSet &univ_,
		const CkViewpoint &univ2texture);
	
	/// Evaluate the root-mean-square error, in pixels, between the
	/// projection of our texture and the projection of the true object.
	double rmsError(const CkViewpoint &univ2screen) const;
	
	/// Evaluate the maximum error, in pixels, between the
	/// projection of our texture and the projection of the true object.
	double maxError(const CkViewpoint &univ2screen) const;
	
	void pup(PUP::er &p);
};



/**
 * A CkInterestViewable is a viewable whose reprojection 
 * (view coherence) is characterized by a set of "interest points".
 * The interest points are used to determine which view to use,
 * and when views are out-of-date.
 */
class CkInterestViewable : public CkViewable {
	CkInterestView *last; // Previous (set of) views
	void flushLast(void) {
		if (last!=NULL) {
			last->unref();
			last=NULL;
		}
	}
	CkViewableID id; /// Our unique, cross-processor ID
	CkInterestSet interest; /// Our 3D interest points
	CkVector3d center; /// Our 3D center point
protected:
	void setViewableID(const CkViewableID &id_) {id=id_;}
	void setUnivPoints(const CkInterestSet &univPoints_) {
		interest=univPoints_;
		center=interest.getMean();
	}
public:
	/// Be sure to set id and interest from your constructor.
	CkInterestViewable();
	~CkInterestViewable();
	
	/**
	 * Search for an appropriate existing view for this viewpoint.
	 * If none exists, render a new view.
	 */
	virtual void view(const CkViewpoint &univ2screen,CkViewConsumer &dest);
	
	/**
	 * Return true if this view is appropriate for use from this viewpoint.
	 * Default implementation checks the rms reprojection error.
	 */
	virtual bool goodView(const CkViewpoint &univ2screen,CkInterestView *testView);
	
	/**
	 * Create a new texture viewpoint for this new universe viewpoint.
	 * Default implementation just windows the universe viewpoint
	 * to fit the new texture.
	 */
	virtual CkViewpoint newViewpoint(const CkViewpoint &univ2screen);
	
	
	/**
	 * Subclass-overridden routine:
	 * Render yourself into this new RGBA image.
	 */
	virtual void render(const CkViewpoint &vp,CkImage &dest) =0;
};





#endif /*def(thisHeader)*/

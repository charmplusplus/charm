/*
Describes a 3d object that can be "viewed"-- that is,
converted into a texture map.

Orion Sky Lawlor, olawlor@acm.org, 8/28/2002
*/

#ifndef __UIUC_PPL_CHARM_VIEWABLE_H
#define __UIUC_PPL_CHARM_VIEWABLE_H

#include "ckvector3d.h"
#include "ckimage.h"
#include "viewpoint.h"

/*
Describes a set of "interest points": 3d locations in space
 that lie on interesting features.
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
	
	void pup(PUP::er &p);
};
PUPmarshall(CkInterestSet);


//Uniquely identifies a CkViewable across processors
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

/*
An object that can be "viewed"-- turned into an image.
*/
class CkViewable {
public:
	virtual ~CkViewable();
	
	/*
	 Get our unique, cross-processor ID
	 */
	virtual const CkViewableID &getViewableID(void) =0;
	
	/*
	 Get our universe-coordinates "interest points"-- points to use
	 when computing whether the image cache is still valid.
	*/
	virtual const CkInterestSet &getInterestPoints(void) =0;
	
	/*
	  Render yourself into a colors==4 image:
	 */
	virtual void view(const CkViewpoint &vp,CkImage &dest) =0;
};


//A view is an image of a viewable from a particular direction:
class CkView {
	//This may or may not be NULL (depending on if we've been flushed)
	CkAllocImage *img;
	
	//These are the universe locations of the four corners of our texture:
	CkVector3d corners[4];
	
	//These are the true universe locations of our interest points:
	CkInterestSet univ;
	//These are the texture-projected locations of the interest points:
	CkInterestSet proj;
	
	//This identifies our viewable
	CkViewableID viewable;

	//Return the squared projection error (pixels) of point i:
	double sqrError(int i,const CkViewpoint &univ2screen) const
	{
		return univ2screen.project_noz(univ[i]) .distSqr(
		       univ2screen.project_noz(proj[i]) );
	}
	
	//Back-project our 4 corners and projected points from this viewpoint:
	void setTexView(const CkViewpoint &univ2texture);
	
	CkView(const CkView &v); //DO NOT USE
	void operator=(const CkView &v);
public:
	/*Most useful constructor: view this object from this viewpoint*/
	CkView(const CkViewpoint &univ2screen, //Source viewpoint 
		CkViewable *viewThis); //Object to render
	
	/*Less useful constructor (for testing):
	Choose a texture viewpoint for rendering these interest points
	as seen from this viewpoint.
	*/
	CkView(const CkInterestSet &univ_,
		const CkViewpoint &univ2screen, //Source viewpoint 
		CkViewpoint &univ2texture);

	//Completely general (and useless) constructor:
	CkView(const CkInterestSet &univ_,const CkInterestSet &proj_,
		const CkVector3d *corners_,CkAllocImage *img_);
	
	~CkView() {delete img;}
	
	void setViewable(const CkViewableID &id) {viewable=id;}
	const CkViewableID &getViewable(void) const {return viewable;}
	CkImage &getImage(void) {return *img;}
	const CkVector3d &getCorner(int c) const {return corners[c];}
	
	//Evaluate the root-mean-square error, in pixels, between the
	// projection of our texture and the projection of the true object.
	double rmsError(const CkViewpoint &univ2screen) const;
	double maxError(const CkViewpoint &univ2screen) const;
	
	//Destructive operation: get rid of our image
	//  (e.g., since it's been converted to OpenGL)
	void flushImage(void) {
		delete img; img=NULL;
	}
	
	CkView();
	void pup(PUP::er &p);
};
PUPmarshall(CkView);



#endif /*def(thisHeader)*/

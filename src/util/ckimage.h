/*
2D flat image class:
This class represents a 2D raster image; a rectangular 2D
array of pixels.

Orion Sky Lawlor, olawlor@acm.org, 5/15/2002
*/
#ifndef __CK_IMAGE_H
#define __CK_IMAGE_H

#include "pup.h"

#undef min
#undef max
inline int min(int a,int b) {return (a<b)?a:b;}
inline int max(int a,int b) {return (a>b)?a:b;}
class CkRect {
public:
	int l,r; //X boundaries of rectangle
	int t,b; //Y boundaries of rectangle
	CkRect() {l=r=t=b=-1;}
	CkRect(int l_,int t_,int r_,int b_) 
		:l(l_), r(r_), t(t_), b(b_) {}
	CkRect(int w,int h)
		:l(0), r(w), t(0), b(h) {}
	//Default copy constructor, assignment operator
	int wid(void) const {return r-l;}
	int ht(void) const {return b-t;}
	int getWidth(void) const {return r-l;}
	int getHeight(void) const {return b-t;}
	inline int operator==(const CkRect &a) 
		{return l==a.l && r==a.r && t==a.t && b==a.b;}
	CkRect getUnion(const CkRect &a) {
		return CkRect(min(l,a.l),min(t,a.t), max(r,a.r),max(b,a.b));
	}
	CkRect getIntersect(const CkRect &a) {
		return CkRect(max(l,a.l),max(t,a.t), min(r,a.r),min(b,a.b));
	}
	CkRect getShift(int dx,int dy) {
		return CkRect(l+dx,t+dy,r+dx,b+dy);
	}
	bool isEmpty(void) const {return ((l>=r) || (t>=b));}
	bool inbounds(int x,int y) const {
		if (x<l || x>=r) return false;
		if (y<t || y>=b) return false;
		return true;
	}
	void makeEmpty(void) {l=t=1000000000; b=r=-1000000000;}
	void empty(void) {makeEmpty();}
	void add(int x,int y) {
		l=min(x,l); r=max(x,r);
		t=min(y,t); b=max(y,b);
	}
	void enlarge(int dx,int dy) {
		l-=dx; r+=dx; t-=dy; b+=dy;
	}
	void zero(void) {l=r=t=b=0;}
	int area(void) const {return (r-l)*(b-t);}
	
	void pup(PUP::er &p) {
		p|l; p|r; p|t; p|b;
	}
};
PUPmarshall(CkRect)

/**
This class describes an image, represented as a flat byte array.
Pixels are stored first by color (e.g., r,g,b), then by row in the usual
raster order.  
*/
class CkImage {
public:
	//This is the data type of a color channel, such as the red channel.
	typedef unsigned char channel_t;
	/// This is the maximum value of a color channel
	enum {channel_max=255};
	
	/// This describes the various data layouts used by image pixels:
	typedef enum {
		/**
		  The default layout: ARGB.
		    With one color, pure luminance.
		    With 3 colors, [0]=R, [1]=G, [2]=B.
		    With 4 colors, [0]=A, [1]=R, [2]=G, [3]=B.
		*/
		layout_default=0,
		/**
		  The "reversed" layout: BGRA.
		    With one color, pure luminance.
		    With 3 colors, [0]=B, [1]=G, [2]=R.
		    With 4 colors, [0]=B, [1]=G, [2]=R, [3]=A.
		*/
		layout_reversed=1
	} layout_t;
private:
	int row,colors; ///< channel_ts per line, channel_ts per pixel
	int layout; ///< Image pixel format.
	int wid,ht; ///< Image size: cols and rows
	channel_t *data; ///< Image pixel data
	
	CkImage(const CkImage &im) ; ///< DO NOT USE
	void operator=(const CkImage &im);
public:
	CkImage() {row=colors=wid=ht=-1; setLayout(layout_default); data=NULL;}
	CkImage(int w_,int h_,int colors_,channel_t *data_)
		:row(w_*colors_), colors(colors_),
		wid(w_), ht(h_), data(data_) { setLayout(layout_default); }
	
	/// Get/set the whole image's data
	channel_t *getData(void) {return data;}
	void setData(channel_t *d) {data=d;}
	
	CkRect getRect(void) const {return CkRect(0,0,wid,ht);}
	/// Return the number of channel_t's per row of the image
	int getRow(void) const {return row;}
	/// Return the number of colors (channel_t's) per pixel
	int getColors(void) const {return colors;}
	
	/// Get/set the pixel format.
	layout_t getLayout(void) const {return (layout_t)layout;}
	void setLayout(layout_t a) {layout=(layout_t)a;}
	
	/// Return the number of pixels per row of the image
	int getWidth(void) const {return wid;}
	/// Return the number of pixels per column of the image
	int getHeight(void) const {return ht;}
	
	//Copy the pixel at src onto the one at dest
	inline void copyPixel(const channel_t *src,channel_t *dest) {
		for (int i=0;i<colors;i++)
			dest[i]=src[i];
	}
	//Set this pixel to this value
	inline void setPixel(const channel_t src,channel_t *dest) {
		for (int i=0;i<colors;i++)
			dest[i]=src;
	}
	//Add the pixel at src to the one at dest, ignoring overflow
	inline void addPixel(const channel_t *src,channel_t *dest) {
		for (int i=0;i<colors;i++)
			dest[i]+=src[i];
	}
	//Add the pixel at src to the one at dest, clipping instead of overflowing
	inline void addPixelClip(const channel_t *src,channel_t *dest,
		const channel_t *clip) 
	{
		for (int i=0;i<colors;i++)
			dest[i]=clip[(int)dest[i]+(int)src[i]];
	}
	
	
	//Get a pixel
	inline channel_t *getPixel(int x,int y) {return data+x*colors+y*row;}
	inline const channel_t *getPixel(int x,int y) const {return data+x*colors+y*row;}
	
	
	/*
	 Clip out this subregion of this image-- make us a subregion
	 */
	void window(const CkRect &src) {
		data+=src.t*row+src.l*colors;
		wid=src.wid(); ht=src.ht();
	}
	
	/*
	Zero out this image-- make it all black.
	*/
	void clear(void);
	
	/*
	 Copy all of src onto this image starting at (x,y).
	 */
	void put(int sx,int sy,const CkImage &src); 
	
	/*
	 Add all of src onto this image starting at (x,y).
	 */
	void add(int sx,int sy,const CkImage &src);
	/*
	 Add all of src onto this image starting at (x,y), clipping
         values instead of overflowing.
	 */
	void addClip(int sx,int sy,const CkImage &src,const channel_t *clip);
	
	//Allocate clipping array for above routine
	static channel_t *newClip(void);
	
	//Pup only the image *size*, not the image *data*.
	void pup(PUP::er &p) {
		p|wid; p|ht; p|colors; p|layout; p|row;
	}
};
PUPmarshall(CkImage)


//A heap-allocated image
class CkAllocImage : public CkImage {
	channel_t *allocData;
public:
	CkAllocImage() {allocData=NULL;}
	CkAllocImage(int w,int h,int c)
		:CkImage(w,h,c,new channel_t[w*h*c]) 
	{
		allocData=getData();
	}
	~CkAllocImage() {delete[] allocData;}
	
	// Allocate the image with its current size.
	void allocate(void) {
		int len=getRect().area()*getColors();
		allocData=new channel_t[len];
		setData(allocData);
	}
	
	// Deallocate the image data (does not change size).
	void deallocate(void) {
		delete[] allocData; allocData=0;
		setData(allocData);
	}
	
	//Pup both image size as well as image data.
	void pup(PUP::er &p);
};
PUPmarshall(CkAllocImage)


#endif


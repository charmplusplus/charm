/*
2D flat image class:
This class represents a 2D raster image; a rectangular 2D
array of pixels.

Orion Sky Lawlor, olawlor@acm.org, 5/15/2002
*/
#ifndef __CK_IMAGE_H
#define __CK_IMAGE_H

#undef min
#undef max
inline int min(int a,int b) {return (a<b)?a:b;}
inline int max(int a,int b) {return (a>b)?a:b;}
class Rect {
public:
	int l,r; //X boundaries of rectangle
	int t,b; //Y boundaries of rectangle
	Rect() {}
	Rect(int l_,int t_,int r_,int b_) 
		:l(l_), r(r_), t(t_), b(b_) {}
	Rect(int w,int h)
		:l(0), r(w), t(0), b(h) {}
	int wid(void) const {return r-l;}
	int ht(void) const {return b-t;}
	inline int operator==(const Rect &a) 
		{return l==a.l && r==a.r && t==a.t && b==a.b;}
	Rect getUnion(const Rect &a) {
		return Rect(min(l,a.l),min(t,a.t), max(r,a.r),max(b,a.b));
	}
	Rect getIntersect(const Rect &a) {
		return Rect(max(l,a.l),max(t,a.t), min(r,a.r),min(b,a.b));
	}
	Rect getShift(int dx,int dy) {
		return Rect(l+dx,t+dy,r+dx,b+dy);
	}
	bool isEmpty(void) const {return (l>=r) || (t>=b);}
	void makeEmpty(void) {l=t=1000000000; b=r=-1000000000;}
	void zero(void) {l=r=t=b=0;}
	int area(void) const {return (r-l)*(b-t);}
};

class Image {
public:
	typedef unsigned char pixel_t;
private:
	int row,bpp; //pixel_ts per line, pixel_ts per pixel
	int wid,ht; //Image size: cols and rows
	pixel_t *data; //Image pixel data
	
public:
	
	Image(int w_,int h_,int bpp_,pixel_t *data_)
		:row(w_*bpp_), bpp(bpp_), wid(w_), ht(h_), data(data_) {}
	
	pixel_t *getData(void) {return data;}
	
	//Copy the pixel at src onto the one at dest
	inline void copyPixel(const pixel_t *src,pixel_t *dest) {
		for (int i=0;i<bpp;i++)
			dest[i]=src[i];
	}
	//Add the pixel at src to the one at dest, ignoring overflow
	inline void addPixel(const pixel_t *src,pixel_t *dest) {
		for (int i=0;i<bpp;i++)
			dest[i]+=src[i];
	}
	//Add the pixel at src to the one at dest, clipping instead of overflowing
	inline void addPixelClip(const pixel_t *src,pixel_t *dest,
		const pixel_t *clip) 
	{
		for (int i=0;i<bpp;i++)
			dest[i]=clip[(int)dest[i]+(int)src[i]];
	}
	
	
	//Get a pixel
	inline pixel_t *getPixel(int x,int y) {return data+x*bpp+y*row;}
	inline const pixel_t *getPixel(int x,int y) const {return data+x*bpp+y*row;}
	
	
	/*
	 Clip out this subregion of this image-- make us a subregion
	 */
	void window(const Rect &src) {
		data+=src.t*row+src.l*bpp;
		wid=src.wid(); ht=src.ht();
	}
	
	/*
	Zero out this image-- make it all black.
	*/
	void clear(void) {
		for (int y=0;y<ht;y++)
		for (int x=0;x<wid;x++)
		for (int i=0;i<bpp;i++)
			data[x*bpp+y*row+i]=(pixel_t)0;
	}
	
	/*
	 Copy all of src onto this image starting at (x,y).
	 */
	void put(int sx,int sy,const Image &src) {
		for (int y=0;y<src.ht;y++)
		for (int x=0;x<src.wid;x++)
			copyPixel(src.getPixel(x,y),getPixel(x+sx,y+sy));
	}
	/*
	 Add all of src onto this image starting at (x,y).
	 */
	void add(int sx,int sy,const Image &src) {
		for (int y=0;y<src.ht;y++)
		for (int x=0;x<src.wid;x++)
			addPixel(src.getPixel(x,y),getPixel(x+sx,y+sy));
	}
	/*
	 Add all of src onto this image starting at (x,y), clipping
         values instead of overflowing.
	 */
	void addClip(int sx,int sy,const Image &src,const pixel_t *clip) {
		for (int y=0;y<src.ht;y++)
		for (int x=0;x<src.wid;x++)
			addPixelClip(src.getPixel(x,y),getPixel(x+sx,y+sy),clip);
	}
};

//A heap-allocated image
class AllocImage : public Image {
	pixel_t *allocData;
public:
	AllocImage(int w,int h,int b)
		:Image(w,h,b,new pixel_t[w*h*b]) 
	{
		allocData=getData();
	}
	~AllocImage() {delete[] allocData;}
};


#endif


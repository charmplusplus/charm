/*
2D flat image class:
This class represents a 2D raster image; a rectangular 2D
array of pixels.

Orion Sky Lawlor, olawlor@acm.org, 5/15/2002
*/
#include "ckimage.h"

/*
Zero out this image-- make it all black.
*/
void CkImage::clear(void) {
	for (int y=0;y<ht;y++)
	for (int x=0;x<wid;x++)
	for (int i=0;i<colors;i++)
		data[x*colors+y*row+i]=(pixel_t)0;
}

	/*
	 Copy all of src onto this image starting at (x,y).
	 */
	void CkImage::put(int sx,int sy,const CkImage &src) {
		for (int y=0;y<src.ht;y++)
		for (int x=0;x<src.wid;x++)
			copyPixel(src.getPixel(x,y),getPixel(x+sx,y+sy));
	}
	
	/*
	 Add all of src onto this image starting at (x,y).
	 */
	void CkImage::add(int sx,int sy,const CkImage &src) {
		for (int y=0;y<src.ht;y++)
		for (int x=0;x<src.wid;x++)
			addPixel(src.getPixel(x,y),getPixel(x+sx,y+sy));
	}
	
	/*
	 Add all of src onto this image starting at (x,y), clipping
         values instead of overflowing.
	 */
	void CkImage::addClip(int sx,int sy,const CkImage &src,const pixel_t *clip) {
		for (int y=0;y<src.ht;y++)
		for (int x=0;x<src.wid;x++)
			addPixelClip(src.getPixel(x,y),getPixel(x+sx,y+sy),clip);
	}
	
	/*
	Allocate clipping array for above routine.
	This array has 512 entries-- it's used to clip off large values
	when summing bytes (like image values) together.  On a machine with
	a small cache, it may be better to use an "if" instead of this table.
	*/
	CkImage::pixel_t *CkImage::newClip(void){
		const int pixelMax=255;
		const int tableLen=2*pixelMax;
		pixel_t *ret=new pixel_t[tableLen];
		int i;
		for (i=0;i<pixelMax;i++) ret[i]=(pixel_t)i;
		for (i=pixelMax;i<tableLen;i++) ret[i]=(pixel_t)pixelMax;
		return ret;
	}

//Pup both image size as well as image data.
void CkAllocImage::pup(PUP::er &p) {
	CkImage::pup(p);
	int len=getRect().area()*getColors();
	if (p.isUnpacking()) {
		allocData=new pixel_t[len];
		setData(allocData);
	}
	p(allocData,len);
}

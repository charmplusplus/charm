#ifndef _IMAGE
#define _IMAGE

#include "ckimage.h"
#include "defines.h"

class Image
{
	public:
		Point m_ulc; // upper-left coordinate
		Point m_lrc; // lower-right coordinate
		byte * m_imgData; // image buffer

		// Constructor Methods
		Image(Point ulc, Point lrc, byte * imgData);
		Image(Rect rect, byte * imgData);
		Image();

		// Destructor Method
		~Image();

		// Packing Helper Methods
		unsigned getImageSize(unsigned bytesPerPixel);
		/*
			Copy image data to the input buffer.
		*/
		void copyImageData(byte *ptr, unsigned bytesPerPixel);

		// Image Overlap Resolution Methods
		bool isOverlapping(Image const &img);
		void getIntersectionCoordinates(Image const& img, Point &ulc, Point &lrc);

		/*
			This function resolves the conflict between the itself and the image passed as parameter.
			It deletes the image passed in as parameter and modifies itself. This merge can result
			in 1 or 2 or 3 images being created. Return value can be NULL or a new image.
		*/
		Image* mergeImages(Image *&img2, unsigned bytesPerPixel);

		/*
			This function is used by mergeImages() to partition the overlapping images into non-
			-overlapping images.
		*/
		Image* partitionImage(Image *&img2, const Point &int_ulc, const Point &int_lrc, unsigned bytesPerPixel);

		/*
			This function is used by mergeImages() to add the intensities of pixels in the input image
			to proper pixels within itself. This is used when one image is completely within boundary
			of another image.
		*/
		void addImageData(Image *img, unsigned bytesPerPixel);

		// Access Methods
		inline unsigned getImageWidth(void);
		inline unsigned getImageHeight(void);

};

unsigned Image::getImageWidth(void)
{
	return (m_lrc.x - m_ulc.x + 1);
}

unsigned Image::getImageHeight(void)
{
	return (m_lrc.y - m_ulc.y + 1);
}
#endif

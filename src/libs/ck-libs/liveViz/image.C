#include "image.h"
#include "stdlib.h"
#include "iostream.h"

// Constructor Methods
Image::Image(Point ulc, Point lrc, byte * imgData)
{
	m_ulc.x = ulc.x;
	m_lrc.x = lrc.x;
	m_ulc.y = ulc.y;
	m_lrc.y = lrc.y;
	m_imgData = imgData;
}

Image::Image(Rect rect, byte * imgData)
{
        m_ulc.x = rect.l;
        m_lrc.x = rect.r;
        m_ulc.y = rect.t;
        m_lrc.y = rect.b;
        m_imgData = imgData;
}


Image::Image()
{
	m_ulc.x = 0;
	m_lrc.x = 0;
	m_ulc.y = 0;
	m_lrc.y = 0;
	m_imgData = NULL;
}

// Destructor Method
Image::~Image()
{
	if(m_imgData != NULL)
		delete[] m_imgData;
	m_imgData = NULL;
}

// Packing Helper Methods
unsigned Image::getImageSize(unsigned bytesPerPixel)
{
	return((m_lrc.x - m_ulc.x + 1)*(m_lrc.y - m_ulc.y + 1)*bytesPerPixel);
}

void Image::copyImageData(byte *ptr, unsigned bytesPerPixel)
{
	for(int i=0; i<getImageSize(bytesPerPixel); i++)
		ptr[i] = m_imgData[i];
}

// Image Overlap Resolution Methods
bool Image::isOverlapping(Image const &img)
{
	if((img.m_ulc.x > m_lrc.x) || (img.m_lrc.x < m_ulc.x) ||
		(img.m_lrc.y < m_ulc.y) || (img.m_ulc.y > m_lrc.y))
		return false;
	else
		return true;
}

void Image::getIntersectionCoordinates(Image const& img, Point &ulc, Point &lrc)
{
	Point ulc1,lrc1,ulc2,lrc2;
	ulc1.x = m_ulc.x;
	lrc1.x = m_lrc.x;
	ulc1.y = m_ulc.y;
	lrc1.y = m_lrc.y;

	ulc2.x = img.m_ulc.x;
	lrc2.x = img.m_lrc.x;
	ulc2.y = img.m_ulc.y;
	lrc2.y = img.m_lrc.y;

	if((ulc1.x<=ulc2.x)&&(ulc2.x<=lrc1.x)&&
		(ulc1.y<=ulc2.y)&&(ulc2.y<=lrc1.y))
	{
		ulc.x=ulc2.x;
		ulc.y=ulc2.y;

		if(lrc2.y<lrc1.y)
			lrc.y = lrc2.y;
		else
			lrc.y = lrc1.y;

		if(lrc2.x < lrc1.x)
			lrc.x = lrc2.x;
		else
			lrc.x = lrc1.x;
	}

	if((ulc1.x<=ulc2.x)&&(ulc2.x<=lrc1.x)&&
		(ulc2.y<=ulc1.y)&&(ulc1.y<=lrc2.y))
	{
		ulc.x = ulc2.x;
		lrc.y = lrc2.y;
		ulc.y = ulc1.y;

		if(lrc2.x < lrc1.x)
			lrc.x = lrc2.x;
		else
			lrc.x = lrc1.x;
	}

	if((ulc2.x<=ulc1.x)&&(ulc1.x<=lrc2.x)&&
		(ulc2.y<=ulc1.y)&&(ulc1.y<=lrc2.y))
	{
		ulc.x = ulc1.x;
		ulc.y = ulc1.y;

		if(lrc1.y < lrc2.y)
			lrc.y = lrc1.y;
		else
			lrc.y = lrc2.y;

		if(lrc1.x < lrc2.x)
			lrc.x = lrc1.x;
		else
			lrc.x = lrc2.x;
	}

	if((ulc2.x<=ulc1.x)&&(ulc1.x<=lrc2.x)&&
		(ulc1.y<=ulc2.y)&&(ulc2.y<=lrc1.y))
	{
		ulc.x = ulc1.x;
		ulc.y = ulc2.y;
		lrc.y = lrc1.y;

		if(lrc1.x < lrc2.x)
			lrc.x = lrc1.x;
		else
			lrc.x = lrc2.x;
	}

	if((ulc2.x<=ulc1.x)&&(ulc1.x<=lrc2.x)&&
		(ulc1.y<=ulc2.y)&&(lrc2.y<=lrc1.y))
	{
		ulc.x = ulc1.x;
		ulc.y = ulc2.y;
		lrc.x = lrc2.x;
		lrc.y = lrc2.y;
	}

	if((ulc1.x<=ulc2.x)&&(ulc2.x<=lrc1.x)&&
		(ulc2.y<=ulc1.y)&&(lrc1.y<=lrc2.y))
	{
		ulc.x = ulc2.x;
		ulc.y = ulc1.y;
		lrc.x = lrc1.x;
		lrc.y = lrc1.y;
	}

	if((ulc2.x<=ulc1.x)&&(lrc2.x>=lrc1.x)&&
		(ulc1.y<=ulc2.y)&&(lrc2.y<=lrc1.y))
	{
		ulc.x = ulc1.x;
		ulc.y = ulc2.y;
		lrc.x = lrc1.x;
		lrc.y = lrc2.y;
	}

	if((ulc1.x<=ulc2.x)&&(lrc2.x<=lrc1.x)&&
		(ulc2.y<=ulc1.y)&&(lrc2.y>=lrc1.y))
	{
		ulc.x = ulc2.x;
		ulc.y = ulc1.y;
		lrc.x = lrc2.x;
		lrc.y = lrc1.y;
	}
}

/* This function deletes all the input image data and image. It allocates new
buffer and allocates memory to Image class object and returns the newly
allocated object*/
Image* Image::mergeImages(Image *&img2, unsigned bytesPerPixel)
{
	Point ulc,lrc;
	Point ulc1,lrc1,ulc2,lrc2;

	ulc1.x = m_ulc.x;
	lrc1.x = m_lrc.x;
	ulc1.y = m_ulc.y;
	lrc1.y = m_lrc.y;

	ulc2.x = img2->m_ulc.x;
	lrc2.x = img2->m_lrc.x;
	ulc2.y = img2->m_ulc.y;
	lrc2.y = img2->m_lrc.y;

	getIntersectionCoordinates(*img2, ulc, lrc);

	if((ulc1.x <= ulc2.x)&&(ulc1.y <= ulc2.y )&&(lrc1.x >= lrc2.x)&&(lrc1.y >= lrc2.y))
	{
		// image1 contains image2
		addImageData(img2, bytesPerPixel);
		delete img2; // free the input image
		img2 = NULL;
		return NULL;
	}
	else
		if((ulc2.x <= ulc1.x)&&(ulc2.y <= ulc1.y )&&(lrc2.x >= lrc1.x)&&(lrc2.y >= lrc1.y))
		{
			// image2 contains image1
			img2->addImageData(this, bytesPerPixel);
			byte* temp = m_imgData;
			m_ulc.x = ulc2.x;
			m_ulc.y = ulc2.y;
			m_lrc.x = lrc2.x;
			m_lrc.y = lrc2.y;
			m_imgData = img2->m_imgData;
			img2->m_imgData = temp;
			delete img2; // free image1
			img2 = NULL;
			return NULL;
		}
		else
			return partitionImage(img2, ulc, lrc, bytesPerPixel);
}

void Image::addImageData(Image *img, unsigned bytesPerPixel)
{
	Point ulc1,lrc1,ulc2,lrc2;

	ulc1.x = m_ulc.x;
	lrc1.x = m_lrc.x;
	ulc1.y = m_ulc.y;
	lrc1.y = m_lrc.y;

	ulc2.x = img->m_ulc.x;
	lrc2.x = img->m_lrc.x;
	ulc2.y = img->m_ulc.y;
	lrc2.y = img->m_lrc.y;

	byte * oldImage1Data = m_imgData;
	byte * oldImage2Data = img->m_imgData;

	int index1, index2;
	int width1, width2;
	width1 = getImageWidth();
	width2 = img->getImageWidth();

	for(int y=ulc2.y; y<=lrc2.y; y++)
		for(int x=ulc2.x; x<=lrc2.x; x++)
			for(int j=0; j<bytesPerPixel; j++)
			{
				index1 = ((y-ulc1.y)*width1 + (x-ulc1.x))*bytesPerPixel + j;
				index2 = ((y-ulc2.y)*width2 + (x-ulc2.x))*bytesPerPixel + j;
				if(oldImage1Data[index1] + oldImage2Data[index2] > 255)
					oldImage1Data[index1] = 255;
				else
					oldImage1Data[index1] += oldImage2Data[index2];
			}
}

Image* Image::partitionImage(Image *&img2, const Point &int_ulc, const Point& int_lrc, unsigned bytesPerPixel)
{
	Image *oldImage1;
	Image *oldImage2;

	if(m_ulc.y < int_ulc.y)
	{
		oldImage1 = this;
		oldImage2 = img2;
	}
	else
	{
		oldImage1 = img2;
		oldImage2 = this;
	}

	Point ulc1,lrc1,ulc2,lrc2;
	Point newImage_ulc1, newImage_ulc2, newImage_ulc3, newImage_lrc1, newImage_lrc2, newImage_lrc3;

	Image *image1=NULL, *image2=NULL, *image3=NULL;

	byte * oldImage1Data = oldImage1->m_imgData;
	byte * oldImage2Data = oldImage2->m_imgData;

	ulc1.x = oldImage1->m_ulc.x;
	lrc1.x = oldImage1->m_lrc.x;
	ulc1.y = oldImage1->m_ulc.y;
	lrc1.y = oldImage1->m_lrc.y;

	ulc2.x = oldImage2->m_ulc.x;
	lrc2.x = oldImage2->m_lrc.x;
	ulc2.y = oldImage2->m_ulc.y;
	lrc2.y = oldImage2->m_lrc.y;

	// image 1 coordinates and image
	if(ulc1.y != ulc2.y)
	{
		newImage_ulc1.x = ulc1.x;
		newImage_ulc1.y = ulc1.y;
		newImage_lrc1.x = lrc1.x;
		newImage_lrc1.y = int_ulc.y - 1;

		image1 = new Image(newImage_ulc1,newImage_lrc1, NULL);

		byte *image1Data = new byte[image1->getImageSize(bytesPerPixel)];

		for(unsigned i=0; i<image1->getImageSize(bytesPerPixel);i++)
			image1Data[i] = oldImage1Data[i];
		image1->m_imgData = image1Data;
	}

	// image 2 containg data for the overlapping images
	newImage_ulc2.y = int_ulc.y;
	newImage_lrc2.y = int_lrc.y;

	if(int_ulc.x > ulc2.x)
		newImage_ulc2.x = ulc2.x;
	else
		if(int_ulc.x > ulc1.x)
			newImage_ulc2.x = ulc1.x;
		else
			newImage_ulc2.x = int_ulc.x;

	if(int_lrc.x < lrc2.x)
		newImage_lrc2.x = lrc2.x;
	else
		if(int_lrc.x < lrc1.x)
			newImage_lrc2.x = lrc1.x;
		else
			newImage_lrc2.x = int_lrc.x;

	image2 = new Image(newImage_ulc2, newImage_lrc2, NULL);

	byte * image2Data = new byte[image2->getImageSize(bytesPerPixel)];

	for(int i=0; i<image2->getImageSize(bytesPerPixel); i++)
		image2Data[i] = 0;

	int width = (newImage_lrc2.x - newImage_ulc2.x + 1);
	for(unsigned y=newImage_ulc2.y; y<=newImage_lrc2.y; y++)
	{
		int startx = 0;
		int image1Width = oldImage1->getImageWidth();

		if(newImage_ulc2.x < ulc1.x)
			startx = (int_ulc.x - newImage_ulc2.x);

		for(int x=0; x<image1Width; x++)
			for(int j=0; j<bytesPerPixel; j++)
				image2Data[((y-newImage_ulc2.y)*width+x+startx)*bytesPerPixel + j] += oldImage1Data[((y-ulc1.y)*image1Width+x)*bytesPerPixel + j];

		startx = 0;
		int image2Width = oldImage2->getImageWidth();

		if(newImage_ulc2.x < ulc2.x)
			startx = (int_ulc.x - newImage_ulc2.x);

		for(int x=0; x<image2Width; x++)
			for(int j=0; j<bytesPerPixel; j++)
				image2Data[((y-newImage_ulc2.y)*width+x+startx)*bytesPerPixel + j] += oldImage2Data[((y-ulc2.y)*image2Width+x)*bytesPerPixel + j];
 	}
	image2->m_imgData = image2Data;

	// image 3 containing remaining image
	if(lrc2.y != lrc1.y)
	{
		newImage_ulc3.y = int_lrc.y+1;
		if(lrc2.y == int_lrc.y)
		{
			newImage_lrc3.y = lrc1.y;
			newImage_ulc3.x = ulc1.x;
			newImage_lrc3.x = lrc1.x;
			image3 = new Image(newImage_ulc3, newImage_lrc3, NULL);

			byte * image3Data = new byte[image3->getImageSize(bytesPerPixel)];
			unsigned index = oldImage1->getImageSize(bytesPerPixel)-1;
			for(int i=image3->getImageSize(bytesPerPixel)-1;i>=0;i--)
			{
				image3Data[i] = oldImage1Data[index];
				index--;
			}
			image3->m_imgData = image3Data;
		}
		else
		{
			newImage_lrc3.y = lrc2.y;
			newImage_lrc3.x = lrc2.x;
			newImage_ulc3.x = ulc2.x;
			image3 = new Image(newImage_ulc3, newImage_lrc3, NULL);

			byte * image3Data = new byte[image3->getImageSize(bytesPerPixel)];
			unsigned index = oldImage2->getImageSize(bytesPerPixel)-1;
			for(int i=image3->getImageSize(bytesPerPixel)-1;i>=0;i--)
			{
				image3Data[i] = oldImage2Data[index];
				index--;
			}
			image3->m_imgData = image3Data;
		}
	}

	if(image1 == NULL)
	{
		if(image2 != NULL)
		{
			image1 = image2;
			image2 = image3;
			image3 = NULL;
		}
		else
		{
			image1 = image3;
			image3 = NULL;
		}
	}

	m_ulc.x = image1->m_ulc.x;
	m_lrc.x = image1->m_lrc.x;
	m_ulc.y = image1->m_ulc.y;
	m_lrc.y = image1->m_lrc.y;

	byte* temp = m_imgData;
	m_imgData = image1->m_imgData;
	image1->m_imgData = temp; //m_imgData;
	delete image1;
	delete img2;
	img2 = image2;

	return image3;
}

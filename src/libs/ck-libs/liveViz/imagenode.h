#ifndef _IMAGENODE
#define _IMAGENODE

#include "image.h"

class ImageNode
{
	public:
		Image *m_img;
		ImageNode *m_next;

		ImageNode();
		ImageNode(Image const& img);
		ImageNode(Image const& img, ImageNode *next);
		ImageNode(Point const& ulc, Point const& lrc, byte *imgData);

		~ImageNode();
};

#endif

#ifndef _XSORTEDIMAGELIST
#define _XSORTEDIMAGELIST

#include "imagelist.h"
#include "image.h"

class XSortedImageList : public ImageList
{
	public:
		XSortedImageList(unsigned bytesPerPixel);
		void add(Image* img, liveVizRequest const& req);
		ImageNode* getOverlappingImage(Image *img);
};

#endif

#ifndef _IMAGELIST
#define _IMAGELIST

#include "defines.h"
#include "imagenode.h"
#include "string.h"
#include "liveViz0.h"
#include "liveViz.decl.h"

class ImageList
{
	protected:
		ImageNode* m_list;
		unsigned m_nodeCount; // number of images in the list.
		unsigned m_bytesPerPixel;
	public:
		ImageList(unsigned bytesPerPixel);
		~ImageList();

		/*
			This function add the input image to the list. This function stores the input image pointer directly.
		*/
		virtual void add(Image *img, liveVizRequest const& req);

		/*
			This function is used by add() to resolve the conflict between images and then add them to list.
		*/
		void addOverlapping(Image *img, ImageNode *imgNode, liveVizRequest const& req);

		// This function removes node from list, it doesn't delete it.
		void removeNodeFromList(ImageNode* node);
		// merges the input list nodes to self and clears the input list.
		// void mergeList(ImageList *imageList);

		// packing methods
		/*
			This function packs the image list into a run of bytes. The byte buffer can be interpreted as:

			 __________________________________________________________________________
			|      |               |   |   |     |   |       |       |       |        |
			|# of  | liveVizRequest|   |   |     |   | image1| image2|       | image n|
			|images|               | h1| h2| ....| hn| buffer| buffer| ..... | buffer |
			|______|_______________|___|___|_____|___|_______|_______|_______|________|

			here:
				h1, h2, ..., hn are the image header for n images in the list.
				#images is an unsigned int.
		*/
		void * pack(const liveVizRequest *req);
		/*
			this function decodes the run of bytes with the structure as shown above and adds all the images to
			itself (list).
		*/
		liveVizRequest* unPack(void *ptr);

		/*
			This returns the number of bytes required to pack the image list.
		*/
		unsigned packedDataSize(void);

		/*
			This function combines the list of image into one big image and returns the composite image.
		*/
		Image * combineImage(void);

		/*
			This function is used by combineImage() to know the buffer size required store the composite image.
		*/
		int getImageSize(Point &ulc, Point &lrc);

		/*
			This function checks if the image is within the window, otherwise it clips the image.
			Return value can be NULL if the input image is completely outside the window, otherwise
			it returns the modified image.
		*/
		Image* getClippedImage(Image *img, liveVizRequest const& req);
};

#endif

#include "imagelist.h"
#include "ckimage.h"

ImageList::ImageList(unsigned bytesPerPixel)
{
	m_list = NULL;
	m_nodeCount = 0;
	m_bytesPerPixel = bytesPerPixel;
}

ImageList::~ImageList()
{
	// delete every node in the list
	ImageNode *temp;

	while(m_list != NULL)
	{
		temp = m_list->m_next;
		delete m_list->m_img;
		delete m_list;
		m_list = temp;
	}
	m_list = NULL;
	m_nodeCount = 0;
}

void ImageList::add(Image *img, liveVizRequest const& req)
{
	Image *image = getClippedImage(img, req);
	if(image == NULL) // image is outside the window bounds
	{
		return;
	}
	ImageNode *temp = m_list;
	while(temp != NULL)
	{
		if(temp->m_img->isOverlapping(*image) == true)
		{
			break;
		}
		temp = temp->m_next;
	}
	if(temp == NULL)
	{
		// add the non-overlapping image directly to image-list
		ImageNode *node = new ImageNode();
		node->m_img = image;
		node->m_next = m_list;
		m_list = node;
		m_nodeCount++;
	}
	else
	{
		addOverlapping(image, temp, req); // temp is deleted by addOverlapping()
	}
}

void ImageList::addOverlapping(Image *img, ImageNode *imgNode, liveVizRequest const& req)
{
	Image *img1 = NULL;
	Image *img2 = img;
	Image *img3 = NULL;

	img1 = imgNode->m_img;

	img3 = img1->mergeImages(img2, m_bytesPerPixel);

	removeNodeFromList(imgNode); // remove node from list
	delete imgNode; // free memory

	add(img1, req);
	if(img2 != NULL)
		add(img2, req);

	if(img3 != NULL)
		add(img3, req);
}

/*
void ImageList::mergeList(ImageList *imageList)
{
	ImageNode *temp = imageList->m_list;
	while(temp != NULL)
	{
		imageList->m_list = temp->m_next;
		add(temp->m_img);
		delete temp;
		temp = imageList->m_list;
	}
}
*/

void ImageList::removeNodeFromList(ImageNode* node)
{
	ImageNode *temp = m_list;
	if(temp == node)
	{
		m_list = temp->m_next;
		m_nodeCount--;
		temp->m_next = NULL;
		return;
	}
	while(temp->m_next != NULL)
	{
		if(temp->m_next == node)
		{
			temp->m_next = temp->m_next->m_next;
			m_nodeCount--;
			return;
		}
		temp = temp->m_next;
	}
}

void ImageList::pack(const liveVizRequest *req,void *dest)
{
	//CkPrintf("Entering ImageList::pack() \n");
	byte * ptr = (byte *)dest;
	memcpy(ptr,&m_nodeCount,sizeof(unsigned));
	memcpy(ptr+sizeof(unsigned),req,sizeof(liveVizRequest));

	byte * headptr = ptr + sizeof(unsigned)+sizeof(liveVizRequest);
	byte * imageptr = headptr + m_nodeCount*sizeof(Rect);

	Rect rect;
	ImageNode *temp = m_list;

	while(temp != NULL)
	{
		rect.l = temp->m_img->m_ulc.x;
		rect.r = temp->m_img->m_lrc.x;
		rect.t = temp->m_img->m_ulc.y;
		rect.b = temp->m_img->m_lrc.y;

		memcpy(headptr, &rect, sizeof(Rect));

		temp->m_img->copyImageData(imageptr, m_bytesPerPixel);

		headptr += sizeof(Rect);
		imageptr += temp->m_img->getImageSize(m_bytesPerPixel);
		temp = temp->m_next;
	}
	//CkPrintf("Exiting ImageList::pack(), nodeCount = %d \n", m_nodeCount);
}

const liveVizRequest* ImageList::unPack(void *ptr)
{
	//CkPrintf("Entering ImageList::unPack() \n");
	unsigned nodeCount;
	nodeCount = ((unsigned int*)ptr)[0];
	const liveVizRequest *req =(const liveVizRequest *)((byte*)ptr + sizeof(unsigned int));

	byte * headptr = (byte*)ptr + sizeof(unsigned int) + sizeof(liveVizRequest);
	byte * imageptr = headptr + sizeof(Rect)*nodeCount;

	Rect rect;
	Image *temp = NULL;
	unsigned imageSize;

	for(int i=0; i<nodeCount; i++)
	{
		memcpy(&rect, headptr, sizeof(Rect));

		// Make a new image that *points to* the data in the input reduction message
		temp = new Image(rect, NULL);
		imageSize = temp->getImageSize(m_bytesPerPixel);
		temp->setData(imageptr,false);

		add(temp, *req);

		headptr += sizeof(Rect);
		imageptr += imageSize;
	}
	//CkPrintf("Exiting ImageList::unPack(), nodeCount = %d \n", nodeCount);
	return req;
}

unsigned ImageList::packedDataSize()
{
	ImageNode *temp = m_list;
	unsigned size = 0;

	while(temp != NULL)
	{
		size += temp->m_img->getImageSize(m_bytesPerPixel);
		temp = temp->m_next;
	}

	return(size + m_nodeCount*sizeof(Rect) + sizeof(unsigned) + sizeof(liveVizRequest));

}

Image * ImageList::combineImage(const liveVizRequest *req)
{
	Point ulc(0,0),lrc(req->wid-1, req->ht-1);

	int bpp=m_bytesPerPixel;
	int destWidth = req->wid;
	int destHeight = req->ht;
	int i, size = destWidth*destHeight*bpp;

	ImageNode *temp = m_list;
	
/* Early exit for (rather common) case of one big image covering everything: */
	if (m_nodeCount==1 && temp->m_img->m_ulc==ulc && temp->m_img->m_lrc==lrc)
	{
		Image *ret=new Image(ulc,lrc,NULL);
		ret->setData(temp->m_img->m_imgData,false);
		return ret;
	}
	
/* Normal case: loop over all our source images */
	byte * destImage = new byte[size];
	for(i=0; i<size; i++)
		destImage[i] = 0;
	
	for(i=0; i<m_nodeCount; i++)
	{
		int srcWidth = temp->m_img->m_lrc.x - temp->m_img->m_ulc.x + 1;
		int srcHeight = temp->m_img->m_lrc.y - temp->m_img->m_ulc.y + 1;
		int dx=temp->m_img->m_ulc.x; /* shift to apply to the output image */
		int dy=temp->m_img->m_ulc.y;
		const byte *srcImage=temp->m_img->m_imgData;
		for(int y=0; y<srcHeight; y++)
			for(int x=0; x<srcWidth; x++)
			{
				for(int j=0; j<m_bytesPerPixel; j++)
					destImage[((dy + y)*destWidth + (dx + x))*bpp + j] =
							srcImage[(y*srcWidth + x)*bpp + j];
 			}
 		temp = temp->m_next;
 	}
	return new Image(ulc, lrc, destImage);
}

Image* ImageList::getClippedImage(Image *img, liveVizRequest const& req)
{
	Point ulc, lrc, int_ulc, int_lrc;
	ulc.x = 0;
	ulc.y = 0;
	lrc.x = req.wid - 1;
	lrc.y = req.ht - 1;
	Image clippingImage(ulc, lrc, NULL);
	if(img->isOverlapping(clippingImage) == false)
	{
		delete img;
		return NULL;
	}
	if((img->m_ulc.x>=0)&&(img->m_ulc.y>=0)&&(img->m_lrc.x<=lrc.x)&&(img->m_lrc.y<=lrc.y))
	{ // image is completely within display
		return img;
	}

	img->getIntersectionCoordinates(clippingImage, int_ulc, int_lrc);

	Image *clippedImage = new Image(int_ulc, int_lrc, NULL);
	byte* clippedImageData = new byte[clippedImage->getImageSize(m_bytesPerPixel)];
	int clippedImageWidth = clippedImage->getImageWidth();
	int completeImageWidth = img->getImageWidth();
	for(int y=int_ulc.y; y<=int_lrc.y; y++)
		for(int x=int_ulc.x; x<=int_lrc.x; x++)
			for(int j=0; j<m_bytesPerPixel; j++)
				clippedImageData[((y-int_ulc.y)*clippedImageWidth+x-int_ulc.x)*m_bytesPerPixel+j] = img->m_imgData[((y-img->m_ulc.y)*completeImageWidth+x-img->m_ulc.x)*m_bytesPerPixel+j];

	clippedImage->m_imgData = clippedImageData;
	delete img;

	return clippedImage;
}

#include "imagelist.h"

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

void * ImageList::pack(const liveVizRequest *req)
{
	unsigned size = packedDataSize();
	if(size == 0)
		return NULL;

	byte * ptr = new byte[size];
	memcpy(ptr,&m_nodeCount,sizeof(unsigned));
	memcpy(ptr+sizeof(unsigned),req,sizeof(liveVizRequest));

	byte * headptr = ptr + sizeof(unsigned)+sizeof(liveVizRequest);
	byte * imageptr = headptr + m_nodeCount*sizeof(Image);
	byte * imageref = imageptr;

	Image image;
	ImageNode *temp = m_list;

	while(temp != NULL)
	{
		image.m_ulc.x = temp->m_img->m_ulc.x;
		image.m_lrc.x = temp->m_img->m_lrc.x;
		image.m_ulc.y = temp->m_img->m_ulc.y;
		image.m_lrc.y = temp->m_img->m_lrc.y;
		image.m_imgData = (byte*)(imageptr - imageref);

		memcpy(headptr,&image,sizeof(Image));

		temp->m_img->copyImageData(imageptr, m_bytesPerPixel);

		headptr += sizeof(Image);
		imageptr += temp->m_img->getImageSize(m_bytesPerPixel);
		temp = temp->m_next;
	}
	image.m_imgData = NULL;

	return ptr;
}

liveVizRequest* ImageList::unPack(void *ptr)
{
	liveVizRequest *req = new liveVizRequest;

	if(ptr == NULL)
		return NULL;

	unsigned nodeCount;
	nodeCount = ((unsigned*)ptr)[0];
	memcpy(req,(byte*)ptr + sizeof(unsigned), sizeof(liveVizRequest));

	byte * headptr = (byte*)ptr + sizeof(unsigned) + sizeof(liveVizRequest);
	byte * imageptr = headptr + sizeof(Image)*nodeCount;

	Image image;
	Image *temp = NULL;
	unsigned imageSize;

	for(int i=0; i<nodeCount; i++)
	{
		memcpy(&image,headptr,sizeof(Image));

		imageSize = image.getImageSize(m_bytesPerPixel);

		byte * imgData = new byte[imageSize];
		memcpy(imgData,imageptr,imageSize);

		temp = new Image(image.m_ulc, image.m_lrc, imgData);

		add(temp, *req);

		headptr += sizeof(Image);
		imageptr += imageSize;
	}
	image.m_imgData = NULL;
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

	return(size + m_nodeCount*sizeof(Image) + sizeof(unsigned) + sizeof(liveVizRequest));

}

Image * ImageList::combineImage(void)
{
	Point ulc,lrc;
	int size = getImageSize(ulc,lrc);
	//printf("%d %d %d %d \n",ulc.x,ulc.y,lrc.x,lrc.y);
	//printf("size = %d #node %d, size of byte=%d\n",size,m_nodeCount,sizeof(byte));
	byte * imageData = new byte[size];
	//printf(" Allocated Image Buffer\n");
	for(int i=0; i<size; i++)
		imageData[i] = 0;
	ImageNode *temp = m_list;
	int newImageWidth = lrc.x - ulc.x + 1;

	for(int i=0; i<m_nodeCount; i++)
	{
		int width = temp->m_img->m_lrc.x - temp->m_img->m_ulc.x + 1;
		int height = temp->m_img->m_lrc.y - temp->m_img->m_ulc.y + 1;
		for(int y=0; y<height; y++)
			for(int x=0; x<width; x++)
			{
				for(int j=0; j<m_bytesPerPixel; j++)
					imageData[((temp->m_img->m_ulc.y - ulc.y + y)*newImageWidth +
							(temp->m_img->m_ulc.x - ulc.x + x))*m_bytesPerPixel + j] =
							temp->m_img->m_imgData[(y*width + x)*m_bytesPerPixel + j];
 			}
 		temp = temp->m_next;
 	}
	return new Image(ulc, lrc, imageData);

}

int ImageList::getImageSize(Point &ulc, Point &lrc)
{
	ulc.x = m_list->m_img->m_ulc.x;
	ulc.y = m_list->m_img->m_ulc.y;
	lrc.x = m_list->m_img->m_lrc.x;
	lrc.y = m_list->m_img->m_lrc.y;
	ImageNode *temp = m_list;
	while(temp != NULL)
	{
		if(ulc.x > temp->m_img->m_ulc.x)
			ulc.x = temp->m_img->m_ulc.x;
		if(ulc.y > temp->m_img->m_ulc.y)
			ulc.y = temp->m_img->m_ulc.y;
		if(lrc.x < temp->m_img->m_lrc.x)
			lrc.x = temp->m_img->m_lrc.x;
		if(lrc.y < temp->m_img->m_lrc.y)
			lrc.y = temp->m_img->m_lrc.y;
		temp = temp->m_next;
	}
	return ((lrc.y - ulc.y + 1)*(lrc.x - ulc.x + 1)*m_bytesPerPixel);
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

#include "xSortedImageList.h"

XSortedImageList::XSortedImageList(unsigned bytesPerPixel):ImageList(bytesPerPixel)
{}
void XSortedImageList::add(Image* img, liveVizRequest const& req)
{

	Image *image = getClippedImage(img, req);
	if(image == NULL) // image is outside the window bounds
	{
		return;
	}

	ImageNode* temp = NULL;
	temp = getOverlappingImage(image);
	if(temp == NULL)
	{
		// add Image to list
		temp = m_list;
		ImageNode *prevNode = NULL;
		while(temp != NULL)
		{
			if(temp->m_img->m_lrc.x > img->m_lrc.x)
			{

				break;
			}
			prevNode = temp;
			temp = temp->m_next;
		}
		ImageNode *node = new ImageNode();
		node->m_img = image;
		node->m_next = temp;
		if(prevNode != NULL)
		{
			prevNode->m_next = node;
		}
		else
		{
			m_list = node;
		}
		m_nodeCount++;
	}
	else
	{
		// resolve the overlap
		addOverlapping(image, temp, req);
	}

}

ImageNode* XSortedImageList::getOverlappingImage(Image *img)
{
	ImageNode *temp = m_list;
	while((temp != NULL)&&(temp->m_img->m_ulc.x <= img->m_lrc.x))
	{
		if(temp->m_img->isOverlapping(*img) == true)
			return temp;
		else
			temp = temp->m_next;
	}
	return NULL;
}

#include "imagenode.h"


// Constructor Methods
ImageNode::ImageNode()
{
	m_img = NULL;
	m_next = NULL;
}

ImageNode::ImageNode(Image const& img)
{
	m_img = new Image(img.m_ulc, img.m_lrc, img.m_imgData);
	m_next = NULL;
}

ImageNode::ImageNode(Image const& img, ImageNode *next)
{
	m_img = new Image(img.m_ulc, img.m_lrc, img.m_imgData);
	m_next = next;
}

ImageNode::ImageNode(Point const& ulc, Point const& lrc, byte *imgData)
{
	m_img = new Image(ulc, lrc, imgData);
	m_next = NULL;
}

// Destructor method
ImageNode::~ImageNode()
{
	m_next = NULL;
}


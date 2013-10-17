#include "ImageData.h"

ImageData::ImageData (int bytesPerPixel)
{
    m_bytesPerPixel         = bytesPerPixel;
    m_size                  = 0;
    m_numDataLines          = 0;
    m_clippedImage          = NULL;
    m_clippedImageAllocated = false;
    m_startx                = 0;
    m_starty                = 0;
    m_sizex                 = 0;
    m_sizey                 = 0;
   	m_header                = NULL;
	   m_headerSize            = 0;
}

ImageData::~ImageData ()
{
    if (true == m_clippedImageAllocated)
    {
        delete [] m_clippedImage;
    }

    if (NULL != m_header)
   	{
	      delete [] m_header;
   	}
}

void ImageData::WriteHeader(ImageDataCombine_t combine,
              const liveVizRequest* req,
              byte* dest)
{
	ImageHeader *ihead=(ImageHeader *)dest;
	ihead->m_combine=combine;
	// since the poll mode does not yet have a req, 
	// just set it to null.
	if(req)
	  ihead->m_req=*req;
}

void ImageData::ReadHeader(ImageDataCombine_t &combine,
              liveVizRequest* req,
              const byte* src)
{
	const ImageHeader *ihead=(const ImageHeader *)src;
	combine=ihead->m_combine;
	*req=ihead->m_req;
}


/******************* AddImage: User Image Input **************
AddImage is used to copy the image from the user's rectangular
array into our "lines" format.  This is the stage at which black
(blank) pixels are eliminated, which has three major steps:
   1.) GetClippedImage sets m_clippedImage with the user's
       image, restricted to the requested output window.
       There's often nothing to do here, as users should normally
       pass in clipped input.
   2.) AddImage(req,NULL) computes the compressed image size.
       Normally called from GetBuffSize, below.
   3.) AddImage(req,dest) called by user copies
       the image data into the compressed output buffer dest.
*/

int ImageData::GetBuffSize (const int& startx,
                            const int& starty,
                            const int& sizex,
                            const int& sizey,
                            const int req_wid,
							const int req_ht,
                            const byte* src)
{
    GetClippedImage (src,
                     startx,
                     starty,
                     sizex,
                     sizey,
                     req_wid,
					 req_ht
					 );

    return AddImage (req_wid);
}


byte* ImageData::GetClippedImage (const byte* img,
                                  const int& startx,
                                  const int& starty,
                                  const int& sizex,
                                  const int& sizey,
                                  const int req_wid,
								  const int req_ht
								  )
{
    bool shift       = false;
    int  newpos      = 0;
    int  oldpos      = 0;
    int  bytesPerRow = 0;
    int  xoffset     = 0;

    // initialize members
    m_startx       = startx;
    m_starty       = starty;
    m_sizex        = sizex;
    m_sizey        = sizey;
    m_clippedImage = (byte*)img;

    if (NULL == img)
    {
        goto EXITPOINT;
    }

    // if image is completely outside display region, then ignore image
    if (((startx+sizex-1) < 0) ||
        ((starty+sizey-1) < 0) ||
        (startx > (req_wid-1)) ||
        (starty > (req_ht-1)))
    {
        m_clippedImage = NULL;
        goto EXITPOINT;
    }

    if (0 > starty)
    {
        m_clippedImage += (0-starty)*sizex*m_bytesPerPixel;
        m_sizey  += starty;
        m_starty  = 0;
    }

    if ((m_starty+m_sizey) > req_ht)
    {
        m_sizey = req_ht - m_starty;
    }

    // need to shift data for other 2 cases
    if (0 > startx)
    {
        m_sizex += startx;
        m_startx = 0;
        shift     = true;
    }

    if ((m_startx + m_sizex) > req_wid)
    {
        m_sizex  = req_wid - m_startx;
        shift    = true;
    }

    if (true == shift)
    {
        int imgSize = m_sizex*m_sizey*m_bytesPerPixel;

        m_clippedImage = new byte [imgSize];

        if (NULL == m_clippedImage)
        {
            // Error
            CmiPrintf ("Memory allocation failure!!!\n");
            goto EXITPOINT;
        }

        m_clippedImageAllocated = true;

        xoffset     = (m_startx - startx) * m_bytesPerPixel;
        bytesPerRow = m_sizex * m_bytesPerPixel;

        for (int y=0; y<m_sizey; y++)
        {
            oldpos = (m_starty + y - starty)*sizex*m_bytesPerPixel + xoffset;
            for (int x=0; x<bytesPerRow; x++)
            {
                m_clippedImage [newpos ++] = img [oldpos ++];
            }
        }
    }

EXITPOINT:
    return m_clippedImage;
}


#ifdef COMPLETE_BLACK_PIXEL_ELIMINATION

int ImageData::AddImage (const int req_wid,
                         byte* dest)
{
    LineHeader head;                 // header of data line being copied
    int    headPos      = 0;     // position in buffer 'dest'
    int    dataPos      = 0;     // position in buffer 'dest'
    int    pos          = 0;     // position in 'src'
    int    numOnPixels  = 0;     // num of on pixels in the chuck
                                 // deposited
    int    numDataLines = 0;     // number of lines of non zero
                                 // data in deposited chunck
    bool   isPixelOn    = false; // pixel scaned is on or off
    bool   foundLine    = false; // a image data line found

    const byte* src = m_clippedImage;
    const int   startx = m_startx;
    const int   starty = m_starty;
    const int   sizex  = m_sizex;
    const int   sizey  = m_sizey;

    if (NULL != dest)
    {
        // prepare to copy out image data
        headPos = sizeof (ImageHeader);
        dataPos = headPos + (sizeof (LineHeader) * m_numDataLines);
	ImageHeader *ihead=(ImageHeader *)dest;
	ihead->m_lines=m_numDataLines;
    }
	
    if (NULL != src)
    {
        for (int y=0; y<sizey; y++)
        {
            for (int x=0; x<sizex; x++)
            {
                isPixelOn = IsPixelOn (src + pos);

                if ((false == foundLine) && (true == isPixelOn))
                {
                    // found a data line
                    if (NULL != dest)
                    {
                        head.m_pos	= ((starty + y)*(req_wid)) + 
                                     (startx + x);
                        head.m_size	= 0;
                    }
				
                    foundLine	= true;
                }

                if (true == isPixelOn)
                {
                    if (NULL != dest)
                    {
                        // update header
                        head.m_size ++;

                        // copy this pixel
                        CopyPixel (dest+dataPos, src+pos);

                        // update dataPos
                        dataPos += m_bytesPerPixel;
                    }

                    numOnPixels ++;
                }
                else
                {
                    if (true == foundLine)
                    {
                        // end of data line
                        if (NULL != dest)
                        {
                            // copy header
                            memcpy (dest + headPos, 
                                    &head,
                                    sizeof (LineHeader));

                            // update headPos
                            headPos += sizeof (LineHeader);
                        }

                        numDataLines ++;
                        foundLine = false;
                    }
                }
			
                pos += m_bytesPerPixel;
            }

            if (true == foundLine)
            {
                // end of data line
                if (NULL != dest)
                {
                     // copy header
                     memcpy (dest + headPos, 
                             &head,
                             sizeof (LineHeader));

                     // update headPos
                     headPos += sizeof (LineHeader);
                }

                numDataLines ++;
                foundLine = false;
            }
        }
    }

    SetSize(numDataLines,numOnPixels);

    return m_size;
}

#endif

#ifdef NO_BLACK_PIXEL_ELIMINATION

int ImageData::AddImage (const int req_wid
                         byte* dest)
{
    LineHeader head;                 // header of data line being copied
    int    headPos      = 0;     // position in buffer 'dest'
    int    dataPos      = 0;     // position in buffer 'dest'
    int    pos          = 0;     // position in 'src'
    int    numOnPixels  = 0;     // num of on pixels in the chuck
                                 // deposited
    int    numDataLines = 0;     // number of lines of non zero
                                 // data in deposited chunck
    bool   isPixelOn    = false; // pixel scaned is on or off
    bool   foundLine    = false; // a image data line found

    const byte* src = m_clippedImage;
    const int   startx = m_startx;
    const int   starty = m_starty;
    const int   sizex  = m_sizex;
    const int   sizey  = m_sizey;

    if (NULL != dest)
    {
        // prepare to copy out image data
        headPos = sizeof (ImageHeader);
        dataPos = headPos + (sizeof (LineHeader) * m_numDataLines);
	ImageHeader *ihead=(ImageHeader *)dest;
	ihead->m_lines=m_numDataLines;
    }
	
    if (NULL != src)
    {
        int bytesToCopy = sizex*m_bytesPerPixel;

        for (int y=0; y<sizey; y++)
        {
            if (NULL != dest)
            {
                head.m_pos	= ((starty + y)*(req_wid)) + 
                               (startx);
                head.m_size	= sizex;

                // copy data line
                memcpy (dest+dataPos, src+pos, bytesToCopy);

                dataPos += bytesToCopy;
                pos     += bytesToCopy;

                // copy header
                memcpy (dest + headPos, 
                        &head,
                        sizeof (LineHeader));

                // update headPos
                headPos += sizeof (LineHeader);
            }

            numOnPixels  += sizex;
            numDataLines ++;
        }
    }

    SetSize(numDataLines,numOnPixels);

    return m_size;
}
#endif

#ifdef EXTERIOR_BLACK_PIXEL_ELIMINATION
/*
   This strategy tries to eliminate black pixels from ends of
   data lines in deposited image chunck.
*/
int ImageData::AddImage (const int req_wid,
                         byte* dest)
{
    LineHeader head;                 // header of data line being copied
    int    headPos      = 0;     // position in buffer 'dest'
    int    dataPos      = 0;     // position in buffer 'dest'
    int    numOnPixels  = 0;     // num of on pixels in the chuck
                                 // deposited
    int    numDataLines = 0;     // number of lines of non zero
                                 // data in deposited chunck
    bool   isPixelOn    = false; // pixel scaned is on or off
    bool   foundLine    = false; // a image data line found

    const int   startx = m_startx;
    const int   starty = m_starty;
    const int   sizex  = m_sizex;
    const int   sizey  = m_sizey;

    if (NULL != dest)
    {
        // prepare to copy out image data
        headPos = sizeof (ImageHeader);
        dataPos = headPos + (sizeof (LineHeader) * m_numDataLines);
	ImageHeader *ihead=(ImageHeader *)dest;
	ihead->m_lines=m_numDataLines;
    }
	
    if (NULL != m_clippedImage)
    {
        for (int y=0; y<sizey; y++)
        {
            int startPos = y*sizex*m_bytesPerPixel;
            int endPos   = startPos + (sizex-1)*m_bytesPerPixel;
            int xoffset  = 0;
            int bytesToCopy = 0;
            bool startPixelFound = false;
            bool endPixelFound = false;

            while (startPos <= endPos)
            {
                startPixelFound = IsPixelOn (m_clippedImage+startPos);
                endPixelFound   = IsPixelOn (m_clippedImage+endPos);

                if (false == startPixelFound)
                {
                    startPos += m_bytesPerPixel;
                    xoffset ++;
                }

                if (false == endPixelFound)
                {
                    endPos -= m_bytesPerPixel;
                }

                if (startPixelFound && endPixelFound)
                {
                    break;
                }
            }

            if (startPos <= endPos)
            {
                numDataLines ++;
                numOnPixels += (endPos - startPos)/m_bytesPerPixel + 1;

                if (NULL != dest)
                {
                    head.m_pos	= ((starty + y)*(req_wid)) + 
                                  (startx + xoffset);
                    head.m_size	= (endPos - startPos)/m_bytesPerPixel + 1;

                    bytesToCopy = (head.m_size)*m_bytesPerPixel;

                    // copy data line
                    memcpy (dest+dataPos, m_clippedImage+startPos, bytesToCopy);

                    dataPos += bytesToCopy;

                    // copy header
                    memcpy (dest + headPos, 
                            &head,
                            sizeof (LineHeader));

                    // update headPos
                    headPos += sizeof (LineHeader);
                }
            }
        }
    }

    SetSize(numDataLines,numOnPixels);

    return m_size;
}
#endif


/********************* Image Merging **************** 
Here's where compressed images are combined.
*/

int ImageData::CombineImageDataSize (int nMsg, CkReductionMsg **msgs)
{
    int     returnVal       = 0;    // size of combined data or '-1'
    int     buffSize        = 0;    // 'buff' size
    int     headPos         = 0;    // in 'buff'
    int     numNonNullLists = 0;    // number of data lists with 
                                    // lines of data to be combined
    int     numDataLines    = 0;    // number of data lines in
                                    // combined data
    int     numPixels       = 0;    // number of pixels in combined
                                    // data
    // varaibles used for n-way merge
    int     minIndex        = 0;    // index of list whose data line
                                    // at 'minPos' is next data line
                                    // to be put into buff
    int     minPos          = 0;    // start pos of next data line
                                    // to be added to buff
    int     minSize         = 0;    // size in pixels
    int     currPos         = 0;    // start position of curr data
                                    // line
    int     currSize        = 0;    // size in pixels
    int*    pos             = NULL; // maintains current position
                                    // in different data lists 
    int*    size            = NULL; // hold size of all data lists
    byte*   buff            = NULL; // to hold combined data
    LineHeader* prevHead        = NULL; // header of prev data line
                                    // added to buff
    LineHeader* currHead        = NULL; // header of current data line
    LineHeader* minHead         = NULL; // header of next data line to 
                                    // be added to buff
    int     i;

/**
  "buff" consists of two totally different things:
     1.) It's got a compressed image header and lineheaders
         for the merged image.
     2.) It's got the "pos" and "size" arrays used for sorting.
*/
    buffSize += sizeof (ImageHeader);

    // find worst-case merged LineHeader size (assuming no overlap)
    for (i=0; i<nMsg; i++)
    {
        buffSize += ((ImageHeader *)(msgs [i]->getData ()))->m_lines*sizeof (LineHeader);
    }

    // include memory needed for 'pos' and 'size' data
    buffSize += sizeof (int) * nMsg * 2;
    

    // allocate buffer
    buff = new byte [buffSize];

    // initialize 'pos' and 'size' pointers using end of "buff":
    pos  = (int*)(buff + buffSize - (sizeof (int)*nMsg*2));
    size = pos + nMsg;

    // calculate 'headPos', position in 'buff' where LineHeaders are
    // placed
    headPos = sizeof (ImageHeader);

    // initialize 'pos' and 'size'
    for (i=0; i<nMsg; i++)
    {
        size [i] = ((ImageHeader *)(msgs[i])->getData ())->m_lines;
        pos [i]  = 0;
    }

    // find num of lists having data lines to be put into buff
    numNonNullLists = NumNonNullLists (pos, size, nMsg);

    // if there are more than 1 lists having data lines then
    // merge (in order) according to 'start pos'
    while (1 < numNonNullLists)
    {
        // find the first list which has more data lines
        minIndex = GetNonNullListIndex (pos, nMsg);

        // find the data line to add to buff
        minHead = getHeader(msgs[minIndex]->getData (), pos [minIndex]);

        // initialize variables
        minPos = minHead->m_pos;
        minSize = minHead->m_size;

        // find the data line with minimum 'start pos'
        for (int i=minIndex+1; i<nMsg; i++)
        {
            if (-1 != pos [i])
            {
                // ith list has more data lines

                currHead = getHeader(msgs [i]->getData (),pos [i]);

                currPos = currHead->m_pos;
                currSize = currHead->m_size;

                if (minPos > currPos)
                {
                    minHead  = currHead;
                    minIndex = i;
                    minPos   = currPos;
                    minSize  = currSize;
                }
            }
        }

        // update position for list with data line having min pos
        pos [minIndex] ++;

        // copy result to buff

        // if this line and prev line can be merged
        if ((NULL != prevHead) && 
            ((prevHead->m_pos + prevHead->m_size) >= minPos))
        {
            // merge
            if ((minPos + minSize) > (prevHead->m_pos + prevHead->m_size))
            {
                numPixels -= prevHead->m_size;
                prevHead->m_size = minPos - prevHead->m_pos + minSize;
                numPixels += prevHead->m_size;
            }
        }
        else
        {
            prevHead = (LineHeader*)(buff + headPos);
            memcpy ((void*) prevHead,
                    (void*) minHead,
                    sizeof (LineHeader));
            headPos += sizeof (LineHeader);
            numDataLines ++;
            numPixels += prevHead->m_size;
        }

        // update numNonNullLists
        numNonNullLists = NumNonNullLists (pos, size, nMsg);
    }

    if (0 < numNonNullLists)
    {
        // find the only list with more data lines
        minIndex = GetNonNullListIndex (pos, nMsg);

        // find the data line to add
        minHead = getHeader(msgs [minIndex]->getData (), pos [minIndex]);

        if (-1 != minIndex)
        {
            for (int i=pos [minIndex]; i<size [minIndex]; i++)
            {
                minPos = minHead->m_pos;
                minSize = minHead->m_size;

                if ((NULL != prevHead) && 
                    ((prevHead->m_pos + prevHead->m_size) >= minPos))
                {
                    if ((minPos + minSize) > (prevHead->m_pos + prevHead->m_size))
                    {
                        numPixels -= prevHead->m_size;
                        prevHead->m_size = minPos - prevHead->m_pos + minSize;
                        numPixels += prevHead->m_size;
                    }

                    minHead ++;  // += sizeof (LineHeader);
                }
                else
                {
                    prevHead = (LineHeader*)(buff + headPos);
                    memcpy ((void*)prevHead, (void*)minHead, sizeof (LineHeader));
                    numPixels += ((LineHeader*)(buff+headPos))->m_size;
                    headPos += sizeof (LineHeader);
                    minHead ++; //+= sizeof (LineHeader);
                    numDataLines ++;
                }
            }
        }
    }

    
    // Save buf so CombineImageData can copy out the merged header.
    m_header=buff;
    memcpy (m_header, 
            msgs[0]->getData (), 
            sizeof (ImageHeader));
    ((ImageHeader *)m_header)->m_lines=numDataLines;
    
    SetSize(numDataLines,numPixels);
    returnVal = m_size;

EXITPOINT:
    return returnVal;
}

void ImageData::CombineImageData (int nMsg, CkReductionMsg **msgs, byte* dest)
{
    // Copy in the merged image header
    memcpy (dest, m_header, m_headerSize);
    
    // Initialize image data to all 0's.
    memset (dest+m_headerSize, 0, m_size-m_headerSize);

    // combine image data into buff
    for (int i=0; i<nMsg; i++)
    {
        CopyImageData (dest, m_numDataLines, msgs[i], 
		((ImageHeader *)dest)->m_combine);
    }
}

// Tiny utility routines used for image combine:
template<class T>
inline void maxArrayT(const T *in,T *out,int n) 
{
	for (int i=0;i<n;i++)
		if (in[i]>out[i]) out[i]=in[i];
}
template<class T>
inline void sumArrayT(const T *in,T *out,int n) 
{
	for (int i=0;i<n;i++)
		out[i]+=in[i];
}
template<class T,class C>
inline void sumArrayClipT(const T *in,T *out,int n,C c) 
{
	for (int i=0;i<n;i++) {
		C sum=out[i];
		sum+=in[i];
		if (sum>c) sum=c;
		out[i]=(T)sum;
	}
}

int ImageData::CopyImageData (byte* dest,
                              int n,
                              const CkReductionMsg* msg,
                              ImageDataCombine_t reducer)
{
    int returnVal       = 0;
    int destHeadPos     = 0;
    int destDataPos     = 0;
    int srcDataPos      = 0;
    int srcHeadPos      = 0; 
    int numSrcDataLines = 0;
    int bytesToCopy     = 0;
    const byte* src     = (const byte*)(msg->getData());
    LineHeader* destLineHeader  = NULL;
    const LineHeader* srcLineHeader   = NULL;
    
    numSrcDataLines = ((ImageHeader*)src)->m_lines;

    destHeadPos = sizeof (ImageHeader);
    srcHeadPos  = sizeof (ImageHeader);
    destDataPos = destHeadPos + (sizeof (LineHeader) * n);
    srcDataPos  = srcHeadPos  + (sizeof (LineHeader) * numSrcDataLines);

    destLineHeader = (LineHeader*)(dest + destHeadPos);
    srcLineHeader  = (const LineHeader*)(src + srcHeadPos);

    for (int i=0; i<numSrcDataLines; i++)
    {
    	// Find the unique dest line that overlaps our src line:
        while ((destLineHeader->m_pos + 
                destLineHeader->m_size - 1) < srcLineHeader->m_pos)
        {
            destDataPos += (m_bytesPerPixel*(destLineHeader->m_size));
            destLineHeader++;
        }
	
        // copy at proper pos
        bytesToCopy = m_bytesPerPixel * (srcLineHeader->m_size);
        int posInDataLine = (srcLineHeader->m_pos - 
                             destLineHeader->m_pos) * m_bytesPerPixel;
	const byte *srcRow=src + srcDataPos; 
	byte *destRow=dest + destDataPos + posInDataLine;
	
	switch(reducer) {
        case sum_image_data: /* Add bytes, and clip result: */
	   sumArrayClipT(srcRow,destRow,bytesToCopy, 0xff);
	   break;
        case max_image_data: /* Take max of input and output data */
	   maxArrayT(srcRow,destRow,bytesToCopy);
           break;
	case sum_float_image_data: /* Take sum as floating-point data */
	   sumArrayT((const float *)srcRow,(float *)destRow,bytesToCopy/sizeof(float));
	   break;
	case max_float_image_data: /* Take max as floating-point data */
	   maxArrayT((const float *)srcRow,(float *)destRow,bytesToCopy/sizeof(float));
	   break;
	default:
	   CkAbort("LiveViz ImageData: Unrecognized image reducer type!\n");
        }
        srcLineHeader++;
	srcDataPos+=bytesToCopy;
    }

    return returnVal;
}


/*************************** Final Image Output ***************
  Reconstruct a rectangular image from this compressed buffer.
  This is the last phase.
*/
byte* ImageData::ConstructImage (byte* src,
				                 liveVizRequest& req)
{
    int i;
    int numDataLines = 0;
    int headPos      = 0;
    int dataPos      = 0;
    int imageSize    = 0;
    int imagePos     = 0;
    int numBytesToCopy = 0;
    byte* image      = NULL;
    const ImageHeader *ihead   = (const ImageHeader *)src;
    LineHeader* head     = NULL;
	
    numDataLines = ihead->m_lines;
    req=ihead->m_req;

    headPos = sizeof (ImageHeader);
    dataPos = headPos + (sizeof (LineHeader) * numDataLines);

    imageSize = req.wid*req.ht*m_bytesPerPixel;

    image = new byte [imageSize];

    if (NULL == image)
    {
        goto EXITPOINT;
    }

    memset (image, 0, imageSize);

    head = (LineHeader*) (src + headPos);

    for (i=0; i<numDataLines; i++)
    {
        numBytesToCopy = head->m_size*m_bytesPerPixel;
        imagePos = head->m_pos*m_bytesPerPixel;
		
        memcpy (image+imagePos, src+dataPos, numBytesToCopy);
        dataPos += numBytesToCopy;
        head ++;
    }
	
EXITPOINT:
    return image;
}

#include "ImageData.h"

ImageData::ImageData (int bytesPerPixel)
{
    m_bytesPerPixel         = bytesPerPixel;
    m_imageData             = NULL;
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

int ImageData::GetBuffSize (const int& startx,
                            const int& starty,
                            const int& sizex,
                            const int& sizey,
                            const liveVizRequest* req,
                            const byte* src)
{
    GetClippedImage (src,
                     startx,
                     starty,
                     sizex,
                     sizey,
                     req);

    return AddImage (req);
}

#ifdef COMPLETE_BLACK_PIXEL_ELIMINATION

int ImageData::AddImage (const liveVizRequest* req,
                         byte* dest)
{
    Header head;                 // header of data line being copied
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
        // initialize the member
        m_imageData = dest;

        // copy data
        headPos = sizeof (int) +
                  sizeof (liveVizRequest);

        dataPos = sizeof (int) +
                  sizeof (liveVizRequest) +
                  (sizeof (Header) * m_numDataLines);

        // copy number of data lines
        memcpy (m_imageData, &m_numDataLines, sizeof (int));

        // copy liveVizRequest object
        memcpy (m_imageData + sizeof (int),
                req,
                sizeof (liveVizRequest));
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
                        head.m_pos	= ((starty + y)*(req->wid)) + 
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
                                    sizeof (Header));

                            // update headPos
                            headPos += sizeof (Header);
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
                             sizeof (Header));

                     // update headPos
                     headPos += sizeof (Header);
                }

                numDataLines ++;
                foundLine = false;
            }
        }
    }

    if (NULL == dest)
    {
        m_size = sizeof (int) +
                 sizeof (liveVizRequest) +
                 (sizeof (Header) * numDataLines) +
                 (m_bytesPerPixel * numOnPixels);

        m_numDataLines = numDataLines;
    }

    return m_size;
}

#endif

#ifdef NO_BLACK_PIXEL_ELIMINATION

int ImageData::AddImage (const liveVizRequest* req,
                         byte* dest)
{
    Header head;                 // header of data line being copied
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
        // initialize the member
        m_imageData = dest;

        // copy data
        headPos = sizeof (int) +
                  sizeof (liveVizRequest);

        dataPos = sizeof (int) +
                  sizeof (liveVizRequest) +
                  (sizeof (Header) * m_numDataLines);

        // copy number of data lines
        memcpy (m_imageData, &m_numDataLines, sizeof (int));

        // copy liveVizRequest object
        memcpy (m_imageData + sizeof (int),
                req,
                sizeof (liveVizRequest));
    }
	
    if (NULL != src)
    {
        int bytesToCopy = sizex*m_bytesPerPixel;

        for (int y=0; y<sizey; y++)
        {
            if (NULL != dest)
            {
                head.m_pos	= ((starty + y)*(req->wid)) + 
                               (startx);
                head.m_size	= sizex;

                // copy data line
                memcpy (dest+dataPos, src+pos, bytesToCopy);

                dataPos += bytesToCopy;
                pos     += bytesToCopy;

                // copy header
                memcpy (dest + headPos, 
                        &head,
                        sizeof (Header));

                // update headPos
                headPos += sizeof (Header);
            }

            numOnPixels  += sizex;
            numDataLines ++;
        }
    }

    if (NULL == dest)
    {
        m_size = sizeof (int) +
                 sizeof (liveVizRequest) +
                 (sizeof (Header) * numDataLines) +
                 (m_bytesPerPixel * numOnPixels);

        m_numDataLines = numDataLines;
    }

    return m_size;
}
#endif

#ifdef EXTERIOR_BLACK_PIXEL_ELIMINATION
/*
   This strategy tries to eliminate black pixels from ends of
   data lines in deposited image chunck.
*/
int ImageData::AddImage (const liveVizRequest* req,
                         byte* dest)
{
    Header head;                 // header of data line being copied
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
        // initialize the member
        m_imageData = dest;

        // copy data
        headPos = sizeof (int) +
                  sizeof (liveVizRequest);

        dataPos = sizeof (int) +
                  sizeof (liveVizRequest) +
                  (sizeof (Header) * m_numDataLines);

        // copy number of data lines
        memcpy (m_imageData, &m_numDataLines, sizeof (int));

        // copy liveVizRequest object
        memcpy (m_imageData + sizeof (int),
                req,
                sizeof (liveVizRequest));
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
                    head.m_pos	= ((starty + y)*(req->wid)) + 
                                  (startx + xoffset);
                    head.m_size	= (endPos - startPos)/m_bytesPerPixel + 1;

                    bytesToCopy = (head.m_size)*m_bytesPerPixel;

                    // copy data line
                    memcpy (dest+dataPos, m_clippedImage+startPos, bytesToCopy);

                    dataPos += bytesToCopy;

                    // copy header
                    memcpy (dest + headPos, 
                            &head,
                            sizeof (Header));

                    // update headPos
                    headPos += sizeof (Header);
                }
            }
        }
    }

    if (NULL == dest)
    {
        m_size = sizeof (int) +
                 sizeof (liveVizRequest) +
                 (sizeof (Header) * numDataLines) +
                 (m_bytesPerPixel * numOnPixels);

        m_numDataLines = numDataLines;
    }

    return m_size;
}
#endif

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
    Header* prevHead        = NULL; // header of prev data line
                                    // added to buff
    Header* currHead        = NULL; // header of current data line
    Header* minHead         = NULL; // header of next data line to 
                                    // be added to buff


    // find optimistic buffer size
    for (int i=0; i<nMsg; i++)
    {
        buffSize += (*((int*)((msgs [i])->getData ())))*sizeof (Header);
    }

    buffSize += sizeof (int) + sizeof (liveVizRequest);
	
    // include memory needed for 'pos' and 'size' data
    buffSize += sizeof (int) * nMsg * 2;

    // allocate buffer
    buff = new byte [buffSize];

    // check for 'new' failure
    if (NULL == buff)
    {
        returnVal = -1; // ERROR
        goto EXITPOINT;
    }

    // initialize 'pos' and 'size' pointers
    pos  = (int*)(buff + buffSize - (sizeof (int)*nMsg*2));
    size = pos + nMsg;

    // initialize member
    m_header = buff;

    // calculate 'headPos', position in 'buff' where Headers are
    // placed
    headPos = sizeof (int) + 
              sizeof (liveVizRequest);

    // initialize 'pos' and 'size'
    for (int i=0; i<nMsg; i++)
    {
        size [i] = *((int*)((msgs[i])->getData ()));
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
        minHead = (Header*)((byte*)((msgs [minIndex])->getData ()) + 
                            sizeof (int) +
                            sizeof (liveVizRequest) + 
                            (sizeof (Header) * pos [minIndex]));

        // initialize variables
        minPos = minHead->m_pos;
        minSize = minHead->m_size;

        // find the data line with minimum 'start pos'
        for (int i=minIndex+1; i<nMsg; i++)
        {
            if (-1 != pos [i])
            {
                // ith list has more data lines

                currHead = (Header*)((byte*)((msgs [i])->getData ()) +
                                     sizeof (int) +
                                     sizeof (liveVizRequest) +
                                     (sizeof (Header)*pos [i]));

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
            prevHead = (Header*)(buff + headPos);
            memcpy (prevHead,
                    minHead,
                    sizeof (Header));
            headPos += sizeof (Header);
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
        minHead = (Header*)((byte*)((msgs [minIndex])->getData ()) +
                           sizeof (int) +
                           sizeof (liveVizRequest) +
                           (sizeof (Header)*pos [minIndex]));

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

                    minHead ++;  // += sizeof (Header);
                }
                else
                {
                    prevHead = (Header*)(buff + headPos);
                    memcpy (prevHead, minHead, sizeof (Header));
                    numPixels += ((Header*)(buff+headPos))->m_size;
                    headPos += sizeof (Header);
                    minHead ++; //+= sizeof (Header);
                    numDataLines ++;
                }
            }
        }
    }

    // copy fixed header to buff
    memcpy (buff, &numDataLines, sizeof (int));
    memcpy (buff + sizeof (int), 
            (byte*)((msgs[0])->getData ()) + sizeof (int), 
            sizeof (liveVizRequest));
	
    // initialize member variables
    m_headerSize = sizeof (int) +
                   sizeof (liveVizRequest) +
                   sizeof (Header) * numDataLines;

    m_numDataLines = numDataLines;
    m_size = sizeof (int) +
             sizeof (liveVizRequest) +
             (sizeof (Header) * numDataLines) +
             (m_bytesPerPixel * numPixels);
    returnVal = m_size;

EXITPOINT:
    return returnVal;
}

void ImageData::CombineImageData (int nMsg, CkReductionMsg **msgs, byte* dest)
{
    int     headPos         = 0;    // in 'buff'
    int     dataPos         = 0;    // in 'buff' 
    int     numPixels       = 0;    // data pixels to be copied

    numPixels = (m_size - m_headerSize)/m_bytesPerPixel;

    // copy header to dest
    memcpy (dest, m_header, m_headerSize);

    // calulate data position in dest buffer
    dataPos = m_headerSize + 1;

    // initialize image data buff
    memset (dest + dataPos, 0, m_bytesPerPixel * numPixels);

    // copy image data to buff
    for (int i=0; i<nMsg; i++)
    {
        CopyImageData (dest, m_numDataLines, msgs[i]);
    }
}

int ImageData::CopyImageData (byte* dest,
                              int n,
                              CkReductionMsg* msg)
{
    int returnVal       = 0;
    int destHeadPos     = 0;
    int destDataPos     = 0;
    int srcDataPos      = 0;
    int srcHeadPos      = 0; 
    int numSrcDataLines = 0;
    int bytesToCopy     = 0;
    int pixelByte       = 0;
    byte* src           = (byte*)(msg->getData ());
    Header* destHeader  = NULL;
    Header* srcHeader   = NULL;

    numSrcDataLines = *((int*)src);

    destHeadPos = sizeof (int) + sizeof (liveVizRequest);
    srcHeadPos  = destHeadPos;
    destDataPos = destHeadPos + (sizeof (Header) * n);
    srcDataPos  = srcHeadPos  + (sizeof (Header) * numSrcDataLines);

    destHeader = (Header*)(dest + destHeadPos);
    srcHeader  = (Header*)(src + srcHeadPos);

    for (int i=0; i<numSrcDataLines; i++)
    {
        while ((destHeader->m_pos + 
                destHeader->m_size - 1) < srcHeader->m_pos)
        {
            destDataPos += (m_bytesPerPixel*(destHeader->m_size));
            destHeader++;
        }

        // copy at proper pos
        bytesToCopy = m_bytesPerPixel * (srcHeader->m_size);
        int posInDataLine = (srcHeader->m_pos - 
                             destHeader->m_pos) * m_bytesPerPixel;
        for (int j=0; j<bytesToCopy; j++)
        {
            pixelByte = *(dest + destDataPos + posInDataLine + j);
            pixelByte += *(src + srcDataPos++);

            if (0xff < pixelByte)
            {
                pixelByte = 0xff;
            }

            *(dest + destDataPos + posInDataLine + j) = pixelByte;
        }

        srcHeader++;
    }

    return returnVal;
}

byte* ImageData::ConstructImage (byte* src,
				                 liveVizRequest& req)
{
    int numDataLines = 0;
    int headPos      = 0;
    int dataPos      = 0;
    int imageSize    = 0;
    int imagePos     = 0;
    int numBytesToCopy = 0;
    byte* image      = NULL;
    Header* head     = NULL;
	
    numDataLines = *((int*)src);
    memcpy (&req, src+sizeof (int), sizeof (liveVizRequest));

    headPos = sizeof (int) +
              sizeof (liveVizRequest);

    dataPos = headPos + (sizeof (Header) * numDataLines);

    imageSize = req.wid*req.ht*m_bytesPerPixel;

    image = new byte [imageSize];

    if (NULL == image)
    {
        goto EXITPOINT;
    }

    memset (image, 0, imageSize);

    head = (Header*) (src + headPos);

    for (int i=0; i<numDataLines; i++)
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

byte* ImageData::GetClippedImage (const byte* img,
                                  const int& startx,
                                  const int& starty,
                                  const int& sizex,
                                  const int& sizey,
                                  const liveVizRequest* req)
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
        (startx > (req->wid-1)) ||
        (starty > (req->ht-1)))
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

    if ((m_starty+m_sizey) > req->ht)
    {
        m_sizey = req->ht - m_starty;
    }

    // need to shift data for other 2 cases
    if (0 > startx)
    {
        m_sizex += startx;
        m_startx = 0;
        shift     = true;
    }

    if ((m_startx + m_sizex) > req->wid)
    {
        m_sizex  = req->wid - m_startx;
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

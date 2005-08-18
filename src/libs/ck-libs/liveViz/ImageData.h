/*
Vikas Mehta's utility routines for packing,
unpacking, and compositing liveViz images.

*/
#ifndef __IMAGEDATA_H
#define __IMAGEDATA_H

#include "liveViz.h"

typedef liveVizCombine_t ImageDataCombine_t;

class ImageData
{
    public:
        ImageData (int bytesPerPixel);
 
        ~ImageData ();

        /*
           This function calculates the size of buffer needed to hold the 
           image data in lines format.
        */
        int GetBuffSize (const int& startx,
                         const int& starty,
                         const int& sizex,
                         const int& sizey,
                         const int req_wid,
						 const int req_ht,
                         const byte* src);

        /*
           This function returns the pre-calculated size of image data buffer
           holding the image data in lines format. To calulate image buff size
           use GetBuffSize ().
        */
        inline int GetImageDataSize (void)
        {
            return m_size;
        }

	/**
	  This header is stored as the first thing in our buffer.
	*/
	class ImageHeader {
	public:
		/// Number of lines of data to follow.
		int m_lines;
		
		/// Image combiner to use.
		ImageDataCombine_t m_combine;
		
		/// Request this image came from
		liveVizRequest m_req;
		
		// ... m_lines LineHeaders follow ...
		// ... m_lines runs of pixel data follow ...
	};
	
	/**
	  Describes a row of image data.
	*/
	class LineHeader {
	public:
	    /// Start position of our row -- offset into image.
	    ///   measured in pixels, e.g., pos = off_y * wid + off_x.
	    int m_pos;  
	    /// Length of our row, in pixels.
	    int m_size;
	};
	
        /*
           This function must be called once GetClippedImage() is called 
           (because GetClippedImage() initializes some member variables used 
           by this function). It has 2 roles, one when "dest == NULL", it 
           returns size of buffer needed to convert the input image (rectangle)
           to lines format. When "dest" is not null, it converts the clipped 
           image into lines format and stores it into the buffer pointed by 
           "dest". Format of image data buffer:
    
           -------------------------------------------------------------------
          |                    |LineHeaders:|          |          |   |       |
          |    ImageHeader     |h1|h2|...|hn|Line1 Data|Line2 Data|...|Line n |
          |                    |  |  |   |  |          |          |   |Data   |
           -------------------------------------------------------------------
        */
        int AddImage (const int req_wid,
                      byte* dest        = NULL);
	
	void WriteHeader(ImageDataCombine_t combine,
                      const liveVizRequest* req,
                      byte* dest);
	void ReadHeader(ImageDataCombine_t &combine,
                      liveVizRequest* req,
                      const byte* src);
				  
        /*
           This function calculates the size of buffer required to fit the
           merged data from nMsg input msgs. It also sets some member 
           variables, it must be called before CombineImageData () is called.           
        */
        int CombineImageDataSize (int nMsg, CkReductionMsg **msgs);

        /*
           This function copies image data from n-input msgs to 'dest' buffer. 
        */
        void CombineImageData (int nMsg, CkReductionMsg **msgs, byte* dest);


        /*
           This function takes image data in lines format as input along, with
           "liveVizRequest" object. It allocates a buffer to hold the 
           image (rectangle) and copies data from input image data buffer
           to image buffer.
        */
        byte* ConstructImage (byte* src, 
                              liveVizRequest& req);

    private:

        /*
           This function copies, image data from input CkReduction msg to
           "dest" buff. This function is called from CombineImageData() once,
           all headers from all input reduction msgs are merged properly into
           "dest" buff. Here 'n' indicates, number of lines of data in 'dest'
           buff.
        */
        int CopyImageData (byte* dest, int n, const CkReductionMsg* msg,
                           ImageDataCombine_t reducer);


        /*
           This function clips the input image. In some cases, it allocates
           a buffer for clipped image (freed by destructor).
        */
        byte* GetClippedImage (const byte* img,
                               const int& startx,
                               const int& starty,
                               const int& sizex,
                               const int& sizey,
                               const int req_wid,
							   const int req_ht
							   );



        /*
           This function checks whether input pixel is black or not.
        */
        inline bool IsPixelOn (const byte* pixel)
        {
            bool isOn = false;
			
            for (int i=0; i<m_bytesPerPixel; i++)
            {
                if (0 != pixel [i])
                {
                    isOn = true;
                }
            }

            return isOn;
        }

        /*
           This function copies src pixel to dest pixel.
        */
        inline void CopyPixel (byte* dest, const byte* src)
        {
            int i = 0;
            while (i < m_bytesPerPixel)
            {
                *dest++ = *src++;
                i++;
            }
        }

        inline int NumNonNullLists (int* pos, 
                                    const int* size,
                                    const int& n)
        {
            int returnVal = 0;

            for (int i=0; i<n; i++)
            {
                if ((-1 != pos [i]) && 
                    (pos [i] != size [i]))
                {
                    returnVal ++;
                }
                else
                {
                    pos [i] = -1;
                }
            }

            return returnVal;
        }

        inline int GetNonNullListIndex (const int* pos,
                                        const int& n)
        {
            for (int i=0; i<n; i++)
            {
                if (-1 != pos [i])
                {
                    return i;
                }
            }
            return -1; 
        }
	

        /*
           Number of bytes per pixel of image data.
	   Normally 1 (grayscale) or 3 (RGB).
        */
        int		   m_bytesPerPixel;
       
        /*
           Total size of image data buffer, in bytes.
	   Includes all headers and image data.
        */
        int		   m_size;

        /*
           Total size of all our header data, in bytes.
        */
        int                m_headerSize;
	
        /*
           number of lines of image data
        */
        int		   m_numDataLines;
	
	
	/// Make our image size be this many lines and pixels
	void SetSize(int nLines,int nPixels) {
		m_numDataLines = nLines;
		m_headerSize=sizeof (ImageHeader) +
		      (sizeof (LineHeader) * nLines);
		m_size=m_headerSize +
		      (m_bytesPerPixel * nPixels);
	}
	
	/// Get the header for this line of this compressed image.
	LineHeader *getHeader(void *src,int lineNo) {
		return (LineHeader *)(((byte *)src)
			+sizeof(ImageHeader)
			+lineNo*sizeof(LineHeader)
		);
	}
	
	
        // members used while image combine operation

        /*
           points to buffer holding merged header
        */
        byte*   m_header;


        // initialized by GetClippedImage ()

        /*
           points to clipped image buffer
        */
        byte*   m_clippedImage;

        /*
           tells if buffer pointed to by "m_clippedImage" was allocated afresh.
        */
        bool    m_clippedImageAllocated;

        /*
           starting x coordinate of clipped image
        */
        int     m_startx;

        /*
           starting y coordinate of clipped image
        */
        int     m_starty;

        /*
           width of clipped image
        */
        int     m_sizex;
 
        /*
           height of clipped image
        */
        int     m_sizey;

        /*
           this constuctor should not be used.
        */
        ImageData () {};
};

#endif


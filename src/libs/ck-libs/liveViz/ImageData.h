#ifndef __IMAGEDATA_H
#define __IMAGEDATA_H

#include "liveViz0.h"
#include "liveViz.decl.h"

typedef struct
{
    int m_pos;  // start position
    int m_size; // size in pixels
} Header;

enum {
  sum_image_pixels,
  max_image_pixels
};

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
                         const liveVizRequest* req,
                         const byte* src);

        /*
           This function returns the image data buffer holding image data in
           lines format.
        */
        inline byte* GetImageData (void)
        {
            return m_imageData;
        }

        /*
           This function returns the pre-calculated size of image data buffer
           holding the image data in lines format. To calulate image buff size
           use GetBuffSize ().
        */
        inline int GetImageDataSize (void)
        {
            return m_size;
        }

		
        /*
           This function must be called once GerClippedImage() is called 
           (because GetClippedImage() initializes some member variables used 
           by this function). It has 2 roles, one when "dest == NULL", it 
           returns size of buffer needed to convert the input image (rectangle)
           to lines format. When "dest" is not null, it converts the clipped 
           image into lines format and stores it into the buffer pointed by 
           "dest". Format of image data buffer:
    
           -------------------------------------------------------------------
          |num  |              |  |  |   |  |          |          |   |       |
          | of  |liveVIzRequest|h1|h2|...|hn|Line1 Data|Line2 Data|...|Line n |
          |Lines|              |  |  |   |  |          |          |   |Data   |
           -------------------------------------------------------------------

        */
        int AddImage (const liveVizRequest* req,
                      byte* dest        = NULL);
				  
        /*
           This function calculates the size of buffer required to fit the
           merged data from nMsg input msgs. It also sets some member 
           variables, it must be called before CombineImageData () is called.           
        */
        int CombineImageDataSize (int nMsg, CkReductionMsg **msgs);

        /*
           This function copies image data from n-input msgs to 'dest' buffer. 
        */
        void CombineImageData (int nMsg, CkReductionMsg **msgs, byte* dest,
                               int reducer);


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
        int CopyImageData (byte* dest, int n, CkReductionMsg* msg,
                           int reducer);


        /*
           This function clips the input image. In some cases, it allocates
           a buffer for clipped image (freed by destructor).
        */
        byte* GetClippedImage (const byte* img,
                               const int& startx,
                               const int& starty,
                               const int& sizex,
                               const int& sizey,
                               const liveVizRequest* req);



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
           number of bytes per pixel
        */
        int		   m_bytesPerPixel;
 
        /*
           points to image data buffer holding image data in lines format.
        */
        byte*	  m_imageData;
       
        /*
           size of image data buffer (m_imageData)
        */
        int		   m_size;

        /*
           number of lines of image data
        */
        int		   m_numDataLines;

        // members used while image combine operation

        /*
           points to buffer holding merged header
        */
        byte*   m_header;

        /*
           holds size of m_header buff in bytes
        */
        int     m_headerSize;


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


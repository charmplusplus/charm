About liveViz
-------------

There are various types of reducers available in charm++, these reducers (mostly) work on numeric data. If array elements compute a small piece of a large 2D-image, then these chucks can be combined to form one large image using liveViz. In other words, liveViz is a reducer for 2D-image data, which combines small chunks of images deposited by chares into one large image.

This visualization library follows client server model, i.e. image assembly is done on server side and final image can be viewed using a java based liveViz client avaliable at:
           .../charm/bin/liveViz localhost 1234 

A sample liveViz server example is available at:
           .../charm/pgms/charm++/ccs/liveViz/server

The Sample liveViz Client:
-------------------------

The source for the sample client is available in .../charm/java/charm/liveViz


How to use liveViz with Charm++ program:
---------------------------------------

A program must provide a chare array with one entry fuction having following prototype:

  void function_name (liveVizRequestMsg *m);

This entry method is supposed to deposit its (array elements) chunck of image. This entry method has following structure:

  void function_name (liveVizRequestMsg *m)
  {
    // prepare image chunk
       ...

    liveVizDeposit (m, startX, startY, width, height, image_buff, this)

    // delete image buffer if it was dynamically allocated
  }

To know the width and height of image data m->req.wid and m->req.ht can be used, here 'm' is the pointer to 'liveVizRequestMsg'. 

Format of deposit image:
-----------------------

image_buff is run of bytes representing image. if image is gray-scale each byte represents one pixel otherwise 3 consecutive bytes (starting at array index which is a multiple of 3) represents a pixel.

liveViz Initialization:
----------------------

liveViz library needs to be initialized before it can be used for visualization.
for initialization follow the following steps:

    - create the chare array (array proxy object 'a') with the entry method
      'function_name' (described above).
    - create a CkCallback object ('c'), specifying 'function_name' as the 
      callback function.
    - create a liveVizConfig object ('cfg'). if image to be deposited is color 
      image (3 bytes per pixel) then liveVizConfig object need to be initialized
      as:

         liveVizConfig cfg (true); // first parameter specifies color/gray-scale
                                   // image

    - call liveVizInit (cfg, a, c)

Compilation:
-----------

Charm++ program using liveViz must be linked with '-module liveViz'. Before
compiling liveViz program, liveViz library needs to be compiled. To compile
liveViz library:

    - go to .../charm/tmp/libs/ck-libs/liveViz
    - make

Extensions:
----------

In above example, liveViz client repeated (or once) asks for image, getting
this request liveViz library asks array elements for image chunks, combines
image chunks and sends the image to client.

There is another method by which server sends image to client (whether it
requested it or not). Fot this mode, second parameter passed to liveVizConfig
constructor must be true. Further details on how to use this mode will be
updated soon.

For 3-d images, third parameter to liveVizConfig constructor needs to be
'true'. Further details for this also will be updated soon. 

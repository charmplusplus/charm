/**
  Builtin, default float to RGB conversion function.
  Users can write their own (with the same name)
  and it will override this version.
*/
#include "liveViz.h"

void liveVizFloatToRGB(liveVizRequest &req, 
	const float *floatSrc, unsigned char *destRgb,
	int nPixels)
{
	for (int i=0;i<nPixels;i++) {
		colorScale(floatSrc[i],0,1,&destRgb[3*i]);
	}
}

/*
  Data types, function prototypes,  etc. exported by liveViz.
  This layer does image assembly, and is the most commonly-used 
  interface to liveViz.
*/
#ifndef __UIUC_CHARM_COLORSCALE_H
#define __UIUC_CHARM_COLORSCALE_H

/*
  A helper routine, call it with the value you have,
  a minimum, a maximum, and 3 unsigned bytes to stuff the RGB result
  in, and it'll compute the color for you.
*/
void colorScale(double val,
                double min,
		double max,
		unsigned char intensity[3]);

#endif /* def(thisHeader) */

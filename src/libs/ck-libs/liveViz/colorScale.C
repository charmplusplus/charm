#include <charm++.h>
#include <math.h>
#include "colorScale.h"

// Scale a value between a min and max bound into a useful color.
// goes red -> purple through the rainbow.
// Out of bound colors get mapped to white.
void colorScale(double val,
                double min,
		double max,
                unsigned char intensity[3])
{
  // Adjust the scale so it runs 0 ... some max
  max -= min;
  val -= min;

  // Compute how far up the scale it is
  double at = (val / max) * 5.0;

  // Compute the intensity based on that
  intensity[0] = intensity[1] = intensity[2] = 0;
  double rem = at - floor(at);
  switch ( (int)floor(at) ) {
    case 0:
      // Start at red and gain green (-> yellow)
      intensity[0] = (unsigned char) 255;
      intensity[1] = (unsigned char) (rem * 255.0);
      break;
    case 1:
      // Start at yellow and lose red (-> green)
      intensity[0] = (unsigned char) (255.0 - rem * 255.0);
      intensity[1] = (unsigned char) 255;
      break;
    case 2:
      // Start at green and add blue (-> cyan)
      intensity[1] = (unsigned char) 255;
      intensity[2] = (unsigned char) (rem * 255.0);
      break;
    case 3:
      // Start at cyan and lose green (-> blue)
      intensity[1] = (unsigned char) (255.0 - rem * 255.0);
      intensity[2] = (unsigned char) 255;
      break;
    case 4:
      // Start at blue and add red (-> purple)
      intensity[0] = (unsigned char) (rem * 255.0);
      intensity[2] = (unsigned char) 255;
      break;
    default:
      if ( rem <= 0.000001 ) {
	intensity[0] = intensity[2] = (unsigned char) 255;
      } else {
        // ckerr << "colorScale error: value out of range."
        //       << "Setting pixel color to white" << endl;
        intensity[0] = intensity[1] = intensity[2] = 255;
      }
      break;
  }
}

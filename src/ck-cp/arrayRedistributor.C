#include <charm++.h>
#include <cmath>
#include <math.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <map>
#include <set>
#include <vector>
#include <utility>
#include <limits>
//#include <sys/time.h>
#include <float.h>

//#include "ControlPoints.decl.h"
#include "trace-controlPoints.h"
#include "LBDatabase.h"
#include "controlPoints.h"
#include "pathHistory.h"
#include "arrayRedistributor.h"


/**
 *  \addtogroup ControlPointFramework
 *   @{
 *
 */

#if CMK_WITH_CONTROLPOINT


using namespace std;



/// The index in the global array for my top row  
int redistributor2D::top_data_idx(){ 
  return (data_height * thisIndex.y) / y_chares; 
} 
 
int redistributor2D::bottom_data_idx(){ 
  return ((data_height * (thisIndex.y+1)) / y_chares) - 1; 
} 
 
int redistributor2D::left_data_idx(){ 
  return (data_width * thisIndex.x) / x_chares; 
} 
 
int redistributor2D::right_data_idx(){ 
  return ((data_width * (thisIndex.x+1)) / x_chares) - 1; 
} 
 
int redistributor2D::top_neighbor(){ 
  return (thisIndex.y + y_chares - 1) % y_chares; 
}  
   
int redistributor2D::bottom_neighbor(){ 
  return (thisIndex.y + 1) % y_chares; 
} 
   
int redistributor2D::left_neighbor(){ 
  return (thisIndex.x + x_chares - 1) % x_chares; 
} 
 
int redistributor2D::right_neighbor(){ 
  return (thisIndex.x + 1) % x_chares; 
} 
  
  
/// the width (X dimension) of the non-ghost part of the local partition 
int redistributor2D::mywidth(){ 
  if(thisElemActive)
    return right_data_idx() - left_data_idx() + 1; 
  else
    return 0;
} 
   
   
/// the height (Y dimension) of the non-ghost part of the local partition 
int redistributor2D::myheight(){ 
  if(thisElemActive)
    return bottom_data_idx() - top_data_idx() + 1; 
  else
    return 0;
} 






/*! @} */

#endif

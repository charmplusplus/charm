
#ifndef __PARFUM_TOPS_TYPES___H
#define __PARFUM_TOPS_TYPES___H

#include <ParFUM_Types.h> // for ElemID

#ifdef FP_TYPE_FLOAT
#warning "Using floats for various things"
typedef float FP_TYPE;
typedef float FP_TYPE_HIGH;
typedef float FP_TYPE_LOW;
typedef float FP_TYPE_SYNC;
#else
#warning "Using doubles for various things"
typedef double FP_TYPE;
typedef double FP_TYPE_HIGH;
typedef double FP_TYPE_LOW;
typedef double FP_TYPE_SYNC;
#endif

/** Hardware device identifiers; used to select which device kernels will be run on */
enum TopDevice {
    DeviceNone,
    DeviceCPU,
    DeviceGPU
};

/** Tops uses some bit patterns for these, but we just use TopNode as a signed value to represent the corresponding ParFUM node. A non-negative value is a local node, while a negative value is a ghost. */
typedef long TopNode;

/** A type for a Vertex (would be different from nodes if using quadratic elements) */
typedef TopNode TopVertex;

/** A type for an element */
typedef ElemID TopElement;

/** A type for a facet */
class TopFacet{
public:
	TopNode node[6];
	TopElement elem[2];
	
	bool operator==( const TopFacet& other){
		return   
					this->node[0] == other.node[0] &&
                	this->node[1] == other.node[1] &&
                	this->node[2] == other.node[2] &&
                	this->node[3] == other.node[3] &&
                	this->node[4] == other.node[4] &&
                	this->node[5] == other.node[5] &&
                	this->elem[0] == other.elem[0] &&
                	this->elem[1] == other.elem[1] ;
	}
	
};


/** Enumerates the possible tops element types. Note that all bulk types come
 * first, then all cohesive types, starting with TOP_ELEMENT_MIN_COHESIVE.
 * This allows us to determine whether an element type is cohesive or bulk
 * by comparing it to TOP_ELEMENT_MIN_COHESIVE
 */
enum TopElementType {
  TOP_ELEMENT_T3 =0,
  TOP_ELEMENT_T6,
  TOP_ELEMENT_Q4,
  TOP_ELEMENT_Q8,
  TOP_ELEMENT_TET4,
  TOP_ELEMENT_TET10,
  TOP_ELEMENT_HEX8,
  TOP_ELEMENT_HEX8_RESERVOIR,
  TOP_ELEMENT_HEX20,
  TOP_ELEMENT_WEDGE15,
  TOP_ELEMENT_WEDGE6,
  TOP_ELEMENT_MIN_COHESIVE,
  TOP_ELEMENT_COH2E2,
  TOP_ELEMENT_COH2E3,
  TOP_ELEMENT_COH3T3,
  TOP_ELEMENT_COH3T6,
  TOP_ELEMENT_COH3Q4,
  TOP_ELEMENT_COH3Q8,
  TOP_ELEMENT_MAX
};

/** used as iterators on CUDA system. See usage!*/
typedef bool TopNodeItr_D;
typedef bool TopElemItr_D;


/** an opaque id for top entities */
typedef int TopID;

/** an enumeration of supported element types */
typedef int TopElemType;



#endif

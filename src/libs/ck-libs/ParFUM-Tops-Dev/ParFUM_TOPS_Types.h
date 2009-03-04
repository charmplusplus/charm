
#ifndef __PARFUM_TOPS_TYPES___H
#define __PARFUM_TOPS_TYPES___H


#ifdef FP_TYPE_FLOAT
//#warning "Using floats for various things"
typedef float FP_TYPE;
typedef float FP_TYPE_HIGH;
typedef float FP_TYPE_LOW;
typedef float FP_TYPE_SYNC;
#else
//#warning "Using doubles for various things"
typedef double FP_TYPE;
typedef double FP_TYPE_HIGH;
typedef double FP_TYPE_LOW;
typedef double FP_TYPE_SYNC;
#endif



/** Tops uses some bit patterns for these, but we just use TopNode as a signed value to represent the corresponding ParFUM node. A non-negative value is a local node, while a negative value is a ghost. */
typedef long TopNode;

/** A type for a Vertex (would be different from nodes if using quadratic elements) */
typedef TopNode TopVertex;

/** A type for a node */
class TopElement{
public:
	long type; // Should be BULK_ELEMENT or COHESIVE_ELEMENT
	long idx; 
};

#define BULK_ELEMENT 0
#define COHESIVE_ELEMENT 1


/** A type for a facet */
class TopFacet{
public:
	TopNode node[6];
	TopElement elem[2];
};


enum {
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
  TOP_ELEMENT_COH2E2,
  TOP_ELEMENT_COH2E3,
  TOP_ELEMENT_COH3T3,
  TOP_ELEMENT_COH3T6,
  TOP_ELEMENT_COH3Q4,
  TOP_ELEMENT_COH3Q8,
  TOP_ELEMENT_WEDGE6,
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

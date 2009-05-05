
#ifndef PARFUM_ITERATORS_TYPES_H
#define PARFUM_ITERATORS_TYPES_H

#include <ParFUM_Types.h> // for ElemID

// Compile-time choice of precision
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

/** Hardware device identifiers; used to select which device kernels will be
 * run on */
enum MeshDevice {
    DeviceNone,
    DeviceCPU,
    DeviceGPU
};

/** MeshNode is a signed value that corresponds to a ParFUM node. A
 * non-negative value is a local node, while a negative value is a ghost. */
typedef long MeshNode;

/** A type for a Vertex (would be different from nodes if using quadratic
 * elements) */
typedef MeshNode MeshVertex;

/** A type for an element */
typedef ElemID MeshElement;

/** A type for a facet */
class MeshFacet {
    public:
        MeshNode node[6];
        MeshElement elem[2];

        bool operator==( const MeshFacet& other){
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


/** Enumerates the possible element types. Note that all bulk types come
 * first, then all cohesive types, starting with MESH_ELEMENT_MIN_COHESIVE.
 * This allows us to determine whether an element type is cohesive or bulk
 * by comparing it to MESH_ELEMENT_MIN_COHESIVE
 */
enum MeshElementType {
  MESH_ELEMENT_T3 =0,
  MESH_ELEMENT_T6,
  MESH_ELEMENT_Q4,
  MESH_ELEMENT_Q8,
  MESH_ELEMENT_TET4,
  MESH_ELEMENT_TET10,
  MESH_ELEMENT_HEX8,
  MESH_ELEMENT_HEX8_RESERVOIR,
  MESH_ELEMENT_HEX20,
  MESH_ELEMENT_WEDGE15,
  MESH_ELEMENT_WEDGE6,
  MESH_ELEMENT_MIN_COHESIVE,
  MESH_ELEMENT_COH2E2,
  MESH_ELEMENT_COH2E3,
  MESH_ELEMENT_COH3T3,
  MESH_ELEMENT_COH3T6,
  MESH_ELEMENT_COH3Q4,
  MESH_ELEMENT_COH3Q8,
  MESH_ELEMENT_MAX
};

/** an opaque id for any entity */
typedef int EntityID;

#endif

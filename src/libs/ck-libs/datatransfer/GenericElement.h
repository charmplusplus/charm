/**
Mesh utility routines:
  - Element shape and interpolation functions

Originally written by Mike Campbell, 2003.
Interface modified for Charm integration by Orion Lawlor, 2004.
*/
#ifndef _CSAR_GenericElement_H_
#define _CSAR_GenericElement_H_

#include "ckvector3d.h"
typedef CkVector3d CVector;
typedef CkVector3d CPoint;

/**
 Class to encapsulate a single actual element's nodes.
 This is GenericElement's interface to real elements.
*/
class ConcreteElement {
public:
	/** Return the location of the i'th node of this element. */
	virtual CPoint getNodeLocation(int i) const =0;
};

/**
 Class to encapsulate an element with nodes and node-centered data.
*/
class ConcreteElementNodeData : public ConcreteElement {
public:
	/** Return the vector of data associated with our i'th node. */
	virtual const double *getNodeData(int i) const =0;
};

/**
 Class to encapsulate all element-type specific methods.
 For example, there would be one GenericElementType for 
 tets, another for hexes, another for 10-node tets, etc.
 
 Terminology:
   Natural Coordinates are a subset of the unit cube that 
     can be mappped onto the element.
   The Shape Function at a given location lists the weights
     for each node used to interpolate values.
   The Jacobian lists the partials of real coordinates with
     respect to natural coordinates.
*/
class GenericElement {
protected:
  enum {maxSize=20}; // Maximum number of nodes per element we can handle
  unsigned int _size; ///< Number of nodes in element
public:
  GenericElement(unsigned int s = 4)
    : _size(s)
  {};
  unsigned int size() const 
  {
    return(_size);
  };
  unsigned int nedges() const
  {
    switch(_size){
    case 4:
    case 10:
      return (6);
    case 8:
    case 20:
      return (12);
    default:
      return(0);
    }
    return(0);
  };
  unsigned int nfaces() const
  {
    switch(_size){
    case 4:
    case 10:
      return(4);
    case 8:
    case 20:
      return(6);
    default:
      return(0);
    }
    return(0);
  };
  void shape_func(const CVector &natc,
		  double []) const;
  void dshape_func(const CVector &natc,
		   double [][3]) const;
  void jacobian(const CPoint p[],
		const CVector &natc,
		CVector J[]) const;
  
  /// Interpolate nValuesPerNode doubles from src element at nc to dest.
  void interpolate_natural(int nValuesPerNode,
  		   const ConcreteElementNodeData &src, // Source element
		   const CVector &nc,
		   double *dest) const;
  
  /// Interpolate nValuesPerNode doubles from src element at point p to dest.
  bool interpolate(int nValuesPerNode,
  		   const ConcreteElementNodeData &src, // Source element
		   const CPoint &p,
		   double *dest) const
  {
    CVector natc;
    if (!element_contains_point(p,src,natc)) return false;
    interpolate_natural(nValuesPerNode,src,natc,dest);
    return true;
  }

  /// Return true if this element contains this point, and return
  ///  the point's natural coordinates.
  bool element_contains_point(const CPoint &p, //    Target Mesh point
	                      const ConcreteElement &e, // Source element
			      CVector &natc) const; // Returns Target point natural coords

  /// Evaluate the element's jacobian at this natural coordinate
  void shapef_jacobian_at(const CPoint &p,
			  CVector &natc,
	                  const ConcreteElement &e, // Source element
			  CVector &fvec,CVector fjac[]) const;
  
};

#endif

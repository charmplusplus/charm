// Magic Software, Inc.
// http://www.magic-software.com
// Copyright (c) 2000-2003.  All Rights Reserved
//
// Source code from Magic Software is supplied under the terms of a license
// agreement and may not be copied or disclosed except in accordance with the
// terms of that agreement.  The various license agreements may be found at
// the Magic Software web site.  This file is subject to the license
//
// FREE SOURCE CODE
// http://www.magic-software.com/License/free.pdf

#ifndef MGCINTR3DTETRTETR_H
#define MGCINTR3DTETRTETR_H

#include "MgcTetrahedron.h"
#include <vector>

namespace Mgc {

/**
  Accept a tetrahedron for further processing.
*/
class TetrahedronConsumer {
public:
	virtual ~TetrahedronConsumer();
	
	/// Take this tet.
	virtual void Add(const Tetrahedron& tet) =0;
};


/**
  Sums up volume of tetrahedra passed to it.
*/
class TetrahedronVolumeConsumer : public TetrahedronConsumer {
	double volume;
/**
 * Return the volume of the tetrahedron with these vertices.
 */
double tetVolume(const Vector3 &A,const Vector3 &B,
                const Vector3 &C,const Vector3 &D) 
{
        const static double oneSixth=1.0/6.0;
        return oneSixth*(B-A).Dot((D-A).Cross(C-A));
}
public:
	TetrahedronVolumeConsumer() :volume(0.0) {}
	void Add(const Tetrahedron &kT2) {
		volume += fabs(tetVolume(kT2[0],kT2[1],kT2[2],kT2[3]));
	}
	operator double () {return volume;}
};


MAGICFM void FindIntersection (const Tetrahedron& rkT0,
    const Tetrahedron& rkT1, TetrahedronConsumer &dest);

}  // namespace Mgc

#endif




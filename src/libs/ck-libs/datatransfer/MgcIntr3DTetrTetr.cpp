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

#include "MgcIntr3DTetrTetr.h"
using namespace Mgc;
using namespace std;

namespace Mgc {
TetrahedronConsumer::~TetrahedronConsumer() {}

/**
  Split tets by this plane, keeping only the peices that are inside the plane.
  A tet is considered inside the plane if all the points
   of the tet have negative distance (rkPlane.DistanceTo).
*/
class PlaneSplitTetrahedronConsumer : public TetrahedronConsumer {
	const Plane& rkPlane;
	TetrahedronConsumer &dest;
public:
	PlaneSplitTetrahedronConsumer(const Plane& rkPlane_,TetrahedronConsumer &dest_)
		:rkPlane(rkPlane_), dest(dest_) {}
	
	/// Pass to dest the fragments of this tet that like on our
	///  side of the plane.
	virtual void Add(const Tetrahedron& tet);
};
}

//----------------------------------------------------------------------------
void Mgc::PlaneSplitTetrahedronConsumer::Add(const Tetrahedron &kTetraP)
{
    // determine on which side of the plane the points of the tetrahedron lie
    Real afC[4];
    int i, aiP[4], aiN[4], aiZ[4];
    int iPositive = 0, iNegative = 0, iZero = 0;

    for (i = 0; i < 4; i++)
    {
        afC[i] = rkPlane.DistanceTo(kTetraP[i]);
        if ( afC[i] > 0.0f )
            aiP[iPositive++] = i;
        else if ( afC[i] < 0.0f )
            aiN[iNegative++] = i;
        else
            aiZ[iZero++] = i;
    }

    // For a split to occur, one of the c_i must be positive and one must
    // be negative.

    if ( iNegative == 0 )
    {
        // tetrahedron is completely on the positive side of plane, full clip
        return;
    }

    if ( iPositive == 0 )
    {
        // tetrahedron is completely on the negative side of plane
        dest.Add(kTetraP);
        return;
    }

    // Tetrahedron is split by plane.  Determine how it is split and how to
    // decompose the negative-side portion into tetrahedra (6 cases).
    Real fW0, fW1, fInvCDiff;
    Vector3 akIntp[4];
    Tetrahedron kTetra(kTetraP); // so we can modify kTetra's points

    if ( iPositive == 3 )
    {
        // +++-
        for (i = 0; i < iPositive; i++)
        {
            fInvCDiff = 1.0f/(afC[aiP[i]] - afC[aiN[0]]);
            fW0 = -afC[aiN[0]]*fInvCDiff;
            fW1 = +afC[aiP[i]]*fInvCDiff;
            kTetra[aiP[i]] = fW0*kTetra[aiP[i]] + fW1*kTetra[aiN[0]];
        }
        dest.Add(kTetra);
    }
    else if ( iPositive == 2 )
    {
        if ( iNegative == 2 )
        {
            // ++--
            for (i = 0; i < iPositive; i++)
            {
                fInvCDiff = 1.0f/(afC[aiP[i]]-afC[aiN[0]]);
                fW0 = -afC[aiN[0]]*fInvCDiff;
                fW1 = +afC[aiP[i]]*fInvCDiff;
                akIntp[i] = fW0*kTetra[aiP[i]] + fW1*kTetra[aiN[0]];
            }
            for (i = 0; i < iNegative; i++)
            {
                fInvCDiff = 1.0f/(afC[aiP[i]]-afC[aiN[1]]);
                fW0 = -afC[aiN[1]]*fInvCDiff;
                fW1 = +afC[aiP[i]]*fInvCDiff;
                akIntp[i+2] = fW0*kTetra[aiP[i]] + fW1*kTetra[aiN[1]];
            }

            kTetra[aiP[0]] = akIntp[2];
            kTetra[aiP[1]] = akIntp[1];
            dest.Add(kTetra);

            dest.Add(Tetrahedron(kTetra[aiN[1]],akIntp[3],akIntp[2],
                akIntp[1]));

            dest.Add(Tetrahedron(kTetra[aiN[0]],akIntp[0],akIntp[1],
                akIntp[2]));
        }
        else
        {
            // ++-0
            for (i = 0; i < iPositive; i++)
            {
                fInvCDiff = 1.0f/(afC[aiP[i]]-afC[aiN[0]]);
                fW0 = -afC[aiN[0]]*fInvCDiff;
                fW1 = +afC[aiP[i]]*fInvCDiff;
                kTetra[aiP[i]] = fW0*kTetra[aiP[i]] + fW1*kTetra[aiN[0]];
            }
            dest.Add(kTetra);
        }
    }
    else if ( iPositive == 1 )
    {
        if ( iNegative == 3 )
        {
            // +---
            for (i = 0; i < iNegative; i++)
            {
                fInvCDiff = 1.0f/(afC[aiP[0]]-afC[aiN[i]]);
                fW0 = -afC[aiN[i]]*fInvCDiff;
                fW1 = +afC[aiP[0]]*fInvCDiff;
                akIntp[i] = fW0*kTetra[aiP[0]] + fW1*kTetra[aiN[i]];
            }

            kTetra[aiP[0]] = akIntp[0];
            dest.Add(kTetra);

            dest.Add(Tetrahedron(akIntp[0],kTetra[aiN[1]],
                kTetra[aiN[2]],akIntp[1]));

            dest.Add(Tetrahedron(kTetra[aiN[2]],akIntp[1],akIntp[2],
                akIntp[0]));
        }
        else if ( iNegative == 2 )
        {
            // +--0
            for (i = 0; i < iNegative; i++)
            {
                fInvCDiff = 1.0f/(afC[aiP[0]]-afC[aiN[i]]);
                fW0 = -afC[aiN[i]]*fInvCDiff;
                fW1 = +afC[aiP[0]]*fInvCDiff;
                akIntp[i] = fW0*kTetra[aiP[0]] + fW1*kTetra[aiN[i]];
            }

            kTetra[aiP[0]] = akIntp[0];
            dest.Add(kTetra);

            dest.Add(Tetrahedron(akIntp[1],kTetra[aiZ[0]],
                kTetra[aiN[1]],akIntp[0]));
        }
        else
        {
            // +-00
            fInvCDiff = 1.0f/(afC[aiP[0]]-afC[aiN[0]]);
            fW0 = -afC[aiN[0]]*fInvCDiff;
            fW1 = +afC[aiP[0]]*fInvCDiff;
            kTetra[aiP[0]] = fW0*kTetra[aiP[0]] + fW1*kTetra[aiN[0]];
            dest.Add(kTetra);
        }
    }
}

//----------------------------------------------------------------------------
// Return true if *all* the points of this tet lie outside of *any* of these planes 
bool allOutside(const Plane *p,int nPlane,const Tetrahedron& rk) 
{
    for (int l=0;l<nPlane;l++) {
        for (int v=0;v<4;v++) {
		if (p[l].DistanceTo(rk[v])<=0.0f) 
			goto tryNextPlane; // This point is inside-- forget it
	}
	// Every vertex is outside plane l: we're done
	return true;
    tryNextPlane:
        ;	
    }
    // If we got here, no plane outs every vertex
    return false;
}

//----------------------------------------------------------------------------
void Mgc::FindIntersection (const Tetrahedron& rkT0, const Tetrahedron& rkT1,
    Mgc::TetrahedronConsumer& dest)
{
    // build planar faces of T0
    const int planePer=4;
    Plane akPlane[planePer];
    rkT0.GetPlanes(akPlane);
    
/* early-exit test for non-overlapping tets */
    if (1) {
        Plane ak1Plane[planePer];
        rkT1.GetPlanes(ak1Plane);
        if (allOutside(akPlane,planePer,rkT1) || allOutside(ak1Plane,planePer,rkT0))
            return; /* tets do not overlap */
    }
    
  /* Build filter to successively clip tets by each plane of T0,
      passing the final tets to the user's destination.
   */
    PlaneSplitTetrahedronConsumer cons0(akPlane[0],dest);
    PlaneSplitTetrahedronConsumer cons1(akPlane[1],cons0);
    PlaneSplitTetrahedronConsumer cons2(akPlane[2],cons1);
    PlaneSplitTetrahedronConsumer cons3(akPlane[3],cons2);
    
  /* Pass T1 through the filter chain */
    cons3.Add(rkT1);
}
//----------------------------------------------------------------------------

#if 0

// Test program.  The first example illustrates when the minimum number of
// tetrahedra in an intersection (1).  The second example illustrates the
// maximum number of tetrahedra in an intersection (19).

#include <fstream>
int main ()
{
    vector<Tetrahedron> kIntr;
    Tetrahedron kT0, kT1;

    kT0[0] = Vector3(0.0f,0.0f,0.0f);
    kT0[1] = Vector3(1.0f,0.0f,0.0f);
    kT0[2] = Vector3(0.0f,1.0f,0.0f);
    kT0[3] = Vector3(0.0f,0.0f,1.0f);

    kT1[0] = Vector3(0.0f,0.0f,0.0f);
    kT1[1] = Vector3(1.0f,1.0f,0.0f);
    kT1[2] = Vector3(0.0f,1.0f,1.0f);
    kT1[3] = Vector3(1.0f,0.0f,1.0f);

    FindIntersection(kT0,kT1,kIntr);

    // kIntr[0]
    // (0.0,0.0,0.0)
    // (0.5,0.5,0.0)
    // (0.0,0.5,0.5)
    // (0.5,0.0,0.5)

    kT1[0] = Vector3(0.4f,0.4f,0.4f);
    kT1[1] = Vector3(-0.1f,0.25f,0.25f);
    kT1[2] = Vector3(0.25f,-0.1f,0.25f);
    kT1[3] = Vector3(0.25f,0.25f,-0.1f);

    FindIntersection(kT0,kT1,kIntr);

    // kIntr[0]
    // (0.275000,0.362500,0.362500)
    // (0.000000,0.280000,0.280000)
    // (0.280000,0.000000,0.280000)
    // (0.280000,0.280000,0.000000)
    //
    // kIntr[1]
    // (0.275000,0.362500,0.362500)
    // (0.280000,0.000000,0.280000)
    // (0.280000,0.280000,0.000000)
    // (0.362500,0.275000,0.362500)
    //
    // kIntr[2]
    // (0.280000,0.280000,0.000000)
    // (0.362500,0.275000,0.362500)
    // (0.362500,0.362500,0.275000)
    // (0.275000,0.362500,0.362500)
    //
    // kIntr[3]
    // (0.000000,0.280000,0.280000)
    // (0.280000,0.000000,0.280000)
    // (0.280000,0.280000,0.000000)
    // (0.000000,0.184211,0.257895)
    //
    // kIntr[4]
    // (0.280000,0.280000,0.000000)
    // (0.000000,0.184211,0.257895)
    // (0.000000,0.257895,0.184211)
    // (0.000000,0.280000,0.280000)
    //
    // kIntr[5]
    // (0.280000,0.000000,0.280000)
    // (0.000000,0.184211,0.257895)
    // (0.280000,0.280000,0.000000)
    // (0.150000,0.000000,0.250000)
    //
    // kIntr[6]
    // (0.000000,0.184211,0.257895)
    // (0.280000,0.280000,0.000000)
    // (0.150000,0.000000,0.250000)
    // (0.000000,0.257895,0.184211)
    //
    // kIntr[7]
    // (0.150000,0.000000,0.250000)
    // (0.000000,0.257895,0.184211)
    // (0.000000,0.150000,0.250000)
    // (0.000000,0.184211,0.257895)
    //
    // kIntr[8]
    // (0.280000,0.280000,0.000000)
    // (0.150000,0.000000,0.250000)
    // (0.257895,0.000000,0.184211)
    // (0.280000,0.000000,0.280000)
    //
    // kIntr[9]
    // (0.280000,0.280000,0.000000)
    // (0.000000,0.257895,0.184211)
    // (0.257895,0.000000,0.184211)
    // (0.150000,0.250000,0.000000)
    //
    // kIntr[10]
    // (0.000000,0.257895,0.184211)
    // (0.257895,0.000000,0.184211)
    // (0.150000,0.250000,0.000000)
    // (0.000000,0.180147,0.231618)
    //
    // kIntr[11]
    // (0.150000,0.250000,0.000000)
    // (0.000000,0.180147,0.231618)
    // (0.000000,0.250000,0.150000)
    // (0.000000,0.257895,0.184211)
    //
    // kIntr[12]
    // (0.257895,0.000000,0.184211)
    // (0.000000,0.180147,0.231618)
    // (0.150000,0.250000,0.000000)
    // (0.150000,0.000000,0.250000)
    //
    // kIntr[13]
    // (0.000000,0.180147,0.231618)
    // (0.150000,0.250000,0.000000)
    // (0.150000,0.000000,0.250000)
    // (0.000000,0.250000,0.150000)
    //
    // kIntr[14]
    // (0.150000,0.000000,0.250000)
    // (0.000000,0.250000,0.150000)
    // (0.000000,0.150000,0.250000)
    // (0.000000,0.180147,0.231618)
    //
    // kIntr[15]
    // (0.150000,0.250000,0.000000)
    // (0.150000,0.000000,0.250000)
    // (0.221429,0.000000,0.178571)
    // (0.257895,0.000000,0.184211)
    //
    // kIntr[16]
    // (0.221429,0.000000,0.178571)
    // (0.150000,0.250000,0.000000)
    // (0.250000,0.150000,0.000000)
    // (0.280000,0.280000,0.000000)
    //
    // kIntr[17]
    // (0.221429,0.000000,0.178571)
    // (0.250000,0.150000,0.000000)
    // (0.280000,0.280000,0.000000)
    // (0.250000,0.000000,0.150000)
    //
    // kIntr[18]
    // (0.280000,0.280000,0.000000)
    // (0.250000,0.000000,0.150000)
    // (0.257895,0.000000,0.184211)
    // (0.221429,0.000000,0.178571)

    return 0;
}

#endif


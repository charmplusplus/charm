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

//----------------------------------------------------------------------------
inline Vector3::Vector3 ()
{
    // For efficiency in construction of large arrays of vectors, the
    // default constructor does not initialize the vector.
}
//----------------------------------------------------------------------------
inline Real& Vector3::operator[] (int i) const
{
    return ((Real*)this)[i];
}
//----------------------------------------------------------------------------
inline Vector3::operator Real* ()
{
    return (Real*)this;
}
//----------------------------------------------------------------------------

// (OSL 2003/8/5: These routines are natural candidates for inlining...)
//----------------------------------------------------------------------------
inline Vector3::Vector3 (Real fX, Real fY, Real fZ)
{
    x = fX;
    y = fY;
    z = fZ;
}
//----------------------------------------------------------------------------
inline Vector3::Vector3 (Real afCoordinate[3])
{
    x = afCoordinate[0];
    y = afCoordinate[1];
    z = afCoordinate[2];
}
//----------------------------------------------------------------------------
inline Vector3::Vector3 (const Vector3& rkVector)
{
    x = rkVector.x;
    y = rkVector.y;
    z = rkVector.z;
}
//----------------------------------------------------------------------------
inline Vector3& Vector3::operator= (const Vector3& rkVector)
{
    x = rkVector.x;
    y = rkVector.y;
    z = rkVector.z;
    return *this;
}

//----------------------------------------------------------------------------
inline Vector3 Vector3::operator+ (const Vector3& rkVector) const
{
    return Vector3(x+rkVector.x,y+rkVector.y,z+rkVector.z);
}
//----------------------------------------------------------------------------
inline Vector3 Vector3::operator- (const Vector3& rkVector) const
{
    return Vector3(x-rkVector.x,y-rkVector.y,z-rkVector.z);
}
//----------------------------------------------------------------------------
inline Vector3 Vector3::operator* (Real fScalar) const
{
    return Vector3(fScalar*x,fScalar*y,fScalar*z);
}

//----------------------------------------------------------------------------
inline Vector3 Vector3::operator- () const
{
    return Vector3(-x,-y,-z);
}
//----------------------------------------------------------------------------
inline Vector3 operator* (Real fScalar, const Vector3& rkVector)
{
    return Vector3(fScalar*rkVector.x,fScalar*rkVector.y,
        fScalar*rkVector.z);
}
//----------------------------------------------------------------------------
inline Vector3& Vector3::operator+= (const Vector3& rkVector)
{
    x += rkVector.x;
    y += rkVector.y;
    z += rkVector.z;
    return *this;
}
//----------------------------------------------------------------------------
inline Vector3& Vector3::operator-= (const Vector3& rkVector)
{
    x -= rkVector.x;
    y -= rkVector.y;
    z -= rkVector.z;
    return *this;
}
//----------------------------------------------------------------------------
inline Vector3& Vector3::operator*= (Real fScalar)
{
    x *= fScalar;
    y *= fScalar;
    z *= fScalar;
    return *this;
}
//----------------------------------------------------------------------------
inline Real Vector3::SquaredLength () const
{
    return x*x + y*y + z*z;
}
//----------------------------------------------------------------------------
inline Real Vector3::Dot (const Vector3& rkVector) const
{
    return x*rkVector.x + y*rkVector.y + z*rkVector.z;
}
//----------------------------------------------------------------------------
inline Vector3 Vector3::Cross (const Vector3& rkVector) const
{
    return Vector3(y*rkVector.z-z*rkVector.y,z*rkVector.x-x*rkVector.z,
        x*rkVector.y-y*rkVector.x);
}



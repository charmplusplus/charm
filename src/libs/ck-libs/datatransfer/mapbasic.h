// $Id$

#ifndef __MAP_BASIC_H_
#define __MAP_BASIC_H_

#define MAP_BEGIN_NAMESPACE  namespace MAP {
#define MAP_END_NAMESPACE    }
#define USE_MAP_NAMESPACE    using namespace MAP;

#include <cstdlib>
#include <iostream>
#include <cmath>
//#include "Element_accessors.h"

MAP_BEGIN_NAMESPACE

/*
using COM::Element_node_enumerator;
using COM::Element_node_enumerator_str_2;
using COM::Element_node_enumerator_uns;
using COM::Facet_node_enumerator;
using COM::Element_vectors_k_const;
using COM::Element_vectors_k;
using COM::Element_node_vectors_k_const;
using COM::Element_node_vectors_k;
*/

class Origin {};
class Null_vector {};

/// Some basic geometric data types.
template <class Type>
class Vector_3 { 
public:
  Vector_3() {}
  explicit Vector_3( Type t) : _x(t), _y(t), _z(t) {}
  Vector_3( Null_vector) : _x(0), _y(0), _z(0) {}
  Vector_3( Type a, Type b, Type c) : _x(a), _y(b), _z(c) {}
  Type &operator[](const int i) { return (&_x)[i]; }
  const Type &operator[](const int i) const { return (&_x)[i]; }

  Type x() const { return _x; }
  Type y() const { return _y; }
  Type z() const { return _z; }

  /// Assign operators
  Vector_3 &operator+=( const Type t)
  { _x+=t; _y+=t; _z+=t; return *this; }
  Vector_3 &operator+=( const Vector_3 &v)
  { _x+=v._x; _y+=v._y; _z+=v._z; return *this; }
  Vector_3 &operator-=( const Type t)
  { _x-=t; _y-=t; _z-=t; return *this; }
  Vector_3 &operator-=( const Vector_3 &v)
  { _x-=v._x; _y-=v._y; _z-=v._z; return *this; }
  Vector_3 &operator*=( const Type t)
  { _x*=t; _y*=t; _z*=t; return *this; }
  Vector_3 &operator*=( const Vector_3 &v)
  { _x*=v._x; _y*=v._y; _z*=v._z; return *this; }
  Vector_3 &operator/=( const Type t)
  { _x/=t; _y/=t; _z/=t; return *this; }
  Vector_3 &operator/=( const Vector_3 &v)
  { _x/=v._x; _y/=v._y; _z/=v._z; return *this; }

  /// Arithmatic operators
  Vector_3 operator+( const Type t) const 
  { return Vector_3(_x+t, _y+t, _z+t); }
  Vector_3 operator+( const Vector_3 &v) const 
  { return Vector_3(_x+v._x, _y+v._y, _z+v._z); }
  Vector_3 operator-( const Type t) const 
  { return Vector_3(_x-t, _y-t, _z-t); }
  Vector_3 operator-( const Vector_3 &v) const 
  { return Vector_3(_x-v._x, _y-v._y, _z-v._z); }
  Vector_3 operator*( const Type t) const 
  { return Vector_3(_x*t, _y*t, _z*t); }
  Vector_3 operator/( const Type t) const 
  { return Vector_3(_x/t, _y/t, _z/t); }
  Type operator*( const Vector_3 &v) const 
  { return _x*v._x + _y*v._y + _z*v._z; }

  Vector_3 operator-() const
  { return Vector_3(-_x, -_y, -_z); }

  static Vector_3 cross_product( const Vector_3 &v, const Vector_3 &w) {
    return Vector_3( v._y * w._z - v._z * w._y ,
                     v._z * w._x - v._x * w._z ,
                     v._x * w._y - v._y * w._x );
  }
  
  Type squared_norm() const { return *this * *this; }

  Type norm() const { return std::sqrt(*this * *this); }

  Vector_3 &normalize() { 
    Type s=squared_norm();
    if ( s != Type(0)) { s=std::sqrt(s); _x/=s; _y/=s; _z/=s; }
    return *this;
  }
  
  Vector_3 &neg() { 
    _x = -_x; _y = -_y; _z = -_z; return *this;
  }
  
  bool operator==( const Vector_3 &p) const 
  { return x()==p.x() && y()==p.y() && z()==p.z(); }
  bool operator!=( const Vector_3 &p) const 
  { return x()!=p.x() || y()!=p.y() || z()!=p.z(); }

  bool operator<( const Vector_3 &v) const
  { return _x<v._x || _x==v._x && _y<v._y || 
      _x==v._x && _y==v._y && _z<v._z; }

  bool is_null() const { return _x==0&&_y==0&&_z==0; }
protected:
  Type _x, _y, _z;
};

template <class T>
Vector_3<T> operator*( T t, const Vector_3<T> &v) { return v*t; }

template <class Type>
std::ostream &operator<<( std::ostream &os, const Vector_3<Type> &p) {
  return os << '(' << p.x() << ',' << p.y() << ',' << p.z() << ')';
}

template <class T>
class Point_3 : protected Vector_3 <T> { 
public:
  Point_3() {}
  explicit Point_3( T t) : Vector_3<T>(t) {}
  Point_3( Origin) : Vector_3<T>(0, 0, 0) {}

  Point_3( T a, T b, T c) : Vector_3<T>(a, b, c) {}
  T &operator[](const int i) { return (&_x)[i]; }
  const T &operator[](const int i) const { return (&_x)[i]; }
  using Vector_3<T>::x;
  using Vector_3<T>::y;
  using Vector_3<T>::z;

  /// Assign operators
  Point_3 &operator+=( const Vector_3<T> &v)
  { _x+=v.x(); _y+=v.y(); _z+=v.z(); return *this; }
  Point_3 &operator-=( const Vector_3<T> &v)
  { _x-=v.x(); _y-=v.y(); _z-=v.z(); return *this; }
  
  /// Arithmatic operators
  Point_3 operator+( const Vector_3<T> &v) const 
  { return Point_3(_x+v.x(), _y+v.y(), _z+v.z()); }
  Point_3 operator-( const Vector_3<T> &v) const 
  { return Point_3(_x-v.x(), _y-v.y(), _z-v.z()); }
  Vector_3<T> operator-( const Point_3 &v) const 
  { return Vector_3<T>(_x-v.x(), _y-v.y(), _z-v.z()); }

  bool operator==( const Point_3 &p) const 
  { return x()==p.x() && y()==p.y() && z()==p.z(); }

  bool operator!=( const Point_3 &p) const 
  { return x()!=p.x() || y()!=p.y() || z()!=p.z(); }

  bool operator<( const Point_3 &v) const
  { return Vector_3<T>::operator<(v); }

  bool is_origin() const { return _x==0&&_y==0&&_z==0; }

#ifndef __SUNPRO_CC
protected:
  using Vector_3<T>::_x;
  using Vector_3<T>::_y;
  using Vector_3<T>::_z;
#endif
};

template <class Type>
std::ostream &operator<<( std::ostream &os, const Point_3<Type> &p) {
  return os << '(' << p.x() << ',' << p.y() << ',' << p.z() << ')';
}

template <class Type>
class Vector_2 { 
public:
  Vector_2() {}
  explicit Vector_2( Type t) : _x(t), _y(t) {}
  Vector_2( Null_vector) : _x(0), _y(0) {}
  Vector_2( Type a, Type b) : _x(a), _y(b) {}
  Type &operator[](const int i) { return i==0?_x:_y; }
  const Type &operator[](const int& i) const { return i==0?_x:_y; }

  Type x() const { return _x; }
  Type y() const { return _y; }

  /// Assign operators
  Vector_2 &operator+=( const Vector_2 &v)
  { _x+=v._x; _y+=v._y; return *this; }
  Vector_2 &operator-=( const Vector_2 &v)
  { _x-=v._x; _y-=v._y; return *this; }
  Vector_2 &operator*=( const Type t)
  { _x*=t; _y*=t; return *this; }
  Vector_2 &operator/=( const Type t)
  { _x/=t; _y/=t; return *this; }

  /// Arithmatic operators
  Vector_2 operator+( const Vector_2 &v) const 
  { return Vector_2(_x+v._x, _y+v._y); }
  Vector_2 operator-( const Vector_2 &v) const 
  { return Vector_2(_x-v._x, _y-v._y); }
  Vector_2 operator*( const Type t) const 
  { return Vector_2(_x*t, _y*t); }
  Vector_2 operator/( const Type t) const 
  { return Vector_2(_x/t, _y/t); }
  Type operator*( const Vector_2 &v) const 
  { return _x*v._x + _y*v._y; }

  Vector_2 operator-() const
  { return Vector_2(-_x, -_y); }
 
  Type squared_norm() const { return *this * *this; }

  Type norm() const { return std::sqrt(*this * *this); }

  Vector_2 &normalize() { 
    Type s=squared_norm();
    if ( s != Type(0)) { s=std::sqrt(s); _x/=s; _y/=s; }
    return *this;
  }
  
  Vector_2 &neg() { 
    _x = -_x; _y = -_y; return *this;
  }
  
  bool operator==( const Vector_2 &p) const 
  { return x()==p.x() && y()==p.y(); }

  bool operator!=( const Vector_2 &p) const 
  { return x()!=p.x() || y()!=p.y(); }

  bool operator<( const Vector_2 &v) const
  { return _x<v._x || _x==v._x && _y<v._y; }

  bool is_null() const { return _x==0&&_y==0; }
  
protected:
  Type _x, _y; 
};

template <class T>
Vector_2<T> operator*( T t, const Vector_2<T> &v) { return v*t; }

template <class Type>
std::ostream &operator<<( std::ostream &os, const Vector_2<Type> &p) {
  return os << '(' << p.x() << ',' << p.y() << ')';
}

template <class T>
class Point_2 : protected Vector_2<T> { 
public:
  Point_2() {}
  Point_2( Origin) : Vector_2<T>( 0, 0) {}
  explicit Point_2( T t) : Vector_2<T>(t) {}
  Point_2( T a, T b) : Vector_2<T>(a, b) {}
  T &operator[](const int i) { return (&_x)[i]; }
  const T &operator[](const int i) const { return (&_x)[i]; }
  using Vector_2<T>::x;
  using Vector_2<T>::y;
  
  /// Assign operators
  Point_2 &operator+=( const Vector_2<T> &v)
  { _x+=v.x(); _y+=v.y(); return *this; }
  Point_2 &operator-=( const Vector_2<T> &v)
  { _x-=v.x(); _y-=v.y(); return *this; }
  
  /// Arithmatic operators
  Point_2 operator+( const Vector_2<T> &v) const 
  { return Point_2(_x+v.x(), _y+v.y()); }
  Point_2 operator-( const Vector_2<T> &v) const 
  { return Point_2(_x-v.x(), _y-v.y()); }
  Vector_2<T> operator-( const Point_2 &v) const 
  { return Vector_2<T>(_x-v.x(), _y-v.y()); }

  bool operator==( const Point_2 &p) const 
  { return x()==p.x() && y()==p.y(); }
  
  bool operator!=( const Point_2 &p) const 
  { return x()!=p.x() || y()!=p.y(); }
  
  bool operator<( const Point_2 &v) const
  { return Vector_2<T>::operator<(v); }

  bool is_origin() const { return _x==0&&_y==0; }

#ifndef __SUNPRO_CC
protected:
  using Vector_2<T>::_x;
  using Vector_2<T>::_y;
#endif
};

template <class Type>
std::ostream &operator<<( std::ostream &os, const Point_2<Type> &p) {
  return os << '(' << p.x() << ',' << p.y() << ')';
}

typedef double Real;

// Modes of element_to_nodes.
enum { E2N_USER=0, E2N_ONE=1, E2N_AREA=2, E2N_ANGLE=3};

MAP_END_NAMESPACE

#endif

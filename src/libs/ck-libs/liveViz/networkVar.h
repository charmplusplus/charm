/*
Classes for easy, transparent use of on-the-wire
network byte order integers and doubles.

Orion Sky Lawlor, olawlor@acm.org, 6/22/2001
*/

#ifndef __OSL_NETWORKVAR_H
#define __OSL_NETWORKVAR_H

//Class for transparent big-endian 4-byte int access--
// treat it like an int; but it will be stored as 
// a network-byte-order 32-bit integer.
//Respectably fast: usually slower than actual integers,
// but a great deal faster (& more convenient) than ASCII.
class networkInt {
  typedef unsigned int uint;
  enum {len=4};
  unsigned char store[len];
  void set(uint i) {
    store[0]=i>>24; store[1]=i>>16; store[2]=i>>8; store[3]=i>>0;
  }
  uint get(void) const {
    return (store[0]<<24)+(store[1]<<16)+(store[2]<<8)+(store[3]<<0);  
  }
public:
  networkInt() {}
  networkInt(int i) {set((uint)i);}
  networkInt &operator=(int i) {set((uint)i); return *this;}
  networkInt(uint i) {set(i);}
  networkInt &operator=(uint i) {set(i); return *this;}
  
  operator uint () const {
    return get();
  }
  operator int () const {
    uint i=get();
    if (sizeof(int)>len) {
      //Can't just cast, because negative integers would map
      // to large positive ones.  Check the sign bit:
      if (i&(1<<31))
        return ((int)i)-0xffFFffFF-1;
    }
    return (int)i;
  }
};

//Class for transparent big-endian 8-byte IEEE access--
// treat it like a double; but it will be stored as 
// a big-endian 64-bit IEEE floating-point number.
//Unlike actual doubles, also has no alignment restrictions.
class networkDouble {
  typedef double real;
  enum {len=8};
  unsigned char store[len];

  static int getFlip(void);
  //Initialize g_doFlip with the result of getFlip
  static int g_doFlip;

  void set(real r) {
    unsigned char *rp=(unsigned char *)&r;
    if (g_doFlip) 
      for (int i=0;i<len;i++) store[i]=rp[len-1-i];
    else
      for (int i=0;i<len;i++) store[i]=rp[i];
  }
  real get(void) const {
    real ret;
    unsigned char *rp=(unsigned char *)&ret;
    if (g_doFlip) 
      for (int i=0;i<len;i++) rp[i]=store[len-1-i];
    else
      for (int i=0;i<len;i++) rp[i]=store[i];
    return ret;
  }
public:
  networkDouble() {}
  networkDouble(real i) {set(i);}
  networkDouble &operator=(real i) {set(i); return *this;}
  
  operator real () const {
    return get();
  }
};

class networkVector3d {
public:
  networkDouble x,y,z;
  networkVector3d() {}
  networkVector3d(double x_,double y_,double z_)
    :x(x_), y(y_), z(z_) { }
  
  void extract(double *dest) const {
    dest[0]=x;dest[1]=y;dest[2]=z;
  }
};


#endif //def(thisHeader)



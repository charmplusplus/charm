/*
Pup routines for STL classes.

After including this header, you can parameter-marshall
an variable consisting of STL vectors, lists, maps,
strings, or pairs.

This includes variables of type "std::list<int>", or even
"std::map<double, std::vector<std::string> >".

NOT included are the rarer types like valarray or slice, 
vector<bool>, set or multiset, or deque.

Orion Sky Lawlor, olawlor@acm.org, 7/22/2002
*/
#ifndef _UIUC_CHARM_PUP_STL_H
#define _UIUC_CHARM_PUP_STL_H

#include <conv-config.h>

/*It's kind of annoying that we have to drag all these headers in
  just so the std:: parameter declarations will compile.
 */
#include <set>
#include <vector>
#include <deque>
#include <list>
#include <map>
#if !CMK_USING_XLC
#include <unordered_map>
#else
#include <tr1/unordered_map>
#endif
#include <string>
#include <complex>
#include <utility> /*for std::pair*/
#include <memory>
#include "pup.h"

namespace PUP {
  /*************** Simple classes ***************/
  // Non-const version is required for puping std::pair
  template <class A,class B>
  inline void operator|(er &p,typename std::pair<A,B> &v);
  template <class A,class B>
  inline void operator|(er &p,typename std::pair<const A,B> &v);
  template <class T>
  inline void operator|(er &p,std::complex<T> &v);
#if !CMK_USING_XLC
  template <class T>
  inline void operator|(er &p, std::unique_ptr<T, std::default_delete<T>> &ptr);
#endif
  template <class charType>
  inline void operator|(er &p,typename std::basic_string<charType> &v);
  inline void operator|(er &p,std::string &v);
  template <class container>
  inline size_t PUP_stl_container_size(er &p,container &c);
  template <class container, class dtype>
  inline void PUP_stl_container_items(er &p,container &c);
  template <> inline void PUP_stl_container_items<std::vector<bool>,bool>(er &p, std::vector<bool> &c);
  template <> inline void PUP_stl_container_items<std::vector<char>,char>(er &p, std::vector<char> &c);
  template <> inline void PUP_stl_container_items<std::vector<unsigned char>,unsigned char>(er &p, std::vector<unsigned char> &c);
  template <> inline void PUP_stl_container_items<std::vector<short>,short>(er &p, std::vector<short> &c);
  template <> inline void PUP_stl_container_items<std::vector<unsigned short>,unsigned short>(er &p, std::vector<unsigned short> &c);
  template <> inline void PUP_stl_container_items<std::vector<int>,int>(er &p, std::vector<int> &c);
  template <> inline void PUP_stl_container_items<std::vector<unsigned int>,unsigned int>(er &p, std::vector<unsigned int> &c);
  template <> inline void PUP_stl_container_items<std::vector<long>,long>(er &p, std::vector<long> &c);
  template <> inline void PUP_stl_container_items<std::vector<unsigned long>,unsigned long>(er &p, std::vector<unsigned long> &c);
  template <> inline void PUP_stl_container_items<std::vector<long long>,long long>(er &p, std::vector<long long> &c);
  template <> inline void PUP_stl_container_items<std::vector<unsigned long long>,unsigned long long>(er &p, std::vector<unsigned long long> &c);
  template <> inline void PUP_stl_container_items<std::vector<float>,float>(er &p, std::vector<float> &c);
  template <> inline void PUP_stl_container_items<std::vector<double>,double>(er &p, std::vector<double> &c);
  template <class container,class dtype>
  inline void PUP_stl_container(er &p,container &c);
  template <class container,class dtype>
  inline void PUP_stl_map(er &p,container &c);
  template <class T>
  inline void operator|(er &p,typename std::vector<T> &v);
  template <class T>
  inline void operator|(er &p,typename std::list<T> &v);
  template <class V,class T,class Cmp>
  inline void operator|(er &p,typename std::map<V,T,Cmp> &m);
  template <class V,class T,class Cmp>
  inline void operator|(er &p,typename std::multimap<V,T,Cmp> &m);
  template <class T>
  inline void operator|(er &p,typename std::set<T> &m);

  template <class A,class B>
  inline void operator|(er &p,typename std::pair<A,B> &v)
  {
    p.syncComment(sync_index);
    p|v.first;
    p.syncComment(sync_item);
    p|v.second;
  }
  // Const version is required for puping std::map
  template <class A,class B> 
  inline void operator|(er &p,typename std::pair<const A,B> &v)
  {
    p.syncComment(sync_index);
    p|*(A *)&v.first; /* cast away constness on A */
    p.syncComment(sync_item);
    p|v.second;
  }
  template <class T>
  inline void operator|(er &p,std::complex<T> &v)
  {
    T re=v.real(), im=v.imag();
    p|re; p|im;
    v=std::complex<T>(re,im);
  }
#if !CMK_USING_XLC
  template <class T>
  inline void operator|(er &p, std::unique_ptr<T, std::default_delete<T>> &ptr)
  {
    bool nonNull = static_cast<bool>(ptr);
    p|nonNull;

    if (nonNull) {
      if (p.isUnpacking())
        ptr.reset(new T);

      p|(*ptr);
    }
  }
#endif
  template <class charType> 
  inline void operator|(er &p,typename std::basic_string<charType> &v)
  {
    size_t nChar=v.length();
    p|nChar;
    if (p.isUnpacking()) { //Unpack to temporary buffer
      charType *buf=new charType[nChar];
      p(buf,nChar);
      v=std::basic_string<charType>(buf,nChar);
      delete[] buf;
    }
    else /*packing*/ { //Do packing in-place from data
      //Have to cast away constness here
      p((charType *)v.data(),nChar);
    }
  }
  inline void operator|(er &p,std::string &v)
  {
    p.syncComment(sync_begin_object,"std::string");
    size_t nChar=v.length();
    p|nChar;
    if (p.isUnpacking()) { //Unpack to temporary buffer
      char *buf=new char[nChar];
      p(buf,nChar);
      v=std::basic_string<char>(buf,nChar);
      delete[] buf;
    }
    else /*packing*/ { //Do packing in-place from data
      //Have to cast away constness here
      p((char *)v.data(),nChar);
    }
    p.syncComment(sync_end_object);
  }

  /**************** Containers *****************/

  //Impl. util: pup the length of a container
  template <class container>
  inline size_t PUP_stl_container_size(er &p,container &c) {
    size_t nElem=c.size();
    p|nElem;
    return nElem; 
  }

  //Impl. util: pup each current item of a container (no allocation)
  template <class container, class dtype>
  inline void PUP_stl_container_items(er &p,container &c) {
    for (typename container::iterator it=c.begin();
	 it!=c.end();
	 ++it) {
      p.syncComment(sync_item);
      // Cast away the constness (needed for std::set)
      p|*(dtype *)&(*it);
    }
  }

  // Specialized to work with vector<bool>
  template<>
  inline void PUP_stl_container_items<std::vector<bool>, bool>(er &p, std::vector<bool> &c)
  {
    std::deque<bool> q(c.begin(), c.end());
    
    for (std::deque<bool>::iterator it = q.begin(); it != q.end(); it++)
    {
      p.syncComment(sync_item);
      p|*it;
    }
  }

// Specialize for vectors of common types
#define PUP_STL_CONTAINER_ITEMS_ARRAY(dtype) \
  template<> \
  inline void PUP_stl_container_items<std::vector<dtype>, dtype>(er &p, std::vector<dtype> &c) \
  { \
    PUParray(p, &c[0], c.size()); \
  }

  PUP_STL_CONTAINER_ITEMS_ARRAY(char)
  PUP_STL_CONTAINER_ITEMS_ARRAY(unsigned char)
  PUP_STL_CONTAINER_ITEMS_ARRAY(short)
  PUP_STL_CONTAINER_ITEMS_ARRAY(unsigned short)
  PUP_STL_CONTAINER_ITEMS_ARRAY(int)
  PUP_STL_CONTAINER_ITEMS_ARRAY(unsigned int)
  PUP_STL_CONTAINER_ITEMS_ARRAY(long)
  PUP_STL_CONTAINER_ITEMS_ARRAY(unsigned long)
  PUP_STL_CONTAINER_ITEMS_ARRAY(long long)
  PUP_STL_CONTAINER_ITEMS_ARRAY(unsigned long long)
  PUP_STL_CONTAINER_ITEMS_ARRAY(float)
  PUP_STL_CONTAINER_ITEMS_ARRAY(double)

  template <class container,class dtype>
  inline void PUP_stl_container(er &p,container &c) {
    p.syncComment(sync_begin_array);
    size_t nElem=PUP_stl_container_size(p,c);
    if (p.isUnpacking()) {
      c.resize(nElem);
    }
    PUP_stl_container_items<container, dtype>(p,c);
    p.syncComment(sync_end_array);
  }

  //Map objects don't have a "push_back", while vector and list
  //  don't have an "insert", so PUP_stl_map isn't PUP_stl_container
  template <class container,class dtype>
  inline void PUP_stl_map(er &p,container &c) {
    p.syncComment(sync_begin_list);
    size_t nElem=PUP_stl_container_size(p,c);
    if (p.isUnpacking()) 
      { //Unpacking: Extract each element and insert:
	for (size_t i=0;i<nElem;i++) {
	  dtype n;
	  p|n;
	  c.insert(n);
	} 
      }
    else PUP_stl_container_items<container, dtype>(p,c);
    p.syncComment(sync_end_list);
  }

  template <class T> 
  inline void operator|(er &p,typename std::vector<T> &v)
  { PUP_stl_container<std::vector<T>,T>(p,v); }
  template <class T> 
  inline void operator|(er &p,typename std::list<T> &v)
  { PUP_stl_container<std::list<T>,T>(p,v); }

  template <class V,class T,class Cmp> 
  inline void operator|(er &p,typename std::map<V,T,Cmp> &m)
  //{ PUP_stl_map<std::map<V,T,Cmp>,std::pair<const V,T> >(p,m); }    // 'const' confuses old version of a SUN CC compiler
  { PUP_stl_map<std::map<V,T,Cmp>,std::pair<V,T> >(p,m); }
#if !CMK_USING_XLC
  template <class V,class T,class Cmp>
  inline void operator|(er &p,typename std::unordered_map<V,T,Cmp> &m)
  //{ PUP_stl_map<std::unordered_map<V,T,Cmp>,std::pair<const V,T> >(p,m); }    // 'const' confuses old version of a SUN CC compiler
  { PUP_stl_map<std::unordered_map<V,T,Cmp>,std::pair<V,T> >(p,m); }
#else
  template <class V,class T,class Cmp>
  inline void operator|(er &p,typename std::tr1::unordered_map<V,T,Cmp> &m)
  //{ PUP_stl_map<std::unordered_map<V,T,Cmp>,std::pair<const V,T> >(p,m); }    // 'const' confuses old version of a SUN CC compiler
  { PUP_stl_map<std::tr1::unordered_map<V,T,Cmp>,std::pair<V,T> >(p,m); }
#endif
  template <class V,class T,class Cmp> 
  inline void operator|(er &p,typename std::multimap<V,T,Cmp> &m)
  { PUP_stl_map<std::multimap<V,T,Cmp>,std::pair<const V,T> >(p,m); }
  template <class T>
  inline void operator|(er &p,typename std::set<T> &m)
  { PUP_stl_map<std::set<T>,T >(p,m); }

}

#endif

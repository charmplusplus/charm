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

/*It's kind of annoying that we have to drag all these headers in
  just so the std:: parameter declarations will compile.
 */
#include <set>
#include <vector>
#include <deque>
#include <list>
#include <map>
#include <string>
#include <complex>
#include <utility> /*for std::pair*/
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
  template <class charType>
  inline void operator|(er &p,typename std::basic_string<charType> &v);
  inline void operator|(er &p,std::string &v);
  template <class container>
  inline int PUP_stl_container_size(er &p,container &c);
  template <class container, class dtype>
  inline void PUP_stl_container_items(er &p,container &c);
  template <> inline void PUP_stl_container_items<std::vector<bool>,bool>(er &p, std::vector<bool> &c);
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
  template <class charType> 
  inline void operator|(er &p,typename std::basic_string<charType> &v)
  {
    int nChar=v.length();
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
    int nChar=v.length();
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
  inline int PUP_stl_container_size(er &p,container &c) {
    int nElem=c.size();
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

  template <class container,class dtype>
  inline void PUP_stl_container(er &p,container &c) {
    p.syncComment(sync_begin_array);
    int nElem=PUP_stl_container_size(p,c);
    if (p.isUnpacking()) 
      { //Unpacking: Extract each element and push_back:
	c.resize(0);
	for (int i=0;i<nElem;i++) {
	  p.syncComment(sync_item);
	  dtype n;
	  p|n;
	  c.push_back(n);
	} 
      }
    else PUP_stl_container_items<container, dtype>(p,c);
    p.syncComment(sync_end_array);
  }
  //Map objects don't have a "push_back", while vector and list
  //  don't have an "insert", so PUP_stl_map isn't PUP_stl_container
  template <class container,class dtype>
  inline void PUP_stl_map(er &p,container &c) {
    p.syncComment(sync_begin_list);
    int nElem=PUP_stl_container_size(p,c);
    if (p.isUnpacking()) 
      { //Unpacking: Extract each element and insert:
	for (int i=0;i<nElem;i++) {
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
  template <class V,class T,class Cmp> 
  inline void operator|(er &p,typename std::multimap<V,T,Cmp> &m)
  { PUP_stl_map<std::multimap<V,T,Cmp>,std::pair<const V,T> >(p,m); }
  template <class T>
  inline void operator|(er &p,typename std::set<T> &m)
  { PUP_stl_map<std::set<T>,T >(p,m); }

}

#endif

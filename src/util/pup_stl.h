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
#include <vector>
#include <list>
#include <map>
#include <string>
#include <utility> /*for std::pair*/
#include "pup.h"

/*************** Simple classes ***************/

template <class A,class B> void operator|(PUP::er &p,std::pair<A,B> &v)
{
  p|v.first;
  p|v.second;
}
template <class charType> 
void operator|(PUP::er &p,std::basic_string<charType> &v)
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

/**************** Containers *****************/

//Impl. util: pup the length of a container
template <class container>
int PUP_stl_container_size(PUP::er &p,container &c) {
  int nElem=c.size();
  p|nElem;
  return nElem; 
}

//Impl. util: pup each current item of a container (no allocation)
template <class container>
void PUP_stl_container_items(PUP::er &p,container &c) {
  for (typename container::iterator it=c.begin();
       it!=c.end();
       ++it)
    p|*it;  
}

template <class container,class dtype>
void PUP_stl_container(PUP::er &p,container &c) {
  int nElem=PUP_stl_container_size(p,c);
  if (p.isUnpacking()) 
  { //Unpacking: Extract each element and push_back:
    for (int i=0;i<nElem;i++) {
      dtype n;
      p|n;
      c.push_back(n);
    } 
  }
  else PUP_stl_container_items(p,c);
}
//Map objects don't have a "push_back", while vector and list
//  don't have an "insert", so PUP_stl_map isn't PUP_stl_container
template <class container,class dtype>
void PUP_stl_map(PUP::er &p,container &c) {
  int nElem=PUP_stl_container_size(p,c);
  if (p.isUnpacking()) 
  { //Unpacking: Extract each element and insert:
    for (int i=0;i<nElem;i++) {
      dtype n;
      p|n;
      c.insert(n);
    } 
  }
  else PUP_stl_container_items(p,c);
}

template <class T> void operator|(PUP::er &p,std::vector<T> &v)
  { PUP_stl_container<std::vector<T>,T>(p,v); }
template <class T> void operator|(PUP::er &p,std::list<T> &v)
  { PUP_stl_container<std::list<T>,T>(p,v); }

template <class V,class T,class Cmp> 
void operator|(PUP::er &p,std::map<V,T,Cmp> &m)
  { PUP_stl_map<std::map<V,T,Cmp>,std::pair<const V,T> >(p,m); }
template <class V,class T,class Cmp> 
void operator|(PUP::er &p,std::multimap<V,T,Cmp> &m)
  { PUP_stl_map<std::multimap<V,T,Cmp>,std::pair<const V,T> >(p,m); }


#endif

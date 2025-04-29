/*
Pup routines for STL classes.

After including this header, you can parameter-marshall
a variable consisting of STL containers such as vectors,
lists, maps, strings, or pairs.

This includes variables of type "std::list<int>", or even
"std::map<double, std::vector<std::string> >".

NOT included are the rarer types like valarray or slice.

Orion Sky Lawlor, olawlor@acm.org, 7/22/2002
*/
#ifndef _UIUC_CHARM_PUP_STL_H
#define _UIUC_CHARM_PUP_STL_H

#include "converse.h"

/*It's kind of annoying that we have to drag all these headers in
  just so the std:: parameter declarations will compile.
 */
#include <algorithm>
#include <array>
#include <set>
#include <vector>
#include <deque>
#include <forward_list>
#include <list>
#include <map>
#include <memory>
#include <random>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <complex>
#include <utility> /*for std::pair*/
#include <chrono>
#include "pup.h"

#include <cstddef>

namespace PUP {
  /*************** Simple classes ***************/
  // Non-const version is required for puping std::pair
  template <class A,class B>
  inline void operator|(er &p,typename std::pair<A,B> &v);
  template <class A,class B>
  inline void operator|(er &p,typename std::pair<const A,B> &v);
  template <class T>
  inline void operator|(er &p,std::complex<T> &v);
  template <class T>
  inline void operator|(er &p, std::shared_ptr<T> &t);
  template <class T>
  inline void operator|(er &p, std::unique_ptr<T, std::default_delete<T>> &ptr);
  template <class charType>
  inline void operator|(er &p,typename std::basic_string<charType> &v);
  inline void operator|(er &p,std::string &v);
  template <class container>
  inline size_t PUP_stl_container_size(er &p,container &c);
  template <class container, class dtype>
  inline void PUP_stl_container_items(er &p, container &c, size_t nElem);
  template <> inline void PUP_stl_container_items<std::vector<bool>,bool>(er &p, std::vector<bool> &c, size_t nElem);
  template <class container,class dtype>
  inline void PUP_stl_container(er &p,container &c);
  template <class container,class dtype>
  inline void PUP_stl_map(er &p,container &c);
  template <class T>
  inline void operator|(er &p,typename std::vector<T> &v);
  template <class T>
  inline void operator|(er &p,typename std::deque<T> &d);
  template <class T>
  inline void operator|(er &p,typename std::list<T> &v);
  template <class T>
  inline void operator|(er &p,typename std::forward_list<T> &fl);
  template <class V,class T,class Cmp>
  inline void operator|(er &p,typename std::map<V,T,Cmp> &m);
  template <class V,class T,class Cmp>
  inline void operator|(er &p,typename std::multimap<V,T,Cmp> &m);
  template <class T>
  inline void operator|(er &p,typename std::set<T> &m);
  template <class T,class Cmp>
  inline void operator|(er &p,typename std::multiset<T,Cmp> &m);
  template <> inline void operator|(er &p,std::vector<bool> &v);

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
      CmiEnforce(buf != nullptr);
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

  template <class container>
  inline void reserve_if_applicable(container &c, size_t nElem)
  {
    c.clear();
    c.reserve(nElem);
  }
  template <class dtype>
  inline void reserve_if_applicable(std::deque<dtype> &c, size_t nElem)
  {
    c.clear();
  }
  template <class dtype>
  inline void reserve_if_applicable(std::list<dtype> &c, size_t nElem)
  {
    c.clear();
  }
  template <class dtype, class cmp>
  inline void reserve_if_applicable(std::set<dtype, cmp> &c, size_t nElem)
  {
    c.clear();
  }
  template <class dtype, class cmp>
  inline void reserve_if_applicable(std::multiset<dtype, cmp> &c, size_t nElem)
  {
    c.clear();
  }
  template <class K, class V, class cmp>
  inline void reserve_if_applicable(std::map<K, V, cmp> &c, size_t nElem)
  {
    c.clear();
  }
  template <class K, class V, class cmp>
  inline void reserve_if_applicable(std::multimap<K, V, cmp> &c, size_t nElem)
  {
    c.clear();
  }

  template <class container, class... Args>
  inline void emplace(container &c, Args&&... args)
  {
    c.emplace(std::forward<Args>(args)...);
  }
  template <class dtype, class... Args>
  inline void emplace(std::vector<dtype> &c, Args&&... args)
  {
    c.emplace_back(std::forward<Args>(args)...);
  }
  template <class dtype, class... Args>
  inline void emplace(std::deque<dtype> &c, Args&&... args)
  {
    c.emplace_back(std::forward<Args>(args)...);
  }
  template <class dtype, class... Args>
  inline void emplace(std::list<dtype> &c, Args&&... args)
  {
    c.emplace_back(std::forward<Args>(args)...);
  }

  //Impl. util: pup the length of a container
  template <class container>
  inline size_t PUP_stl_container_size(er &p,container &c) {
    size_t nElem=c.size();
    p|nElem;
    return nElem; 
  }

  //Impl. util: pup each current item of a container (no allocation)
  template <class container, class dtype>
  inline void PUP_stl_container_items(er &p, container &c, size_t nElem)
  {
    if (p.isUnpacking())
    {
      reserve_if_applicable(c, nElem);
      for (size_t i = 0; i < nElem; ++i)
      {
        p.syncComment(sync_item);
        detail::TemporaryObjectHolder<dtype> n;
        p|n;
        emplace(c, std::move(n.t));
      }
    }
    else
    {
      for (typename container::iterator it=c.begin(); it!=c.end(); ++it)
      {
        p.syncComment(sync_item);
        // Cast away the constness (needed for std::set)
        p|*(dtype *)&(*it);
      }
    }
  }

  // Specialized to work with vector<bool>
  template<>
  inline void PUP_stl_container_items<std::vector<bool>, bool>(er &p, std::vector<bool> &c, size_t nElem)
  {
    // iterators of std::vector<bool> are read-only temporaries so we need special handling for unpacking
    if (p.isUnpacking()) {
      c.resize(nElem);
      for (size_t i = 0; i < nElem; ++i)
      {
        p.syncComment(sync_item);
        detail::TemporaryObjectHolder<bool> n;
        p|n;
        c[i] = n.t;
      }
    }
    else {
      for (bool n : c)
      {
        p.syncComment(sync_item);
        p|n;
      }
    }
  }

  template <class container,class dtype>
  inline void PUP_stl_container(er &p,container &c) {
    p.syncComment(sync_begin_array);
    size_t nElem=PUP_stl_container_size(p,c);
    PUP_stl_container_items<container, dtype>(p, c, nElem);
    p.syncComment(sync_end_array);
  }

  // forward_list does not have: .size(), .emplace(), .emplace_back()
  template <class dtype>
  inline void PUP_stl_forward_list(er &p,std::forward_list<dtype> &c) {
    p.syncComment(sync_begin_array);
    size_t nElem;
    if (p.isUnpacking())
    {
      p | nElem;
      auto iter = c.before_begin();
      for (size_t i = 0; i < nElem; ++i)
      {
        p.syncComment(sync_item);
        detail::TemporaryObjectHolder<dtype> n;
        p|n;
        iter = c.emplace_after(iter, std::move(n.t));
      }
    }
    else
    {
      nElem = 0;
      for (auto& n: c)
      {
        ++nElem;
      }
      p | nElem;
      for (auto& n : c)
      {
        p.syncComment(sync_item);
        p | n;
      }
    }
    p.syncComment(sync_end_array);
  }

  template <class container, class K, class V>
  inline void PUP_stl_map(er &p,container &c) {
    p.syncComment(sync_begin_list);
    size_t nElem=PUP_stl_container_size(p,c);
    if (p.isUnpacking())
      { //Unpacking: Extract each element and insert:
        reserve_if_applicable(c, nElem);
        for (size_t i=0;i<nElem;i++)
        {
          detail::TemporaryObjectHolder<K> k;
          detail::TemporaryObjectHolder<V> v;

          // keep in sync with std::pair
          p.syncComment(sync_index);
          p | k;
          p.syncComment(sync_item);
          p | v;

          c.emplace(std::piecewise_construct, std::forward_as_tuple(std::move(k.t)), std::forward_as_tuple(std::move(v.t)));
        }
      }
    else
    {
      for (auto& kv : c)
      {
        p | kv;
      }
    }
    p.syncComment(sync_end_list);
  }

  template <class T>
  inline void operator|(er &p, typename std::vector<T> &v) {
    if (PUP::as_bytes<T>::value) {
      size_t nElem = PUP_stl_container_size(p, v);
      if (p.isUnpacking()) {
        v.resize(nElem);
        v.shrink_to_fit();
      }
      PUParray(p, v.data(), nElem);
    } else {
      PUP_stl_container<std::vector<T>, T>(p, v);
    }
  }

  template <class T>
  inline void operator|(er &p,typename std::deque<T> &d)
  { PUP_stl_container<std::deque<T>,T>(p,d); }
  template <class T> 
  inline void operator|(er &p,typename std::list<T> &v)
  { PUP_stl_container<std::list<T>,T>(p,v); }
  template <class T>
  inline void operator|(er &p,typename std::forward_list<T> &fl)
  { PUP_stl_forward_list<T>(p,fl); }

  template <class V,class T,class Cmp> 
  inline void operator|(er &p,typename std::map<V,T,Cmp> &m)
  { PUP_stl_map<std::map<V,T,Cmp>,V,T >(p,m); }
  template <class V,class T,class Cmp> 
  inline void operator|(er &p,typename std::multimap<V,T,Cmp> &m)
  { PUP_stl_map<std::multimap<V,T,Cmp>,V,T >(p,m); }
  /// \warning This does not work with custom hash functions that have state
  template <class V,class T,class Cmp>
  inline void operator|(er &p,typename std::unordered_map<V,T,Cmp> &m)
  { PUP_stl_map<std::unordered_map<V,T,Cmp>,V,T >(p,m); }
  template <class V,class T,class Cmp>
  inline void operator|(er &p,typename std::unordered_multimap<V,T,Cmp> &m)
  { PUP_stl_map<std::unordered_multimap<V,T,Cmp>,V,T >(p,m); }

  template <class T>
  inline void operator|(er &p,typename std::set<T> &m)
  { PUP_stl_container<std::set<T>,T >(p,m); }
  template <class T,class Cmp>
  inline void operator|(er &p,typename std::multiset<T,Cmp> &m)
  { PUP_stl_container<std::multiset<T,Cmp>,T >(p,m); }
  template <class T>
  inline void operator|(er &p,typename std::unordered_set<T> &m)
  { PUP_stl_container<std::unordered_set<T>,T >(p,m); }
  template <class T,class Cmp>
  inline void operator|(er &p,typename std::unordered_multiset<T,Cmp> &m)
  { PUP_stl_container<std::unordered_multiset<T,Cmp>,T >(p,m); }

  // Specialized to work with vector<bool>, which doesn't
  // have data() or shrink_to_fit() members
  template <>
  inline void operator|(er &p,std::vector<bool> &v) {
    PUP_stl_container<std::vector<bool>, bool>(p, v);
  }

// Distributed under the MIT License.
// The following allows for pupping STL structures with pointers to abstract
// base classes. Requires is used in place of enable_if_t to enforce
// requirements on template parameters for the following PUP methods.
template <bool B>
struct requires_impl {
  using template_error_type_failed_to_meet_requirements_on_template_parameters
      = std::nullptr_t;
};

template <>
struct requires_impl<false> {};

template <bool B>
using Requires = typename requires_impl<
    B>::template_error_type_failed_to_meet_requirements_on_template_parameters;

  template <typename T, std::size_t N,
            Requires<!PUP::as_bytes<T>::value> = nullptr>
  inline void pup(PUP::er& p, std::array<T, N>& a) {
    std::for_each(a.begin(), a.end(), [&p](T& t) { p | t; });
  }

  template <typename T, std::size_t N,
            Requires<PUP::as_bytes<T>::value> = nullptr>
  inline void pup(PUP::er& p, std::array<T, N>& a) {
    PUParray(p, a.data(), N);
  }

  template <typename T, std::size_t N>
  inline void operator|(er& p, std::array<T, N>& a) {
    pup(p, a);
  }

  template <typename T, Requires<std::is_enum<T>::value> = nullptr>
  inline void operator|(PUP::er& p, T& s) {
    pup_bytes(&p, static_cast<void*>(&s), sizeof(T));
  }

  template <size_t N = 0, typename... Args,
            Requires<0 == sizeof...(Args)> = nullptr>
  void pup_tuple_impl(PUP::er& /* p */, std::tuple<Args...>& /* t */) {
  }

  template <size_t N = 0, typename... Args,
            Requires<(0 < sizeof...(Args) && 0 == N)> = nullptr>
  void pup_tuple_impl(PUP::er& p, std::tuple<Args...>& t) {
    p | std::get<N>(t);
  }

  template <size_t N, typename... Args,
            Requires<(sizeof...(Args) > 0 && N > 0)> = nullptr>
  void pup_tuple_impl(PUP::er& p, std::tuple<Args...>& t) {
    p | std::get<N>(t);
    pup_tuple_impl<N - 1>(p, t);
  }

  template <typename... Args>
  inline void pup(PUP::er& p, std::tuple<Args...>& t) {
    pup_tuple_impl<sizeof...(Args) - 1>(p, t);
  }

  template <typename... Args>
  inline void operator|(PUP::er& p, std::tuple<Args...>& t) {
    pup(p, t);
  }

  template <typename T,
            Requires<!std::is_base_of<PUP::able, T>::value> = nullptr>
  inline void pup(PUP::er& p, std::unique_ptr<T>& t) {
    T* t1 = t.get();
    PUP::ptr_helper<T>()(p, t1);
    if (p.isUnpacking()) {
      t.reset(t1);
    }
  }

  template <typename T, Requires<std::is_base_of<PUP::able, T>::value> = nullptr>
  inline void pup(PUP::er& p, std::unique_ptr<T>& t) {
    PUP::able* t1 = nullptr;
    if (p.isUnpacking()) {
      p | t1;
      t.reset(dynamic_cast<T*>(t1));
    } else {
      t1 = dynamic_cast<PUP::able*>(t.get());
      p | t1;
    }
  }

  template <typename T>
  inline void operator|(PUP::er& p, std::unique_ptr<T>& t) {
    pup(p, t);
  }

  template <typename T,
            Requires<!std::is_base_of<PUP::able, T>::value> = nullptr>
  inline void pup(PUP::er& p, std::shared_ptr<T>& t) {
    T* t1 = t.get();
    PUP::ptr_helper<T>()(p, t1);
    if (p.isUnpacking()) {
      t.reset(t1);
    }
  }

  template <class T, Requires<std::is_base_of<PUP::able, T>::value> = nullptr>
  inline void pup(PUP::er &p, std::shared_ptr<T> &t) {
    PUP::able* _ = (p.isUnpacking()) ? nullptr : t.get();
    p(&_);
    if (p.isUnpacking()) {
      // the shared ptr must be created with the original PUP::able ptr
      // otherwise the static cast can lead to invalid frees
      // (since it can change the pointer's value)
      t = std::static_pointer_cast<T>(std::shared_ptr<PUP::able>(_));
    }
  }

  template <typename T>
  inline void operator|(PUP::er& p, std::shared_ptr<T>& t) {
    pup(p, t);
  }


  //Adding random numberengines defined in the header <Random>.
  //To pup an engine we need to pup it's state and create a new engine with the same state after unpacking
  template<
    class UIntType, size_t w, size_t n, size_t m, size_t r,
    UIntType a, size_t u, UIntType d, size_t s,UIntType b, size_t t,UIntType c, size_t l, UIntType f>
  inline void pup(PUP::er& p, std::mersenne_twister_engine<UIntType, w, n, m, r,
                             a, u, d, s, b, t, c, l, f>& engine){
    std::stringstream o;
    std::string state;

    if(p.isUnpacking()){
      p | state;
      o.str(state);
      o>>engine;
    }
    else{
      o<<engine;
      state=o.str();
      p | state;
    }
  }

  template<class UIntType, size_t w, size_t n, size_t m, size_t r,
    UIntType a, size_t u, UIntType d, size_t s,UIntType b, size_t t,UIntType c, size_t l, UIntType f>
  inline void operator|(PUP::er& p, std::mersenne_twister_engine<UIntType, w, n, m, r,
                        a, u, d, s, b, t, c, l, f>& engine) {
    pup(p,engine);
  }

  template<class UIntType, UIntType a, UIntType c, UIntType m>
  inline void pup(PUP::er& p, std::linear_congruential_engine<UIntType, a, c, m>& engine){
    std::stringstream o;
    std::string state;

    if(p.isUnpacking()){
      p | state;
      o.str(state);
      o>>engine;
    }
    else{
      o<<engine;
      state=o.str();
      p | state;
    }
  }

  template<class UIntType, UIntType a, UIntType c, UIntType m>
  inline void operator|(PUP::er& p, std::linear_congruential_engine<UIntType, a, c, m>& engine) {
    pup(p,engine);
  }

  template<class UIntType, size_t w, size_t s, size_t r>
  inline void pup(PUP::er& p, std::subtract_with_carry_engine<UIntType, w, s, r>& engine){
    std::stringstream o;
    std::string state;

    if(p.isUnpacking()){
      p | state;
      o.str(state);
      o>>engine;
    }
    else{
      o<<engine;
      state=o.str();
      p | state;
    }
  }

  template<class UIntType, size_t w, size_t s, size_t r>
  inline void operator|(PUP::er& p, std::subtract_with_carry_engine<UIntType, w, s, r>& engine) {
    pup(p,engine);
  }

  template<class Engine, size_t P, size_t R>
  inline void pup(PUP::er& p, std::discard_block_engine<Engine, P, R>& engine) {
    std::stringstream o;
    std::string state;

    if(p.isUnpacking()){
      p | state;
      o.str(state);
      o>>engine;
    }
    else{
      o<<engine;
      state=o.str();
      p | state;
    }
  }

  template<class Engine, size_t P, size_t R>
  inline void operator|(PUP::er& p, std::discard_block_engine<Engine, P, R>& engine) {
    pup(p,engine);
  }

  template<class Engine, std::size_t W, class UIntType>
  inline void pup(PUP::er& p, std::independent_bits_engine<Engine, W, UIntType>& engine) {
    std::stringstream o;
    std::string state;

    if(p.isUnpacking()){
      p | state;
      o.str(state);
      o>>engine;
    }
    else{
      o<<engine;
      state=o.str();
      p | state;
    }
  }

  template<class Engine, std::size_t W, class UIntType>
  inline void operator|(PUP::er& p, std::independent_bits_engine<Engine, W, UIntType>& engine) {
    pup(p,engine);
  }

  template<class Engine, std::size_t K>
  inline void pup(PUP::er& p, std::shuffle_order_engine<Engine, K>& engine) {
    std::stringstream o;
    std::string state;

    if(p.isUnpacking()){
      p | state;
      o.str(state);
      o>>engine;
    }
    else{
      o<<engine;
      state=o.str();
      p | state;
    }
  }

  template<class Engine, std::size_t K>
  inline void operator|(PUP::er& p, std::shuffle_order_engine<Engine, K>& engine) {
    pup(p,engine);
  }

  template <class Rep, class Period>
  inline void pup(PUP::er& p, std::chrono::duration<Rep, Period>& duration)
  {
    Rep count;
    if (p.isUnpacking())
    {
      p | count;
      duration = std::chrono::duration<Rep, Period>(count);
    }
    else
    {
      count = duration.count();
      p | count;
    }
  }

  template <class Rep, class Period>
  inline void operator|(PUP::er& p, std::chrono::duration<Rep, Period>& duration)
  {
    pup(p, duration);
  }

  template <class Clock, class Duration>
  inline void pup(PUP::er& p, std::chrono::time_point<Clock, Duration>& tp)
  {
    Duration sinceEpoch;
    if (p.isUnpacking())
    {
      p | sinceEpoch;
      tp = std::chrono::duration<Clock, Duration>(sinceEpoch);
    }
    else
    {
      sinceEpoch = tp.time_since_epoch();
      p | sinceEpoch;
    }
  }

  template <class Clock, class Duration>
  inline void operator|(PUP::er& p, std::chrono::time_point<Clock, Duration>& tp)
  {
    pup(p, tp);
  }
} // end of namespace PUP

#endif

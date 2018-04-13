/*
 * Utilities for compile-time checks on Charm++ traits,
 * such as if an object is a Chare or has a PUP routine.
 */

#ifndef _CHARM_TYPE_TRAITS_H_
#define _CHARM_TYPE_TRAITS_H_

#include <charm++.h>
#include <pup.h>


namespace charmxx {

/// Check if `T` is a Charm++ proxy for an array chare
template <typename T>
struct is_array_proxy : std::is_base_of<CProxy_ArrayElement, T>::type {};

/// Check if `T` is a Charm++ proxy for a chare
template <typename T>
struct is_chare_proxy : std::is_base_of<CProxy_Chare, T>::type {};

/// Check if `T` is a Charm++ proxy for a group chare
template <typename T>
struct is_group_proxy : std::is_base_of<CProxy_IrrGroup, T>::type {};

/// Check if `T` is a Charm++ proxy for a node group chare
template <typename T>
struct is_node_group_proxy : std::is_base_of<CProxy_NodeGroup, T>::type {};

namespace cpp17 {

#if CMK_HAS_STD_VOID_T // std::void_t is C++17
using std::void_t;
#else // CMK_HAS_STD_VOID_T
template <typename... Ts>
using void_t = void;
#endif // CMK_HAS_STD_VOID_T

} // namespace cpp17

/// Check if `T` is a Charm++ bound array
template <typename T, typename = cpp17::void_t<>>
struct is_bound_array : std::false_type {};

template <typename T>
struct is_bound_array<T, cpp17::void_t<typename T::bind_to>> : std::true_type {
  static_assert(charmxx::is_array_proxy<typename T::type>::value,
                "Can only bind a chare array");
  static_assert(charmxx::is_array_proxy<typename T::bind_to::type>::value,
                "Can only bind to a chare array");
};

template <typename T, typename = cpp17::void_t<>>
struct has_pup_member : std::false_type {};

template <typename T>
struct has_pup_member<
    T, cpp17::void_t<decltype(std::declval<T>().pup(std::declval<PUP::er&>()))>>
    : std::true_type {};

template <typename T>
constexpr bool has_pup_member_v = has_pup_member<T>::value;

template <typename T>
using has_pup_member_t = typename has_pup_member<T>::type;

template <typename T, typename U = void>
struct is_pupable : std::false_type {};

template <typename T>
struct is_pupable<
    T, cpp17::void_t<decltype(std::declval<PUP::er&>() | std::declval<T&>())>>
    : std::true_type {};

template <typename T>
constexpr bool is_pupable_v = is_pupable<T>::value;

template <typename T>
using is_pupable_t = typename is_pupable<T>::type;

} // namespace charmxx

#endif //_CHARM_TYPE_TRAITS_H_

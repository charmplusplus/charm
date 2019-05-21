/*
 * ck128bitHash.h
 *
 *  Created on: Apr 12, 2017
 *      Author: Tim Haines (thaines.astro@gmail.com)
 */

#ifndef CK128BITHASH_H_
#define CK128BITHASH_H_

#include "converse.h"

/*
 * Explicit template specialization of C++11's std::hash for CmiUInt16 (128-bit integers)
 *
 * The code is based on the std::unordered_set benchmark from libc++
 * https://github.com/llvm-mirror/libcxx/blob/master/benchmarks/unordered_set_operations.bench.cpp#L69
 *
 * NOTE: In GNU-extension mode (e.g., --std=gnu++11), libstdc++ already defines this explicit
 * 		 specialization. However, that implementation uses a "null" hash which has a larger bias
 * 		 than this implementation. Don't use it unless you really have to.
 *
 */
#if !defined __GLIBCXX_TYPE_INT_N_0
namespace std {
	template<> struct hash<CmiUInt16> {
		using argument_type = CmiUInt16;
		using result_type = std::size_t;
		static_assert(sizeof(CmiUInt16) == 16, "128-bit integers are not supported");

		CMI_FORCE_INLINE
		result_type operator()(argument_type data) const {
			const argument_type mask = static_cast<std::size_t>(-1);
			const auto a = static_cast<std::size_t>(data & mask);
			const auto b = static_cast<std::size_t>((data & (mask << 64)) >> 64);
			return hash_len_16(a, rotate_by_at_least_1(b + 16, 16)) ^ b;
		}
		CMI_FORCE_INLINE
		result_type rotate_by_at_least_1(std::size_t val, int shift) const {
			return (val >> shift) | (val << (64 - shift));
		}
		CMI_FORCE_INLINE
		result_type hash_len_16(std::size_t u, std::size_t v) const {
			const std::size_t mul = 0x9ddfea08eb382d69ULL;
			auto a = (u ^ v) * mul;
			a ^= (a >> 47);
			auto b = (v ^ a) * mul;
			b ^= (b >> 47);
			return b * mul;
		}
	};
}
#endif

#endif /* CK128BITHASH_H_ */

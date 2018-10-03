#ifndef __STDC_FORMAT_MACROS
# define __STDC_FORMAT_MACROS
#endif
#ifndef __STDC_LIMIT_MACROS
# define __STDC_LIMIT_MACROS
#endif

#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <random>
#include <chrono>
#include <algorithm>
#include <functional>
#include <type_traits>

// for getrusage
#include <sys/time.h>
#include <sys/resource.h>

#include "mpi.h"

#define STRINGIZE(x) #x
#define STRINGIZE_VALUE_OF(x) STRINGIZE(x)
#define isomalloc_method_str STRINGIZE_VALUE_OF(isomalloc_method)

static const uint32_t primes[] =
{
  2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
  73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151,
  157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233,
  239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317,
  331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419,
  421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503,
  509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607,
  613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701,
  709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811,
  821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911,
  919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997, 1009, 1013, 1019,
  1021, 1031, 1033, 1039, 1049, 1051, 1061, 1063, 1069, 1087, 1091, 1093, 1097,
  1103, 1109, 1117, 1123, 1129, 1151, 1153, 1163, 1171, 1181, 1187, 1193, 1201,
  1213, 1217, 1223, 1229, 1231, 1237, 1249, 1259, 1277, 1279, 1283, 1289, 1291,
  1297, 1301, 1303, 1307, 1319, 1321, 1327, 1361, 1367, 1373, 1381, 1399, 1409,
  1423, 1427, 1429, 1433, 1439, 1447, 1451, 1453, 1459, 1471, 1481, 1483, 1487,
  1489, 1493, 1499, 1511, 1523, 1531, 1543, 1549, 1553, 1559, 1567, 1571, 1579,
  1583, 1597, 1601, 1607, 1609, 1613, 1619, 1621, 1627, 1637, 1657, 1663, 1667,
  1669, 1693, 1697, 1699, 1709, 1721, 1723, 1733, 1741, 1747, 1753, 1759, 1777,
  1783, 1787, 1789, 1801, 1811, 1823, 1831, 1847, 1861, 1867, 1871, 1873, 1877,
  1879, 1889, 1901, 1907, 1913, 1931, 1933, 1949, 1951, 1973, 1979, 1987, 1993,
  1997, 1999, 2003, 2011, 2017, 2027, 2029, 2039, 2053, 2063, 2069, 2081, 2083,
  2087, 2089, 2099, 2111, 2113, 2129, 2131, 2137, 2141, 2143, 2153, 2161, 2179,
  2203, 2207, 2213, 2221, 2237, 2239, 2243, 2251, 2267, 2269, 2273, 2281, 2287,
  2293, 2297, 2309, 2311, 2333, 2339, 2341, 2347, 2351, 2357, 2371, 2377, 2381,
  2383, 2389, 2393, 2399, 2411, 2417, 2423, 2437, 2441, 2447, 2459, 2467, 2473,
  2477, 2503, 2521, 2531, 2539, 2543, 2549, 2551, 2557, 2579, 2591, 2593, 2609,
  2617, 2621, 2633, 2647, 2657, 2659, 2663, 2671, 2677, 2683, 2687, 2689, 2693,
  2699, 2707, 2711, 2713, 2719, 2729, 2731, 2741, 2749, 2753, 2767, 2777, 2789,
  2791, 2797, 2801, 2803, 2819, 2833, 2837, 2843, 2851, 2857, 2861, 2879, 2887,
  2897, 2903, 2909, 2917, 2927, 2939, 2953, 2957, 2963, 2969, 2971, 2999, 3001,
  3011, 3019, 3023, 3037, 3041, 3049, 3061, 3067, 3079, 3083, 3089, 3109, 3119,
  3121, 3137, 3163, 3167, 3169, 3181, 3187, 3191, 3203, 3209, 3217, 3221, 3229,
  3251, 3253, 3257, 3259, 3271, 3299, 3301, 3307, 3313, 3319, 3323, 3329, 3331,
  3343, 3347, 3359, 3361, 3371, 3373, 3389, 3391, 3407, 3413, 3433, 3449, 3457,
  3461, 3463, 3467, 3469, 3491, 3499, 3511, 3517, 3527, 3529, 3533, 3539, 3541,
  3547, 3557, 3559, 3571, 3581, 3583, 3593, 3607, 3613, 3617, 3623, 3631, 3637,
  3643, 3659, 3671, 3673, 3677, 3691, 3697, 3701, 3709, 3719, 3727, 3733, 3739,
  3761, 3767, 3769, 3779, 3793, 3797, 3803, 3821, 3823, 3833, 3847, 3851, 3853,
  3863, 3877, 3881, 3889, 3907, 3911, 3917, 3919, 3923, 3929, 3931, 3943, 3947,
  3967, 3989, 4001, 4003, 4007, 4013, 4019, 4021, 4027, 4049, 4051, 4057, 4073,
  4079, 4091, 4093, 4099, 4111, 4127, 4129, 4133, 4139, 4153, 4157, 4159, 4177,
  4201, 4211, 4217, 4219, 4229, 4231, 4241, 4243, 4253, 4259, 4261, 4271, 4273,
  4283, 4289, 4297, 4327, 4337, 4339, 4349, 4357, 4363, 4373, 4391, 4397, 4409,
  4421, 4423, 4441, 4447, 4451, 4457, 4463, 4481, 4483, 4493, 4507, 4513, 4517,
  4519, 4523, 4547, 4549, 4561, 4567, 4583, 4591, 4597, 4603, 4621, 4637, 4639,
  4643, 4649, 4651, 4657, 4663, 4673, 4679, 4691, 4703, 4721, 4723, 4729, 4733,
  4751, 4759, 4783, 4787, 4789, 4793, 4799, 4801, 4813, 4817, 4831, 4861, 4871,
  4877, 4889, 4903, 4909, 4919, 4931, 4933, 4937, 4943, 4951, 4957, 4967, 4969,
  4973, 4987, 4993, 4999, 5003, 5009, 5011, 5021, 5023, 5039, 5051, 5059, 5077,
  5081, 5087, 5099, 5101, 5107, 5113, 5119, 5147, 5153, 5167, 5171, 5179, 5189,
  5197, 5209, 5227, 5231, 5233, 5237, 5261, 5273, 5279, 5281, 5297, 5303, 5309,
  5323, 5333, 5347, 5351, 5381, 5387, 5393, 5399, 5407, 5413, 5417, 5419, 5431,
  5437, 5441, 5443, 5449, 5471, 5477, 5479, 5483, 5501, 5503, 5507, 5519, 5521,
  5527, 5531, 5557, 5563, 5569, 5573, 5581, 5591, 5623, 5639, 5641, 5647, 5651,
  5653, 5657, 5659, 5669, 5683, 5689, 5693, 5701, 5711, 5717, 5737, 5741, 5743,
  5749, 5779, 5783, 5791, 5801, 5807, 5813, 5821, 5827, 5839, 5843, 5849, 5851,
  5857, 5861, 5867, 5869, 5879, 5881, 5897, 5903, 5923, 5927, 5939, 5953, 5981,
  5987, 6007, 6011, 6029, 6037, 6043, 6047, 6053, 6067, 6073, 6079, 6089, 6091,
  6101, 6113, 6121, 6131, 6133, 6143, 6151, 6163, 6173, 6197, 6199, 6203, 6211,
  6217, 6221, 6229, 6247, 6257, 6263, 6269, 6271, 6277, 6287, 6299, 6301, 6311,
  6317, 6323, 6329, 6337, 6343, 6353, 6359, 6361, 6367, 6373, 6379, 6389, 6397,
  6421, 6427, 6449, 6451, 6469, 6473, 6481, 6491, 6521, 6529, 6547, 6551, 6553,
  6563, 6569, 6571, 6577, 6581, 6599, 6607, 6619, 6637, 6653, 6659, 6661, 6673,
  6679, 6689, 6691, 6701, 6703, 6709, 6719, 6733, 6737, 6761, 6763, 6779, 6781,
  6791, 6793, 6803, 6823, 6827, 6829, 6833, 6841, 6857, 6863, 6869, 6871, 6883,
  6899, 6907, 6911, 6917, 6947, 6949, 6959, 6961, 6967, 6971, 6977, 6983, 6991,
  6997, 7001, 7013, 7019, 7027, 7039, 7043, 7057, 7069, 7079, 7103, 7109, 7121,
  7127, 7129, 7151, 7159, 7177, 7187, 7193, 7207, 7211, 7213, 7219, 7229, 7237,
  7243, 7247, 7253, 7283, 7297, 7307, 7309, 7321, 7331, 7333, 7349, 7351, 7369,
  7393, 7411, 7417, 7433, 7451, 7457, 7459, 7477, 7481, 7487, 7489, 7499, 7507,
  7517, 7523, 7529, 7537, 7541, 7547, 7549, 7559, 7561, 7573, 7577, 7583, 7589,
  7591, 7603, 7607, 7621, 7639, 7643, 7649, 7669, 7673, 7681, 7687, 7691, 7699,
  7703, 7717, 7723, 7727, 7741, 7753, 7757, 7759, 7789, 7793, 7817, 7823, 7829,
  7841, 7853, 7867, 7873, 7877, 7879, 7883, 7901, 7907, 7919, 7927, 7933, 7937,
  7949, 7951, 7963, 7993, 8009, 8011, 8017, 8039, 8053, 8059, 8069, 8081, 8087,
  8089, 8093, 8101, 8111, 8117, 8123, 8147, 8161, 8167, 8171, 8179, 8191, 8209,
  8219, 8221, 8231, 8233, 8237, 8243, 8263, 8269, 8273, 8287, 8291, 8293, 8297,
  8311, 8317, 8329, 8353, 8363, 8369, 8377, 8387, 8389, 8419, 8423, 8429, 8431,
  8443, 8447, 8461, 8467, 8501, 8513, 8521, 8527, 8537, 8539, 8543, 8563, 8573,
  8581, 8597, 8599, 8609, 8623, 8627, 8629, 8641, 8647, 8663, 8669, 8677, 8681,
  8689, 8693, 8699, 8707, 8713, 8719, 8731, 8737, 8741, 8747, 8753, 8761, 8779,
  8783, 8803, 8807, 8819, 8821, 8831, 8837, 8839, 8849, 8861, 8863, 8867, 8887,
  8893, 8923, 8929, 8933, 8941, 8951, 8963, 8969, 8971, 8999, 9001, 9007, 9011,
  9013, 9029, 9041, 9043, 9049, 9059, 9067, 9091, 9103, 9109, 9127, 9133, 9137,
  9151, 9157, 9161, 9173, 9181, 9187, 9199, 9203, 9209, 9221, 9227, 9239, 9241,
  9257, 9277, 9281, 9283, 9293, 9311, 9319, 9323, 9337, 9341, 9343, 9349, 9371,
  9377, 9391, 9397, 9403, 9413, 9419, 9421, 9431, 9433, 9437, 9439, 9461, 9463,
  9467, 9473, 9479, 9491, 9497, 9511, 9521, 9533, 9539, 9547, 9551, 9587, 9601,
  9613, 9619, 9623, 9629, 9631, 9643, 9649, 9661, 9677, 9679, 9689, 9697, 9719,
  9721, 9733, 9739, 9743, 9749, 9767, 9769, 9781, 9787, 9791, 9803, 9811, 9817,
  9829, 9833, 9839, 9851, 9857, 9859, 9871, 9883, 9887, 9901, 9907, 9923, 9929,
  9931, 9941, 9949, 9967, 9973,
};

static const uint32_t largeprimes[] =
{
  143687, 266083, 393073, 523417, 656683, 791447, 927853, 1066909, 1205653,
  1346567, 1488133, 1630987, 1774937, 1918439, 2063573, 2209499, 2356573,
  2503597, 2650931, 2799451, 2948161, 3097141, 3246119, 3396661, 3548471,
  3698881, 3849943, 4002547, 4154791, 4307453, 4459951, 4613237, 4766999,
  4920613, 5074633, 5229331, 5383601, 5539363, 5695831, 5851717, 6007247,
  6163567, 6319483, 6477301, 6633877, 6791437, 6947753, 7104341, 7263143,
  7421383, 7580459, 7738559, 7896949, 8056423, 8215633, 8375273, 8534489,
  8693653, 8853359, 9014101, 9174793, 9334487, 9495601, 9655759, 9817057,
  9979111, 10140307,
};

template <typename T, size_t N>
static inline constexpr size_t countof(T const (&)[N]) noexcept
{
  return N;
}

static inline void request_migration()
{
  AMPI_Migrate(AMPI_INFO_LB_SYNC);
}

using clock_type = typename std::conditional<
  std::chrono::high_resolution_clock::is_steady,
  std::chrono::high_resolution_clock, std::chrono::steady_clock>::type;
using time_point = typename std::chrono::time_point<clock_type>;
using duration_type = typename std::chrono::duration<unsigned long long, std::nano>;

static void malloc_stress(int rank, int p, uint32_t randomseed)
{
  // This function avoids heap allocations other than the test itself.

  std::mt19937 generator{randomseed};

  static constexpr size_t maxallocs = 2048;
  static constexpr size_t maxsegments = 8;
  static constexpr size_t maxtests = 32;

  struct alloc_type
  {
    unsigned char *ptr;
    unsigned int size;
  };
  alloc_type allocations[maxallocs]{};
  alloc_type * allocation_queue[maxallocs*2]{};

  unsigned int segments[maxsegments]{}; // segment boundaries

  struct test_type
  {
    const char *description1, *description2;
    unsigned long long duration_count;
  };
  test_type tests[maxtests]{};

  size_t numallocs = 0;
  size_t numsegments = 1;
  segments[0] = 0;


  static_assert(countof(primes) + 3*countof(largeprimes) <= maxallocs, "increase maxallocs");

  for (size_t i = rank; i < countof(primes); i += p)
  {
    allocations[numallocs++].size = primes[i];
  }
  segments[numsegments++] = numallocs;

  for (size_t i = rank; i < countof(largeprimes); i += p)
  {
    allocations[numallocs++].size = largeprimes[i];
  }
  segments[numsegments++] = numallocs;

  for (size_t i = rank; i < countof(largeprimes); i += p)
  {
    allocations[numallocs++].size = primes[countof(primes)-1-i];
    allocations[numallocs++].size = largeprimes[i];
  }
  segments[numsegments++] = numallocs;


  const size_t queuesize = numallocs * 2;

#ifdef MIGRATION_OVERKILL
  const size_t migrationinterval = numallocs / 31; // prime number divisor
  size_t eventnum = 0;
#endif

  size_t testnum = 0;

  auto alloc_loop = [&](const size_t a_begin, const size_t a_end)
  {
    for (size_t a = a_begin; a < a_end; ++a)
    {
      alloc_type & allocation = *(allocation_queue[a]);
      const size_t size = allocation.size;
      const unsigned char rankbyte = rank & 0xFF;

      const auto ptr = allocation.ptr = (unsigned char*)malloc(size);
      if (ptr == nullptr)
      {
        printf("Error: malloc() failure in rank %d, test %zu, allocation %zu\n", rank, testnum, a);
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
      ptr[0] = ptr[size-1] = rankbyte;

#ifdef MIGRATION_OVERKILL
      ++eventnum;
      if (eventnum == migrationinterval)
      {
        eventnum = 0;
        request_migration();
      }
#endif
    }
  };

  auto free_loop = [&](const size_t a_begin, const size_t a_end)
  {
    for (size_t a = a_begin; a < a_end; ++a)
    {
      alloc_type & allocation = *(allocation_queue[a]);
      const size_t size = allocation.size;
      const unsigned char rankbyte = rank & 0xFF;

      const auto ptr = allocation.ptr;
      if (ptr[0] != rankbyte || ptr[size-1] != rankbyte)
      {
        printf("Error: Correctness failure in rank %d, test %zu, allocation %zu\n", rank, testnum, a);
        MPI_Abort(MPI_COMM_WORLD, 2);
      }
      free(ptr);
      allocation.ptr = nullptr;

#ifdef MIGRATION_OVERKILL
      ++eventnum;
      if (eventnum == migrationinterval)
      {
        eventnum = 0;
        request_migration();
      }
#endif
    }
  };

  auto undetermined_loop = [&](const size_t a_begin, const size_t a_end)
  {
    for (size_t a = a_begin; a < a_end; ++a)
    {
      alloc_type & allocation = *(allocation_queue[a]);
      const size_t size = allocation.size;
      const unsigned char rankbyte = rank & 0xFF;

      auto ptr = allocation.ptr;
      if (ptr == nullptr)
      {
        ptr = allocation.ptr = (unsigned char*)malloc(size);
        if (ptr == nullptr)
        {
          printf("Error: malloc() failure in rank %d, test %zu, allocation %zu\n", rank, testnum, a);
          MPI_Abort(MPI_COMM_WORLD, 1);
        }
        ptr[0] = ptr[size-1] = rankbyte;
      }
      else
      {
        if (ptr[0] != rankbyte || ptr[size-1] != rankbyte)
        {
          printf("Error: Correctness failure in rank %d, test %zu, allocation %zu\n", rank, testnum, a);
          MPI_Abort(MPI_COMM_WORLD, 2);
        }
        allocation.ptr = nullptr;
        free(ptr);
      }

#ifdef MIGRATION_OVERKILL
      ++eventnum;
      if (eventnum == migrationinterval)
      {
        eventnum = 0;
        request_migration();
      }
#endif
    }
  };

  auto reset_queue = [&]()
  {
    alloc_type ** q = allocation_queue;
    for (alloc_type * al = allocations, * al_end = al + numallocs; al < al_end; ++al)
      *(q++) = al;
  };

  // tests freeing after every segment
  auto test_by_segment = [&](std::function<void(size_t)> between = [](size_t){}, std::function<void(size_t)> after = [](size_t){})
  {
#ifdef MIGRATION_OVERKILL
    eventnum = 0;
#endif

    if (testnum == maxtests)
    {
      printf("Error: Increase maxtests.\n");
      MPI_Abort(MPI_COMM_WORLD, 10);
    }

    unsigned long long test_duration_count = 0;

    MPI_Barrier(MPI_COMM_WORLD);

    for (size_t s = 1; s < numsegments; ++s)
    {
      const time_point t_0 = clock_type::now();
      alloc_loop(segments[s-1], segments[s]);
      const time_point t_1 = clock_type::now();

      request_migration();
      between(s);

      const time_point t_2 = clock_type::now();
      free_loop(segments[s-1], segments[s]);
      const time_point t_3 = clock_type::now();

      request_migration();
      after(s);

      const duration_type test_duration = (t_3 - t_2) + (t_1 - t_0);
      test_duration_count += test_duration.count();
    }

    unsigned long long duration_count = 0;
    MPI_Reduce(&test_duration_count, &duration_count, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    tests[testnum].duration_count = duration_count;

    ++testnum;
  };

  // tests freeing at the end
  auto test_all = [&](std::function<void()> between = [](){}, std::function<void()> after = [](){})
  {
#ifdef MIGRATION_OVERKILL
    eventnum = 0;
#endif

    if (testnum == maxtests)
    {
      printf("Error: Increase maxtests.\n");
      MPI_Abort(MPI_COMM_WORLD, 10);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    const time_point t_0 = clock_type::now();
    alloc_loop(0, numallocs);
    const time_point t_1 = clock_type::now();

    request_migration();
    between();

    const time_point t_2 = clock_type::now();
    free_loop(0, numallocs);
    const time_point t_3 = clock_type::now();

    request_migration();
    after();

    const duration_type test_duration = (t_3 - t_2) + (t_1 - t_0);
    unsigned long long test_duration_count = test_duration.count();

    unsigned long long duration_count = 0;
    MPI_Reduce(&test_duration_count, &duration_count, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    tests[testnum].duration_count = duration_count;

    ++testnum;
  };

  // tests with alloc and free interspersed
  auto test_mixed = [&]()
  {
#ifdef MIGRATION_OVERKILL
    eventnum = 0;
#endif

    if (testnum == maxtests)
    {
      printf("Error: Increase maxtests.\n");
      MPI_Abort(MPI_COMM_WORLD, 10);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    const time_point t_0 = clock_type::now();
    undetermined_loop(0, numallocs*2);
    const time_point t_1 = clock_type::now();

    request_migration();

    const duration_type test_duration = t_1 - t_0;
    unsigned long long test_duration_count = test_duration.count();

    unsigned long long duration_count = 0;
    MPI_Reduce(&test_duration_count, &duration_count, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    tests[testnum].duration_count = duration_count;

    ++testnum;
  };

  auto shuffle_segment = [&](size_t s)
  {
    std::shuffle(allocation_queue + segments[s-1], allocation_queue + segments[s], generator);
  };
  auto shuffle_segments = [&]()
  {
    for (size_t s = 1; s < numsegments; ++s)
      shuffle_segment(s);
  };
  auto shuffle_all = [&]()
  {
    std::shuffle(allocation_queue, allocation_queue + numallocs, generator);
  };

  auto reverse_segment = [&](size_t s)
  {
    std::reverse(allocation_queue + segments[s-1], allocation_queue + segments[s]);
  };
  auto reverse_segments = [&]()
  {
    for (size_t s = 1; s < numsegments; ++s)
      reverse_segment(s);
  };
  auto reverse_all = [&]()
  {
    std::reverse(allocation_queue, allocation_queue + numallocs);
  };

  auto shuffle_segments_and_reverse_all = [&]()
  {
    shuffle_segments();
    reverse_all();
  };

  auto tests_static = [&](const char *desc1)
  {
    // allocate and free in the original order

    tests[testnum].description1 = desc1;
    tests[testnum].description2 = "FIFO free, batched by segment";
    test_by_segment();

    tests[testnum].description1 = desc1;
    tests[testnum].description2 = "FIFO free, one batch";
    test_all();


    // allocate in the original order, free in the reverse order

    tests[testnum].description1 = desc1;
    tests[testnum].description2 = "LIFO free, batched by segment";
    test_by_segment(reverse_segment, reverse_segment);

    tests[testnum].description1 = desc1;
    tests[testnum].description2 = "LIFO free within segments, one batch";
    test_all(reverse_segments, reverse_segments);

    tests[testnum].description1 = desc1;
    tests[testnum].description2 = "LIFO free overall, one batch";
    test_all(reverse_all, reverse_all);
  };

  auto tests_dynamic = [&](const char *desc1)
  {
    tests[testnum].description1 = desc1;
    tests[testnum].description2 = "within segments, batched by segment";
    test_by_segment(shuffle_segment);

    tests[testnum].description1 = desc1;
    tests[testnum].description2 = "within segments, FIFO segment order, one batch";
    test_all(shuffle_segments);

    tests[testnum].description1 = desc1;
    tests[testnum].description2 = "within segments, LIFO segment order, one batch";
    test_all(shuffle_segments_and_reverse_all); // reverse the segment order

    tests[testnum].description1 = desc1;
    tests[testnum].description2 = "overall, one batch";
    test_all(shuffle_all);
  };

  auto tests_mixed = [&](const char *desc1)
  {
    {
      alloc_type ** q = allocation_queue;
      for (size_t s = 1; s < numsegments; ++s)
      {
        const size_t a_begin = segments[s-1], a_end = segments[s];

        alloc_type ** const q_begin = q;

        alloc_type * const al_begin = allocations + a_begin, * const al_end = allocations + a_end;
        for (alloc_type * al = al_begin; al < al_end; ++al)
          *(q++) = al;
        for (alloc_type * al = al_begin; al < al_end; ++al)
          *(q++) = al;

        std::shuffle(q_begin, q, generator);
      }
    }
    tests[testnum].description1 = desc1;
    tests[testnum].description2 = "within segments, batched by segment";
    test_mixed();

    {
      alloc_type ** q = allocation_queue;
      for (alloc_type * al = allocations, * al_end = al + numallocs; al < al_end; ++al)
        *(q++) = al;
      for (alloc_type * al = allocations, * al_end = al + numallocs; al < al_end; ++al)
        *(q++) = al;
    }
    std::shuffle(allocation_queue, allocation_queue + queuesize, generator);
    tests[testnum].description1 = desc1;
    tests[testnum].description2 = "overall, one batch";
    test_mixed();
  };


  MPI_Barrier(MPI_COMM_WORLD);
  const time_point t_0 = clock_type::now();

  reset_queue();

  tests_static("distinct phases, without shuffling, ");

  shuffle_segments();
  tests_static("distinct phases, static shuffle within segments, ");

  shuffle_all();
  tests_static("distinct phases, static shuffle overall, ");

  shuffle_all();
  tests_dynamic("distinct phases, shuffling between phases ");

  tests_mixed("one phase, shuffled ");

  MPI_Barrier(MPI_COMM_WORLD);
  const time_point t_1 = clock_type::now();


  const duration_type overall_duration = t_1 - t_0;
  unsigned long long overall_duration_count = overall_duration.count();

  // print results
  if (rank == 0)
  {
    unsigned long long total_duration_count = 0;

    printf("malloc stress test results:\n");
    printf("   # - duration (ns) - description\n");
    for (size_t t = 0; t < testnum; ++t)
    {
      const test_type & test = tests[t];

      total_duration_count += test.duration_count;

      printf("%4zu - %13llu - %s%s\n", t+1, test.duration_count, test.description1, test.description2);
    }
    printf(" all - %13llu - total\n", total_duration_count);

    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);

    printf("memory usage high water mark ("
#ifdef __APPLE__
      "bytes"
#else
      "kilobytes"
#endif
      "): %lu\n", usage.ru_maxrss);
    printf("overall test duration (ns) including migration, shuffling, barriers, etc: %llu\n", overall_duration_count);
  }
}

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);

  int rank;
  int p;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  uint32_t randomseed;

  if (argc > 1)
  {
    randomseed = (uint32_t)atol(argv[1]);
  }
  else
  {
    if (rank == 0)
      randomseed = std::chrono::system_clock::now().time_since_epoch().count();

    MPI_Bcast(&randomseed, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
  }

  if (rank == 0)
    printf("memory test: ./" isomalloc_method_str " (...) +vp%i %" PRIu32 "\n", p, randomseed);

  malloc_stress(rank, p, randomseed);

  MPI_Finalize();

  return 0;
}

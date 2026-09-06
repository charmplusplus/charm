// hashcheck: prints the example's routing and init values so ref/selftest.py
// can check the Python port bit for bit. The functions are copied from moe.C
// (host hashing, buildPhase, route) and moe.cu (device init, emulated on the
// host in the same float arithmetic).
//   hashcheck E T K zipf drift seed pe steps
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static inline uint64_t sm64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}
static inline uint64_t hmix(uint64_t a, uint64_t b) {
  return sm64(a ^ (b * 0x9E3779B97F4A7C15ULL + 0x632BE59BD9B4E019ULL));
}
static inline double u01(uint64_t h) {
  return ((double)(h >> 11) + 0.5) / 9007199254740992.0;
}
// moe.cu
static inline uint64_t splitmix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}
static inline float u01f(uint64_t h) {
  return ((float)(unsigned int)(h >> 40) + 0.5f) * (1.0f / 16777216.0f);
}
static float initVal(size_t i, uint64_t seed, float bound) {
  uint64_t h = splitmix64(seed ^ (i * 0x9E3779B97F4A7C15ULL));
  h = splitmix64(h + i);
  return bound * (2.0f * u01f(h) - 1.0f);
}

int main(int argc, char** argv) {
  if (argc < 9) { fprintf(stderr, "usage: E T K zipf drift seed pe steps\n"); return 1; }
  const int E = atoi(argv[1]), T = atoi(argv[2]), K = atoi(argv[3]);
  const double zipf_z = atof(argv[4]);
  const int drift = atoi(argv[5]);
  const uint64_t seed = strtoull(argv[6], 0, 10);
  const int pe = atoi(argv[7]), steps = atoi(argv[8]);

  printf("hmix %llu %llu %llu\n", (unsigned long long)hmix(seed, 1),
         (unsigned long long)hmix(hmix(seed, 1000), pe),
         (unsigned long long)hmix(hmix(hmix(hmix(hmix(seed, 3), pe), 7), 0), 2));

  std::vector<int> perm(E);
  std::vector<double> cdf(E);
  int cur_phase = -1;
  for (int step = 1; step <= steps; step++) {
    const int phase = drift > 0 ? (step - 1) / drift : 0;
    if (phase != cur_phase) {
      cur_phase = phase;
      for (int i = 0; i < E; i++) perm[i] = i;
      for (int i = E - 1; i > 0; i--) {
        const uint64_t h = hmix(hmix(seed, 77), (uint64_t)phase * E + i);
        const int j = (int)(h % (uint64_t)(i + 1));
        std::swap(perm[i], perm[j]);
      }
      double s = 0.0;
      for (int r = 0; r < E; r++) {
        s += (zipf_z > 0.0) ? pow((double)(r + 1), -zipf_z) : 1.0;
        cdf[r] = s;
      }
      for (int r = 0; r < E; r++) cdf[r] /= s;
    }
    printf("route %d", step);
    for (int t = 0; t < T; t++) {
      int chosen[8];
      for (int j = 0; j < K; j++) {
        int e = -1;
        for (int attempt = 0; attempt < 16 && e < 0; attempt++) {
          const uint64_t h = hmix(hmix(hmix(hmix(hmix(seed, step), pe), t), j), attempt);
          int r = (int)(std::upper_bound(cdf.begin(), cdf.end(), u01(h)) - cdf.begin());
          if (r >= E) r = E - 1;
          const int cand = perm[r];
          bool dup = false;
          for (int q = 0; q < j; q++) if (chosen[q] == cand) dup = true;
          if (!dup) e = cand;
        }
        if (e < 0) {
          e = (chosen[j - 1] + 1) % E;
          for (;;) {
            bool dup = false;
            for (int q = 0; q < j; q++) if (chosen[q] == e) dup = true;
            if (!dup) break;
            e = (e + 1) % E;
          }
        }
        chosen[j] = e;
        printf(" %d", e);
      }
    }
    printf("\n");
  }
  // init: W1 of expert 0 and the inputs of this PE, first 8 and some far values
  const uint64_t s1 = hmix(hmix(seed, 1), 0), sx = hmix(hmix(seed, 1000), pe);
  const size_t idx[12] = {0, 1, 2, 3, 4, 5, 6, 7, 1000, 123456, 16777215, 16777216};
  printf("init");
  for (int q = 0; q < 12; q++) {
    float v = initVal(idx[q], s1, 1.0f / sqrtf(2048.0f));
    uint32_t b; memcpy(&b, &v, 4);
    printf(" %u", b);
  }
  for (int q = 0; q < 12; q++) {
    float v = initVal(idx[q], sx, sqrtf(3.0f));
    uint32_t b; memcpy(&b, &v, 4);
    printf(" %u", b);
  }
  printf("\n");
  return 0;
}

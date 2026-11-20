# Testing before a release

What to run before tagging a release (alpha, beta or final), on top of the
pull-request CI in `.github/workflows/reconverse-ci.yaml`. CI runs on hosted
runners: one network (TCP), a few compilers, unoptimized builds without memory
checking. The runs below cover what it does not.

Record each run on the release's tracking issue: machine, compiler, charm and
reconverse commits, the build command, and the script's `RESULT` and `SUMMARY`
lines.

| # | run | where | what it covers |
|---|---|---|---|
| 1 | CI green on the commit to be tagged | GitHub | the test tier on Linux x86_64 and arm64, macOS, Windows |
| 2 | Production site run | one InfiniBand machine and one Slingshot machine | real networks, several nodes, the site launcher |
| 3 | Checking site run | one cluster, x86_64 | out-of-bounds accesses, use after free and double frees |
| 4 | Newest compilers | x86_64, cluster or container | what a new GCC or clang release changes |
| 5 | Build from the release tarball | any machine | the archive a user downloads, without git |

## 2. Production site run

Build with `--with-production` and run `tests/reconverse-site-run.sh` in a
batch allocation of at least 2 nodes; its header gives the usage and the site
profiles. Run the multi-process pass with several processes per node as well
as one (`NODES=2 PROCS=4`), since that is how applications are launched.

## 3. Checking site run

A production build runs through many memory errors without a symptom, and so
does CI's unoptimized build. Build once with AddressSanitizer and libstdc++'s
assertions and run the same tier with `CHECK=1`:

    S="-fsanitize=address -fno-omit-frame-pointer -D_GLIBCXX_ASSERTIONS"
    CFLAGS="$S" CXXFLAGS="$S" LDFLAGS=-fsanitize=address ASAN_OPTIONS=detect_leaks=0 \
      ./build charm++ reconverse-linux-x86_64 -j8 -g -O1 $S
    SITE=<site> CHECK=1 CHARM_BUILD=$PWD/reconverse-linux-x86_64 tests/reconverse-site-run.sh

Every sanitizer or assertion report is a defect to fix or to file before the
tag, whether or not the test passes without the sanitizer. Two things differ
from a production run:

- The script unsets `LD_PRELOAD`: a library the site preloads (XALT on Anvil)
  makes a sanitized program abort at startup. Unset it for the build as well,
  because the build runs the sanitized `charmxi`.
- Tests that fail only on their own memory-growth limit (`io_sessions`,
  `zc_send_noleak`) are reported but not counted: the sanitizer keeps freed
  blocks for a while, so the memory in use grows without a leak.

## 4. Newest compilers

Build and run the tier with the newest released GCC and the newest released
clang on x86_64, unoptimized (`-g`, as CI builds), in 1 and 2 processes. A site
module, Spack, or a container image (`gcc:<version>`) all serve. Compiler
releases change code generation for C++20 coroutines, turn on library
assertions and default to newer language standards; arm64 does not stand in
for x86_64 here.

Record the compiler's exact version: the coroutine tests (`ckcoro`,
`jacobi3d-coro`, `sparse-coro`) behave differently under GCC 14.2, 14.3 and
15.2, and GCC 14.2 does not compile them.

## 5. Release tarball

Download the release asset, check its `sha256sum`, and build `LIBS` from it
with `--with-production` in a directory that is not a git checkout.

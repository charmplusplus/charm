include(CheckCXXSymbolExists)
include(CheckCXXSourceCompiles)
include(CheckCXXCompilerFlag)

# Workaround for CMake bug when calling check_include_file_cxx() with C++11:
# https://stackoverflow.com/uestions/47213356/cmake-using-corrext-c-standard-when-checking-for-header-files
if(${CMAKE_VERSION} VERSION_LESS "3.8.0")
  set(CMAKE_REQUIRED_FLAGS "-std=c++11")
else()
  cmake_policy(SET CMP0067 NEW)
endif()


# C++ type sizes
check_type_size("std::void_t" CMK_HAS_STD_VOID_T LANGUAGE CXX)

# C++ header files
check_include_file_cxx(atomic CMK_HAS_CXX11_ATOMIC)
check_include_file_cxx(cstdatomic CMK_HAS_CXX0X_CSTDATOMIC)
check_include_file_cxx(regex CMK_HAS_REGEX)

# C++ compiler flags
check_cxx_compiler_flag("-mno-tls-direct-seg-refs" CMK_COMPILER_KNOWS_TLSDIRECTSEGREFS)

check_cxx_compiler_flag("-fno-stack-protector" CMK_COMPILER_KNOWS_FNOSTACKPROTECTOR)
if(${CMK_COMPILER_KNOWS_FNOSTACKPROTECTOR})
  set(OPTS_CC "${OPTS_CC} -fno-stack-protector")
  set(OPTS_CXX "${OPTS_CXX} -fno-stack-protector")
endif()

check_cxx_compiler_flag("-fno-lifetime-dse" CMK_COMPILER_KNOWS_LIFETIMEDSE)
if(${CMK_COMPILER_KNOWS_LIFETIMEDSE})
  set(OPTS_CXX "${OPTS_CXX} -fno-lifetime-dse")
endif()

# C++ complex tests
check_cxx_source_compiles("
#include <type_traits>
struct s { s(int a) { } };
int main()
{
  return std::is_constructible<s, int>::value;
}" CMK_HAS_IS_CONSTRUCTIBLE)

check_cxx_source_compiles("
#include <vector>
#include <iterator>
int main()
{
  std::vector<int> tree;
  return std::distance(tree.begin(), tree.end());
}
" CMK_HAS_STD_DISTANCE)

check_cxx_source_compiles("
#include <stdlib.h>
class er {
 protected:
   void operator()(char &v,const char *desc=NULL) {};
   void operator()(signed char &v,const char *desc=NULL) {};
};
int main() {}
" CMK_SIGNEDCHAR_DIFF_CHAR)

check_cxx_source_compiles("
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <spawn.h>
int main() {
    return posix_spawn(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
}
" CMK_HAS_POSIX_SPAWN)

check_cxx_source_compiles("
#include <type_traits>
int main()
{
  return std::alignment_of<int>::value;
}
" CMK_HAS_ALIGNMENT_OF)

check_cxx_source_compiles("
#include <iterator>
template <typename T> // T models Input Iterator
typename std::iterator_traits<T>::value_type accumulate(T first, T last)
{
      typename std::iterator_traits<T>::value_type result = 0;
      while(first != last)
            result += *first++;
      return result;
}
int main() {}
" CMK_HAS_ITERATOR_TRAITS)

check_cxx_source_compiles("
#include <list>
#include <iterator>
int main()
{
  using namespace std;
  list<int> L;
  inserter ( L, L.end ( ) ) = 500;
}
" CMK_HAS_STD_INSERTER )

check_cxx_source_compiles("
#include <typeinfo>
int main() {
  int x;
  typeid(x).name();
}
" CMK_HAS_TYPEINFO )

check_cxx_source_compiles("
class foo {
public:
  void operator delete(void*p){};
  void operator delete(void*p,int*){};
};
int main() {}
" CMK_MULTIPLE_DELETE)

file(READ ${CMAKE_SOURCE_DIR}/src/util/ckdll_system.C ckdll_system)
check_cxx_source_compiles("${ckdll_system}\n int main(){}" CMK_SIGSAFE_SYSTEM)

file(READ ${CMAKE_SOURCE_DIR}/src/util/ckdll_win32.C ckdll_win32)
check_cxx_source_compiles("${ckdll_win32}\n int main(){}" CMK_DLL_USE_WIN32)

check_cxx_source_compiles("
#include <cstddef>
int main() {
  extern void *(*__morecore)(ptrdiff_t);
  __morecore(0);
  return 0;
}
" CMK_EXPECTS_MORECORE)

# Unset workaround from above
if(${CMAKE_VERSION} VERSION_LESS "3.8.0")
  set(CMAKE_REQUIRED_FLAGS "")
endif()

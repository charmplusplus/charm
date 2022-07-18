enable_language(C ASM)

set(ABI sysv)
set(SYNTAX gas)
set(EXTENSION S)
if(CHARM_OS STREQUAL "darwin")
  set(FORMAT macho)
elseif(CHARM_OS STREQUAL "win")
  set(ABI ms)
  set(FORMAT pe)
  if(CMK_COMPILER STREQUAL "msvc")
    set(SYNTAX masm)
    set(EXTENSION asm)
  endif()
else()
  set(FORMAT elf)
endif()
if(CHARM_CPU STREQUAL "arm7")
  set(ARCHITECTURE arm)
  set(ABI aapcs)
elseif(CHARM_CPU STREQUAL "arm8")
  set(ARCHITECTURE arm64)
  set(ABI aapcs)
elseif(CHARM_CPU STREQUAL "ppc64le")
  set(ARCHITECTURE ppc64)
elseif(CHARM_CPU STREQUAL "x86_64")
  set(ARCHITECTURE x86_64)
else()
  set(ARCHITECTURE i386)
endif()

set(ufcontext_sources
    jump_${ARCHITECTURE}_${ABI}_${FORMAT}_${SYNTAX}.${EXTENSION}
    make_${ARCHITECTURE}_${ABI}_${FORMAT}_${SYNTAX}.${EXTENSION}
    ontop_${ARCHITECTURE}_${ABI}_${FORMAT}_${SYNTAX}.${EXTENSION}
)

add_library(boost-context OBJECT ${ufcontext_sources})
set_target_properties(boost-context PROPERTIES LINKER_LANGUAGE CXX)

if(CMK_COMPILER STREQUAL "msvc")
    target_compile_options(boost-context PRIVATE -DBOOST_CONTEXT_EXPORT)
endif()

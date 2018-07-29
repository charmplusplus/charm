set(tcharm-c-sources compat_funs.c compat_fus.c compat_regmm.c compat_uns.c compat_us.c)

set(tcharm-h-sources tcharm.h tcharm_impl.h tcharmc.h tcharmf.h)

include(../../../ci-files.cmake)

add_library(moduletcharm tcharm.C ${tcharm-h-sources} ${CMAKE_BINARY_DIR}/include/tcharm.decl.h)

add_library(moduletcharmmain tcharmmain.C ${CMAKE_BINARY_DIR}/include/tcharm.decl.h ${CMAKE_BINARY_DIR}/include/tcharmmain.decl.h)
add_library(tcharm-compat ${tcharm-c-sources})

configure_file(libmoduletcharm.dep     ${CMAKE_BINARY_DIR}/lib/ COPYONLY)
configure_file(libmoduletcharmmain.dep ${CMAKE_BINARY_DIR}/lib/ COPYONLY)

foreach(f ${tcharm-h-sources})
    configure_file(${f} ${CMAKE_BINARY_DIR}/include COPYONLY)
endforeach(f)

# packlib

set(packlib-cxx-sources packlib/packc.C packlib/packf77.C)
set(packlib-h-sources packlib/PackLib.h packlib/cpacklib.h)

add_library(packlib ${packlib-cxx-sources} ${packlib-h-sources})

foreach(file ${packlib-h-sources})
    configure_file(${file} ${CMAKE_BINARY_DIR}/include/ COPYONLY)
endforeach()

# master-slave

set(master-slave-cxx-sources master-slave/cms.C)
set(master-slave-h-sources master-slave/cms.h)

foreach(file ${master-slave-h-sources})
    configure_file(${file} ${CMAKE_BINARY_DIR}/include/ COPYONLY)
endforeach()

add_library(cms ${master-slave-cxx-sources} ${master-slave-h-sources})

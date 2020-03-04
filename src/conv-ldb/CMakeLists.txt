set(conv-ldb-c-sources cldb.C cldb.neighbor.C cldb.none.C
    cldb.prioritycentralized.C cldb.rand.C cldb.spray.C cldb.test.C
    cldb.workstealing.C edgelist.C generate.C priorityqueue.C)

set(conv-ldb-cxx-sources topology.C)

set(conv-ldb-h-sources
    cldb.h cldb.neighbor.h cldb.prioritycentralized.h cldb.workstealing.h
    graphdefs.h topology.h)

add_library(conv-ldb topology.C edgelist.C generate.C)

foreach(filename ${conv-ldb-h-sources})
    configure_file(${filename} ${CMAKE_BINARY_DIR}/include/ COPYONLY)
endforeach()

add_library(ldb-none cldb.none.C ${conv-ldb-h-sources})
add_library(ldb-test cldb.test.C ${conv-ldb-h-sources})
add_library(ldb-rand cldb.rand.C cldb.h ${conv-ldb-h-sources})
add_library(ldb-neighbor cldb.neighbor.C cldb.neighbor.h ${conv-ldb-h-sources})
add_library(ldb-workstealing cldb.workstealing.C cldb.workstealing.h ${conv-ldb-h-sources})
add_library(ldb-spray cldb.spray.C ${conv-ldb-h-sources})

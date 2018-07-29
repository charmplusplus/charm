set(conv-ldb-c-sources cldb.c cldb.neighbor.c cldb.none.c
    cldb.prioritycentralized.c cldb.rand.c cldb.spray.c cldb.test.c
    cldb.workstealing.c edgelist.c generate.c priorityqueue.c)

set(conv-ldb-cxx-sources topology.C)

set(conv-ldb-h-sources ../langs/bluegene/blue.h ../langs/bluegene/blue-conv.h
    cldb.h cldb.neighbor.h cldb.prioritycentralized.h cldb.workstealing.h
    graphdefs.h topology.h)

add_library(conv-ldb topology.C edgelist.c generate.c)

foreach(filename ${conv-ldb-h-sources})
    configure_file(${filename} ${CMAKE_BINARY_DIR}/include/ COPYONLY)
endforeach()

add_library(ldb-none cldb.none.c ${conv-ldb-h-sources})
add_library(ldb-test cldb.test.c ${conv-ldb-h-sources})
add_library(ldb-rand cldb.rand.c cldb.h ${conv-ldb-h-sources})
add_library(ldb-neighbor cldb.neighbor.c cldb.neighbor.h ${conv-ldb-h-sources})
add_library(ldb-workstealing cldb.workstealing.c cldb.workstealing.h ${conv-ldb-h-sources})
add_library(ldb-spray cldb.spray.c ${conv-ldb-h-sources})
add_library(ldb-bluegene cldb.bluegene.c ${conv-ldb-h-sources})

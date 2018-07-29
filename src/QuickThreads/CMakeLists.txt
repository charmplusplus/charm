# Find out which version of Quickthreads we are using
execute_process(COMMAND sh -c "CHARMINC=${CMAKE_BINARY_DIR}/include; . ${CMAKE_BINARY_DIR}/include/conv-config.sh; echo $CMK_QT" OUTPUT_VARIABLE CMK_QT OUTPUT_STRIP_TRAILING_WHITESPACE)

if(${CMK_QT} STREQUAL "generic64-light")
	set(ckqt-c-sources md/setjmp64_.c)
elseif(${CMK_QT} STREQUAL "generic")
	set(ckqt-c-sources md/setjmp.c)
elseif(${CMK_QT} STREQUAL "linux-smp-32")
    set(ckqt-c-sources md/setjmp.c)
elseif(${CMK_QT} STREQUAL "generic-mic64")
	set(ckqt-c-sources md/setjmp64.c)
elseif(${CMK_QT} STREQUAL "aix") # BG
	set(ckqt-c-sources md/setjmp.c)
elseif(${CMK_QT} STREQUAL "none") # Win
	message(FATAL_ERROR "Unsupported Quickthreads version \"${CMK_QT}\"")
elseif(${CMK_QT} STREQUAL "i386-gcc")
	message(FATAL_ERROR "Unsupported Quickthreads version \"${CMK_QT}\"")
else()
	message(FATAL_ERROR "Unknown Quickthreads version \"${CMK_QT}\"")
endif()

configure_file(md/setjmp_d.h include/qtmd.h COPYONLY)
configure_file(md/setjmp_d.h ${CMAKE_BINARY_DIR}/include/qtmd.h COPYONLY)
configure_file(qt.h ${CMAKE_BINARY_DIR}/include/ COPYONLY)
set(ckqt-h-sources qt.h include/qtmd.h md/setjmp_d.h)
list(APPEND ckqt-c-sources qt.c)

add_library(ckqt ${ckqt-c-sources} ${ckqt-h-sources})
target_include_directories(ckqt PRIVATE . ${CMAKE_CURRENT_BINARY_DIR}/include)

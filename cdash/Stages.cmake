
if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/PyCTestPreInit.cmake")
    include("${CMAKE_CURRENT_LIST_DIR}/PyCTestPreInit.cmake")
endif(EXISTS "${CMAKE_CURRENT_LIST_DIR}/PyCTestPreInit.cmake")

if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/Init.cmake")

    include("${CMAKE_CURRENT_LIST_DIR}/Init.cmake")
    include_if("${CMAKE_CURRENT_LIST_DIR}/PyCTestPostInit.cmake")

    list(FIND STAGES "Start" DO_START)
    list(FIND STAGES "Update" DO_UPDATE)
    list(FIND STAGES "Configure" DO_CONFIGURE)
    list(FIND STAGES "Build" DO_BUILD)
    list(FIND STAGES "Test" DO_TEST)
    list(FIND STAGES "Coverage" DO_COVERAGE)
    list(FIND STAGES "MemCheck" DO_MEMCHECK)
    list(FIND STAGES "Submit" DO_SUBMIT)

    message(STATUS "")
    message(STATUS "STAGES = ${STAGES}")
    message(STATUS "")

    ctest_read_custom_files("${CMAKE_CURRENT_LIST_DIR}")

    set_if_defined(CTEST_START           START           _CTEST_START)
    set_if_defined(CTEST_END             END             _CTEST_END)
    set_if_defined(CTEST_STRIDE          STRIDE          _CTEST_STRIDE)
    set_if_defined(CTEST_INCLUDE         INCLUDE         _CTEST_INCLUDE)
    set_if_defined(CTEST_EXCLUDE         EXCLUDE         _CTEST_EXCLUDE)
    set_if_defined(CTEST_INCLUDE_LABEL   INCLUDE_LABEL   _CTEST_INCLUDE_LABEL)
    set_if_defined(CTEST_EXCLUDE_LABEL   EXCLUDE_LABEL   _CTEST_EXCLUDE_LABEL)
    set_if_defined(CTEST_PARALLEL_LEVEL  PARALLEL_LEVEL  _CTEST_PARALLEL_LEVEL)
    set_if_defined(CTEST_STOP_TIME       STOP_TIME       _CTEST_STOP_TIME)
    set_if_defined(CTEST_COVERAGE_LABELS LABELS          _CTEST_COVERAGE_LABELS)

    if(CTEST_APPEND)
        set(DO_START -1)
    endif()

    #-------------------------------------------------------------------------#
    # if CTEST_{BZR,CVS,GIT,HG,P4,SVN}_COMMAND was specified instead of
    #   CTEST_UPDATE_COMMAND
    #
    if(NOT "${DO_UPDATE}" GREATER -1)
        foreach(_VC_TYPE BZR CVS GIT HG P4 SVN)
            string(TOLOWER ${_VC_TYPE} _LOWER_VC_TYPE)
            if(DEFINED CTEST_${_VC_TYPE}_COMMAND OR
               CTEST_UPDATE_TYPE STREQUAL "${_LOWER_VC_TYPE}")
                set(DO_UPDATE 0)
            endif()
        endforeach()
    endif()

    #-------------------------------------------------------------------------#
    # Start
    #
    if("${DO_START}" GREATER -1)
        message(STATUS "")
        message(STATUS "[${CTEST_BUILD_NAME}] Running CTEST_START stage...")
        message(STATUS "")
        set(_CTEST_APPEND )
        set(_CTEST_VERB "Running")
    else()
        message(STATUS "")
        message(STATUS "[${CTEST_BUILD_NAME}] Appending CTEST_START stage...")
        message(STATUS "")
        set(_CTEST_APPEND APPEND)
        set(_CTEST_VERB "Appending")
    endif()
    ctest_start(${CTEST_MODEL} TRACK ${CTEST_MODEL} ${_CTEST_APPEND}
        ${CTEST_SOURCE_DIRECTORY} ${CTEST_BINARY_DIRECTORY})

    #-------------------------------------------------------------------------#
    # Config
    #
    copy_ctest_config_files()
    ctest_read_custom_files("${CTEST_BINARY_DIRECTORY}")

    #-------------------------------------------------------------------------#
    # Update
    #
    if("${DO_UPDATE}" GREATER -1 AND (NOT "${CTEST_UPDATE_COMMAND}" STREQUAL "" OR
                                      NOT "${CTEST_UPDATE_TYPE}" STREQUAL ""))
        message(STATUS "")
        message(STATUS "[${CTEST_BUILD_NAME}] ${_CTEST_VERB} CTEST_UPDATE stage...")
        message(STATUS "")
        ctest_update(SOURCE "${CTEST_SOURCE_DIRECTORY}"
                    RETURN_VALUE up_ret)
        # if update failed
        if(up_ret GREATER 0)
            set(DO_BUILD -1)
            set(DO_TEST -1)
            set(DO_COVERAGE -1)
            set(DO_MEMCHECK -1)
        endif()
    else()
        message(STATUS "")
        message(STATUS "[${CTEST_BUILD_NAME}] Skipping CTEST_UPDATE stage...")
        message(STATUS "")
    endif()

    #-------------------------------------------------------------------------#
    # Configure
    #
    if("${DO_CONFIGURE}" GREATER -1 AND NOT "${CTEST_CONFIGURE_COMMAND}" STREQUAL "")
        message(STATUS "")
        message(STATUS "[${CTEST_BUILD_NAME}] ${_CTEST_VERB} CTEST_CONFIGURE stage...")
        message(STATUS "")
        ctest_configure(BUILD "${CTEST_BINARY_DIRECTORY}"
                    SOURCE ${CTEST_SOURCE_DIRECTORY}
                    ${_CTEST_APPEND}
                    OPTIONS "${CTEST_CONFIGURE_OPTIONS}"
                    RETURN_VALUE config_ret)
        if(config_ret GREATER 0)
            set(DO_BUILD -1)
            set(DO_TEST -1)
            set(DO_COVERAGE -1)
            set(DO_MEMCHECK -1)
        endif()
    else()
        message(STATUS "")
        message(STATUS "[${CTEST_BUILD_NAME}] Skipping CTEST_CONFIGURE stage...")
        message(STATUS "")
    endif()

    #-------------------------------------------------------------------------#
    # Build
    #
    if("${DO_BUILD}" GREATER -1 AND NOT "${CTEST_BUILD_COMMAND}" STREQUAL "")
        message(STATUS "")
        message(STATUS "[${CTEST_BUILD_NAME}] ${_CTEST_VERB} CTEST_BUILD stage...")
        message(STATUS "")
        ctest_build(BUILD "${CTEST_BINARY_DIRECTORY}"
                    ${_CTEST_APPEND}
                    RETURN_VALUE build_ret)
        if(build_ret GREATER 0)
            set(DO_TEST -1)
            set(DO_COVERAGE -1)
            set(DO_MEMCHECK -1)
        endif()
    else()
        message(STATUS "")
        message(STATUS "[${CTEST_BUILD_NAME}] Skipping CTEST_BUILD stage...")
        message(STATUS "")
    endif()

    #-------------------------------------------------------------------------#
    # Test
    #
    if("${DO_TEST}" GREATER -1)
        message(STATUS "")
        message(STATUS "[${CTEST_BUILD_NAME}] ${_CTEST_VERB} CTEST_TEST stage...")
        message(STATUS "")
        ctest_test(RETURN_VALUE test_ret
                    ${_CTEST_APPEND}
                    ${_CTEST_START}
                    ${_CTEST_END}
                    ${_CTEST_STRIDE}
                    ${_CTEST_INCLUDE}
                    ${_CTEST_EXCLUDE}
                    ${_CTEST_INCLUDE_LABEL}
                    ${_CTEST_EXCLUDE_LABEL}
                    ${_CTEST_PARALLEL_LEVEL}
                    ${_CTEST_STOP_TIME}
                    SCHEDULE_RANDOM OFF)
    else()
        message(STATUS "")
        message(STATUS "[${CTEST_BUILD_NAME}] Skipping CTEST_TEST stage...")
        message(STATUS "")
    endif()

    #-------------------------------------------------------------------------#
    # Coverage
    #
    if("${DO_COVERAGE}" GREATER -1 AND NOT "${CTEST_COVERAGE_COMMAND}" STREQUAL "")
        message(STATUS "")
        message(STATUS "[${CTEST_BUILD_NAME}] ${_CTEST_VERB} CTEST_COVERAGE stage...")
        message(STATUS "")
        execute_process(COMMAND ${CTEST_COVERAGE_COMMAND} ${CTEST_COVERAGE_EXTRA_FLAGS}
            WORKING_DIRECTORY ${CTEST_BINARY_DIRECTORY}
            ERROR_QUIET)
        ctest_coverage(${_CTEST_APPEND}
                    ${_CTEST_COVERAGE_LABELS}
                    RETURN_VALUE cov_ret)
        # remove the "coverage.xml" file after it has been processed
        # on macOS because the file-system is not case-sensitive
        if(APPLE AND EXISTS "${CTEST_BINARY_DIRECTORY}/coverage.xml")
            execute_process(COMMAND ${CTEST_CMAKE_COMMAND}
                -E remove ${CTEST_BINARY_DIRECTORY}/coverage.xml
                ERROR_QUIET)
        endif()
    else()
        message(STATUS "")
        message(STATUS "[${CTEST_BUILD_NAME}] Skipping CTEST_COVERAGE stage...")
        message(STATUS "")
    endif()

    #-------------------------------------------------------------------------#
    # MemCheck
    #
    if("${DO_MEMCHECK}" GREATER -1 AND NOT "${CTEST_MEMORYCHECK_COMMAND}" STREQUAL "")
        message(STATUS "")
        message(STATUS "[${CTEST_BUILD_NAME}] ${_CTEST_VERB} CTEST_MEMCHECK stage...")
        message(STATUS "")
        ctest_memcheck(RETURN_VALUE mem_ret
                    ${_CTEST_APPEND}
                    ${_CTEST_START}
                    ${_CTEST_END}
                    ${_CTEST_STRIDE}
                    ${_CTEST_INCLUDE}
                    ${_CTEST_EXCLUDE}
                    ${_CTEST_INCLUDE_LABEL}
                    ${_CTEST_EXCLUDE_LABEL}
                    ${_CTEST_PARALLEL_LEVEL})
    else()
        message(STATUS "")
        message(STATUS "[${CTEST_BUILD_NAME}] Skipping CTEST_MEMCHECK stage...")
        message(STATUS "")
    endif()

    #-------------------------------------------------------------------------#
    # Submit
    #
    if("${DO_SUBMIT}" GREATER -1)
        message(STATUS "")
        message(STATUS "[${CTEST_BUILD_NAME}] Running CTEST_SUBMIT stage...")
        message(STATUS "")
        submit_command()
    else()
        message(STATUS "")
        message(STATUS "[${CTEST_BUILD_NAME}] Skipping CTEST_SUBMIT stage...")
        message(STATUS "")
    endif()

    message(STATUS "")
    message(STATUS "[${CTEST_BUILD_NAME}] Finished ${CTEST_MODEL} Stages (${STAGES})")
    message(STATUS "")

endif(EXISTS "${CMAKE_CURRENT_LIST_DIR}/Init.cmake")

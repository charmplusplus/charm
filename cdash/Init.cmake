
# include guard
if(__pyctestinit_is_loaded)
    return()
endif()
set(__pyctestinit_is_loaded ON)

include(ProcessorCount)
ProcessorCount(CTEST_PROCESSOR_COUNT)

cmake_policy(SET CMP0009 NEW)
cmake_policy(SET CMP0011 NEW)

# ---------------------------------------------------------------------------- #
# -- Commands
# ---------------------------------------------------------------------------- #
find_program(CTEST_CMAKE_COMMAND    NAMES cmake)
find_program(CTEST_UNAME_COMMAND    NAMES uname)

find_program(CTEST_BZR_COMMAND      NAMES bzr)
find_program(CTEST_CVS_COMMAND      NAMES cvs)
find_program(CTEST_GIT_COMMAND      NAMES git)
find_program(CTEST_HG_COMMAND       NAMES hg)
find_program(CTEST_P4_COMMAND       NAMES p4)
find_program(CTEST_SVN_COMMAND      NAMES svn)

find_program(VALGRIND_COMMAND       NAMES valgrind)
find_program(GCOV_COMMAND           NAMES gcov)
find_program(LCOV_COMMAND           NAMES llvm-cov)
find_program(MEMORYCHECK_COMMAND    NAMES valgrind )

set(MEMORYCHECK_TYPE Valgrind)
# set(MEMORYCHECK_TYPE Purify)
# set(MEMORYCHECK_TYPE BoundsChecker)
# set(MEMORYCHECK_TYPE ThreadSanitizer)
# set(MEMORYCHECK_TYPE AddressSanitizer)
# set(MEMORYCHECK_TYPE LeakSanitizer)
# set(MEMORYCHECK_TYPE MemorySanitizer)
# set(MEMORYCHECK_TYPE UndefinedBehaviorSanitizer)
set(MEMORYCHECK_COMMAND_OPTIONS "--trace-children=yes --leak-check=full")

# ---------------------------------------------------------------------------- #
# -- Include file if exists
# ---------------------------------------------------------------------------- #
macro(include_if _TESTFILE)
    if(EXISTS "${_TESTFILE}")
        include("${_TESTFILE}")
    else(EXISTS "${_TESTFILE}")
        if(NOT "${ARGN}" STREQUAL "")
            include("${ARGN}")
        endif(NOT "${ARGN}" STREQUAL "")
    endif(EXISTS "${_TESTFILE}")
endmacro(include_if _TESTFILE)


# ---------------------------------------------------------------------------- #
# -- Settings
# ---------------------------------------------------------------------------- #
## -- Process timeout in seconds
set(CTEST_TIMEOUT           "7200")
## -- Set output to English
set(ENV{LC_MESSAGES}        "en_EN" )


# ---------------------------------------------------------------------------- #
# -- Set if defined
# ---------------------------------------------------------------------------- #
macro(SET_IF_DEFINED IF_DEF_VAR PREFIX_VAR SET_VAR)
    if(DEFINED "${IF_DEF_VAR}")
        set(${SET_VAR} "${PREFIX_VAR} \"${${IF_DEF_VAR}}\"")
    endif()
endmacro()


# ---------------------------------------------------------------------------- #
# -- Clean/Reset
# ---------------------------------------------------------------------------- #
macro(CLEAN_DIRECTORIES)
    message(STATUS "Cleaning ${CTEST_BINARY_DIRECTORY}...")
    execute_process(
        COMMAND ${CTEST_CMAKE_COMMAND} -E remove_directory ${CTEST_BINARY_DIRECTORY}
        WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR})
endmacro(CLEAN_DIRECTORIES)


# ---------------------------------------------------------------------------- #
# -- Copy ctest configuration file
# ---------------------------------------------------------------------------- #
macro(COPY_CTEST_CONFIG_FILES)
    if(NOT "${CMAKE_CURRENT_LIST_DIR}" STREQUAL "${CTEST_BINARY_DIRECTORY}" AND
       NOT "${CTEST_SOURCE_DIRECTORY}" STREQUAL "${CTEST_BINARY_DIRECTORY}")
        ## -- CTest Config
        configure_file(${CMAKE_CURRENT_LIST_DIR}/CTestConfig.cmake
                       ${CTEST_BINARY_DIRECTORY}/CTestConfig.cmake COPYONLY)
        ## -- CTest Custom
        configure_file(${CMAKE_CURRENT_LIST_DIR}/CTestCustom.cmake
                       ${CTEST_BINARY_DIRECTORY}/CTestCustom.cmake COPYONLY)
    endif(NOT "${CMAKE_CURRENT_LIST_DIR}" STREQUAL "${CTEST_BINARY_DIRECTORY}" AND
          NOT "${CTEST_SOURCE_DIRECTORY}" STREQUAL "${CTEST_BINARY_DIRECTORY}")
endmacro(COPY_CTEST_CONFIG_FILES)


# ---------------------------------------------------------------------------- #
# -- Run submit scripts
# ---------------------------------------------------------------------------- #
macro(READ_PRESUBMIT_SCRIPTS)
    ## check
    file(GLOB_RECURSE PRESUBMIT_SCRIPTS "${CTEST_BINARY_DIRECTORY}/*CTestPreSubmitScript.cmake")
    if("${PRESUBMIT_SCRIPTS}" STREQUAL "")
        message(STATUS "No scripts matching '${CTEST_BINARY_DIRECTORY}/*CTestPreSubmitScript.cmake' were found")
    endif()
    foreach(_FILE ${PRESUBMIT_SCRIPTS})
        message(STATUS "Including pre-submit script: \"${_FILE}\"...")
        include("${_FILE}")
    endforeach(_FILE ${PRESUBMIT_SCRIPTS})
endmacro(READ_PRESUBMIT_SCRIPTS)


# ---------------------------------------------------------------------------- #
# -- Read CTestNotes.cmake file
# ---------------------------------------------------------------------------- #
macro(READ_NOTES)
    ## check
    file(GLOB_RECURSE NOTE_FILES "${CTEST_BINARY_DIRECTORY}/*CTestNotes.cmake")
    foreach(_FILE ${NOTE_FILES})
        message(STATUS "Including CTest notes files: \"${_FILE}\"...")
        include("${_FILE}")
    endforeach(_FILE ${NOTE_FILES})
endmacro(READ_NOTES)


# ---------------------------------------------------------------------------- #
# -- Check to see if there is a ctest token (for CDash submission)
# ---------------------------------------------------------------------------- #
macro(CHECK_FOR_CTEST_TOKEN)

    # set using token to off
    set(CTEST_USE_TOKEN OFF)
    # set token to empty
    set(CTEST_TOKEN     "")

    if(NOT "${CTEST_TOKEN_FILE}" STREQUAL "")
        string(REGEX REPLACE "^~" "$ENV{HOME}" CTEST_TOKEN_FILE "${CTEST_TOKEN_FILE}")
    endif(NOT "${CTEST_TOKEN_FILE}" STREQUAL "")

    # check for a file containing token
    if(NOT "${CTEST_TOKEN_FILE}" STREQUAL "" AND EXISTS "${CTEST_TOKEN_FILE}")
        message(STATUS "Reading CTest token file: ${CTEST_TOKEN_FILE}")
        file(READ "${CTEST_TOKEN_FILE}" CTEST_TOKEN)
        string(REPLACE "\n" "" CTEST_TOKEN "${CTEST_TOKEN}")
    endif(NOT "${CTEST_TOKEN_FILE}" STREQUAL "" AND EXISTS "${CTEST_TOKEN_FILE}")

    # if no file, check the environment
    if("${CTEST_TOKEN}" STREQUAL "" AND NOT "$ENV{CTEST_TOKEN}" STREQUAL "")
        set(CTEST_TOKEN "$ENV{CTEST_TOKEN}")
    endif("${CTEST_TOKEN}" STREQUAL "" AND NOT "$ENV{CTEST_TOKEN}" STREQUAL "")

    # if non-empty token, set CTEST_USE_TOKEN to ON
    if(NOT "${CTEST_TOKEN}" STREQUAL "")
        set(CTEST_USE_TOKEN ON)
    endif(NOT "${CTEST_TOKEN}" STREQUAL "")

endmacro(CHECK_FOR_CTEST_TOKEN)


# ---------------------------------------------------------------------------- #
# -- Submit command
# ---------------------------------------------------------------------------- #
macro(SUBMIT_COMMAND)

    # check for token
    check_for_ctest_token()
    read_notes()
    read_presubmit_scripts()

    if(NOT CTEST_USE_TOKEN)
        # standard submit
        ctest_submit(RETURN_VALUE ret ${ARGN})
    else(NOT CTEST_USE_TOKEN)
        # submit with token
        ctest_submit(RETURN_VALUE ret
                     HTTPHEADER "Authorization: Bearer ${CTEST_TOKEN}"
                     ${ARGN})
    endif(NOT CTEST_USE_TOKEN)

endmacro(SUBMIT_COMMAND)

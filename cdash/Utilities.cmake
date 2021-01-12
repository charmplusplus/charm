# - Include guard
if(__pyctest_utilities_isloaded)
  return()
endif()
set(__pyctest_utilities_isloaded YES)

include(CMakeParseArguments)


################################################################################
# macro usage:
#   set a variable to a default value if not set
#
macro(set_if_empty VAR VAL)
    if("${${VAR}}" STREQUAL "")
        set(${VAR} "${VAL}")
    endif()
endmacro()


################################################################################
# macro usage:
#
#   download_conda([ANACONDA]
#       PYTHON_VERSION <value>  (default: 2.7)
#       INSTALL_PREFIX <value>  (default: ${CMAKE_CURRENT_LIST_DIR}/{Anaconda,Miniconda})
#       ARCH <value>            (default: x86_64)
#       VERSION <value>         (default: {5.0.0,latest})
#       URL_PREFIX <value>      (default: https://repo.continuum.io/{archive,miniconda})
#       DOWNLOAD_DIR <value>    (default: ${CMAKE_CURRENT_LIST_DIR})
#
#   Unless ANACONDA option is specified, Miniconda is downloaded
#   If ANACONDA option is used it is recommended to specify VERSION since
#   there is not a "latest" tag in https://repo.continuum.io/archive/
#
#   PYTHON_VESION is used to determine whether to download Miniconda2, Anaconda3, etc.
#
#   ARCH is used to specify x86_64, x86, ppc641e, etc.
#
#
macro(DOWNLOAD_CONDA)
    set(_options            "ANACONDA")
    set(_one_val_keywords   PYTHON_VERSION
                            INSTALL_PREFIX
                            ARCH
                            VERSION
                            URL_PREFIX
                            DOWNLOAD_DIR)
    set(_mult_val_keywords  "")

    cmake_parse_arguments(CONDA         # prefix
        "${_options}"                   # options
        "${_one_val_keywords}"          # one_value_keywords
        "${_mult_val_keywords}"         # multiple_value_keywords
        ${ARGN})

    if(CONDA_ANACONDA)
        set(CONDA_NAME "Anaconda")
        set(CONDA_URL_FOLDER "archive")
        set_if_empty(CONDA_VERSION "5.0.0")
    else()
        set(CONDA_MINICONDA ON)
        set(CONDA_NAME "Miniconda")
        set(CONDA_URL_FOLDER "miniconda")
        set_if_empty(CONDA_VERSION "latest")
    endif()

    set_if_empty(CONDA_PYTHON_VERSION "2.7")
    set_if_empty(CONDA_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/${CONDA_NAME}")
    set_if_empty(CONDA_ARCH "x86_64")
    set_if_empty(CONDA_URL_PREFIX "https://repo.continuum.io/${CONDA_URL_FOLDER}")
    set_if_empty(CONDA_DOWNLOAD_DIR "${CMAKE_CURRENT_LIST_DIR}")

    string(REPLACE "." ";" CONDA_PYTHON_MAJOR_VERSION "${CONDA_PYTHON_VERSION}")
    list(GET CONDA_PYTHON_MAJOR_VERSION 0 CONDA_PYTHON_MAJOR_VERSION)
    set(CONDA_PYVERSION "${CONDA_NAME}${CONDA_PYTHON_MAJOR_VERSION}")
    set(INSTALLER_EXTENSION "sh")

    if(APPLE)
        set(CONDA_OS "MacOSX")
    elseif(WIN32)
        set(CONDA_OS "Windows")
        set(INSTALLER_EXTENSION "exe")
    else()
        set(CONDA_OS "Linux")
    endif(APPLE)

    set(CONDA_ARCHIVE "${CONDA_PYVERSION}-${CONDA_VERSION}-${CONDA_OS}-${CONDA_ARCH}")
    set(CONDA_URL "${CONDA_URL_PREFIX}/${CONDA_ARCHIVE}.${INSTALLER_EXTENSION}")

    file(DOWNLOAD
         "${CONDA_URL}"
         "${CONDA_DOWNLOAD_DIR}/${CONDA_NAME}.${INSTALLER_EXTENSION}")

    if(WIN32)
        execute_process(COMMAND
            start /wait "" ${CONDA_DOWNLOAD_DIR}/${CONDA_NAME}.${INSTALLER_EXTENSION}
            /InstallationType=JustMe /RegisterPython=0 /S /D=${CONDA_INSTALL_PREFIX})
    else(WIN32)
        find_program(BASH_EXE NAMES bash)
        execute_process(COMMAND
            ${BASH_EXE} ${CONDA_DOWNLOAD_DIR}/${CONDA_NAME}.${INSTALLER_EXTENSION}
            -b -p ${CONDA_INSTALL_PREFIX})
    endif(WIN32)

endmacro()


################################################################################
# macro usage:
#
#   find_conda(CONDA_PREFIX <value> [<other-search-paths-if-needed>])
#
macro(FIND_CONDA CONDA_PREFIX CONDA_ENVIRONMENT)

    set_if_empty(CONDA_PREFIX "${CMAKE_CURRENT_LIST_DIR}/Miniconda")

    # search paths
    set(ENV{PATH} "${CONDA_PREFIX}/envs/${CONDA_ENVIRONMENT}/bin:${CONDA_PREFIX}/bin:$ENV{PATH}")
    set(SEARCH_PATHS
        ${CONDA_PREFIX}/envs/${CONDA_ENVIRONMENT}
        ${CONDA_PREFIX}
        ${ARGN}
    )

    #--------------------------------------------------------------------------#
    #   local `conda`
    #--------------------------------------------------------------------------#
    find_program(CONDA_EXE
        NAMES           conda
        PATHS           ${SEARCH_PATHS}
        HINTS           ${SEARCH_PATHS}
        PATH_SUFFIXES   bin Scripts Library/bin
        NO_DEFAULT_PATH)

    if(NOT CONDA_EXE OR NOT EXISTS "${CONDA_EXE}")
        message(FATAL_ERROR "Error! Could not find 'conda' command in \"${SEARCH_PATHS}\"")
    endif()

    get_filename_component(CONDA_DIR "${CONDA_EXE}" PATH)
    set(SEARCH_PATHS ${SEARCH_PATHS} ${CONDA_DIR})

    #--------------------------------------------------------------------------#
    #   local `python`
    #--------------------------------------------------------------------------#
    find_program(PYTHON_EXE
        NAMES           python
        PATHS           ${SEARCH_PATHS}
        HINTS           ${SEARCH_PATHS}
        PATH_SUFFIXES   bin Scripts Library/bin
        NO_DEFAULT_PATH)

    if(NOT PYTHON_EXE OR NOT EXISTS "${PYTHON_EXE}")
        message(FATAL_ERROR "Error! Could not find 'python' command in \"${SEARCH_PATHS}\"")
    endif()

    set(ENV{CONDA_EXE}          ${CONDA_EXE})
    set(ENV{CONDA_DEFAULT_ENV}  ${CONDA_ENVIRONMENT})
    set(ENV{CONDA_PREFIX}       ${CONDA_PREFIX})
    set(ENV{CONDA_PYTHON_EXE}   ${PYTHON_EXE})
    set(ENV{PYTHON_ROOT}        ${CONDA_PREFIX})

    message(STATUS "")
    message(STATUS "Using conda: '${CONDA_EXE}' and python: '${PYTHON_EXE}'")
    message(STATUS "")
    execute_process(COMMAND ${CONDA_EXE} info -a)
    message(STATUS "")
    message(STATUS "Using conda: '${CONDA_EXE}' and python: '${PYTHON_EXE}'")
    message(STATUS "")

endmacro()


################################################################################
# macro usage:
#
#   configure_conda([CHANGE_PS1] [UPLOAD] [NO_PIN] [UPDATE]
#           PREFIX <value>
#           ENVIRONMENT <value>                 (default: base)
#           PYTHON_VERSION <value>              (default: 2.7)
#           PACKAGES
#
#   CHANGE_PS1 option will run: "conda config --set changeps1 no"
#   UPLOAD option will run: "conda config --set anaconda_upload yes"
#   PACKAGES will run:
#       conda install -n ${ENVIRONMENT} python=${PYTHON_VERSION} ${PACKAGES}
#
#   NOTE: if not installing to "root" or "base" environment, be aware that
#       command execution runs in a subprocess and environment changes are
#       from "activate" might not possibly be propagated
#
macro(CONFIGURE_CONDA)

    set(_options            CHANGE_PS1
                            UPLOAD
                            NO_PIN
                            UPDATE)
    set(_one_val_keywords   PREFIX
                            PYTHON_VERSION
                            ENVIRONMENT)
    set(_mult_val_keywords  PACKAGES CHANNELS)

    cmake_parse_arguments(CONDA         # prefix
        "${_options}"                   # options
        "${_one_val_keywords}"          # one_value_keywords
        "${_mult_val_keywords}"         # multiple_value_keywords
        ${ARGN})

    # defaults
    set_if_empty(CONDA_PREFIX "${CMAKE_CURRENT_LIST_DIR}/Miniconda")
    set_if_empty(CONDA_ENVIRONMENT "pyctest-unknown-project")
    set_if_empty(CONDA_PYTHON_VERSION "2.7")
    list(APPEND CONDA_CHANNELS "conda-forge" "jrmadsen")
    list(REMOVE_DUPLICATES CONDA_CHANNELS)

    # get CONDA_EXE
    find_conda(${CONDA_PREFIX} ${CONDA_ENVIRONMENT})

    # add channels
    set(CHANNEL_ARGS )
    foreach(_CHANNEL ${CONDA_CHANNELS})
        list(APPEND CHANNEL_ARGS -c ${_CHANNEL})
    endforeach()

    # changeps1
    if(CONDA_CHANGE_PS1)
        execute_process(COMMAND ${CONDA_EXE} config --set changeps1 yes)
    endif()

    # update conda
    if(CONDA_UPDATE)
        execute_process(COMMAND ${CONDA_EXE} update -y -n ${CONDA_ENVIRONMENT} conda
            ERROR_QUIET)
    endif()

    set(NO_PIN_OPT "")
    if(CONDA_NO_PIN)
        set(NO_PIN_OPT "--no-pin")
    endif()

    # remove message
    message(STATUS "")
    message(STATUS "Running: '${CONDA_EXE} remove -y -n ${CONDA_ENVIRONMENT} --all'...")
    message(STATUS "")

    # remove existing environment
    execute_process(COMMAND ${CONDA_EXE} remove -y -n ${CONDA_ENVIRONMENT} --all
        ERROR_QUIET)

    # install message
    message(STATUS "")
    message(STATUS "Running: '${CONDA_EXE} create -y ${CHANNEL_ARGS} -n ${CONDA_ENVIRONMENT} python=${CONDA_PYTHON_VERSION} ${CONDA_PACKAGES} ${NO_PIN_OPT}'...")
    message(STATUS "")

    # install packages
    execute_process(COMMAND ${CONDA_EXE} create -y ${CHANNEL_ARGS} -n ${CONDA_ENVIRONMENT}
        python=${CONDA_PYTHON_VERSION} ${CONDA_PACKAGES} ${NO_PIN_OPT})

endmacro()



macro(CHARMC_TARGET Name CharmInput CharmOutputDirectory CharmModules CharmTarget)
    set(CHARMC_TARGET_modules ${${CharmModules}})

    set(CHARMC_TARGET_output )

    foreach(module ${CHARMC_TARGET_modules})
      set(MODULE_DECL "${module}.decl.h")
      set(MODULE_DEF  "${module}.def.h")
      list(
        APPEND
        CHARMC_TARGET_output
        "${CharmOutputDirectory}/${MODULE_DECL}"
      )
      list(APPEND
        CHARMC_TARGET_output
        "${CharmOutputDirectory}/${MODULE_DEF}"
      )
    endforeach(module)

    set(${CharmTarget} ${CHARMC_TARGET_output})

    set(
      BuildCommand
      "[charmc] Compiling Charm++ Interface file: ${CharmInput}, Modules=[${${CharmModules}}]"
    )

    # DEPENDS on "Charm++" assumes that the top-level CMake wrapper for Charm++ is used via add_subdirectory
    add_custom_command(
      OUTPUT ${CHARMC_TARGET_output}
      COMMAND ${CHARMC_EXECUTABLE} ${CharmInput}
      VERBATIM
      DEPENDS ${CharmInput} Charm++
      COMMENT ${BuildCommand}
      WORKING_DIRECTORY ${CharmOutputDirectory}
    )
endmacro()

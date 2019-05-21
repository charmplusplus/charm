#!/bin/bash
# This script can be run on a compiled file (.o, .so, .a, linked ELF
# binary) to produce a list of symbols representing global variables
# potentially of concern for AMPI from a privatization stand-point

if [[ $# -lt 1 ]]; then
    echo "$(basename $0): Display global writable variables in object files, libraries, and executables"
    echo "Usage: $0 <files>"
    exit 2
fi

symlist=$(nm --format posix --print-file-name --demangle $* | egrep ' [BDGS] ')

if [[ -z $symlist ]]; then
    echo "No global writable variables in '$*' found."
    exit 0
else
    echo "The following global writable variables in '$*' were found:"
    echo
    out="File\tType\tName\n"

    while read -r line; do
        loc=${line%% *}
        loc=${loc##.*[}
        loc=${loc%%]:}
        type=$(echo $line | awk '{print $(NF-2)}')
        name=$(echo $line | awk '{$1 = $NF = $(NF-1) = $(NF-2) = ""; print}')

        # Filter out things we don't need
        [[
               "$name" =~ ^.typeinfo.name.for
            || "$name" =~ ^.typeinfo.for
            || "$name" =~ ^.vtable.for
            || "$name" =~ ^.construction.vtable.for
            || "$name" =~ ^.VTT.for
            || "$name" =~ ^._?Csv_ # Already privatized
        ]] && continue

        out+="$loc\t$type\t$name\n"
    done <<< "$symlist"

    # Print table with variables
    printf "$out" | column -t -s $'\t'

    echo
    echo "Legend:"
    echo "  B - The symbol is in the uninitialized data section (BSS)."
    echo "  D - The symbol is in the initialized data section."
    echo "  G - The symbol is in the initialized data section for small objects."
    echo "  S - The symbol is in the uninitialized data section for small objects."

    echo
    echo "To support virtualization and migration, AMPI requires that an application"
    echo "uses no global variables, or that they are read-only. Please see the section on"
    echo "\"Global Variable Privatization\" in the AMPI manual for further information."
    exit 1
fi

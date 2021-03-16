#!/bin/bash

# a default set of environment variables for Visual Studio under Cygwin or MSYS2


# VS 2015
# VCVER="14.0"
# SDKVER="10.0.15063.0"

# TOOLARCH="amd64"
# SDKARCH="x64"

# export VCINSTALLDIR="C:\\Program Files (x86)\\Microsoft Visual Studio $VCVER\\VC"
# VSBIN="$(cygpath -u "$VCINSTALLDIR\\BIN\\$TOOLARCH")"


# VS 2017
# VSYEAR="2017"
# VCVER="15.0"
# SDKVER="10.0.17763.0"
# TOOLVER=""

# VS 2019
VSYEAR="2019"
VCVER="16.0"
SDKVER="10.0.19041.0"
TOOLVER="14.28.29910"

TOOLARCH="x64"
SDKARCH="x64"
EDITION="Enterprise"

export VCINSTALLDIR="C:\\Program Files (x86)\\Microsoft Visual Studio\\$VSYEAR\\$EDITION\\VC\\Tools\\MSVC\\$TOOLVER"
VSBIN="$(cygpath -u "$VCINSTALLDIR\\bin\\Hostx64\\$TOOLARCH")"


export WindowsSdkDir="C:\\Program Files (x86)\\Windows Kits\\10"
export PATH="$VSBIN:$PATH"

INCLUDE=""
INCLUDE="$VCINSTALLDIR\\INCLUDE;$INCLUDE"
INCLUDE="$VCINSTALLDIR\\ATLMFC\\INCLUDE;$INCLUDE"
INCLUDE="$WindowsSdkDir\\include\\$SDKVER\\ucrt;$INCLUDE"
INCLUDE="$WindowsSdkDir\\include\\$SDKVER\\shared;$INCLUDE"
INCLUDE="$WindowsSdkDir\\include\\$SDKVER\\um;$INCLUDE"
INCLUDE="$WindowsSdkDir\\include\\$SDKVER\\winrt;$INCLUDE"
export INCLUDE

LIB=""
LIB="$VCINSTALLDIR\\LIB\\$TOOLARCH;$LIB"
LIB="$VCINSTALLDIR\\ATLMFC\\LIB\\$TOOLARCH;$LIB"
LIB="$WindowsSdkDir\\lib\\$SDKVER\\ucrt\\$SDKARCH;$LIB"
LIB="$WindowsSdkDir\\lib\\$SDKVER\\um\\$SDKARCH;$LIB"
export LIB

LIBPATH=""
LIBPATH="$WindowsSdkDir\\References;$LIBPATH"
LIBPATH="$WindowsSdkDir\\UnionMetadata;$LIBPATH"
LIBPATH="$VCINSTALLDIR\\ATLMFC\\LIB\\$TOOLARCH;$LIBPATH"
LIBPATH="$VCINSTALLDIR\\LIB\\$TOOLARCH;$LIBPATH"
export LIBPATH

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
# SDKVER="10.0.17763.0"
# TOOLVER=""

# VS 2019
VSYEAR="2019"
SDKVER="10.0.19041.0"
TOOLVER="14.28.29910"

# Common to 2017 and 2019
SDKARCH="x64"
TOOLARCH="x64"
EDITION="Enterprise"

# Identify how to call vswhere if it exists
VSWHERE="vswhere"
if ! command -v "$VSWHERE" &> /dev/null
then
  VSWHERE="$(cygpath -u "C:\\Program Files (x86)\\Microsoft Visual Studio\\Installer\\vswhere.exe")"
fi

# If vswhere exists, use that to get up to date info, otherwise fall back on the hardcoded values
if command -v "$VSWHERE" &> /dev/null
then
  VSROOTINSTALLDIR="$("$VSWHERE" -latest -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath | tr -d '\r')"
  # Get tool version from this file if it exists
  TOOLVERFILE="$(cygpath -u "$VSROOTINSTALLDIR\\VC\\Auxiliary\\Build\\Microsoft.VCToolsVersion.default.txt")"
  [[ -f "$TOOLVERFILE" ]] && TOOLVER="$(tr -d '\r' < "$TOOLVERFILE")"

  export VCINSTALLDIR="$VSROOTINSTALLDIR\\VC\\Tools\\MSVC\\$TOOLVER"
else
  export VCINSTALLDIR="C:\\Program Files (x86)\\Microsoft Visual Studio\\$VSYEAR\\$EDITION\\VC\\Tools\\MSVC\\$TOOLVER"
fi

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

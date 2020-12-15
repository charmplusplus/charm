#!/bin/bash

# a default set of environment variables for Visual Studio under Cygwin or MSYS2

# VS 2015
# VCVER="14.0"
# SDKVER="10.0.15063.0"
# export VCINSTALLDIR="C:\Program Files (x86)\Microsoft Visual Studio $VCVER\VC"

# VS 2017
# VSYEAR="2017"
# VCVER="15.0"
# SDKVER="10.0.17763.0"

# VS 2019
VSYEAR="2019"
VCVER="16.0"
SDKVER="10.0.19030.0"

export VCINSTALLDIR="C:\Program Files (x86)\Microsoft Visual Studio $VSYEAR\Enterprise\VC"
export WindowsSdkDir="C:\Program Files (x86)\Windows Kits\10"

# debug
find "$(cygpath -u "C:\Program Files (x86)\Microsoft Visual Studio $VSYEAR")"
find "$(cygpath -u "$WindowsSdkDir")"

INCLUDE=""
INCLUDE="$VCINSTALLDIR\INCLUDE;$INCLUDE"
INCLUDE="$VCINSTALLDIR\ATLMFC\INCLUDE;$INCLUDE"
INCLUDE="$WindowsSdkDir\include\$SDKVER\ucrt;$INCLUDE"
INCLUDE="$WindowsSdkDir\include\$SDKVER\shared;$INCLUDE"
INCLUDE="$WindowsSdkDir\include\$SDKVER\um;$INCLUDE"
INCLUDE="$WindowsSdkDir\include\$SDKVER\winrt;$INCLUDE"
export INCLUDE

LIB=""
LIB="$VCINSTALLDIR\LIB\amd64;$LIB"
LIB="$VCINSTALLDIR\ATLMFC\LIB\amd64;$LIB"
LIB="$WindowsSdkDir\lib\$SDKVER\ucrt\x64;$LIB"
LIB="$WindowsSdkDir\lib\$SDKVER\um\x64;$LIB"
export LIB

LIBPATH=""
LIBPATH="$WindowsSdkDir\References;$LIBPATH"
LIBPATH="$WindowsSdkDir\UnionMetadata;$LIBPATH"
LIBPATH="$VCINSTALLDIR\ATLMFC\LIB\amd64;$LIBPATH"
LIBPATH="$VCINSTALLDIR\LIB\amd64;$LIBPATH"
export LIBPATH

VSBIN="$(cygpath -u "$VCINSTALLDIR\BIN\amd64")"
export PATH="$VSBIN:$PATH"

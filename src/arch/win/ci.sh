#!/bin/bash

# a default set of environment variables for Visual Studio under Cygwin or MSYS2

# VS 2015
# VCVER="14.0"
# SDKVER="10.0.15063.0"

# VS 2019
VCVER="16.0"
SDKVER="10.0.19030.0"

INCLUDE=""
INCLUDE="C:\Program Files (x86)\Microsoft Visual Studio $VCVER\VC\INCLUDE;$INCLUDE"
INCLUDE="C:\Program Files (x86)\Microsoft Visual Studio $VCVER\VC\ATLMFC\INCLUDE;$INCLUDE"
INCLUDE="C:\Program Files (x86)\Windows Kits\10\include\$SDKVER\ucrt;$INCLUDE"
INCLUDE="C:\Program Files (x86)\Windows Kits\10\include\$SDKVER\shared;$INCLUDE"
INCLUDE="C:\Program Files (x86)\Windows Kits\10\include\$SDKVER\um;$INCLUDE"
INCLUDE="C:\Program Files (x86)\Windows Kits\10\include\$SDKVER\winrt;$INCLUDE"
export INCLUDE

LIB=""
LIB="C:\Program Files (x86)\Microsoft Visual Studio $VCVER\VC\LIB\amd64;$LIB"
LIB="C:\Program Files (x86)\Microsoft Visual Studio $VCVER\VC\ATLMFC\LIB\amd64;$LIB"
LIB="C:\Program Files (x86)\Windows Kits\10\lib\$SDKVER\ucrt\x64;$LIB"
LIB="C:\Program Files (x86)\Windows Kits\10\lib\$SDKVER\um\x64;$LIB"
export LIB

LIBPATH=""
LIBPATH="\Microsoft.VCLibs\$VCVER\References\CommonConfiguration\neutral;$LIBPATH"
LIBPATH="C:\Program Files (x86)\Windows Kits\10\References;$LIBPATH"
LIBPATH="C:\Program Files (x86)\Windows Kits\10\UnionMetadata;$LIBPATH"
LIBPATH="C:\Program Files (x86)\Microsoft Visual Studio $VCVER\VC\ATLMFC\LIB\amd64;$LIBPATH"
LIBPATH="C:\Program Files (x86)\Microsoft Visual Studio $VCVER\VC\LIB\amd64;$LIBPATH"
export LIBPATH

export VCINSTALLDIR="C:\Program Files (x86)\Microsoft Visual Studio $VCVER\VC"
export WindowsSdkDir="C:\Program Files (x86)\Windows Kits\10"

VSPATH=$(cygpath -u "C:\Program Files (x86)\Microsoft Visual Studio $VCVER\VC\BIN\amd64")
export PATH="$VSPATH:$PATH"

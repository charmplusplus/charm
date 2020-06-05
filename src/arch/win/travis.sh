#!/bin/bash

set -o errexit

# Script to set environment variables for Travis Windows VMs

export INCLUDE=""
export INCLUDE="C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\INCLUDE;$INCLUDE"
export INCLUDE="C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\ATLMFC\INCLUDE;$INCLUDE"
export INCLUDE="C:\Program Files (x86)\Windows Kits\10\include\10.0.15063.0\ucrt;$INCLUDE"
export INCLUDE="C:\Program Files (x86)\Windows Kits\10\include\10.0.15063.0\shared;$INCLUDE"
export INCLUDE="C:\Program Files (x86)\Windows Kits\10\include\10.0.15063.0\um;$INCLUDE"
export INCLUDE="C:\Program Files (x86)\Windows Kits\10\include\10.0.15063.0\winrt;$INCLUDE"

export LIB=""
export LIB="C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\LIB\amd64;$LIB"
export LIB="C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\ATLMFC\LIB\amd64;$LIB"
export LIB="C:\Program Files (x86)\Windows Kits\10\lib\10.0.15063.0\ucrt\x64;$LIB"
export LIB="C:\Program Files (x86)\Windows Kits\10\lib\10.0.15063.0\um\x64;$LIB"

export LIBPATH=""
export LIBPATH="\Microsoft.VCLibs\14.0\References\CommonConfiguration\neutral;$LIBPATH"
export LIBPATH="C:\Program Files (x86)\Windows Kits\10\References;$LIBPATH"
export LIBPATH="C:\Program Files (x86)\Windows Kits\10\UnionMetadata;$LIBPATH"
export LIBPATH="C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\ATLMFC\LIB\amd64;$LIBPATH"
export LIBPATH="C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\LIB\amd64;$LIBPATH"
export LIBPATH="C:\Windows\Microsoft.NET\Framework64\v4.0.30319;$LIBPATH"

export VCINSTALLDIR="C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC"
export WindowsSdkDir="C:\Program Files (x86)\Windows Kits\10"

export PATH="/cygdrive/c/Program Files (x86)/Microsoft Visual Studio 14.0/VC/BIN/amd64:$PATH"

echo $PATH

ls /cygdrive/c

"/cygdrive/c/Program Files (x86)/Microsoft Visual Studio 14.0/VC/BIN/amd64/cl.exe"

cl.exe

export TESTOPTS="++local"

"$@"

@echo off
goto begin

:syntax
  echo.
  echo Usage: NT-INSTALL (target) (version) [charmc-options ...]
  echo.
  echo targets: converse, charm, charm++
  echo.
  echo versions:  ( cd src ; dir *-* )
  echo.
  echo example charmc-options: -g -save -verbose'
  echo.
  echo.
  echo Note: This script is trivial.  It
  echo.
  echo  1. Creates directories (version) and (version)/tmp
  echo  2. Copies src/Common/scripts/Makefile into (version)/tmp
  echo  3. Does a "nmake (target) (version) OPTS=(charmc-options)" in (version)/tmp.
  echo.
  echo That's all NT-INSTALL does.  The rest is handled by the Makefile.
  echo.
  goto done

:begin
  
  if "%1"=="" goto syntax
  set program=%1
  shift
  if "%1"=="" goto syntax 
  set version=%1

  echo Creating directories: %version% and %version%\tmp
  if not exist net-winnt\nul mkdir %version%
  if not exist net-winnt\tmp\nul mkdir %version%\tmp

  echo Copying src\net-winnt\Makefile.nt to %version%\tmp
  xcopy /q src\net-winnt\Makefile.nt %version%\tmp 

  echo Copying files...
  xcopy /d /q     src\Common\conv-core\*  net-winnt\tmp
  xcopy /d /q     src\Common\ck-core\*    net-winnt\tmp
  xcopy /d /q     src\Common\ck-ldb\*     net-winnt\tmp
  xcopy /d /q     src\Common\ck-perf\*    net-winnt\tmp
  xcopy /d /q     src\Common\xlat-i\*     net-winnt\tmp
  xcopy /d /q     src\Common\xlatcpm\*     net-winnt\tmp
  xcopy /d /q     src\Common\conv-ldb\*   net-winnt\tmp
  xcopy /d /q     src\net-winnt\*.c      net-winnt\tmp
  xcopy /d /q     src\net-winnt\*.h      net-winnt\tmp  
  
:done

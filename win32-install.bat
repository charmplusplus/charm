@echo off
goto begin

:syntax
  echo.
  echo Usage: WIN32-INSTALL (target) (version)
  echo.
  echo targets: converse, charm++
  echo.
  echo versions:  ( net-win32 net-win32-smp )
  echo.
  echo Note: This script is trivial.  It
  echo.
  echo  1. Creates directories (version) and (version)\tmp
  echo  2. Copies src\Common\scripts\Makefile.win32 into (version)\tmp
  echo  3. Does a "nmake (target) (version) in (version)\tmp.
  echo.
  echo That's all WIN32-INSTALL does.  The rest is handled by the Makefile.
  echo.
  goto done

:begin
  
  if "%1"=="" goto syntax
  set program=%1
  shift
  if "%1"=="" goto syntax 
  set version=%1

  echo Creating directories: %version% and %version%\tmp
  if not exist %version%\nul mkdir %version%
  if not exist %version%\tmp\nul mkdir %version%\tmp

  echo Copying src\Common.win32\Makefile.win32 to %version%\tmp
  xcopy /q src\Common.win32\Makefile.win32 %version%\tmp 

  echo Copying files...
  xcopy /d /q     src\Common\conv-core\*  %version%\tmp
  xcopy /d /q     src\Common\ck-core\*    %version%\tmp
  xcopy /d /q     src\Common\ck-ldb\*     %version%\tmp
  xcopy /d /q     src\Common\ck-perf\*    %version%\tmp
  xcopy /d /q     src\Common\xlat-i\*     %version%\tmp
  xcopy /d /q     src\Common\xlatcpm\*    %version%\tmp
  xcopy /d /q     src\Common\conv-ldb\*   %version%\tmp
  xcopy /d /q     src\Common.win32\*.*    %version%\tmp
  xcopy /d /q     src\%version%\*.*       %version%\tmp

  echo You are ready to do a nmake now.
  cd %version%\tmp 
  nmake /nologo /f Makefile.win32 %program%
  cd ..\..
:done

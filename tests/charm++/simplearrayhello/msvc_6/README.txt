This directory contains a Microsoft Visual Studio
6.0 project file for a simple Charm++ application.

To use it, first build Charm++, then translate the hello.ci
interface file into the hello.decl.h and hello.def.h headers
manually.  To do this, open a DOS prompt and type:

	cd net-win32\pgms\charm++\simplearrayhello
	..\..\..\bin\charmxi.exe hello.ci

You should now be able to open hello.dsw in Visual Studio
by double-clicking on the hello.dsw icon.  Build the 
executable under the Build menu, and then execute it.


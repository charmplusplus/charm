/*
Portable Dynamically Linked Libraries (DLL) interface

Orion Sky Lawlor, olawlor@acm.org, 7/26/2002
*/
#ifndef __UIUC_CHARM_DLL_H
#define __UIUC_CHARM_DLL_H

/**
 * Abstraction for a DLL (Dynamically Linked Library) that is linked in
 * to the running program.
 */
class CkDll {
	void *handle; /*Opaque handle to opened library*/
public:
	/**
	 * Open this DLL and link it into this program.
	 */
	CkDll(const char *sharedLibraryName);
	
	///Return true if the link-in was successful
	int valid(void) const {return handle!=0;}
	
	/**
	 * Resolve this symbol in the DLL, and return its pointer.
	 *  If the symbol is a function, you can cast this to a 
	 *   function pointer and call it.
	 *  If that symbol name is not found, returns NULL.
	 */
	void *lookup(const char *symbolName);
	
	/**
	 * Remove this DLL from the program.
	 * This invalidates any pointers created with lookup--
	 *  any references to this dll's function pointers will crash
	 *  the program (or worse).
	 */
	~CkDll();
	
	///Filename extension used by DLLs on this machine (e.g., ".dll" or ".so")
	static const char *extension;
};

/**
 * An interpreter for C++ code.  The string passed in will
 *  be written to a file, compiled into a shared library, and
 *  then linked into the running program.
 */
class CkCppInterpreter {
	char libraryFile[256]; //e.g., "/tmp/sharedLib.123.so"
	CkDll *library;
public:
	////Compile "cppCode", making available the includes at inclPath
	CkCppInterpreter(const char *cppCode,const char *inclPath=0);
	
	///Return true if the compilation was a success
	int valid(void) const {return library!=0;}
	
	///Get the name of the compiled library (e.g., to copy it somewhere)
	inline const char *getLibraryName(void) const {return libraryFile;}
	
	///Return a function pointer you can cast to the appropriate
	///  type to call the interpreted code.
	inline void *lookup(const char *symbolName) 
	{
		if (!library) return 0;
		else return library->lookup(symbolName);
	}
	
	///Remove "cppCode" from the program.
	///  This invalidates any function pointers created with lookup
	~CkCppInterpreter();
};


#endif

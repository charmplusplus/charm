/* This is a Doxygen documentation page--
   It contains no actual executable source code. */
/**
\page Charm Code Guidelines
<!-- This HTML is generated from charm/doc/doxygen/code_guidelines.C -->

\section intro Introduction
Charm is a portable runtime system, and also a library.
To survive in Charm, code has to be portable and library-friendly,
which is what these guidelines are intended to insure.


\section requirements Requirements

<ul>
<li>Never re-use variables declared in a loop:
\code
for (int i=...) {...}
for (int i=...) {...}
\endcode
Even though the ISO standard for C++ says this
is legal, older compilers (like Visual C++ 6.0)
complain about a "duplicate definition i".

Instead, you have to declare the variable once
and use it in subsequent loops:
\code
int i;
for (i=...) {...}
for (i=...) {...}
\endcode

<li>Never use C++-style comments in C code, or C headers:
\code
void myFunction(void) { //First I do... 
   ...
}
\endcode
This compiles under C++ or in C with gcc, but fails 
with a syntax error on most C compilers.  C++-style 
comments are only acceptable in C++ code.

</ul>


\section guidelines Guidelines
<ul>
<li>Fear advanced C++.
<p>It actually is possible to use complicated, nested templates;
multiple inheritance; exceptions; dynamic_cast<>; and the 
Standard Template Library and yet still write portable C++ programs.  
It's just error-prone and painful.  Unless you're 
willing to invest hours making your
code work on that bizarre (insert favorite OEM here) compiler, 
upon which you don't even have an account, stick with basic C++.

<p>Things that work well everywhere:
<ul>
<li>classes, constructors, destructors, and methods
<li>public, protected, private members
<li>inline methods
<li>virtual methods
</ul>

<p>Things that can be made to work:
<ul>
<li>simple templates, declared inline in headers
<li>simple function and operator overloading
<li>inner classes and types, declared public
<li>multiple and virtual inheritance
<li>bool (this was once not true, but now bool works everywhere)
</ul>

<p>Things that never seem to work:
<ul>
<li>nested, "extern", or non-inline templates
<li>complicated uses of the Standard Template Library or iostreams
<li>dynamic_cast<>
</ul>


</ul>

\section guidelines Library Guidelines
<ul>

<li>Punctuate library routine names consistently.
<p>In Converse and Charm++ and related libraries, use 
a short prefix and initial capital letters, like "CkMyPe".
This applies to classes as well as routine calls.

<p>In AMPI, FEM, and Fortran-callable libraries, use 
an all-caps prefix, underscore, and one initial capital, 
with all subsequent words in lowercase separated by underscores,
like "MPI_Comm_size" or "FEM_Update_ghost_field".
Not only does this match with the the MPI standard names,
it's actually carefully designed to work well under Fortran.
The underscores separate words even in case-insensitive Fortran, 
and the mixed case ensures the Fortran and C bindings will map 
to different names regardless of whether the Fortran compiler 
uppercases or lowercases external names.

<li>You don't want to use global variables in libraries.
<p>Libraries are meant to be used in different places, 
but in big programs they can be used from several different 
places at the same time.  This means you have to store 
all your library state in per-object variables rather than
globals (or readonlies).  Note that this affects your 
interface, since the user will have to pass in some sort of 
identifier so you can tell which problem you're working on.

</ul>

\section dontcare Don't Care
<ul>

<li>Indentation, location of braces, spacing around 
operators, and local variable names.
<p>These contentious, personal choices are left to the programmer.
Pick a style and stick with it.

</ul>


Orion Lawlor, 4/2003.

*/

#undef CMK_DLL_CC
#define CMK_DLL_CC  "g++3 -shared -O3 -o "

#undef  CMK_COMPILEMODE_ORIG
#undef  CMK_COMPILEMODE_ANSI
#define CMK_COMPILEMODE_ORIG                               0
#define CMK_COMPILEMODE_ANSI                               1

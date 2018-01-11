/* A Bison parser, made by GNU Bison 3.0.4.  */

/* Bison implementation for Yacc-like parsers in C

   Copyright (C) 1984, 1989-1990, 2000-2015 Free Software Foundation, Inc.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.

   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */

/* C LALR(1) parser skeleton written by Richard Stallman, by
   simplifying the original so-called "semantic" parser.  */

/* All symbols defined below should begin with yy or YY, to avoid
   infringing on user name space.  This should be done even for local
   variables, as they might otherwise be expanded by user macros.
   There are some unavoidable exceptions within include files to
   define necessary library symbols; they are noted "INFRINGES ON
   USER NAME SPACE" below.  */

/* Identify Bison output.  */
#define YYBISON 1

/* Bison version.  */
#define YYBISON_VERSION "3.0.4"

/* Skeleton name.  */
#define YYSKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define YYPURE 0

/* Push parsers.  */
#define YYPUSH 0

/* Pull parsers.  */
#define YYPULL 1

/* Copy the first part of user declarations.  */
#line 2 "xi-grammar.y" /* yacc.c:339  */

#include "EToken.h"
#include "sdag/constructs/Constructs.h"
#include "xi-Chare.h"
#include "xi-symbol.h"
#include <iostream>
#include <string.h>
#include <string>

// Has to be a macro since YYABORT can only be used within rule actions.
#define ERROR(...)                              \
  if (xi::num_errors++ == xi::MAX_NUM_ERRORS) { \
    YYABORT;                                    \
  } else {                                      \
    xi::pretty_msg("error", __VA_ARGS__);       \
  }

#define WARNING(...)                        \
  if (enable_warnings) {                    \
    xi::pretty_msg("warning", __VA_ARGS__); \
  }

using namespace xi;
extern int yylex(void);
extern unsigned char in_comment;
extern unsigned int lineno;
extern int in_bracket, in_braces, in_int_expr;
extern std::list<Entry*> connectEntries;
extern char* yytext;
AstChildren<Module>* modlist;

void yyerror(const char*);

namespace xi {

const int MAX_NUM_ERRORS = 10;
int num_errors = 0;
bool firstRdma = true;

bool enable_warnings = true;

extern int macroDefined(const char* str, int istrue);
extern const char* python_doc;
extern char* fname;
void splitScopedName(const char* name, const char** scope, const char** basename);
void ReservedWord(int token, int fCol, int lCol);
}  // namespace xi

#line 115 "y.tab.c" /* yacc.c:339  */

#ifndef YY_NULLPTR
#if defined __cplusplus && 201103L <= __cplusplus
#define YY_NULLPTR nullptr
#else
#define YY_NULLPTR 0
#endif
#endif

/* Enabling verbose error messages.  */
#ifdef YYERROR_VERBOSE
#undef YYERROR_VERBOSE
#define YYERROR_VERBOSE 1
#else
#define YYERROR_VERBOSE 0
#endif

/* In a future release of Bison, this section will be replaced
   by #include "y.tab.h".  */
#ifndef YY_YY_Y_TAB_H_INCLUDED
#define YY_YY_Y_TAB_H_INCLUDED
/* Debug traces.  */
#ifndef YYDEBUG
#define YYDEBUG 0
#endif
#if YYDEBUG
extern int yydebug;
#endif

/* Token type.  */
#ifndef YYTOKENTYPE
#define YYTOKENTYPE
enum yytokentype {
  MODULE = 258,
  MAINMODULE = 259,
  EXTERN = 260,
  READONLY = 261,
  INITCALL = 262,
  INITNODE = 263,
  INITPROC = 264,
  PUPABLE = 265,
  CHARE = 266,
  MAINCHARE = 267,
  GROUP = 268,
  NODEGROUP = 269,
  ARRAY = 270,
  MESSAGE = 271,
  CONDITIONAL = 272,
  CLASS = 273,
  INCLUDE = 274,
  STACKSIZE = 275,
  THREADED = 276,
  TEMPLATE = 277,
  SYNC = 278,
  IGET = 279,
  EXCLUSIVE = 280,
  IMMEDIATE = 281,
  SKIPSCHED = 282,
  INLINE = 283,
  VIRTUAL = 284,
  MIGRATABLE = 285,
  AGGREGATE = 286,
  CREATEHERE = 287,
  CREATEHOME = 288,
  NOKEEP = 289,
  NOTRACE = 290,
  APPWORK = 291,
  VOID = 292,
  CONST = 293,
  NOCOPY = 294,
  PACKED = 295,
  VARSIZE = 296,
  ENTRY = 297,
  FOR = 298,
  FORALL = 299,
  WHILE = 300,
  WHEN = 301,
  OVERLAP = 302,
  SERIAL = 303,
  IF = 304,
  ELSE = 305,
  PYTHON = 306,
  LOCAL = 307,
  NAMESPACE = 308,
  USING = 309,
  IDENT = 310,
  NUMBER = 311,
  LITERAL = 312,
  CPROGRAM = 313,
  HASHIF = 314,
  HASHIFDEF = 315,
  INT = 316,
  LONG = 317,
  SHORT = 318,
  CHAR = 319,
  FLOAT = 320,
  DOUBLE = 321,
  UNSIGNED = 322,
  ACCEL = 323,
  READWRITE = 324,
  WRITEONLY = 325,
  ACCELBLOCK = 326,
  MEMCRITICAL = 327,
  REDUCTIONTARGET = 328,
  CASE = 329
};
#endif
/* Tokens.  */
#define MODULE 258
#define MAINMODULE 259
#define EXTERN 260
#define READONLY 261
#define INITCALL 262
#define INITNODE 263
#define INITPROC 264
#define PUPABLE 265
#define CHARE 266
#define MAINCHARE 267
#define GROUP 268
#define NODEGROUP 269
#define ARRAY 270
#define MESSAGE 271
#define CONDITIONAL 272
#define CLASS 273
#define INCLUDE 274
#define STACKSIZE 275
#define THREADED 276
#define TEMPLATE 277
#define SYNC 278
#define IGET 279
#define EXCLUSIVE 280
#define IMMEDIATE 281
#define SKIPSCHED 282
#define INLINE 283
#define VIRTUAL 284
#define MIGRATABLE 285
#define AGGREGATE 286
#define CREATEHERE 287
#define CREATEHOME 288
#define NOKEEP 289
#define NOTRACE 290
#define APPWORK 291
#define VOID 292
#define CONST 293
#define NOCOPY 294
#define PACKED 295
#define VARSIZE 296
#define ENTRY 297
#define FOR 298
#define FORALL 299
#define WHILE 300
#define WHEN 301
#define OVERLAP 302
#define SERIAL 303
#define IF 304
#define ELSE 305
#define PYTHON 306
#define LOCAL 307
#define NAMESPACE 308
#define USING 309
#define IDENT 310
#define NUMBER 311
#define LITERAL 312
#define CPROGRAM 313
#define HASHIF 314
#define HASHIFDEF 315
#define INT 316
#define LONG 317
#define SHORT 318
#define CHAR 319
#define FLOAT 320
#define DOUBLE 321
#define UNSIGNED 322
#define ACCEL 323
#define READWRITE 324
#define WRITEONLY 325
#define ACCELBLOCK 326
#define MEMCRITICAL 327
#define REDUCTIONTARGET 328
#define CASE 329

/* Value type.  */
#if !defined YYSTYPE && !defined YYSTYPE_IS_DECLARED

union YYSTYPE {
#line 53 "xi-grammar.y" /* yacc.c:355  */

  AstChildren<Module>* modlist;
  Module* module;
  ConstructList* conslist;
  Construct* construct;
  TParam* tparam;
  TParamList* tparlist;
  Type* type;
  PtrType* ptype;
  NamedType* ntype;
  FuncType* ftype;
  Readonly* readonly;
  Message* message;
  Chare* chare;
  Entry* entry;
  EntryList* entrylist;
  Parameter* pname;
  ParamList* plist;
  Template* templat;
  TypeList* typelist;
  AstChildren<Member>* mbrlist;
  Member* member;
  TVar* tvar;
  TVarList* tvarlist;
  Value* val;
  ValueList* vallist;
  MsgVar* mv;
  MsgVarList* mvlist;
  PUPableClass* pupable;
  IncludeFile* includeFile;
  const char* strval;
  int intval;
  unsigned int cattr;  // actually Chare::attrib_t, but referring to that creates nasty
                       // #include issues
  SdagConstruct* sc;
  IntExprConstruct* intexpr;
  WhenConstruct* when;
  SListConstruct* slist;
  CaseListConstruct* clist;
  OListConstruct* olist;
  SdagEntryConstruct* sentry;
  XStr* xstrptr;
  AccelBlock* accelBlock;

#line 347 "y.tab.c" /* yacc.c:355  */
};

typedef union YYSTYPE YYSTYPE;
#define YYSTYPE_IS_TRIVIAL 1
#define YYSTYPE_IS_DECLARED 1
#endif

/* Location type.  */
#if !defined YYLTYPE && !defined YYLTYPE_IS_DECLARED
typedef struct YYLTYPE YYLTYPE;
struct YYLTYPE {
  int first_line;
  int first_column;
  int last_line;
  int last_column;
};
#define YYLTYPE_IS_DECLARED 1
#define YYLTYPE_IS_TRIVIAL 1
#endif

extern YYSTYPE yylval;
extern YYLTYPE yylloc;
int yyparse(void);

#endif /* !YY_YY_Y_TAB_H_INCLUDED  */

/* Copy the second part of user declarations.  */

#line 378 "y.tab.c" /* yacc.c:358  */

#ifdef short
#undef short
#endif

#ifdef YYTYPE_UINT8
typedef YYTYPE_UINT8 yytype_uint8;
#else
typedef unsigned char yytype_uint8;
#endif

#ifdef YYTYPE_INT8
typedef YYTYPE_INT8 yytype_int8;
#else
typedef signed char yytype_int8;
#endif

#ifdef YYTYPE_UINT16
typedef YYTYPE_UINT16 yytype_uint16;
#else
typedef unsigned short int yytype_uint16;
#endif

#ifdef YYTYPE_INT16
typedef YYTYPE_INT16 yytype_int16;
#else
typedef short int yytype_int16;
#endif

#ifndef YYSIZE_T
#ifdef __SIZE_TYPE__
#define YYSIZE_T __SIZE_TYPE__
#elif defined size_t
#define YYSIZE_T size_t
#elif !defined YYSIZE_T
#include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#define YYSIZE_T size_t
#else
#define YYSIZE_T unsigned int
#endif
#endif

#define YYSIZE_MAXIMUM ((YYSIZE_T)-1)

#ifndef YY_
#if defined YYENABLE_NLS && YYENABLE_NLS
#if ENABLE_NLS
#include <libintl.h> /* INFRINGES ON USER NAME SPACE */
#define YY_(Msgid) dgettext("bison-runtime", Msgid)
#endif
#endif
#ifndef YY_
#define YY_(Msgid) Msgid
#endif
#endif

#ifndef YY_ATTRIBUTE
#if (defined __GNUC__ && (2 < __GNUC__ || (__GNUC__ == 2 && 96 <= __GNUC_MINOR__))) || \
    defined __SUNPRO_C && 0x5110 <= __SUNPRO_C
#define YY_ATTRIBUTE(Spec) __attribute__(Spec)
#else
#define YY_ATTRIBUTE(Spec) /* empty */
#endif
#endif

#ifndef YY_ATTRIBUTE_PURE
#define YY_ATTRIBUTE_PURE YY_ATTRIBUTE((__pure__))
#endif

#ifndef YY_ATTRIBUTE_UNUSED
#define YY_ATTRIBUTE_UNUSED YY_ATTRIBUTE((__unused__))
#endif

#if !defined _Noreturn && (!defined __STDC_VERSION__ || __STDC_VERSION__ < 201112)
#if defined _MSC_VER && 1200 <= _MSC_VER
#define _Noreturn __declspec(noreturn)
#else
#define _Noreturn YY_ATTRIBUTE((__noreturn__))
#endif
#endif

/* Suppress unused-variable warnings by "using" E.  */
#if !defined lint || defined __GNUC__
#define YYUSE(E) ((void)(E))
#else
#define YYUSE(E) /* empty */
#endif

#if defined __GNUC__ && 407 <= __GNUC__ * 100 + __GNUC_MINOR__
/* Suppress an incorrect diagnostic about yylval being uninitialized.  */
#define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN                                            \
  _Pragma("GCC diagnostic push") _Pragma("GCC diagnostic ignored \"-Wuninitialized\"") \
      _Pragma("GCC diagnostic ignored \"-Wmaybe-uninitialized\"")
#define YY_IGNORE_MAYBE_UNINITIALIZED_END _Pragma("GCC diagnostic pop")
#else
#define YY_INITIAL_VALUE(Value) Value
#endif
#ifndef YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
#define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
#define YY_IGNORE_MAYBE_UNINITIALIZED_END
#endif
#ifndef YY_INITIAL_VALUE
#define YY_INITIAL_VALUE(Value) /* Nothing. */
#endif

#if !defined yyoverflow || YYERROR_VERBOSE

/* The parser invokes alloca or malloc; define the necessary symbols.  */

#ifdef YYSTACK_USE_ALLOCA
#if YYSTACK_USE_ALLOCA
#ifdef __GNUC__
#define YYSTACK_ALLOC __builtin_alloca
#elif defined __BUILTIN_VA_ARG_INCR
#include <alloca.h> /* INFRINGES ON USER NAME SPACE */
#elif defined _AIX
#define YYSTACK_ALLOC __alloca
#elif defined _MSC_VER
#include <malloc.h> /* INFRINGES ON USER NAME SPACE */
#define alloca _alloca
#else
#define YYSTACK_ALLOC alloca
#if !defined _ALLOCA_H && !defined EXIT_SUCCESS
#include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
/* Use EXIT_SUCCESS as a witness for stdlib.h.  */
#ifndef EXIT_SUCCESS
#define EXIT_SUCCESS 0
#endif
#endif
#endif
#endif
#endif

#ifdef YYSTACK_ALLOC
/* Pacify GCC's 'empty if-body' warning.  */
#define YYSTACK_FREE(Ptr) \
  do { /* empty */        \
    ;                     \
  } while (0)
#ifndef YYSTACK_ALLOC_MAXIMUM
/* The OS might guarantee only one guard page at the bottom of the stack,
   and a page size can be as small as 4096 bytes.  So we cannot safely
   invoke alloca (N) if N exceeds 4096.  Use a slightly smaller number
   to allow for a few compiler-allocated temporary stack slots.  */
#define YYSTACK_ALLOC_MAXIMUM 4032 /* reasonable circa 2006 */
#endif
#else
#define YYSTACK_ALLOC YYMALLOC
#define YYSTACK_FREE YYFREE
#ifndef YYSTACK_ALLOC_MAXIMUM
#define YYSTACK_ALLOC_MAXIMUM YYSIZE_MAXIMUM
#endif
#if (defined __cplusplus && !defined EXIT_SUCCESS && \
     !((defined YYMALLOC || defined malloc) && (defined YYFREE || defined free)))
#include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#ifndef EXIT_SUCCESS
#define EXIT_SUCCESS 0
#endif
#endif
#ifndef YYMALLOC
#define YYMALLOC malloc
#if !defined malloc && !defined EXIT_SUCCESS
void* malloc(YYSIZE_T); /* INFRINGES ON USER NAME SPACE */
#endif
#endif
#ifndef YYFREE
#define YYFREE free
#if !defined free && !defined EXIT_SUCCESS
void free(void*);       /* INFRINGES ON USER NAME SPACE */
#endif
#endif
#endif
#endif /* ! defined yyoverflow || YYERROR_VERBOSE */

#if (!defined yyoverflow &&                                                        \
     (!defined __cplusplus || (defined YYLTYPE_IS_TRIVIAL && YYLTYPE_IS_TRIVIAL && \
                               defined YYSTYPE_IS_TRIVIAL && YYSTYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union yyalloc {
  yytype_int16 yyss_alloc;
  YYSTYPE yyvs_alloc;
  YYLTYPE yyls_alloc;
};

/* The size of the maximum gap between one aligned stack and the next.  */
#define YYSTACK_GAP_MAXIMUM (sizeof(union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
#define YYSTACK_BYTES(N)                                              \
  ((N) * (sizeof(yytype_int16) + sizeof(YYSTYPE) + sizeof(YYLTYPE)) + \
   2 * YYSTACK_GAP_MAXIMUM)

#define YYCOPY_NEEDED 1

/* Relocate STACK from its old location to the new one.  The
   local variables YYSIZE and YYSTACKSIZE give the old and new number of
   elements in the stack, and YYPTR gives the new location of the
   stack.  Advance YYPTR to a properly aligned location for the next
   stack.  */
#define YYSTACK_RELOCATE(Stack_alloc, Stack)                         \
  do {                                                               \
    YYSIZE_T yynewbytes;                                             \
    YYCOPY(&yyptr->Stack_alloc, Stack, yysize);                      \
    Stack = &yyptr->Stack_alloc;                                     \
    yynewbytes = yystacksize * sizeof(*Stack) + YYSTACK_GAP_MAXIMUM; \
    yyptr += yynewbytes / sizeof(*yyptr);                            \
  } while (0)

#endif

#if defined YYCOPY_NEEDED && YYCOPY_NEEDED
/* Copy COUNT objects from SRC to DST.  The source and destination do
   not overlap.  */
#ifndef YYCOPY
#if defined __GNUC__ && 1 < __GNUC__
#define YYCOPY(Dst, Src, Count) __builtin_memcpy(Dst, Src, (Count) * sizeof(*(Src)))
#else
#define YYCOPY(Dst, Src, Count)                                  \
  do {                                                           \
    YYSIZE_T yyi;                                                \
    for (yyi = 0; yyi < (Count); yyi++) (Dst)[yyi] = (Src)[yyi]; \
  } while (0)
#endif
#endif
#endif /* !YYCOPY_NEEDED */

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL 56
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST 1517

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS 91
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS 117
/* YYNRULES -- Number of rules.  */
#define YYNRULES 371
/* YYNSTATES -- Number of states.  */
#define YYNSTATES 726

/* YYTRANSLATE[YYX] -- Symbol number corresponding to YYX as returned
   by yylex, with out-of-bounds checking.  */
#define YYUNDEFTOK 2
#define YYMAXUTOK 329

#define YYTRANSLATE(YYX) \
  ((unsigned int)(YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[TOKEN-NUM] -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex, without out-of-bounds checking.  */
static const yytype_uint8 yytranslate[] = {
    0,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
    2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  85, 2,  83, 84,
    82, 2,  79, 89, 90, 2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  76, 75, 80, 88, 81,
    2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
    2,  2,  2,  2,  2,  2,  2,  86, 2,  87, 2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
    2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  77, 2,  78,
    2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
    2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
    2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
    2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
    2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
    2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
    2,  2,  2,  2,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17,
    18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
    39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
    60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74};

#if YYDEBUG
/* YYRLINE[YYN] -- Source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] = {
    0,    195,  195,  200,  203,  208,  209,  213,  215,  220,  221,  226,  228,  229,
    230,  232,  233,  234,  236,  237,  238,  239,  240,  244,  245,  246,  247,  248,
    249,  250,  251,  252,  253,  254,  255,  256,  257,  258,  259,  260,  261,  264,
    265,  266,  267,  268,  269,  270,  271,  272,  273,  274,  276,  278,  279,  282,
    283,  284,  285,  289,  291,  298,  302,  309,  311,  316,  317,  321,  323,  325,
    327,  329,  342,  344,  346,  348,  354,  356,  358,  360,  362,  364,  366,  368,
    370,  372,  380,  382,  384,  388,  390,  395,  396,  401,  402,  406,  408,  410,
    412,  414,  416,  418,  420,  422,  424,  426,  428,  430,  432,  434,  438,  439,
    446,  448,  452,  456,  458,  462,  466,  468,  470,  472,  474,  476,  480,  482,
    484,  486,  488,  492,  494,  498,  500,  504,  508,  513,  514,  518,  522,  527,
    528,  533,  534,  544,  546,  550,  552,  557,  558,  562,  564,  569,  570,  574,
    579,  580,  584,  586,  590,  592,  597,  598,  602,  603,  606,  610,  612,  616,
    618,  620,  625,  626,  630,  632,  636,  638,  642,  646,  650,  656,  660,  662,
    666,  668,  672,  676,  680,  684,  686,  691,  692,  697,  698,  700,  702,  711,
    713,  715,  719,  721,  725,  729,  731,  733,  735,  737,  741,  743,  748,  755,
    759,  761,  763,  764,  766,  768,  770,  774,  776,  778,  784,  790,  799,  801,
    803,  809,  817,  819,  822,  826,  830,  832,  837,  839,  847,  849,  851,  853,
    855,  857,  859,  861,  863,  865,  867,  870,  880,  897,  914,  916,  920,  925,
    926,  928,  935,  937,  941,  943,  945,  947,  949,  951,  953,  955,  957,  959,
    961,  963,  965,  967,  969,  971,  973,  985,  994,  996,  998,  1003, 1004, 1006,
    1015, 1016, 1018, 1024, 1030, 1036, 1044, 1051, 1059, 1066, 1068, 1070, 1072, 1077,
    1089, 1090, 1091, 1094, 1095, 1096, 1097, 1104, 1110, 1119, 1126, 1132, 1138, 1146,
    1148, 1152, 1154, 1158, 1160, 1164, 1166, 1171, 1172, 1176, 1178, 1180, 1184, 1186,
    1190, 1192, 1196, 1198, 1200, 1208, 1211, 1214, 1216, 1218, 1222, 1224, 1226, 1228,
    1230, 1232, 1234, 1236, 1238, 1240, 1242, 1244, 1248, 1250, 1252, 1254, 1256, 1258,
    1260, 1263, 1266, 1268, 1270, 1272, 1274, 1276, 1287, 1288, 1290, 1294, 1298, 1302,
    1306, 1311, 1318, 1320, 1324, 1327, 1331, 1335};
#endif

#if YYDEBUG || YYERROR_VERBOSE || 0
/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char* const yytname[] = {"$end",
                                      "error",
                                      "$undefined",
                                      "MODULE",
                                      "MAINMODULE",
                                      "EXTERN",
                                      "READONLY",
                                      "INITCALL",
                                      "INITNODE",
                                      "INITPROC",
                                      "PUPABLE",
                                      "CHARE",
                                      "MAINCHARE",
                                      "GROUP",
                                      "NODEGROUP",
                                      "ARRAY",
                                      "MESSAGE",
                                      "CONDITIONAL",
                                      "CLASS",
                                      "INCLUDE",
                                      "STACKSIZE",
                                      "THREADED",
                                      "TEMPLATE",
                                      "SYNC",
                                      "IGET",
                                      "EXCLUSIVE",
                                      "IMMEDIATE",
                                      "SKIPSCHED",
                                      "INLINE",
                                      "VIRTUAL",
                                      "MIGRATABLE",
                                      "AGGREGATE",
                                      "CREATEHERE",
                                      "CREATEHOME",
                                      "NOKEEP",
                                      "NOTRACE",
                                      "APPWORK",
                                      "VOID",
                                      "CONST",
                                      "NOCOPY",
                                      "PACKED",
                                      "VARSIZE",
                                      "ENTRY",
                                      "FOR",
                                      "FORALL",
                                      "WHILE",
                                      "WHEN",
                                      "OVERLAP",
                                      "SERIAL",
                                      "IF",
                                      "ELSE",
                                      "PYTHON",
                                      "LOCAL",
                                      "NAMESPACE",
                                      "USING",
                                      "IDENT",
                                      "NUMBER",
                                      "LITERAL",
                                      "CPROGRAM",
                                      "HASHIF",
                                      "HASHIFDEF",
                                      "INT",
                                      "LONG",
                                      "SHORT",
                                      "CHAR",
                                      "FLOAT",
                                      "DOUBLE",
                                      "UNSIGNED",
                                      "ACCEL",
                                      "READWRITE",
                                      "WRITEONLY",
                                      "ACCELBLOCK",
                                      "MEMCRITICAL",
                                      "REDUCTIONTARGET",
                                      "CASE",
                                      "';'",
                                      "':'",
                                      "'{'",
                                      "'}'",
                                      "','",
                                      "'<'",
                                      "'>'",
                                      "'*'",
                                      "'('",
                                      "')'",
                                      "'&'",
                                      "'['",
                                      "']'",
                                      "'='",
                                      "'-'",
                                      "'.'",
                                      "$accept",
                                      "File",
                                      "ModuleEList",
                                      "OptExtern",
                                      "OneOrMoreSemiColon",
                                      "OptSemiColon",
                                      "Name",
                                      "QualName",
                                      "Module",
                                      "ConstructEList",
                                      "ConstructList",
                                      "ConstructSemi",
                                      "Construct",
                                      "TParam",
                                      "TParamList",
                                      "TParamEList",
                                      "OptTParams",
                                      "BuiltinType",
                                      "NamedType",
                                      "QualNamedType",
                                      "SimpleType",
                                      "OnePtrType",
                                      "PtrType",
                                      "FuncType",
                                      "BaseType",
                                      "BaseDataType",
                                      "RestrictedType",
                                      "Type",
                                      "ArrayDim",
                                      "Dim",
                                      "DimList",
                                      "Readonly",
                                      "ReadonlyMsg",
                                      "OptVoid",
                                      "MAttribs",
                                      "MAttribList",
                                      "MAttrib",
                                      "CAttribs",
                                      "CAttribList",
                                      "PythonOptions",
                                      "ArrayAttrib",
                                      "ArrayAttribs",
                                      "ArrayAttribList",
                                      "CAttrib",
                                      "OptConditional",
                                      "MsgArray",
                                      "Var",
                                      "VarList",
                                      "Message",
                                      "OptBaseList",
                                      "BaseList",
                                      "Chare",
                                      "Group",
                                      "NodeGroup",
                                      "ArrayIndexType",
                                      "Array",
                                      "TChare",
                                      "TGroup",
                                      "TNodeGroup",
                                      "TArray",
                                      "TMessage",
                                      "OptTypeInit",
                                      "OptNameInit",
                                      "TVar",
                                      "TVarList",
                                      "TemplateSpec",
                                      "Template",
                                      "MemberEList",
                                      "MemberList",
                                      "NonEntryMember",
                                      "InitNode",
                                      "InitProc",
                                      "PUPableClass",
                                      "IncludeFile",
                                      "Member",
                                      "MemberBody",
                                      "UnexpectedToken",
                                      "Entry",
                                      "AccelBlock",
                                      "EReturn",
                                      "EAttribs",
                                      "EAttribList",
                                      "EAttrib",
                                      "DefaultParameter",
                                      "CPROGRAM_List",
                                      "CCode",
                                      "ParamBracketStart",
                                      "ParamBraceStart",
                                      "ParamBraceEnd",
                                      "Parameter",
                                      "AccelBufferType",
                                      "AccelInstName",
                                      "AccelArrayParam",
                                      "AccelParameter",
                                      "ParamList",
                                      "AccelParamList",
                                      "EParameters",
                                      "AccelEParameters",
                                      "OptStackSize",
                                      "OptSdagCode",
                                      "Slist",
                                      "Olist",
                                      "CaseList",
                                      "OptTraceName",
                                      "WhenConstruct",
                                      "NonWhenConstruct",
                                      "SingleConstruct",
                                      "HasElse",
                                      "IntExpr",
                                      "EndIntExpr",
                                      "StartIntExpr",
                                      "SEntry",
                                      "SEntryList",
                                      "SParamBracketStart",
                                      "SParamBracketEnd",
                                      "HashIFComment",
                                      "HashIFDefComment",
                                      YY_NULLPTR};
#endif

#ifdef YYPRINT
/* YYTOKNUM[NUM] -- (External) token number corresponding to the
   (internal) symbol number NUM (which must be that of a token).  */
static const yytype_uint16 yytoknum[] = {
    0,   256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270,
    271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286,
    287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302,
    303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318,
    319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 59,  58,  123, 125, 44,
    60,  62,  42,  40,  41,  38,  91,  93,  61,  45,  46};
#endif

#define YYPACT_NINF -603

#define yypact_value_is_default(Yystate) (!!((Yystate) == (-603)))

#define YYTABLE_NINF -323

#define yytable_value_is_error(Yytable_value) 0

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
static const yytype_int16 yypact[] = {
    249,  1344, 1344, 34,   -603, 249,  -603, -603, -603, -603, -603, -603, -603, -603,
    -603, -603, -603, -603, -603, -603, -603, -603, -603, -603, -603, -603, -603, -603,
    -603, -603, -603, -603, -603, -603, -603, -603, -603, -603, -603, -603, -603, -603,
    -603, -603, -603, -603, -603, -603, -603, -603, -603, -603, -603, -603, 42,   42,
    -603, -603, -603, 768,  -35,  -603, -603, -603, 7,    1344, 149,  1344, 1344, 142,
    932,  -23,  910,  768,  -603, -603, -603, -603, 1431, -17,  101,  -603, 116,  -603,
    -603, -603, -35,  -21,  604,  136,  136,  -9,   101,  138,  138,  138,  138,  160,
    169,  1344, 171,  155,  768,  -603, -603, -603, -603, -603, -603, -603, -603, 257,
    -603, -603, -603, -603, 182,  -603, -603, -603, -603, -603, -603, -603, -603, -603,
    -603, -603, -35,  -603, -603, -603, 1431, -603, 97,   -603, -603, -603, -603, 176,
    58,   -603, -603, 184,  192,  196,  15,   -603, 101,  768,  116,  207,  68,   -21,
    208,  931,  1450, 184,  192,  196,  -603, 20,   101,  -603, 101,  101,  229,  101,
    221,  -603, 21,   1344, 1344, 1344, 1344, 1128, 222,  223,  255,  1344, -603, -603,
    -603, 1364, 233,  138,  138,  138,  138,  222,  169,  -603, -603, -603, -603, -603,
    -35,  -603, 277,  -603, -603, -603, 237,  -603, -603, 1396, -603, -603, -603, -603,
    -603, -603, 1344, 252,  278,  -21,  276,  -21,  256,  -603, 182,  260,  25,   -603,
    261,  -603, 46,   38,   125,  266,  140,  101,  -603, -603, 267,  268,  259,  272,
    272,  272,  272,  -603, 1344, 271,  281,  274,  1200, 1344, 311,  1344, -603, -603,
    283,  284,  287,  1344, 62,   1344, 286,  290,  182,  1344, 1344, 1344, 1344, 1344,
    1344, -603, -603, -603, -603, 293,  -603, 292,  -603, 259,  -603, -603, 301,  309,
    296,  302,  -21,  -35,  101,  1344, -603, 305,  -603, -21,  136,  1396, 136,  136,
    1396, 136,  -603, -603, 21,   -603, 101,  154,  154,  154,  154,  313,  -603, 311,
    -603, 272,  272,  -603, 255,  0,    304,  225,  -603, 314,  1364, -603, -603, 272,
    272,  272,  272,  272,  172,  1396, -603, 318,  -21,  276,  -21,  -21,  -603, 46,
    319,  -603, 317,  -603, 323,  328,  327,  101,  331,  329,  -603, 335,  -603, 368,
    -35,  -603, -603, -603, -603, -603, -603, 154,  154,  -603, -603, -603, 1450, 9,
    337,  1450, -603, -603, -603, -603, -603, -603, 154,  154,  154,  154,  154,  399,
    -35,  -603, 1383, -603, -603, -603, -603, -603, -603, 334,  -603, -603, -603, 336,
    -603, 96,   338,  -603, 101,  -603, 684,  381,  347,  182,  368,  -603, -603, -603,
    -603, 1344, -603, -603, -603, -603, -603, -603, -603, -603, 348,  1450, -603, 1344,
    -21,  353,  351,  1417, 136,  136,  136,  -603, -603, 948,  1056, -603, 182,  -35,
    -603, 355,  182,  1344, -21,  2,    352,  1417, -603, 358,  359,  360,  361,  -603,
    -603, -603, -603, -603, -603, -603, -603, -603, -603, -603, -603, -603, -603, 377,
    -603, 362,  -603, -603, 363,  369,  364,  318,  1344, -603, 370,  182,  -35,  365,
    371,  -603, 236,  -603, -603, -603, -603, -603, -603, -603, -603, -603, 422,  -603,
    1001, 535,  318,  -603, -35,  -603, -603, -603, 116,  -603, 1344, -603, -603, 380,
    378,  380,  413,  393,  414,  380,  395,  258,  -35,  -21,  -603, -603, -603, 453,
    318,  -603, -21,  419,  -21,  107,  397,  545,  555,  -603, 400,  -21,  275,  403,
    481,  208,  390,  535,  406,  -603, 408,  415,  411,  -603, -21,  413,  350,  -603,
    423,  496,  -21,  411,  380,  420,  380,  428,  414,  380,  430,  -21,  431,  275,
    -603, 182,  -603, 182,  452,  -603, 424,  400,  -21,  380,  -603, 607,  317,  -603,
    -603, 435,  -603, -603, 208,  752,  -21,  459,  -21,  555,  400,  -21,  275,  208,
    -603, -603, -603, -603, -603, -603, -603, -603, -603, 1344, 439,  437,  433,  -21,
    442,  -21,  258,  -603, 318,  -603, 182,  258,  468,  447,  444,  411,  454,  -21,
    411,  460,  182,  457,  1450, 787,  -603, 208,  -21,  471,  470,  -603, -603, 473,
    759,  -603, -21,  380,  766,  -603, 208,  815,  -603, -603, 1344, 1344, -21,  469,
    -603, 1344, 411,  -21,  -603, 468,  258,  -603, 477,  -21,  258,  -603, 182,  258,
    468,  -603, 72,   -48,  466,  1344, 182,  822,  475,  -603, 483,  -21,  486,  485,
    -603, 487,  -603, -603, 1344, 1272, 488,  1344, 1344, -603, 94,   -35,  258,  -603,
    -21,  -603, 411,  -21,  -603, 468,  81,   479,  99,   1344, -603, 144,  -603, 490,
    411,  829,  493,  -603, -603, -603, -603, -603, -603, -603, 836,  258,  -603, -21,
    258,  -603, 497,  411,  498,  -603, 885,  -603, 258,  -603, 499,  -603};

/* YYDEFACT[STATE-NUM] -- Default reduction number in state STATE-NUM.
   Performed when YYTABLE does not specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint16 yydefact[] = {
    3,   0,   0,   0,   2,   3,   12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22,
    23,  24,  25,  26,  27,  28,  29,  30,  31,  33,  34,  35,  36,  37,  38,  39,  40,
    32,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  11,  54,  55,
    56,  57,  58,  0,   0,   1,   4,   7,   0,   63,  61,  62,  85,  6,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   84,  82,  83,  8,   0,   0,   0,   59,  68,  370, 371,
    287, 249, 280, 0,   139, 139, 139, 0,   147, 147, 147, 147, 0,   141, 0,   0,   0,
    0,   76,  210, 211, 70,  77,  78,  79,  80,  0,   81,  69,  213, 212, 9,   244, 236,
    237, 238, 239, 240, 242, 243, 241, 234, 235, 74,  75,  66,  109, 0,   95,  96,  97,
    98,  106, 107, 0,   93,  112, 113, 124, 125, 126, 130, 250, 0,   0,   67,  0,   281,
    280, 0,   0,   0,   118, 119, 120, 121, 132, 0,   140, 0,   0,   0,   0,   226, 214,
    0,   0,   0,   0,   0,   0,   0,   154, 0,   0,   216, 228, 215, 0,   0,   147, 147,
    147, 147, 0,   141, 201, 202, 203, 204, 205, 10,  64,  127, 105, 108, 99,  100, 103,
    104, 91,  111, 114, 115, 116, 128, 129, 0,   0,   0,   280, 277, 280, 0,   288, 0,
    0,   122, 123, 0,   131, 135, 220, 217, 0,   222, 0,   158, 159, 0,   149, 93,  170,
    170, 170, 170, 153, 0,   0,   156, 0,   0,   0,   0,   0,   145, 146, 0,   143, 167,
    0,   121, 0,   198, 0,   9,   0,   0,   0,   0,   0,   0,   101, 102, 87,  88,  89,
    92,  0,   86,  93,  73,  60,  0,   278, 0,   0,   280, 248, 0,   0,   368, 135, 137,
    280, 139, 0,   139, 139, 0,   139, 227, 148, 0,   110, 0,   0,   0,   0,   0,   0,
    179, 0,   155, 170, 170, 142, 0,   160, 189, 0,   196, 191, 0,   200, 72,  170, 170,
    170, 170, 170, 0,   0,   94,  0,   280, 277, 280, 280, 285, 135, 0,   136, 0,   133,
    0,   0,   0,   0,   0,   0,   150, 172, 171, 0,   206, 174, 175, 176, 177, 178, 157,
    0,   0,   144, 161, 168, 0,   160, 0,   0,   195, 192, 193, 194, 197, 199, 0,   0,
    0,   0,   0,   160, 187, 90,  0,   71,  283, 279, 284, 282, 138, 0,   369, 134, 221,
    0,   218, 0,   0,   223, 0,   233, 0,   0,   0,   0,   0,   229, 230, 180, 181, 0,
    166, 169, 190, 182, 183, 184, 185, 186, 0,   0,   312, 289, 280, 307, 0,   0,   139,
    139, 139, 173, 253, 0,   0,   231, 9,   232, 209, 162, 0,   0,   280, 160, 0,   0,
    311, 0,   0,   0,   0,   273, 256, 257, 258, 259, 265, 266, 267, 272, 260, 261, 262,
    263, 264, 151, 268, 0,   270, 271, 0,   254, 59,  0,   0,   207, 0,   0,   188, 0,
    0,   286, 0,   290, 292, 308, 117, 219, 225, 224, 152, 269, 0,   252, 0,   0,   0,
    163, 164, 293, 275, 274, 276, 291, 0,   255, 357, 0,   0,   0,   0,   0,   328, 0,
    0,   0,   317, 280, 246, 346, 318, 315, 0,   363, 280, 0,   280, 0,   366, 0,   0,
    327, 0,   280, 0,   0,   0,   0,   0,   0,   0,   361, 0,   0,   0,   364, 280, 0,
    0,   330, 0,   0,   280, 0,   0,   0,   0,   0,   328, 0,   0,   280, 0,   324, 326,
    9,   321, 9,   0,   245, 0,   0,   280, 0,   362, 0,   0,   367, 329, 0,   345, 323,
    0,   0,   280, 0,   280, 0,   0,   280, 0,   0,   347, 325, 319, 356, 316, 294, 295,
    296, 314, 0,   0,   309, 0,   280, 0,   280, 0,   354, 0,   331, 9,   0,   358, 0,
    0,   0,   0,   280, 0,   0,   9,   0,   0,   0,   313, 0,   280, 0,   0,   365, 344,
    0,   0,   352, 280, 0,   0,   333, 0,   0,   334, 343, 0,   0,   280, 0,   310, 0,
    0,   280, 355, 358, 0,   359, 0,   280, 0,   341, 9,   0,   358, 297, 0,   0,   0,
    0,   0,   0,   0,   353, 0,   280, 0,   0,   332, 0,   339, 305, 0,   0,   0,   0,
    0,   303, 0,   247, 0,   349, 280, 360, 0,   280, 342, 358, 0,   0,   0,   0,   299,
    0,   306, 0,   0,   0,   0,   340, 302, 301, 300, 298, 304, 348, 0,   0,   336, 280,
    0,   350, 0,   0,   0,   335, 0,   351, 0,   337, 0,   338};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] = {
    -603, -603, 559,  -603, -51,  -251, -1,   -58,  515,  531,  -50,  -603, -603,
    -603, -179, -603, -216, -603, -129, -79,  -71,  -64,  -62,  -171, 441,  463,
    -603, -86,  -603, -603, -262, -603, -603, -80,  416,  299,  -603, 102,  316,
    -603, -603, 438,  310,  -603, 177,  -603, -603, -238, -603, 4,    227,  -603,
    -603, -603, -66,  -603, -603, -603, -603, -603, -603, -603, 307,  -603, 300,
    551,  -603, 80,   224,  557,  -603, -603, 394,  -603, -603, -603, -603, 231,
    -603, 198,  -603, 143,  -603, -603, 303,  -82,  -402, -63,  -492, -603, -603,
    -550, -603, -603, -312, 14,   -438, -603, -603, 103,  -508, 53,   -529, 83,
    -484, -603, -443, -602, -487, -522, -476, -603, 100,  122,  74,   -603, -603};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] = {
    -1,  3,   4,   70,  350, 197, 236, 140, 5,   61,  71,  72,  73,  271, 272, 273, 206,
    141, 237, 142, 157, 158, 159, 160, 161, 146, 147, 274, 338, 287, 288, 104, 105, 164,
    179, 252, 253, 171, 234, 487, 244, 176, 245, 235, 362, 473, 363, 364, 106, 301, 348,
    107, 108, 109, 177, 110, 191, 192, 193, 194, 195, 366, 316, 258, 259, 399, 112, 351,
    400, 401, 114, 115, 169, 182, 402, 403, 129, 404, 74,  148, 430, 466, 467, 499, 280,
    537, 420, 513, 220, 421, 598, 660, 643, 599, 422, 600, 381, 567, 535, 514, 531, 546,
    558, 528, 515, 560, 532, 631, 538, 571, 520, 524, 525, 289, 389, 75,  76};

/* YYTABLE[YYPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule whose
   number is the opposite.  If YYTABLE_NINF, syntax error.  */
static const yytype_int16 yytable[] = {
    54,   55,   162,  60,   60,   154,  87,   143,  82,   320,  165,  167,  256,  168,
    144,  438,  145,  360,  86,   360,  299,  128,  150,  130,  562,  337,  360,  579,
    163,  522,  589,  491,  680,  529,  56,   540,  575,  152,  477,  577,  77,   563,
    549,  238,  239,  240,  559,  667,  516,  78,   254,  232,  184,  210,  517,  116,
    674,  617,  223,  329,  149,  143,  153,  223,  79,   196,  83,   84,   144,  212,
    145,  218,  233,  386,  580,  559,  582,  166,  361,  585,  536,  602,  545,  547,
    221,  541,  608,  -165, 477,  703,  478,  634,  516,  603,  637,  618,  626,  257,
    180,  213,  211,  629,  559,  224,  226,  225,  227,  228,  224,  230,  247,  443,
    341,  682,  151,  344,  625,  58,   309,  59,   310,  290,  151,  265,  665,  408,
    692,  694,  605,  481,  697,  645,  286,  278,  151,  281,  610,  646,  205,  416,
    547,  -191, 668,  -191, 656,  215,  671,  256,  379,  673,  315,  216,  168,  675,
    217,  676,  81,   653,  677,  199,  666,  678,  679,  200,  676,  704,  701,  677,
    627,  283,  678,  679,  151,  163,  243,  698,  699,  676,  710,  425,  677,  471,
    676,  678,  679,  677,  706,  651,  678,  679,  380,  655,  151,  286,  658,  720,
    172,  173,  174,  700,  334,  151,  80,   716,  81,   291,  718,  339,  292,  196,
    340,  275,  342,  343,  724,  345,  151,  58,   642,  85,   294,  347,  685,  295,
    170,  708,  335,  676,  181,  58,   677,  349,  257,  678,  679,  183,  369,  201,
    202,  203,  204,  305,  302,  303,  304,  243,  175,  58,   382,  377,  384,  385,
    1,    2,    314,  178,  317,  58,   712,  502,  321,  322,  323,  324,  325,  326,
    207,  715,  185,  186,  187,  188,  189,  190,  208,  378,  407,  723,  209,  410,
    81,   367,  368,  214,  336,  393,  219,  261,  262,  263,  264,  81,   496,  497,
    419,  250,  251,  229,  267,  268,  231,  503,  504,  505,  506,  507,  508,  509,
    246,  248,  590,  260,  591,  357,  358,  210,  -287, 347,  550,  551,  552,  506,
    553,  554,  555,  372,  373,  374,  375,  376,  276,  437,  510,  277,  279,  85,
    -287, 419,  440,  205,  282,  -287, 284,  285,  444,  445,  446,  298,  300,  556,
    433,  502,  85,   293,  297,  419,  476,  628,  306,  143,  307,  308,  241,  312,
    313,  318,  144,  639,  145,  397,  311,  319,  327,  328,  88,   89,   90,   91,
    92,   330,  332,  196,  352,  353,  354,  474,  99,   100,  331,  333,  101,  286,
    365,  503,  504,  505,  506,  507,  508,  509,  355,  380,  315,  387,  388,  672,
    435,  390,  -287, 391,  398,  392,  394,  395,  396,  409,  360,  423,  439,  424,
    498,  426,  494,  398,  510,  432,  436,  85,   574,  469,  593,  533,  441,  -287,
    486,  442,  475,  405,  406,  480,  512,  472,  482,  483,  484,  485,  -208, -11,
    490,  488,  489,  477,  411,  412,  413,  414,  415,  493,  495,  500,  572,  131,
    156,  519,  521,  548,  578,  557,  523,  492,  526,  527,  530,  534,  539,  587,
    543,  85,   564,  81,   597,  561,  502,  568,  512,  133,  134,  135,  136,  137,
    138,  139,  566,  594,  595,  570,  557,  502,  611,  518,  613,  576,  569,  616,
    601,  583,  581,  586,  592,  588,  196,  596,  196,  607,  612,  620,  621,  624,
    630,  623,  622,  615,  632,  557,  503,  504,  505,  506,  507,  508,  509,  633,
    635,  636,  641,  597,  502,  640,  638,  503,  504,  505,  506,  507,  508,  509,
    502,  647,  648,  663,  652,  649,  669,  681,  686,  510,  502,  196,  85,   -320,
    662,  687,  689,  690,  57,   691,  705,  196,  709,  695,  510,  670,  713,  85,
    -322, 719,  721,  725,  503,  504,  505,  506,  507,  508,  509,  103,  62,   688,
    503,  504,  505,  506,  507,  508,  509,  198,  619,  222,  503,  504,  505,  506,
    507,  508,  509,  196,  266,  702,  502,  510,  58,   359,  511,  683,  346,  249,
    479,  356,  371,  510,  155,  111,  544,  427,  370,  296,  434,  113,  470,  510,
    431,  717,  85,   501,  383,  644,  614,  584,  565,  659,  661,  131,  156,  573,
    664,  542,  606,  0,    0,    0,    503,  504,  505,  506,  507,  508,  509,  0,
    0,    81,   0,    0,    659,  0,    0,    133,  134,  135,  136,  137,  138,  139,
    0,    0,    0,    659,  659,  0,    696,  659,  0,    510,  0,    0,    604,  428,
    0,    -251, -251, -251, 0,    -251, -251, -251, 707,  -251, -251, -251, -251, -251,
    0,    0,    0,    -251, -251, -251, -251, -251, -251, -251, -251, -251, -251, -251,
    -251, 0,    -251, -251, -251, -251, -251, -251, -251, -251, -251, -251, -251, -251,
    -251, -251, -251, -251, -251, -251, -251, 0,    -251, 0,    -251, -251, 0,    0,
    0,    0,    0,    -251, -251, -251, -251, -251, -251, -251, -251, 502,  0,    -251,
    -251, -251, -251, 0,    502,  0,    0,    0,    0,    0,    0,    502,  0,    63,
    429,  -5,   -5,   64,   -5,   -5,   -5,   -5,   -5,   -5,   -5,   -5,   -5,   -5,
    -5,   0,    -5,   -5,   0,    0,    -5,   0,    0,    593,  0,    503,  504,  505,
    506,  507,  508,  509,  503,  504,  505,  506,  507,  508,  509,  503,  504,  505,
    506,  507,  508,  509,  502,  0,    0,    0,    0,    65,   66,   502,  131,  156,
    510,  67,   68,   609,  502,  0,    0,    510,  0,    0,    650,  502,  0,    69,
    510,  0,    81,   654,  0,    -5,   -65,  0,    133,  134,  135,  136,  137,  138,
    139,  0,    594,  595,  503,  504,  505,  506,  507,  508,  509,  503,  504,  505,
    506,  507,  508,  509,  503,  504,  505,  506,  507,  508,  509,  503,  504,  505,
    506,  507,  508,  509,  502,  0,    0,    510,  0,    0,    657,  0,    0,    0,
    510,  0,    0,    684,  0,    0,    0,    510,  0,    0,    711,  0,    0,    0,
    510,  0,    0,    714,  0,    0,    117,  118,  119,  120,  0,    121,  122,  123,
    124,  125,  0,    0,    503,  504,  505,  506,  507,  508,  509,  1,    2,    0,
    88,   89,   90,   91,   92,   93,   94,   95,   96,   97,   98,   447,  99,   100,
    126,  0,    101,  0,    0,    0,    0,    510,  0,    0,    722,  0,    0,    0,
    0,    0,    131,  448,  0,    449,  450,  451,  452,  453,  454,  0,    0,    455,
    456,  457,  458,  459,  460,  58,   81,   0,    127,  0,    0,    0,    133,  134,
    135,  136,  137,  138,  139,  461,  462,  0,    447,  0,    0,    0,    0,    0,
    0,    102,  0,    0,    0,    0,    0,    0,    463,  0,    0,    0,    464,  465,
    448,  0,    449,  450,  451,  452,  453,  454,  0,    0,    455,  456,  457,  458,
    459,  460,  0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    461,  462,  0,    0,    0,    0,    0,    6,    7,    8,    0,    9,
    10,   11,   0,    12,   13,   14,   15,   16,   0,    464,  465,  17,   18,   19,
    20,   21,   22,   23,   24,   25,   26,   27,   28,   0,    29,   30,   31,   32,
    33,   131,  132,  34,   35,   36,   37,   38,   39,   40,   41,   42,   43,   44,
    45,   0,    46,   0,    47,   468,  0,    0,    0,    0,    0,    133,  134,  135,
    136,  137,  138,  139,  49,   0,    0,    50,   51,   52,   53,   6,    7,    8,
    0,    9,    10,   11,   0,    12,   13,   14,   15,   16,   0,    0,    0,    17,
    18,   19,   20,   21,   22,   23,   24,   25,   26,   27,   28,   0,    29,   30,
    31,   32,   33,   0,    0,    34,   35,   36,   37,   38,   39,   40,   41,   42,
    43,   44,   45,   241,  46,   0,    47,   48,   242,  0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    49,   0,    0,    50,   51,   52,   53,   6,
    7,    8,    0,    9,    10,   11,   0,    12,   13,   14,   15,   16,   0,    0,
    0,    17,   18,   19,   20,   21,   22,   23,   24,   25,   26,   27,   28,   0,
    29,   30,   31,   32,   33,   0,    0,    34,   35,   36,   37,   38,   39,   40,
    41,   42,   43,   44,   45,   0,    46,   0,    47,   48,   242,  0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    49,   0,    0,    50,   51,   52,
    53,   6,    7,    8,    0,    9,    10,   11,   0,    12,   13,   14,   15,   16,
    0,    0,    0,    17,   18,   19,   20,   21,   22,   23,   24,   25,   26,   27,
    28,   0,    29,   30,   31,   32,   33,   0,    0,    34,   35,   36,   37,   38,
    39,   40,   41,   42,   43,   44,   45,   0,    46,   0,    47,   48,   693,  0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    49,   0,    0,    50,
    51,   52,   53,   6,    7,    8,    0,    9,    10,   11,   0,    12,   13,   14,
    15,   16,   0,    0,    0,    17,   18,   19,   20,   21,   22,   23,   24,   25,
    26,   27,   28,   0,    29,   30,   31,   32,   33,   0,    255,  34,   35,   36,
    37,   38,   39,   40,   41,   42,   43,   44,   45,   0,    46,   0,    47,   48,
    0,    131,  156,  0,    0,    0,    0,    0,    0,    0,    0,    0,    49,   0,
    0,    50,   51,   52,   53,   81,   131,  156,  417,  0,    0,    133,  134,  135,
    136,  137,  138,  139,  0,    131,  156,  0,    0,    0,    81,   0,    0,    0,
    0,    0,    133,  134,  135,  136,  137,  138,  139,  81,   269,  270,  131,  156,
    417,  133,  134,  135,  136,  137,  138,  139,  0,    0,    0,    418,  131,  132,
    0,    0,    81,   0,    0,    0,    0,    0,    133,  134,  135,  136,  137,  138,
    139,  0,    81,   131,  156,  0,    0,    0,    133,  134,  135,  136,  137,  138,
    139,  0,    0,    0,    0,    0,    0,    81,   0,    0,    0,    0,    0,    133,
    134,  135,  136,  137,  138,  139};

static const yytype_int16 yycheck[] = {
    1,   2,   88,  54,  55,  87,  69,  78,  66,  260, 90,  91,  183, 92,  78,  417, 78,
    17,  69,  17,  236, 72,  80,  73,  532, 287, 17,  549, 37,  505, 559, 469, 80,  509,
    0,   522, 544, 58,  86,  547, 75,  533, 529, 172, 173, 174, 530, 649, 491, 42,  179,
    30,  102, 38,  492, 78,  658, 586, 38,  275, 77,  132, 83,  38,  65,  116, 67,  68,
    132, 148, 132, 153, 51,  335, 550, 559, 552, 86,  78,  555, 518, 568, 525, 526, 155,
    523, 578, 78,  86,  691, 88,  613, 535, 569, 616, 587, 604, 183, 99,  149, 85,  609,
    586, 83,  162, 85,  164, 165, 83,  167, 176, 423, 291, 663, 76,  294, 603, 75,  247,
    77,  249, 83,  76,  189, 646, 363, 676, 677, 571, 441, 680, 623, 86,  215, 76,  217,
    579, 624, 80,  377, 583, 79,  650, 81,  636, 77,  654, 318, 327, 657, 88,  83,  231,
    81,  86,  83,  55,  633, 86,  62,  647, 89,  90,  66,  83,  84,  688, 86,  606, 220,
    89,  90,  76,  37,  175, 81,  684, 83,  700, 83,  86,  432, 83,  89,  90,  86,  87,
    630, 89,  90,  83,  634, 76,  86,  637, 717, 94,  95,  96,  686, 282, 76,  53,  711,
    55,  80,  714, 289, 83,  260, 290, 212, 292, 293, 722, 295, 76,  75,  620, 77,  80,
    300, 665, 83,  86,  81,  284, 83,  57,  75,  86,  77,  318, 89,  90,  80,  315, 61,
    62,  63,  64,  242, 238, 239, 240, 246, 86,  75,  330, 77,  332, 333, 3,   4,   255,
    86,  257, 75,  701, 1,   261, 262, 263, 264, 265, 266, 82,  710, 11,  12,  13,  14,
    15,  16,  82,  326, 362, 720, 82,  365, 55,  56,  57,  76,  285, 343, 78,  185, 186,
    187, 188, 55,  56,  57,  380, 40,  41,  68,  61,  62,  79,  43,  44,  45,  46,  47,
    48,  49,  86,  86,  561, 78,  563, 309, 310, 38,  58,  396, 43,  44,  45,  46,  47,
    48,  49,  321, 322, 323, 324, 325, 78,  417, 74,  55,  58,  77,  78,  423, 420, 80,
    84,  83,  82,  82,  424, 425, 426, 79,  76,  74,  401, 1,   77,  87,  87,  441, 438,
    608, 87,  430, 79,  87,  51,  79,  77,  79,  430, 618, 430, 1,   87,  81,  79,  81,
    6,   7,   8,   9,   10,  78,  84,  432, 302, 303, 304, 436, 18,  19,  79,  87,  22,
    86,  88,  43,  44,  45,  46,  47,  48,  49,  87,  83,  88,  84,  87,  656, 407, 84,
    58,  81,  42,  84,  81,  84,  79,  78,  17,  83,  419, 83,  478, 83,  473, 42,  74,
    78,  78,  77,  78,  430, 6,   513, 79,  83,  57,  84,  437, 357, 358, 87,  491, 86,
    84,  84,  84,  84,  78,  83,  79,  87,  87,  86,  372, 373, 374, 375, 376, 87,  87,
    37,  542, 37,  38,  83,  86,  528, 548, 530, 55,  470, 77,  57,  77,  20,  55,  557,
    79,  77,  88,  55,  566, 78,  1,   75,  535, 61,  62,  63,  64,  65,  66,  67,  86,
    69,  70,  84,  559, 1,   580, 500, 582, 78,  87,  585, 567, 77,  86,  77,  56,  78,
    561, 87,  563, 78,  55,  76,  79,  75,  50,  601, 87,  584, 75,  586, 43,  44,  45,
    46,  47,  48,  49,  87,  78,  615, 620, 621, 1,   80,  78,  43,  44,  45,  46,  47,
    48,  49,  1,   76,  78,  80,  632, 78,  75,  87,  79,  74,  1,   608, 77,  78,  642,
    78,  76,  78,  5,   78,  87,  618, 78,  81,  74,  653, 79,  77,  78,  78,  78,  78,
    43,  44,  45,  46,  47,  48,  49,  70,  55,  669, 43,  44,  45,  46,  47,  48,  49,
    132, 597, 156, 43,  44,  45,  46,  47,  48,  49,  656, 190, 689, 1,   74,  75,  312,
    77,  664, 298, 177, 439, 307, 318, 74,  16,  70,  77,  396, 317, 231, 402, 70,  430,
    74,  399, 713, 77,  490, 331, 621, 583, 554, 535, 640, 641, 37,  38,  543, 645, 523,
    572, -1,  -1,  -1,  43,  44,  45,  46,  47,  48,  49,  -1,  -1,  55,  -1,  -1,  663,
    -1,  -1,  61,  62,  63,  64,  65,  66,  67,  -1,  -1,  -1,  676, 677, -1,  679, 680,
    -1,  74,  -1,  -1,  77,  1,   -1,  3,   4,   5,   -1,  7,   8,   9,   695, 11,  12,
    13,  14,  15,  -1,  -1,  -1,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,
    30,  -1,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,
    47,  48,  49,  50,  -1,  52,  -1,  54,  55,  -1,  -1,  -1,  -1,  -1,  61,  62,  63,
    64,  65,  66,  67,  68,  1,   -1,  71,  72,  73,  74,  -1,  1,   -1,  -1,  -1,  -1,
    -1,  -1,  1,   -1,  1,   86,  3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,
    14,  15,  16,  -1,  18,  19,  -1,  -1,  22,  -1,  -1,  6,   -1,  43,  44,  45,  46,
    47,  48,  49,  43,  44,  45,  46,  47,  48,  49,  43,  44,  45,  46,  47,  48,  49,
    1,   -1,  -1,  -1,  -1,  53,  54,  1,   37,  38,  74,  59,  60,  77,  1,   -1,  -1,
    74,  -1,  -1,  77,  1,   -1,  71,  74,  -1,  55,  77,  -1,  77,  78,  -1,  61,  62,
    63,  64,  65,  66,  67,  -1,  69,  70,  43,  44,  45,  46,  47,  48,  49,  43,  44,
    45,  46,  47,  48,  49,  43,  44,  45,  46,  47,  48,  49,  43,  44,  45,  46,  47,
    48,  49,  1,   -1,  -1,  74,  -1,  -1,  77,  -1,  -1,  -1,  74,  -1,  -1,  77,  -1,
    -1,  -1,  74,  -1,  -1,  77,  -1,  -1,  -1,  74,  -1,  -1,  77,  -1,  -1,  6,   7,
    8,   9,   -1,  11,  12,  13,  14,  15,  -1,  -1,  43,  44,  45,  46,  47,  48,  49,
    3,   4,   -1,  6,   7,   8,   9,   10,  11,  12,  13,  14,  15,  16,  1,   18,  19,
    42,  -1,  22,  -1,  -1,  -1,  -1,  74,  -1,  -1,  77,  -1,  -1,  -1,  -1,  -1,  37,
    21,  -1,  23,  24,  25,  26,  27,  28,  -1,  -1,  31,  32,  33,  34,  35,  36,  75,
    55,  -1,  78,  -1,  -1,  -1,  61,  62,  63,  64,  65,  66,  67,  51,  52,  -1,  1,
    -1,  -1,  -1,  -1,  -1,  -1,  77,  -1,  -1,  -1,  -1,  -1,  -1,  68,  -1,  -1,  -1,
    72,  73,  21,  -1,  23,  24,  25,  26,  27,  28,  -1,  -1,  31,  32,  33,  34,  35,
    36,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  51,  52,
    -1,  -1,  -1,  -1,  -1,  3,   4,   5,   -1,  7,   8,   9,   -1,  11,  12,  13,  14,
    15,  -1,  72,  73,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  -1,
    32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,  48,
    49,  50,  -1,  52,  -1,  54,  55,  -1,  -1,  -1,  -1,  -1,  61,  62,  63,  64,  65,
    66,  67,  68,  -1,  -1,  71,  72,  73,  74,  3,   4,   5,   -1,  7,   8,   9,   -1,
    11,  12,  13,  14,  15,  -1,  -1,  -1,  19,  20,  21,  22,  23,  24,  25,  26,  27,
    28,  29,  30,  -1,  32,  33,  34,  35,  36,  -1,  -1,  39,  40,  41,  42,  43,  44,
    45,  46,  47,  48,  49,  50,  51,  52,  -1,  54,  55,  56,  -1,  -1,  -1,  -1,  -1,
    -1,  -1,  -1,  -1,  -1,  -1,  68,  -1,  -1,  71,  72,  73,  74,  3,   4,   5,   -1,
    7,   8,   9,   -1,  11,  12,  13,  14,  15,  -1,  -1,  -1,  19,  20,  21,  22,  23,
    24,  25,  26,  27,  28,  29,  30,  -1,  32,  33,  34,  35,  36,  -1,  -1,  39,  40,
    41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  -1,  52,  -1,  54,  55,  56,  -1,
    -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  68,  -1,  -1,  71,  72,  73,  74,
    3,   4,   5,   -1,  7,   8,   9,   -1,  11,  12,  13,  14,  15,  -1,  -1,  -1,  19,
    20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  -1,  32,  33,  34,  35,  36,
    -1,  -1,  39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  -1,  52,  -1,
    54,  55,  56,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  68,  -1,  -1,
    71,  72,  73,  74,  3,   4,   5,   -1,  7,   8,   9,   -1,  11,  12,  13,  14,  15,
    -1,  -1,  -1,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  -1,  32,
    33,  34,  35,  36,  -1,  18,  39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,
    50,  -1,  52,  -1,  54,  55,  -1,  37,  38,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
    -1,  68,  -1,  -1,  71,  72,  73,  74,  55,  37,  38,  39,  -1,  -1,  61,  62,  63,
    64,  65,  66,  67,  -1,  37,  38,  -1,  -1,  -1,  55,  -1,  -1,  -1,  -1,  -1,  61,
    62,  63,  64,  65,  66,  67,  55,  56,  57,  37,  38,  39,  61,  62,  63,  64,  65,
    66,  67,  -1,  -1,  -1,  84,  37,  38,  -1,  -1,  55,  -1,  -1,  -1,  -1,  -1,  61,
    62,  63,  64,  65,  66,  67,  -1,  55,  37,  38,  -1,  -1,  -1,  61,  62,  63,  64,
    65,  66,  67,  -1,  -1,  -1,  -1,  -1,  -1,  55,  -1,  -1,  -1,  -1,  -1,  61,  62,
    63,  64,  65,  66,  67};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] = {
    0,   3,   4,   92,  93,  99,  3,   4,   5,   7,   8,   9,   11,  12,  13,  14,  15,
    19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  32,  33,  34,  35,  36,
    39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  52,  54,  55,  68,  71,
    72,  73,  74,  97,  97,  0,   93,  75,  77,  95,  100, 100, 1,   5,   53,  54,  59,
    60,  71,  94,  101, 102, 103, 169, 206, 207, 75,  42,  97,  53,  55,  98,  97,  97,
    77,  95,  178, 6,   7,   8,   9,   10,  11,  12,  13,  14,  15,  16,  18,  19,  22,
    77,  99,  122, 123, 139, 142, 143, 144, 146, 156, 157, 160, 161, 162, 78,  6,   7,
    8,   9,   11,  12,  13,  14,  15,  42,  78,  95,  167, 101, 37,  38,  61,  62,  63,
    64,  65,  66,  67,  98,  108, 110, 111, 112, 113, 116, 117, 170, 77,  98,  76,  58,
    83,  176, 16,  38,  111, 112, 113, 114, 115, 118, 37,  124, 124, 86,  124, 110, 163,
    86,  128, 128, 128, 128, 86,  132, 145, 86,  125, 97,  57,  164, 80,  101, 11,  12,
    13,  14,  15,  16,  147, 148, 149, 150, 151, 95,  96,  116, 62,  66,  61,  62,  63,
    64,  80,  107, 82,  82,  82,  38,  85,  110, 101, 76,  77,  83,  86,  176, 78,  179,
    111, 115, 38,  83,  85,  98,  98,  98,  68,  98,  79,  30,  51,  129, 134, 97,  109,
    109, 109, 109, 51,  56,  97,  131, 133, 86,  145, 86,  132, 40,  41,  126, 127, 109,
    18,  114, 118, 154, 155, 78,  128, 128, 128, 128, 145, 125, 61,  62,  56,  57,  104,
    105, 106, 118, 97,  78,  55,  176, 58,  175, 176, 84,  95,  82,  82,  86,  120, 121,
    204, 83,  80,  83,  87,  80,  83,  163, 87,  79,  107, 76,  140, 140, 140, 140, 97,
    87,  79,  87,  109, 109, 87,  79,  77,  97,  88,  153, 97,  79,  81,  96,  97,  97,
    97,  97,  97,  97,  79,  81,  107, 78,  79,  84,  87,  176, 98,  97,  121, 119, 176,
    124, 105, 124, 124, 105, 124, 129, 110, 141, 77,  95,  158, 158, 158, 158, 87,  133,
    140, 140, 126, 17,  78,  135, 137, 138, 88,  152, 56,  57,  110, 153, 155, 140, 140,
    140, 140, 140, 77,  95,  105, 83,  187, 176, 175, 176, 176, 121, 84,  87,  205, 84,
    81,  84,  98,  81,  84,  79,  1,   42,  156, 159, 160, 165, 166, 168, 158, 158, 118,
    138, 78,  118, 158, 158, 158, 158, 158, 138, 39,  84,  118, 177, 180, 185, 83,  83,
    83,  83,  141, 1,   86,  171, 168, 78,  95,  159, 97,  78,  118, 177, 97,  176, 79,
    84,  185, 124, 124, 124, 1,   21,  23,  24,  25,  26,  27,  28,  31,  32,  33,  34,
    35,  36,  51,  52,  68,  72,  73,  172, 173, 55,  97,  170, 96,  86,  136, 95,  97,
    176, 86,  88,  135, 87,  185, 84,  84,  84,  84,  57,  130, 87,  87,  79,  187, 97,
    87,  95,  87,  56,  57,  98,  174, 37,  172, 1,   43,  44,  45,  46,  47,  48,  49,
    74,  77,  95,  178, 190, 195, 197, 187, 97,  83,  201, 86,  201, 55,  202, 203, 77,
    57,  194, 201, 77,  191, 197, 176, 20,  189, 187, 176, 199, 55,  199, 187, 204, 79,
    77,  197, 192, 197, 178, 199, 43,  44,  45,  47,  48,  49,  74,  178, 193, 195, 196,
    78,  191, 179, 88,  190, 86,  188, 75,  87,  84,  200, 176, 203, 78,  191, 78,  191,
    176, 200, 201, 86,  201, 77,  194, 201, 77,  176, 78,  193, 96,  96,  56,  6,   69,
    70,  87,  118, 181, 184, 186, 178, 199, 201, 77,  197, 205, 78,  179, 77,  197, 176,
    55,  176, 192, 178, 176, 193, 179, 97,  76,  79,  87,  176, 75,  199, 191, 187, 96,
    191, 50,  198, 75,  87,  200, 78,  176, 200, 78,  96,  80,  118, 177, 183, 186, 179,
    199, 76,  78,  78,  77,  197, 176, 201, 77,  197, 179, 77,  197, 97,  182, 97,  176,
    80,  97,  200, 199, 198, 191, 75,  176, 191, 96,  191, 198, 81,  83,  86,  89,  90,
    80,  87,  182, 95,  77,  197, 79,  78,  176, 76,  78,  78,  182, 56,  182, 81,  97,
    182, 81,  191, 199, 200, 176, 198, 84,  87,  87,  97,  81,  78,  200, 77,  197, 79,
    77,  197, 191, 176, 191, 78,  200, 78,  77,  197, 191, 78};

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] = {
    0,   91,  92,  93,  93,  94,  94,  95,  95,  96,  96,  97,  97,  97,  97,  97,  97,
    97,  97,  97,  97,  97,  97,  97,  97,  97,  97,  97,  97,  97,  97,  97,  97,  97,
    97,  97,  97,  97,  97,  97,  97,  97,  97,  97,  97,  97,  97,  97,  97,  97,  97,
    97,  97,  97,  97,  97,  97,  97,  97,  98,  98,  99,  99,  100, 100, 101, 101, 102,
    102, 102, 102, 102, 103, 103, 103, 103, 103, 103, 103, 103, 103, 103, 103, 103, 103,
    103, 104, 104, 104, 105, 105, 106, 106, 107, 107, 108, 108, 108, 108, 108, 108, 108,
    108, 108, 108, 108, 108, 108, 108, 108, 109, 110, 111, 111, 112, 113, 113, 114, 115,
    115, 115, 115, 115, 115, 116, 116, 116, 116, 116, 117, 117, 118, 118, 119, 120, 121,
    121, 122, 123, 124, 124, 125, 125, 126, 126, 127, 127, 128, 128, 129, 129, 130, 130,
    131, 132, 132, 133, 133, 134, 134, 135, 135, 136, 136, 137, 138, 138, 139, 139, 139,
    140, 140, 141, 141, 142, 142, 143, 144, 145, 145, 146, 146, 147, 147, 148, 149, 150,
    151, 151, 152, 152, 153, 153, 153, 153, 154, 154, 154, 155, 155, 156, 157, 157, 157,
    157, 157, 158, 158, 159, 159, 160, 160, 160, 160, 160, 160, 160, 161, 161, 161, 161,
    161, 162, 162, 162, 162, 163, 163, 164, 165, 166, 166, 166, 166, 167, 167, 167, 167,
    167, 167, 167, 167, 167, 167, 167, 168, 168, 168, 169, 169, 170, 171, 171, 171, 172,
    172, 173, 173, 173, 173, 173, 173, 173, 173, 173, 173, 173, 173, 173, 173, 173, 173,
    173, 173, 174, 174, 174, 175, 175, 175, 176, 176, 176, 176, 176, 176, 177, 178, 179,
    180, 180, 180, 180, 180, 181, 181, 181, 182, 182, 182, 182, 182, 182, 183, 184, 184,
    184, 185, 185, 186, 186, 187, 187, 188, 188, 189, 189, 190, 190, 190, 191, 191, 192,
    192, 193, 193, 193, 194, 194, 195, 195, 195, 196, 196, 196, 196, 196, 196, 196, 196,
    196, 196, 196, 196, 197, 197, 197, 197, 197, 197, 197, 197, 197, 197, 197, 197, 197,
    197, 198, 198, 198, 199, 200, 201, 202, 202, 203, 203, 204, 205, 206, 207};

/* YYR2[YYN] -- Number of symbols on the right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] = {
    0,  2,  1, 0, 2, 0, 1, 1, 2, 0, 1,  1, 1,  1,  1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,
    1,  1,  1, 1, 1, 1, 1, 1, 1, 1, 1,  1, 1,  1,  1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,
    1,  1,  1, 1, 1, 1, 1, 1, 1, 1, 4,  3, 3,  1,  4, 0, 2, 3, 2, 2, 2, 7, 5,  5,  2,
    2,  2,  2, 2, 2, 2, 2, 1, 1, 1, 1,  1, 1,  1,  1, 3, 0, 1, 0, 3, 1, 1, 1,  1,  2,
    2,  3,  3, 2, 2, 2, 1, 1, 2, 1, 2,  2, 1,  1,  2, 2, 2, 8, 1, 1, 1, 1, 2,  2,  1,
    1,  1,  2, 2, 2, 1, 2, 1, 1, 3, 0,  2, 4,  6,  0, 1, 0, 3, 1, 3, 1, 1, 0,  3,  1,
    3,  0,  1, 1, 0, 3, 1, 3, 1, 1, 0,  1, 0,  2,  5, 1, 2, 3, 5, 6, 0, 2, 1,  3,  5,
    5,  5,  5, 4, 3, 6, 6, 5, 5, 5, 5,  5, 4,  7,  0, 2, 0, 2, 2, 2, 3, 2, 3,  1,  3,
    4,  2,  2, 2, 2, 2, 1, 4, 0, 2, 1,  1, 1,  1,  2, 2, 2, 3, 6, 9, 3, 6, 3,  6,  9,
    9,  1,  3, 1, 1, 1, 2, 2, 1, 1, 1,  1, 1,  1,  1, 1, 1, 1, 1, 1, 7, 5, 13, 5,  2,
    1,  0,  3, 1, 1, 3, 1, 1, 1, 1, 1,  1, 1,  1,  1, 1, 1, 1, 1, 2, 1, 1, 1,  1,  1,
    1,  1,  0, 1, 3, 0, 1, 5, 5, 5, 4,  3, 1,  1,  1, 3, 4, 3, 4, 1, 1, 1, 1,  4,  3,
    4,  4,  4, 3, 7, 5, 6, 1, 3, 1, 3,  3, 2,  3,  2, 0, 3, 1, 1, 4, 1, 2, 1,  2,  1,
    2,  1,  1, 0, 4, 3, 5, 6, 4, 4, 11, 9, 12, 14, 6, 8, 5, 7, 4, 6, 4, 1, 4,  11, 9,
    12, 14, 6, 8, 5, 7, 4, 1, 0, 2, 4,  1, 1,  1,  2, 5, 1, 3, 1, 1, 2, 2};

#define yyerrok (yyerrstatus = 0)
#define yyclearin (yychar = YYEMPTY)
#define YYEMPTY (-2)
#define YYEOF 0

#define YYACCEPT goto yyacceptlab
#define YYABORT goto yyabortlab
#define YYERROR goto yyerrorlab

#define YYRECOVERING() (!!yyerrstatus)

#define YYBACKUP(Token, Value)                      \
  do                                                \
    if (yychar == YYEMPTY) {                        \
      yychar = (Token);                             \
      yylval = (Value);                             \
      YYPOPSTACK(yylen);                            \
      yystate = *yyssp;                             \
      goto yybackup;                                \
    } else {                                        \
      yyerror(YY_("syntax error: cannot back up")); \
      YYERROR;                                      \
    }                                               \
  while (0)

/* Error token number */
#define YYTERROR 1
#define YYERRCODE 256

/* YYLLOC_DEFAULT -- Set CURRENT to span from RHS[1] to RHS[N].
   If N is 0, then set CURRENT to the empty location which ends
   the previous symbol: RHS[0] (always defined).  */

#ifndef YYLLOC_DEFAULT
#define YYLLOC_DEFAULT(Current, Rhs, N)                                              \
  do                                                                                 \
    if (N) {                                                                         \
      (Current).first_line = YYRHSLOC(Rhs, 1).first_line;                            \
      (Current).first_column = YYRHSLOC(Rhs, 1).first_column;                        \
      (Current).last_line = YYRHSLOC(Rhs, N).last_line;                              \
      (Current).last_column = YYRHSLOC(Rhs, N).last_column;                          \
    } else {                                                                         \
      (Current).first_line = (Current).last_line = YYRHSLOC(Rhs, 0).last_line;       \
      (Current).first_column = (Current).last_column = YYRHSLOC(Rhs, 0).last_column; \
    }                                                                                \
  while (0)
#endif

#define YYRHSLOC(Rhs, K) ((Rhs)[K])

/* Enable debugging if requested.  */
#if YYDEBUG

#ifndef YYFPRINTF
#include <stdio.h> /* INFRINGES ON USER NAME SPACE */
#define YYFPRINTF fprintf
#endif

#define YYDPRINTF(Args)          \
  do {                           \
    if (yydebug) YYFPRINTF Args; \
  } while (0)

/* YY_LOCATION_PRINT -- Print the location on the stream.
   This macro was not mandated originally: define only if we know
   we won't break user code: when these are the locations we know.  */

#ifndef YY_LOCATION_PRINT
#if defined YYLTYPE_IS_TRIVIAL && YYLTYPE_IS_TRIVIAL

/* Print *YYLOCP on YYO.  Private, do not rely on its existence. */

YY_ATTRIBUTE_UNUSED
static unsigned yy_location_print_(FILE* yyo, YYLTYPE const* const yylocp) {
  unsigned res = 0;
  int end_col = 0 != yylocp->last_column ? yylocp->last_column - 1 : 0;
  if (0 <= yylocp->first_line) {
    res += YYFPRINTF(yyo, "%d", yylocp->first_line);
    if (0 <= yylocp->first_column) res += YYFPRINTF(yyo, ".%d", yylocp->first_column);
  }
  if (0 <= yylocp->last_line) {
    if (yylocp->first_line < yylocp->last_line) {
      res += YYFPRINTF(yyo, "-%d", yylocp->last_line);
      if (0 <= end_col) res += YYFPRINTF(yyo, ".%d", end_col);
    } else if (0 <= end_col && yylocp->first_column < end_col)
      res += YYFPRINTF(yyo, "-%d", end_col);
  }
  return res;
}

#define YY_LOCATION_PRINT(File, Loc) yy_location_print_(File, &(Loc))

#else
#define YY_LOCATION_PRINT(File, Loc) ((void)0)
#endif
#endif

#define YY_SYMBOL_PRINT(Title, Type, Value, Location) \
  do {                                                \
    if (yydebug) {                                    \
      YYFPRINTF(stderr, "%s ", Title);                \
      yy_symbol_print(stderr, Type, Value, Location); \
      YYFPRINTF(stderr, "\n");                        \
    }                                                 \
  } while (0)

/*----------------------------------------.
| Print this symbol's value on YYOUTPUT.  |
`----------------------------------------*/

static void yy_symbol_value_print(FILE* yyoutput, int yytype,
                                  YYSTYPE const* const yyvaluep,
                                  YYLTYPE const* const yylocationp) {
  FILE* yyo = yyoutput;
  YYUSE(yyo);
  YYUSE(yylocationp);
  if (!yyvaluep) return;
#ifdef YYPRINT
  if (yytype < YYNTOKENS) YYPRINT(yyoutput, yytoknum[yytype], *yyvaluep);
#endif
  YYUSE(yytype);
}

/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

static void yy_symbol_print(FILE* yyoutput, int yytype, YYSTYPE const* const yyvaluep,
                            YYLTYPE const* const yylocationp) {
  YYFPRINTF(yyoutput, "%s %s (", yytype < YYNTOKENS ? "token" : "nterm", yytname[yytype]);

  YY_LOCATION_PRINT(yyoutput, *yylocationp);
  YYFPRINTF(yyoutput, ": ");
  yy_symbol_value_print(yyoutput, yytype, yyvaluep, yylocationp);
  YYFPRINTF(yyoutput, ")");
}

/*------------------------------------------------------------------.
| yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

static void yy_stack_print(yytype_int16* yybottom, yytype_int16* yytop) {
  YYFPRINTF(stderr, "Stack now");
  for (; yybottom <= yytop; yybottom++) {
    int yybot = *yybottom;
    YYFPRINTF(stderr, " %d", yybot);
  }
  YYFPRINTF(stderr, "\n");
}

#define YY_STACK_PRINT(Bottom, Top)               \
  do {                                            \
    if (yydebug) yy_stack_print((Bottom), (Top)); \
  } while (0)

/*------------------------------------------------.
| Report that the YYRULE is going to be reduced.  |
`------------------------------------------------*/

static void yy_reduce_print(yytype_int16* yyssp, YYSTYPE* yyvsp, YYLTYPE* yylsp,
                            int yyrule) {
  unsigned long int yylno = yyrline[yyrule];
  int yynrhs = yyr2[yyrule];
  int yyi;
  YYFPRINTF(stderr, "Reducing stack by rule %d (line %lu):\n", yyrule - 1, yylno);
  /* The symbols being reduced.  */
  for (yyi = 0; yyi < yynrhs; yyi++) {
    YYFPRINTF(stderr, "   $%d = ", yyi + 1);
    yy_symbol_print(stderr, yystos[yyssp[yyi + 1 - yynrhs]],
                    &(yyvsp[(yyi + 1) - (yynrhs)]), &(yylsp[(yyi + 1) - (yynrhs)]));
    YYFPRINTF(stderr, "\n");
  }
}

#define YY_REDUCE_PRINT(Rule)                                \
  do {                                                       \
    if (yydebug) yy_reduce_print(yyssp, yyvsp, yylsp, Rule); \
  } while (0)

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int yydebug;
#else /* !YYDEBUG */
#define YYDPRINTF(Args)
#define YY_SYMBOL_PRINT(Title, Type, Value, Location)
#define YY_STACK_PRINT(Bottom, Top)
#define YY_REDUCE_PRINT(Rule)
#endif /* !YYDEBUG */

/* YYINITDEPTH -- initial size of the parser's stacks.  */
#ifndef YYINITDEPTH
#define YYINITDEPTH 200
#endif

/* YYMAXDEPTH -- maximum size the stacks can grow to (effective only
   if the built-in stack extension method is used).

   Do not make this value too large; the results are undefined if
   YYSTACK_ALLOC_MAXIMUM < YYSTACK_BYTES (YYMAXDEPTH)
   evaluated with infinite-precision integer arithmetic.  */

#ifndef YYMAXDEPTH
#define YYMAXDEPTH 10000
#endif

#if YYERROR_VERBOSE

#ifndef yystrlen
#if defined __GLIBC__ && defined _STRING_H
#define yystrlen strlen
#else
/* Return the length of YYSTR.  */
static YYSIZE_T yystrlen(const char* yystr) {
  YYSIZE_T yylen;
  for (yylen = 0; yystr[yylen]; yylen++) continue;
  return yylen;
}
#endif
#endif

#ifndef yystpcpy
#if defined __GLIBC__ && defined _STRING_H && defined _GNU_SOURCE
#define yystpcpy stpcpy
#else
/* Copy YYSRC to YYDEST, returning the address of the terminating '\0' in
   YYDEST.  */
static char* yystpcpy(char* yydest, const char* yysrc) {
  char* yyd = yydest;
  const char* yys = yysrc;

  while ((*yyd++ = *yys++) != '\0') continue;

  return yyd - 1;
}
#endif
#endif

#ifndef yytnamerr
/* Copy to YYRES the contents of YYSTR after stripping away unnecessary
   quotes and backslashes, so that it's suitable for yyerror.  The
   heuristic is that double-quoting is unnecessary unless the string
   contains an apostrophe, a comma, or backslash (other than
   backslash-backslash).  YYSTR is taken from yytname.  If YYRES is
   null, do not copy; instead, return the length of what the result
   would have been.  */
static YYSIZE_T yytnamerr(char* yyres, const char* yystr) {
  if (*yystr == '"') {
    YYSIZE_T yyn = 0;
    char const* yyp = yystr;

    for (;;) switch (*++yyp) {
        case '\'':
        case ',':
          goto do_not_strip_quotes;

        case '\\':
          if (*++yyp != '\\') goto do_not_strip_quotes;
        /* Fall through.  */
        default:
          if (yyres) yyres[yyn] = *yyp;
          yyn++;
          break;

        case '"':
          if (yyres) yyres[yyn] = '\0';
          return yyn;
      }
  do_not_strip_quotes:;
  }

  if (!yyres) return yystrlen(yystr);

  return yystpcpy(yyres, yystr) - yyres;
}
#endif

/* Copy into *YYMSG, which is of size *YYMSG_ALLOC, an error message
   about the unexpected token YYTOKEN for the state stack whose top is
   YYSSP.

   Return 0 if *YYMSG was successfully written.  Return 1 if *YYMSG is
   not large enough to hold the message.  In that case, also set
   *YYMSG_ALLOC to the required number of bytes.  Return 2 if the
   required number of bytes is too large to store.  */
static int yysyntax_error(YYSIZE_T* yymsg_alloc, char** yymsg, yytype_int16* yyssp,
                          int yytoken) {
  YYSIZE_T yysize0 = yytnamerr(YY_NULLPTR, yytname[yytoken]);
  YYSIZE_T yysize = yysize0;
  enum { YYERROR_VERBOSE_ARGS_MAXIMUM = 5 };
  /* Internationalized format string. */
  const char* yyformat = YY_NULLPTR;
  /* Arguments of yyformat. */
  char const* yyarg[YYERROR_VERBOSE_ARGS_MAXIMUM];
  /* Number of reported tokens (one for the "unexpected", one per
     "expected"). */
  int yycount = 0;

  /* There are many possibilities here to consider:
     - If this state is a consistent state with a default action, then
       the only way this function was invoked is if the default action
       is an error action.  In that case, don't check for expected
       tokens because there are none.
     - The only way there can be no lookahead present (in yychar) is if
       this state is a consistent state with a default action.  Thus,
       detecting the absence of a lookahead is sufficient to determine
       that there is no unexpected or expected token to report.  In that
       case, just report a simple "syntax error".
     - Don't assume there isn't a lookahead just because this state is a
       consistent state with a default action.  There might have been a
       previous inconsistent state, consistent state with a non-default
       action, or user semantic action that manipulated yychar.
     - Of course, the expected token list depends on states to have
       correct lookahead information, and it depends on the parser not
       to perform extra reductions after fetching a lookahead from the
       scanner and before detecting a syntax error.  Thus, state merging
       (from LALR or IELR) and default reductions corrupt the expected
       token list.  However, the list is correct for canonical LR with
       one exception: it will still contain any token that will not be
       accepted due to an error action in a later state.
  */
  if (yytoken != YYEMPTY) {
    int yyn = yypact[*yyssp];
    yyarg[yycount++] = yytname[yytoken];
    if (!yypact_value_is_default(yyn)) {
      /* Start YYX at -YYN if negative to avoid negative indexes in
         YYCHECK.  In other words, skip the first -YYN actions for
         this state because they are default actions.  */
      int yyxbegin = yyn < 0 ? -yyn : 0;
      /* Stay within bounds of both yycheck and yytname.  */
      int yychecklim = YYLAST - yyn + 1;
      int yyxend = yychecklim < YYNTOKENS ? yychecklim : YYNTOKENS;
      int yyx;

      for (yyx = yyxbegin; yyx < yyxend; ++yyx)
        if (yycheck[yyx + yyn] == yyx && yyx != YYTERROR &&
            !yytable_value_is_error(yytable[yyx + yyn])) {
          if (yycount == YYERROR_VERBOSE_ARGS_MAXIMUM) {
            yycount = 1;
            yysize = yysize0;
            break;
          }
          yyarg[yycount++] = yytname[yyx];
          {
            YYSIZE_T yysize1 = yysize + yytnamerr(YY_NULLPTR, yytname[yyx]);
            if (!(yysize <= yysize1 && yysize1 <= YYSTACK_ALLOC_MAXIMUM)) return 2;
            yysize = yysize1;
          }
        }
    }
  }

  switch (yycount) {
#define YYCASE_(N, S) \
  case N:             \
    yyformat = S;     \
    break
    YYCASE_(0, YY_("syntax error"));
    YYCASE_(1, YY_("syntax error, unexpected %s"));
    YYCASE_(2, YY_("syntax error, unexpected %s, expecting %s"));
    YYCASE_(3, YY_("syntax error, unexpected %s, expecting %s or %s"));
    YYCASE_(4, YY_("syntax error, unexpected %s, expecting %s or %s or %s"));
    YYCASE_(5, YY_("syntax error, unexpected %s, expecting %s or %s or %s or %s"));
#undef YYCASE_
  }

  {
    YYSIZE_T yysize1 = yysize + yystrlen(yyformat);
    if (!(yysize <= yysize1 && yysize1 <= YYSTACK_ALLOC_MAXIMUM)) return 2;
    yysize = yysize1;
  }

  if (*yymsg_alloc < yysize) {
    *yymsg_alloc = 2 * yysize;
    if (!(yysize <= *yymsg_alloc && *yymsg_alloc <= YYSTACK_ALLOC_MAXIMUM))
      *yymsg_alloc = YYSTACK_ALLOC_MAXIMUM;
    return 1;
  }

  /* Avoid sprintf, as that infringes on the user's name space.
     Don't have undefined behavior even if the translation
     produced a string with the wrong number of "%s"s.  */
  {
    char* yyp = *yymsg;
    int yyi = 0;
    while ((*yyp = *yyformat) != '\0')
      if (*yyp == '%' && yyformat[1] == 's' && yyi < yycount) {
        yyp += yytnamerr(yyp, yyarg[yyi++]);
        yyformat += 2;
      } else {
        yyp++;
        yyformat++;
      }
  }
  return 0;
}
#endif /* YYERROR_VERBOSE */

/*-----------------------------------------------.
| Release the memory associated to this symbol.  |
`-----------------------------------------------*/

static void yydestruct(const char* yymsg, int yytype, YYSTYPE* yyvaluep,
                       YYLTYPE* yylocationp) {
  YYUSE(yyvaluep);
  YYUSE(yylocationp);
  if (!yymsg) yymsg = "Deleting";
  YY_SYMBOL_PRINT(yymsg, yytype, yyvaluep, yylocationp);

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  YYUSE(yytype);
  YY_IGNORE_MAYBE_UNINITIALIZED_END
}

/* The lookahead symbol.  */
int yychar;

/* The semantic value of the lookahead symbol.  */
YYSTYPE yylval;
/* Location data for the lookahead symbol.  */
YYLTYPE yylloc
#if defined YYLTYPE_IS_TRIVIAL && YYLTYPE_IS_TRIVIAL
    = {1, 1, 1, 1}
#endif
;
/* Number of syntax errors so far.  */
int yynerrs;

/*----------.
| yyparse.  |
`----------*/

int yyparse(void) {
  int yystate;
  /* Number of tokens to shift before error messages enabled.  */
  int yyerrstatus;

  /* The stacks and their tools:
     'yyss': related to states.
     'yyvs': related to semantic values.
     'yyls': related to locations.

     Refer to the stacks through separate pointers, to allow yyoverflow
     to reallocate them elsewhere.  */

  /* The state stack.  */
  yytype_int16 yyssa[YYINITDEPTH];
  yytype_int16* yyss;
  yytype_int16* yyssp;

  /* The semantic value stack.  */
  YYSTYPE yyvsa[YYINITDEPTH];
  YYSTYPE* yyvs;
  YYSTYPE* yyvsp;

  /* The location stack.  */
  YYLTYPE yylsa[YYINITDEPTH];
  YYLTYPE* yyls;
  YYLTYPE* yylsp;

  /* The locations where the error started and ended.  */
  YYLTYPE yyerror_range[3];

  YYSIZE_T yystacksize;

  int yyn;
  int yyresult;
  /* Lookahead token as an internal (translated) token number.  */
  int yytoken = 0;
  /* The variables used to return semantic value and location from the
     action routines.  */
  YYSTYPE yyval;
  YYLTYPE yyloc;

#if YYERROR_VERBOSE
  /* Buffer for error messages, and its allocated size.  */
  char yymsgbuf[128];
  char* yymsg = yymsgbuf;
  YYSIZE_T yymsg_alloc = sizeof yymsgbuf;
#endif

#define YYPOPSTACK(N) (yyvsp -= (N), yyssp -= (N), yylsp -= (N))

  /* The number of symbols on the RHS of the reduced rule.
     Keep to zero when no symbol should be popped.  */
  int yylen = 0;

  yyssp = yyss = yyssa;
  yyvsp = yyvs = yyvsa;
  yylsp = yyls = yylsa;
  yystacksize = YYINITDEPTH;

  YYDPRINTF((stderr, "Starting parse\n"));

  yystate = 0;
  yyerrstatus = 0;
  yynerrs = 0;
  yychar = YYEMPTY; /* Cause a token to be read.  */
  yylsp[0] = yylloc;
  goto yysetstate;

/*------------------------------------------------------------.
| yynewstate -- Push a new state, which is found in yystate.  |
`------------------------------------------------------------*/
yynewstate:
  /* In all cases, when you get here, the value and location stacks
     have just been pushed.  So pushing a state here evens the stacks.  */
  yyssp++;

yysetstate:
  *yyssp = yystate;

  if (yyss + yystacksize - 1 <= yyssp) {
    /* Get the current used size of the three stacks, in elements.  */
    YYSIZE_T yysize = yyssp - yyss + 1;

#ifdef yyoverflow
    {
      /* Give user a chance to reallocate the stack.  Use copies of
         these so that the &'s don't force the real ones into
         memory.  */
      YYSTYPE* yyvs1 = yyvs;
      yytype_int16* yyss1 = yyss;
      YYLTYPE* yyls1 = yyls;

      /* Each stack pointer address is followed by the size of the
         data in use in that stack, in bytes.  This used to be a
         conditional around just the two extra args, but that might
         be undefined if yyoverflow is a macro.  */
      yyoverflow(YY_("memory exhausted"), &yyss1, yysize * sizeof(*yyssp), &yyvs1,
                 yysize * sizeof(*yyvsp), &yyls1, yysize * sizeof(*yylsp), &yystacksize);

      yyls = yyls1;
      yyss = yyss1;
      yyvs = yyvs1;
    }
#else /* no yyoverflow */
#ifndef YYSTACK_RELOCATE
    goto yyexhaustedlab;
#else
    /* Extend the stack our own way.  */
    if (YYMAXDEPTH <= yystacksize) goto yyexhaustedlab;
    yystacksize *= 2;
    if (YYMAXDEPTH < yystacksize) yystacksize = YYMAXDEPTH;

    {
      yytype_int16* yyss1 = yyss;
      union yyalloc* yyptr = (union yyalloc*)YYSTACK_ALLOC(YYSTACK_BYTES(yystacksize));
      if (!yyptr) goto yyexhaustedlab;
      YYSTACK_RELOCATE(yyss_alloc, yyss);
      YYSTACK_RELOCATE(yyvs_alloc, yyvs);
      YYSTACK_RELOCATE(yyls_alloc, yyls);
#undef YYSTACK_RELOCATE
      if (yyss1 != yyssa) YYSTACK_FREE(yyss1);
    }
#endif
#endif /* no yyoverflow */

    yyssp = yyss + yysize - 1;
    yyvsp = yyvs + yysize - 1;
    yylsp = yyls + yysize - 1;

    YYDPRINTF((stderr, "Stack size increased to %lu\n", (unsigned long int)yystacksize));

    if (yyss + yystacksize - 1 <= yyssp) YYABORT;
  }

  YYDPRINTF((stderr, "Entering state %d\n", yystate));

  if (yystate == YYFINAL) YYACCEPT;

  goto yybackup;

/*-----------.
| yybackup.  |
`-----------*/
yybackup:

  /* Do appropriate processing given the current state.  Read a
     lookahead token if we need one and don't already have one.  */

  /* First try to decide what to do without reference to lookahead token.  */
  yyn = yypact[yystate];
  if (yypact_value_is_default(yyn)) goto yydefault;

  /* Not known => get a lookahead token if don't already have one.  */

  /* YYCHAR is either YYEMPTY or YYEOF or a valid lookahead symbol.  */
  if (yychar == YYEMPTY) {
    YYDPRINTF((stderr, "Reading a token: "));
    yychar = yylex();
  }

  if (yychar <= YYEOF) {
    yychar = yytoken = YYEOF;
    YYDPRINTF((stderr, "Now at end of input.\n"));
  } else {
    yytoken = YYTRANSLATE(yychar);
    YY_SYMBOL_PRINT("Next token is", yytoken, &yylval, &yylloc);
  }

  /* If the proper action on seeing token YYTOKEN is to reduce or to
     detect an error, take that action.  */
  yyn += yytoken;
  if (yyn < 0 || YYLAST < yyn || yycheck[yyn] != yytoken) goto yydefault;
  yyn = yytable[yyn];
  if (yyn <= 0) {
    if (yytable_value_is_error(yyn)) goto yyerrlab;
    yyn = -yyn;
    goto yyreduce;
  }

  /* Count tokens shifted since error; after three, turn off error
     status.  */
  if (yyerrstatus) yyerrstatus--;

  /* Shift the lookahead token.  */
  YY_SYMBOL_PRINT("Shifting", yytoken, &yylval, &yylloc);

  /* Discard the shifted token.  */
  yychar = YYEMPTY;

  yystate = yyn;
  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END
  *++yylsp = yylloc;
  goto yynewstate;

/*-----------------------------------------------------------.
| yydefault -- do the default action for the current state.  |
`-----------------------------------------------------------*/
yydefault:
  yyn = yydefact[yystate];
  if (yyn == 0) goto yyerrlab;
  goto yyreduce;

/*-----------------------------.
| yyreduce -- Do a reduction.  |
`-----------------------------*/
yyreduce:
  /* yyn is the number of a rule to reduce with.  */
  yylen = yyr2[yyn];

  /* If YYLEN is nonzero, implement the default value of the action:
     '$$ = $1'.

     Otherwise, the following line sets YYVAL to garbage.
     This behavior is undocumented and Bison
     users should not rely upon it.  Assigning to YYVAL
     unconditionally makes the parser a bit smaller, and it avoids a
     GCC warning that YYVAL may be used uninitialized.  */
  yyval = yyvsp[1 - yylen];

  /* Default location.  */
  YYLLOC_DEFAULT(yyloc, (yylsp - yylen), yylen);
  YY_REDUCE_PRINT(yyn);
  switch (yyn) {
    case 2:
#line 196 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.modlist) = (yyvsp[0].modlist);
      modlist = (yyvsp[0].modlist);
    }
#line 2242 "y.tab.c" /* yacc.c:1646  */
    break;

    case 3:
#line 200 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.modlist) = 0;
    }
#line 2250 "y.tab.c" /* yacc.c:1646  */
    break;

    case 4:
#line 204 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.modlist) =
          new AstChildren<Module>(lineno, (yyvsp[-1].module), (yyvsp[0].modlist));
    }
#line 2256 "y.tab.c" /* yacc.c:1646  */
    break;

    case 5:
#line 208 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.intval) = 0;
    }
#line 2262 "y.tab.c" /* yacc.c:1646  */
    break;

    case 6:
#line 210 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.intval) = 1;
    }
#line 2268 "y.tab.c" /* yacc.c:1646  */
    break;

    case 7:
#line 214 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.intval) = 1;
    }
#line 2274 "y.tab.c" /* yacc.c:1646  */
    break;

    case 8:
#line 216 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.intval) = 2;
    }
#line 2280 "y.tab.c" /* yacc.c:1646  */
    break;

    case 9:
#line 220 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.intval) = 0;
    }
#line 2286 "y.tab.c" /* yacc.c:1646  */
    break;

    case 10:
#line 222 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.intval) = 1;
    }
#line 2292 "y.tab.c" /* yacc.c:1646  */
    break;

    case 11:
#line 227 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.strval) = (yyvsp[0].strval);
    }
#line 2298 "y.tab.c" /* yacc.c:1646  */
    break;

    case 12:
#line 228 "xi-grammar.y" /* yacc.c:1646  */
    {
      ReservedWord(MODULE, (yyloc).first_column, (yyloc).last_column);
      YYABORT;
    }
#line 2304 "y.tab.c" /* yacc.c:1646  */
    break;

    case 13:
#line 229 "xi-grammar.y" /* yacc.c:1646  */
    {
      ReservedWord(MAINMODULE, (yyloc).first_column, (yyloc).last_column);
      YYABORT;
    }
#line 2310 "y.tab.c" /* yacc.c:1646  */
    break;

    case 14:
#line 230 "xi-grammar.y" /* yacc.c:1646  */
    {
      ReservedWord(EXTERN, (yyloc).first_column, (yyloc).last_column);
      YYABORT;
    }
#line 2316 "y.tab.c" /* yacc.c:1646  */
    break;

    case 15:
#line 232 "xi-grammar.y" /* yacc.c:1646  */
    {
      ReservedWord(INITCALL, (yyloc).first_column, (yyloc).last_column);
      YYABORT;
    }
#line 2322 "y.tab.c" /* yacc.c:1646  */
    break;

    case 16:
#line 233 "xi-grammar.y" /* yacc.c:1646  */
    {
      ReservedWord(INITNODE, (yyloc).first_column, (yyloc).last_column);
      YYABORT;
    }
#line 2328 "y.tab.c" /* yacc.c:1646  */
    break;

    case 17:
#line 234 "xi-grammar.y" /* yacc.c:1646  */
    {
      ReservedWord(INITPROC, (yyloc).first_column, (yyloc).last_column);
      YYABORT;
    }
#line 2334 "y.tab.c" /* yacc.c:1646  */
    break;

    case 18:
#line 236 "xi-grammar.y" /* yacc.c:1646  */
    {
      ReservedWord(CHARE, (yyloc).first_column, (yyloc).last_column);
      YYABORT;
    }
#line 2340 "y.tab.c" /* yacc.c:1646  */
    break;

    case 19:
#line 237 "xi-grammar.y" /* yacc.c:1646  */
    {
      ReservedWord(MAINCHARE, (yyloc).first_column, (yyloc).last_column);
      YYABORT;
    }
#line 2346 "y.tab.c" /* yacc.c:1646  */
    break;

    case 20:
#line 238 "xi-grammar.y" /* yacc.c:1646  */
    {
      ReservedWord(GROUP, (yyloc).first_column, (yyloc).last_column);
      YYABORT;
    }
#line 2352 "y.tab.c" /* yacc.c:1646  */
    break;

    case 21:
#line 239 "xi-grammar.y" /* yacc.c:1646  */
    {
      ReservedWord(NODEGROUP, (yyloc).first_column, (yyloc).last_column);
      YYABORT;
    }
#line 2358 "y.tab.c" /* yacc.c:1646  */
    break;

    case 22:
#line 240 "xi-grammar.y" /* yacc.c:1646  */
    {
      ReservedWord(ARRAY, (yyloc).first_column, (yyloc).last_column);
      YYABORT;
    }
#line 2364 "y.tab.c" /* yacc.c:1646  */
    break;

    case 23:
#line 244 "xi-grammar.y" /* yacc.c:1646  */
    {
      ReservedWord(INCLUDE, (yyloc).first_column, (yyloc).last_column);
      YYABORT;
    }
#line 2370 "y.tab.c" /* yacc.c:1646  */
    break;

    case 24:
#line 245 "xi-grammar.y" /* yacc.c:1646  */
    {
      ReservedWord(STACKSIZE, (yyloc).first_column, (yyloc).last_column);
      YYABORT;
    }
#line 2376 "y.tab.c" /* yacc.c:1646  */
    break;

    case 25:
#line 246 "xi-grammar.y" /* yacc.c:1646  */
    {
      ReservedWord(THREADED, (yyloc).first_column, (yyloc).last_column);
      YYABORT;
    }
#line 2382 "y.tab.c" /* yacc.c:1646  */
    break;

    case 26:
#line 247 "xi-grammar.y" /* yacc.c:1646  */
    {
      ReservedWord(TEMPLATE, (yyloc).first_column, (yyloc).last_column);
      YYABORT;
    }
#line 2388 "y.tab.c" /* yacc.c:1646  */
    break;

    case 27:
#line 248 "xi-grammar.y" /* yacc.c:1646  */
    {
      ReservedWord(SYNC, (yyloc).first_column, (yyloc).last_column);
      YYABORT;
    }
#line 2394 "y.tab.c" /* yacc.c:1646  */
    break;

    case 28:
#line 249 "xi-grammar.y" /* yacc.c:1646  */
    {
      ReservedWord(IGET, (yyloc).first_column, (yyloc).last_column);
      YYABORT;
    }
#line 2400 "y.tab.c" /* yacc.c:1646  */
    break;

    case 29:
#line 250 "xi-grammar.y" /* yacc.c:1646  */
    {
      ReservedWord(EXCLUSIVE, (yyloc).first_column, (yyloc).last_column);
      YYABORT;
    }
#line 2406 "y.tab.c" /* yacc.c:1646  */
    break;

    case 30:
#line 251 "xi-grammar.y" /* yacc.c:1646  */
    {
      ReservedWord(IMMEDIATE, (yyloc).first_column, (yyloc).last_column);
      YYABORT;
    }
#line 2412 "y.tab.c" /* yacc.c:1646  */
    break;

    case 31:
#line 252 "xi-grammar.y" /* yacc.c:1646  */
    {
      ReservedWord(SKIPSCHED, (yyloc).first_column, (yyloc).last_column);
      YYABORT;
    }
#line 2418 "y.tab.c" /* yacc.c:1646  */
    break;

    case 32:
#line 253 "xi-grammar.y" /* yacc.c:1646  */
    {
      ReservedWord(NOCOPY, (yyloc).first_column, (yyloc).last_column);
      YYABORT;
    }
#line 2424 "y.tab.c" /* yacc.c:1646  */
    break;

    case 33:
#line 254 "xi-grammar.y" /* yacc.c:1646  */
    {
      ReservedWord(INLINE, (yyloc).first_column, (yyloc).last_column);
      YYABORT;
    }
#line 2430 "y.tab.c" /* yacc.c:1646  */
    break;

    case 34:
#line 255 "xi-grammar.y" /* yacc.c:1646  */
    {
      ReservedWord(VIRTUAL, (yyloc).first_column, (yyloc).last_column);
      YYABORT;
    }
#line 2436 "y.tab.c" /* yacc.c:1646  */
    break;

    case 35:
#line 256 "xi-grammar.y" /* yacc.c:1646  */
    {
      ReservedWord(MIGRATABLE, (yyloc).first_column, (yyloc).last_column);
      YYABORT;
    }
#line 2442 "y.tab.c" /* yacc.c:1646  */
    break;

    case 36:
#line 257 "xi-grammar.y" /* yacc.c:1646  */
    {
      ReservedWord(CREATEHERE, (yyloc).first_column, (yyloc).last_column);
      YYABORT;
    }
#line 2448 "y.tab.c" /* yacc.c:1646  */
    break;

    case 37:
#line 258 "xi-grammar.y" /* yacc.c:1646  */
    {
      ReservedWord(CREATEHOME, (yyloc).first_column, (yyloc).last_column);
      YYABORT;
    }
#line 2454 "y.tab.c" /* yacc.c:1646  */
    break;

    case 38:
#line 259 "xi-grammar.y" /* yacc.c:1646  */
    {
      ReservedWord(NOKEEP, (yyloc).first_column, (yyloc).last_column);
      YYABORT;
    }
#line 2460 "y.tab.c" /* yacc.c:1646  */
    break;

    case 39:
#line 260 "xi-grammar.y" /* yacc.c:1646  */
    {
      ReservedWord(NOTRACE, (yyloc).first_column, (yyloc).last_column);
      YYABORT;
    }
#line 2466 "y.tab.c" /* yacc.c:1646  */
    break;

    case 40:
#line 261 "xi-grammar.y" /* yacc.c:1646  */
    {
      ReservedWord(APPWORK, (yyloc).first_column, (yyloc).last_column);
      YYABORT;
    }
#line 2472 "y.tab.c" /* yacc.c:1646  */
    break;

    case 41:
#line 264 "xi-grammar.y" /* yacc.c:1646  */
    {
      ReservedWord(PACKED, (yyloc).first_column, (yyloc).last_column);
      YYABORT;
    }
#line 2478 "y.tab.c" /* yacc.c:1646  */
    break;

    case 42:
#line 265 "xi-grammar.y" /* yacc.c:1646  */
    {
      ReservedWord(VARSIZE, (yyloc).first_column, (yyloc).last_column);
      YYABORT;
    }
#line 2484 "y.tab.c" /* yacc.c:1646  */
    break;

    case 43:
#line 266 "xi-grammar.y" /* yacc.c:1646  */
    {
      ReservedWord(ENTRY, (yyloc).first_column, (yyloc).last_column);
      YYABORT;
    }
#line 2490 "y.tab.c" /* yacc.c:1646  */
    break;

    case 44:
#line 267 "xi-grammar.y" /* yacc.c:1646  */
    {
      ReservedWord(FOR, (yyloc).first_column, (yyloc).last_column);
      YYABORT;
    }
#line 2496 "y.tab.c" /* yacc.c:1646  */
    break;

    case 45:
#line 268 "xi-grammar.y" /* yacc.c:1646  */
    {
      ReservedWord(FORALL, (yyloc).first_column, (yyloc).last_column);
      YYABORT;
    }
#line 2502 "y.tab.c" /* yacc.c:1646  */
    break;

    case 46:
#line 269 "xi-grammar.y" /* yacc.c:1646  */
    {
      ReservedWord(WHILE, (yyloc).first_column, (yyloc).last_column);
      YYABORT;
    }
#line 2508 "y.tab.c" /* yacc.c:1646  */
    break;

    case 47:
#line 270 "xi-grammar.y" /* yacc.c:1646  */
    {
      ReservedWord(WHEN, (yyloc).first_column, (yyloc).last_column);
      YYABORT;
    }
#line 2514 "y.tab.c" /* yacc.c:1646  */
    break;

    case 48:
#line 271 "xi-grammar.y" /* yacc.c:1646  */
    {
      ReservedWord(OVERLAP, (yyloc).first_column, (yyloc).last_column);
      YYABORT;
    }
#line 2520 "y.tab.c" /* yacc.c:1646  */
    break;

    case 49:
#line 272 "xi-grammar.y" /* yacc.c:1646  */
    {
      ReservedWord(SERIAL, (yyloc).first_column, (yyloc).last_column);
      YYABORT;
    }
#line 2526 "y.tab.c" /* yacc.c:1646  */
    break;

    case 50:
#line 273 "xi-grammar.y" /* yacc.c:1646  */
    {
      ReservedWord(IF, (yyloc).first_column, (yyloc).last_column);
      YYABORT;
    }
#line 2532 "y.tab.c" /* yacc.c:1646  */
    break;

    case 51:
#line 274 "xi-grammar.y" /* yacc.c:1646  */
    {
      ReservedWord(ELSE, (yyloc).first_column, (yyloc).last_column);
      YYABORT;
    }
#line 2538 "y.tab.c" /* yacc.c:1646  */
    break;

    case 52:
#line 276 "xi-grammar.y" /* yacc.c:1646  */
    {
      ReservedWord(LOCAL, (yyloc).first_column, (yyloc).last_column);
      YYABORT;
    }
#line 2544 "y.tab.c" /* yacc.c:1646  */
    break;

    case 53:
#line 278 "xi-grammar.y" /* yacc.c:1646  */
    {
      ReservedWord(USING, (yyloc).first_column, (yyloc).last_column);
      YYABORT;
    }
#line 2550 "y.tab.c" /* yacc.c:1646  */
    break;

    case 54:
#line 279 "xi-grammar.y" /* yacc.c:1646  */
    {
      ReservedWord(ACCEL, (yyloc).first_column, (yyloc).last_column);
      YYABORT;
    }
#line 2556 "y.tab.c" /* yacc.c:1646  */
    break;

    case 55:
#line 282 "xi-grammar.y" /* yacc.c:1646  */
    {
      ReservedWord(ACCELBLOCK, (yyloc).first_column, (yyloc).last_column);
      YYABORT;
    }
#line 2562 "y.tab.c" /* yacc.c:1646  */
    break;

    case 56:
#line 283 "xi-grammar.y" /* yacc.c:1646  */
    {
      ReservedWord(MEMCRITICAL, (yyloc).first_column, (yyloc).last_column);
      YYABORT;
    }
#line 2568 "y.tab.c" /* yacc.c:1646  */
    break;

    case 57:
#line 284 "xi-grammar.y" /* yacc.c:1646  */
    {
      ReservedWord(REDUCTIONTARGET, (yyloc).first_column, (yyloc).last_column);
      YYABORT;
    }
#line 2574 "y.tab.c" /* yacc.c:1646  */
    break;

    case 58:
#line 285 "xi-grammar.y" /* yacc.c:1646  */
    {
      ReservedWord(CASE, (yyloc).first_column, (yyloc).last_column);
      YYABORT;
    }
#line 2580 "y.tab.c" /* yacc.c:1646  */
    break;

    case 59:
#line 290 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.strval) = (yyvsp[0].strval);
    }
#line 2586 "y.tab.c" /* yacc.c:1646  */
    break;

    case 60:
#line 292 "xi-grammar.y" /* yacc.c:1646  */
    {
      char* tmp = new char[strlen((yyvsp[-3].strval)) + strlen((yyvsp[0].strval)) + 3];
      sprintf(tmp, "%s::%s", (yyvsp[-3].strval), (yyvsp[0].strval));
      (yyval.strval) = tmp;
    }
#line 2596 "y.tab.c" /* yacc.c:1646  */
    break;

    case 61:
#line 299 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.module) = new Module(lineno, (yyvsp[-1].strval), (yyvsp[0].conslist));
    }
#line 2604 "y.tab.c" /* yacc.c:1646  */
    break;

    case 62:
#line 303 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.module) = new Module(lineno, (yyvsp[-1].strval), (yyvsp[0].conslist));
      (yyval.module)->setMain();
    }
#line 2613 "y.tab.c" /* yacc.c:1646  */
    break;

    case 63:
#line 310 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.conslist) = 0;
    }
#line 2619 "y.tab.c" /* yacc.c:1646  */
    break;

    case 64:
#line 312 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.conslist) = (yyvsp[-2].conslist);
    }
#line 2625 "y.tab.c" /* yacc.c:1646  */
    break;

    case 65:
#line 316 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.conslist) = 0;
    }
#line 2631 "y.tab.c" /* yacc.c:1646  */
    break;

    case 66:
#line 318 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.conslist) =
          new ConstructList(lineno, (yyvsp[-1].construct), (yyvsp[0].conslist));
    }
#line 2637 "y.tab.c" /* yacc.c:1646  */
    break;

    case 67:
#line 322 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.construct) = new UsingScope((yyvsp[0].strval), false);
    }
#line 2643 "y.tab.c" /* yacc.c:1646  */
    break;

    case 68:
#line 324 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.construct) = new UsingScope((yyvsp[0].strval), true);
    }
#line 2649 "y.tab.c" /* yacc.c:1646  */
    break;

    case 69:
#line 326 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyvsp[0].member)->setExtern((yyvsp[-1].intval));
      (yyval.construct) = (yyvsp[0].member);
    }
#line 2655 "y.tab.c" /* yacc.c:1646  */
    break;

    case 70:
#line 328 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyvsp[0].message)->setExtern((yyvsp[-1].intval));
      (yyval.construct) = (yyvsp[0].message);
    }
#line 2661 "y.tab.c" /* yacc.c:1646  */
    break;

    case 71:
#line 330 "xi-grammar.y" /* yacc.c:1646  */
    {
      Entry* e =
          new Entry(lineno, 0, (yyvsp[-4].type), (yyvsp[-2].strval), (yyvsp[0].plist), 0,
                    0, 0, (yylsp[-6]).first_line, (yyloc).last_line);
      int isExtern = 1;
      e->setExtern(isExtern);
      e->targs = (yyvsp[-1].tparlist);
      e->label = new XStr;
      (yyvsp[-3].ntype)->print(*e->label);
      (yyval.construct) = e;
      firstRdma = true;
    }
#line 2676 "y.tab.c" /* yacc.c:1646  */
    break;

    case 72:
#line 343 "xi-grammar.y" /* yacc.c:1646  */
    {
      if ((yyvsp[-2].conslist))
        (yyvsp[-2].conslist)->recurse<int&>((yyvsp[-4].intval), &Construct::setExtern);
      (yyval.construct) = (yyvsp[-2].conslist);
    }
#line 2682 "y.tab.c" /* yacc.c:1646  */
    break;

    case 73:
#line 345 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.construct) = new Scope((yyvsp[-3].strval), (yyvsp[-1].conslist));
    }
#line 2688 "y.tab.c" /* yacc.c:1646  */
    break;

    case 74:
#line 347 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.construct) = (yyvsp[-1].construct);
    }
#line 2694 "y.tab.c" /* yacc.c:1646  */
    break;

    case 75:
#line 349 "xi-grammar.y" /* yacc.c:1646  */
    {
      ERROR("preceding construct must be semicolon terminated", (yyloc).first_column,
            (yyloc).last_column);
      YYABORT;
    }
#line 2704 "y.tab.c" /* yacc.c:1646  */
    break;

    case 76:
#line 355 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyvsp[0].module)->setExtern((yyvsp[-1].intval));
      (yyval.construct) = (yyvsp[0].module);
    }
#line 2710 "y.tab.c" /* yacc.c:1646  */
    break;

    case 77:
#line 357 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyvsp[0].chare)->setExtern((yyvsp[-1].intval));
      (yyval.construct) = (yyvsp[0].chare);
    }
#line 2716 "y.tab.c" /* yacc.c:1646  */
    break;

    case 78:
#line 359 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyvsp[0].chare)->setExtern((yyvsp[-1].intval));
      (yyval.construct) = (yyvsp[0].chare);
    }
#line 2722 "y.tab.c" /* yacc.c:1646  */
    break;

    case 79:
#line 361 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyvsp[0].chare)->setExtern((yyvsp[-1].intval));
      (yyval.construct) = (yyvsp[0].chare);
    }
#line 2728 "y.tab.c" /* yacc.c:1646  */
    break;

    case 80:
#line 363 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyvsp[0].chare)->setExtern((yyvsp[-1].intval));
      (yyval.construct) = (yyvsp[0].chare);
    }
#line 2734 "y.tab.c" /* yacc.c:1646  */
    break;

    case 81:
#line 365 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyvsp[0].templat)->setExtern((yyvsp[-1].intval));
      (yyval.construct) = (yyvsp[0].templat);
    }
#line 2740 "y.tab.c" /* yacc.c:1646  */
    break;

    case 82:
#line 367 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.construct) = NULL;
    }
#line 2746 "y.tab.c" /* yacc.c:1646  */
    break;

    case 83:
#line 369 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.construct) = NULL;
    }
#line 2752 "y.tab.c" /* yacc.c:1646  */
    break;

    case 84:
#line 371 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.construct) = (yyvsp[0].accelBlock);
    }
#line 2758 "y.tab.c" /* yacc.c:1646  */
    break;

    case 85:
#line 373 "xi-grammar.y" /* yacc.c:1646  */
    {
      ERROR("invalid construct", (yyloc).first_column, (yyloc).last_column);
      YYABORT;
    }
#line 2768 "y.tab.c" /* yacc.c:1646  */
    break;

    case 86:
#line 381 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.tparam) = new TParamType((yyvsp[0].type));
    }
#line 2774 "y.tab.c" /* yacc.c:1646  */
    break;

    case 87:
#line 383 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.tparam) = new TParamVal((yyvsp[0].strval));
    }
#line 2780 "y.tab.c" /* yacc.c:1646  */
    break;

    case 88:
#line 385 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.tparam) = new TParamVal((yyvsp[0].strval));
    }
#line 2786 "y.tab.c" /* yacc.c:1646  */
    break;

    case 89:
#line 389 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.tparlist) = new TParamList((yyvsp[0].tparam));
    }
#line 2792 "y.tab.c" /* yacc.c:1646  */
    break;

    case 90:
#line 391 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.tparlist) = new TParamList((yyvsp[-2].tparam), (yyvsp[0].tparlist));
    }
#line 2798 "y.tab.c" /* yacc.c:1646  */
    break;

    case 91:
#line 395 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.tparlist) = new TParamList(0);
    }
#line 2804 "y.tab.c" /* yacc.c:1646  */
    break;

    case 92:
#line 397 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.tparlist) = (yyvsp[0].tparlist);
    }
#line 2810 "y.tab.c" /* yacc.c:1646  */
    break;

    case 93:
#line 401 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.tparlist) = 0;
    }
#line 2816 "y.tab.c" /* yacc.c:1646  */
    break;

    case 94:
#line 403 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.tparlist) = (yyvsp[-1].tparlist);
    }
#line 2822 "y.tab.c" /* yacc.c:1646  */
    break;

    case 95:
#line 407 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.type) = new BuiltinType("int");
    }
#line 2828 "y.tab.c" /* yacc.c:1646  */
    break;

    case 96:
#line 409 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.type) = new BuiltinType("long");
    }
#line 2834 "y.tab.c" /* yacc.c:1646  */
    break;

    case 97:
#line 411 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.type) = new BuiltinType("short");
    }
#line 2840 "y.tab.c" /* yacc.c:1646  */
    break;

    case 98:
#line 413 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.type) = new BuiltinType("char");
    }
#line 2846 "y.tab.c" /* yacc.c:1646  */
    break;

    case 99:
#line 415 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.type) = new BuiltinType("unsigned int");
    }
#line 2852 "y.tab.c" /* yacc.c:1646  */
    break;

    case 100:
#line 417 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.type) = new BuiltinType("unsigned long");
    }
#line 2858 "y.tab.c" /* yacc.c:1646  */
    break;

    case 101:
#line 419 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.type) = new BuiltinType("unsigned long");
    }
#line 2864 "y.tab.c" /* yacc.c:1646  */
    break;

    case 102:
#line 421 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.type) = new BuiltinType("unsigned long long");
    }
#line 2870 "y.tab.c" /* yacc.c:1646  */
    break;

    case 103:
#line 423 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.type) = new BuiltinType("unsigned short");
    }
#line 2876 "y.tab.c" /* yacc.c:1646  */
    break;

    case 104:
#line 425 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.type) = new BuiltinType("unsigned char");
    }
#line 2882 "y.tab.c" /* yacc.c:1646  */
    break;

    case 105:
#line 427 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.type) = new BuiltinType("long long");
    }
#line 2888 "y.tab.c" /* yacc.c:1646  */
    break;

    case 106:
#line 429 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.type) = new BuiltinType("float");
    }
#line 2894 "y.tab.c" /* yacc.c:1646  */
    break;

    case 107:
#line 431 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.type) = new BuiltinType("double");
    }
#line 2900 "y.tab.c" /* yacc.c:1646  */
    break;

    case 108:
#line 433 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.type) = new BuiltinType("long double");
    }
#line 2906 "y.tab.c" /* yacc.c:1646  */
    break;

    case 109:
#line 435 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.type) = new BuiltinType("void");
    }
#line 2912 "y.tab.c" /* yacc.c:1646  */
    break;

    case 110:
#line 438 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.ntype) = new NamedType((yyvsp[-1].strval), (yyvsp[0].tparlist));
    }
#line 2918 "y.tab.c" /* yacc.c:1646  */
    break;

    case 111:
#line 439 "xi-grammar.y" /* yacc.c:1646  */
    {
      const char *basename, *scope;
      splitScopedName((yyvsp[-1].strval), &scope, &basename);
      (yyval.ntype) = new NamedType(basename, (yyvsp[0].tparlist), scope);
    }
#line 2928 "y.tab.c" /* yacc.c:1646  */
    break;

    case 112:
#line 447 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.type) = (yyvsp[0].type);
    }
#line 2934 "y.tab.c" /* yacc.c:1646  */
    break;

    case 113:
#line 449 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.type) = (yyvsp[0].ntype);
    }
#line 2940 "y.tab.c" /* yacc.c:1646  */
    break;

    case 114:
#line 453 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.ptype) = new PtrType((yyvsp[-1].type));
    }
#line 2946 "y.tab.c" /* yacc.c:1646  */
    break;

    case 115:
#line 457 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyvsp[-1].ptype)->indirect();
      (yyval.ptype) = (yyvsp[-1].ptype);
    }
#line 2952 "y.tab.c" /* yacc.c:1646  */
    break;

    case 116:
#line 459 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyvsp[-1].ptype)->indirect();
      (yyval.ptype) = (yyvsp[-1].ptype);
    }
#line 2958 "y.tab.c" /* yacc.c:1646  */
    break;

    case 117:
#line 463 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.ftype) =
          new FuncType((yyvsp[-7].type), (yyvsp[-4].strval), (yyvsp[-1].plist));
    }
#line 2964 "y.tab.c" /* yacc.c:1646  */
    break;

    case 118:
#line 467 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.type) = (yyvsp[0].type);
    }
#line 2970 "y.tab.c" /* yacc.c:1646  */
    break;

    case 119:
#line 469 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.type) = (yyvsp[0].ptype);
    }
#line 2976 "y.tab.c" /* yacc.c:1646  */
    break;

    case 120:
#line 471 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.type) = (yyvsp[0].ptype);
    }
#line 2982 "y.tab.c" /* yacc.c:1646  */
    break;

    case 121:
#line 473 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.type) = (yyvsp[0].ftype);
    }
#line 2988 "y.tab.c" /* yacc.c:1646  */
    break;

    case 122:
#line 475 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.type) = new ConstType((yyvsp[0].type));
    }
#line 2994 "y.tab.c" /* yacc.c:1646  */
    break;

    case 123:
#line 477 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.type) = new ConstType((yyvsp[-1].type));
    }
#line 3000 "y.tab.c" /* yacc.c:1646  */
    break;

    case 124:
#line 481 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.type) = (yyvsp[0].type);
    }
#line 3006 "y.tab.c" /* yacc.c:1646  */
    break;

    case 125:
#line 483 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.type) = (yyvsp[0].ptype);
    }
#line 3012 "y.tab.c" /* yacc.c:1646  */
    break;

    case 126:
#line 485 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.type) = (yyvsp[0].ptype);
    }
#line 3018 "y.tab.c" /* yacc.c:1646  */
    break;

    case 127:
#line 487 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.type) = new ConstType((yyvsp[0].type));
    }
#line 3024 "y.tab.c" /* yacc.c:1646  */
    break;

    case 128:
#line 489 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.type) = new ConstType((yyvsp[-1].type));
    }
#line 3030 "y.tab.c" /* yacc.c:1646  */
    break;

    case 129:
#line 493 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.type) = new ReferenceType((yyvsp[-1].type));
    }
#line 3036 "y.tab.c" /* yacc.c:1646  */
    break;

    case 130:
#line 495 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.type) = (yyvsp[0].type);
    }
#line 3042 "y.tab.c" /* yacc.c:1646  */
    break;

    case 131:
#line 499 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.type) = new ReferenceType((yyvsp[-1].type));
    }
#line 3048 "y.tab.c" /* yacc.c:1646  */
    break;

    case 132:
#line 501 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.type) = (yyvsp[0].type);
    }
#line 3054 "y.tab.c" /* yacc.c:1646  */
    break;

    case 133:
#line 505 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.val) = new Value((yyvsp[0].strval));
    }
#line 3060 "y.tab.c" /* yacc.c:1646  */
    break;

    case 134:
#line 509 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.val) = (yyvsp[-1].val);
    }
#line 3066 "y.tab.c" /* yacc.c:1646  */
    break;

    case 135:
#line 513 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.vallist) = 0;
    }
#line 3072 "y.tab.c" /* yacc.c:1646  */
    break;

    case 136:
#line 515 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.vallist) = new ValueList((yyvsp[-1].val), (yyvsp[0].vallist));
    }
#line 3078 "y.tab.c" /* yacc.c:1646  */
    break;

    case 137:
#line 519 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.readonly) =
          new Readonly(lineno, (yyvsp[-2].type), (yyvsp[-1].strval), (yyvsp[0].vallist));
    }
#line 3084 "y.tab.c" /* yacc.c:1646  */
    break;

    case 138:
#line 523 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.readonly) = new Readonly(lineno, (yyvsp[-3].type), (yyvsp[-1].strval),
                                      (yyvsp[0].vallist), 1);
    }
#line 3090 "y.tab.c" /* yacc.c:1646  */
    break;

    case 139:
#line 527 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.intval) = 0;
    }
#line 3096 "y.tab.c" /* yacc.c:1646  */
    break;

    case 140:
#line 529 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.intval) = 0;
    }
#line 3102 "y.tab.c" /* yacc.c:1646  */
    break;

    case 141:
#line 533 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.intval) = 0;
    }
#line 3108 "y.tab.c" /* yacc.c:1646  */
    break;

    case 142:
#line 535 "xi-grammar.y" /* yacc.c:1646  */
    {
      /*
      printf("Warning: Message attributes are being phased out.\n");
      printf("Warning: Please remove them from interface files.\n");
      */
      (yyval.intval) = (yyvsp[-1].intval);
    }
#line 3120 "y.tab.c" /* yacc.c:1646  */
    break;

    case 143:
#line 545 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.intval) = (yyvsp[0].intval);
    }
#line 3126 "y.tab.c" /* yacc.c:1646  */
    break;

    case 144:
#line 547 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.intval) = (yyvsp[-2].intval) | (yyvsp[0].intval);
    }
#line 3132 "y.tab.c" /* yacc.c:1646  */
    break;

    case 145:
#line 551 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.intval) = 0;
    }
#line 3138 "y.tab.c" /* yacc.c:1646  */
    break;

    case 146:
#line 553 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.intval) = 0;
    }
#line 3144 "y.tab.c" /* yacc.c:1646  */
    break;

    case 147:
#line 557 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.cattr) = 0;
    }
#line 3150 "y.tab.c" /* yacc.c:1646  */
    break;

    case 148:
#line 559 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.cattr) = (yyvsp[-1].cattr);
    }
#line 3156 "y.tab.c" /* yacc.c:1646  */
    break;

    case 149:
#line 563 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.cattr) = (yyvsp[0].cattr);
    }
#line 3162 "y.tab.c" /* yacc.c:1646  */
    break;

    case 150:
#line 565 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.cattr) = (yyvsp[-2].cattr) | (yyvsp[0].cattr);
    }
#line 3168 "y.tab.c" /* yacc.c:1646  */
    break;

    case 151:
#line 569 "xi-grammar.y" /* yacc.c:1646  */
    {
      python_doc = NULL;
      (yyval.intval) = 0;
    }
#line 3174 "y.tab.c" /* yacc.c:1646  */
    break;

    case 152:
#line 571 "xi-grammar.y" /* yacc.c:1646  */
    {
      python_doc = (yyvsp[0].strval);
      (yyval.intval) = 0;
    }
#line 3180 "y.tab.c" /* yacc.c:1646  */
    break;

    case 153:
#line 575 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.cattr) = Chare::CPYTHON;
    }
#line 3186 "y.tab.c" /* yacc.c:1646  */
    break;

    case 154:
#line 579 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.cattr) = 0;
    }
#line 3192 "y.tab.c" /* yacc.c:1646  */
    break;

    case 155:
#line 581 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.cattr) = (yyvsp[-1].cattr);
    }
#line 3198 "y.tab.c" /* yacc.c:1646  */
    break;

    case 156:
#line 585 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.cattr) = (yyvsp[0].cattr);
    }
#line 3204 "y.tab.c" /* yacc.c:1646  */
    break;

    case 157:
#line 587 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.cattr) = (yyvsp[-2].cattr) | (yyvsp[0].cattr);
    }
#line 3210 "y.tab.c" /* yacc.c:1646  */
    break;

    case 158:
#line 591 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.cattr) = Chare::CMIGRATABLE;
    }
#line 3216 "y.tab.c" /* yacc.c:1646  */
    break;

    case 159:
#line 593 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.cattr) = Chare::CPYTHON;
    }
#line 3222 "y.tab.c" /* yacc.c:1646  */
    break;

    case 160:
#line 597 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.intval) = 0;
    }
#line 3228 "y.tab.c" /* yacc.c:1646  */
    break;

    case 161:
#line 599 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.intval) = 1;
    }
#line 3234 "y.tab.c" /* yacc.c:1646  */
    break;

    case 162:
#line 602 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.intval) = 0;
    }
#line 3240 "y.tab.c" /* yacc.c:1646  */
    break;

    case 163:
#line 604 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.intval) = 1;
    }
#line 3246 "y.tab.c" /* yacc.c:1646  */
    break;

    case 164:
#line 607 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.mv) = new MsgVar((yyvsp[-3].type), (yyvsp[-2].strval), (yyvsp[-4].intval),
                              (yyvsp[-1].intval));
    }
#line 3252 "y.tab.c" /* yacc.c:1646  */
    break;

    case 165:
#line 611 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.mvlist) = new MsgVarList((yyvsp[0].mv));
    }
#line 3258 "y.tab.c" /* yacc.c:1646  */
    break;

    case 166:
#line 613 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.mvlist) = new MsgVarList((yyvsp[-1].mv), (yyvsp[0].mvlist));
    }
#line 3264 "y.tab.c" /* yacc.c:1646  */
    break;

    case 167:
#line 617 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.message) = new Message(lineno, (yyvsp[0].ntype));
    }
#line 3270 "y.tab.c" /* yacc.c:1646  */
    break;

    case 168:
#line 619 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.message) = new Message(lineno, (yyvsp[-2].ntype));
    }
#line 3276 "y.tab.c" /* yacc.c:1646  */
    break;

    case 169:
#line 621 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.message) = new Message(lineno, (yyvsp[-3].ntype), (yyvsp[-1].mvlist));
    }
#line 3282 "y.tab.c" /* yacc.c:1646  */
    break;

    case 170:
#line 625 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.typelist) = 0;
    }
#line 3288 "y.tab.c" /* yacc.c:1646  */
    break;

    case 171:
#line 627 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.typelist) = (yyvsp[0].typelist);
    }
#line 3294 "y.tab.c" /* yacc.c:1646  */
    break;

    case 172:
#line 631 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.typelist) = new TypeList((yyvsp[0].ntype));
    }
#line 3300 "y.tab.c" /* yacc.c:1646  */
    break;

    case 173:
#line 633 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.typelist) = new TypeList((yyvsp[-2].ntype), (yyvsp[0].typelist));
    }
#line 3306 "y.tab.c" /* yacc.c:1646  */
    break;

    case 174:
#line 637 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.chare) =
          new Chare(lineno, (yyvsp[-3].cattr) | Chare::CCHARE, (yyvsp[-2].ntype),
                    (yyvsp[-1].typelist), (yyvsp[0].mbrlist));
    }
#line 3312 "y.tab.c" /* yacc.c:1646  */
    break;

    case 175:
#line 639 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.chare) = new MainChare(lineno, (yyvsp[-3].cattr), (yyvsp[-2].ntype),
                                    (yyvsp[-1].typelist), (yyvsp[0].mbrlist));
    }
#line 3318 "y.tab.c" /* yacc.c:1646  */
    break;

    case 176:
#line 643 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.chare) = new Group(lineno, (yyvsp[-3].cattr), (yyvsp[-2].ntype),
                                (yyvsp[-1].typelist), (yyvsp[0].mbrlist));
    }
#line 3324 "y.tab.c" /* yacc.c:1646  */
    break;

    case 177:
#line 647 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.chare) = new NodeGroup(lineno, (yyvsp[-3].cattr), (yyvsp[-2].ntype),
                                    (yyvsp[-1].typelist), (yyvsp[0].mbrlist));
    }
#line 3330 "y.tab.c" /* yacc.c:1646  */
    break;

    case 178:
#line 651 "xi-grammar.y" /* yacc.c:1646  */
    {                    /*Stupid special case for [1D] indices*/
      char* buf = new char[40];
      sprintf(buf, "%sD", (yyvsp[-2].strval));
      (yyval.ntype) = new NamedType(buf);
    }
#line 3340 "y.tab.c" /* yacc.c:1646  */
    break;

    case 179:
#line 657 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.ntype) = new NamedType((yyvsp[-1].strval));
    }
#line 3346 "y.tab.c" /* yacc.c:1646  */
    break;

    case 180:
#line 661 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.chare) =
          new Array(lineno, (yyvsp[-4].cattr), (yyvsp[-3].ntype), (yyvsp[-2].ntype),
                    (yyvsp[-1].typelist), (yyvsp[0].mbrlist));
    }
#line 3352 "y.tab.c" /* yacc.c:1646  */
    break;

    case 181:
#line 663 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.chare) =
          new Array(lineno, (yyvsp[-3].cattr), (yyvsp[-4].ntype), (yyvsp[-2].ntype),
                    (yyvsp[-1].typelist), (yyvsp[0].mbrlist));
    }
#line 3358 "y.tab.c" /* yacc.c:1646  */
    break;

    case 182:
#line 667 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.chare) = new Chare(lineno, (yyvsp[-3].cattr) | Chare::CCHARE,
                                new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist),
                                (yyvsp[0].mbrlist));
    }
#line 3364 "y.tab.c" /* yacc.c:1646  */
    break;

    case 183:
#line 669 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.chare) =
          new MainChare(lineno, (yyvsp[-3].cattr), new NamedType((yyvsp[-2].strval)),
                        (yyvsp[-1].typelist), (yyvsp[0].mbrlist));
    }
#line 3370 "y.tab.c" /* yacc.c:1646  */
    break;

    case 184:
#line 673 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.chare) =
          new Group(lineno, (yyvsp[-3].cattr), new NamedType((yyvsp[-2].strval)),
                    (yyvsp[-1].typelist), (yyvsp[0].mbrlist));
    }
#line 3376 "y.tab.c" /* yacc.c:1646  */
    break;

    case 185:
#line 677 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.chare) =
          new NodeGroup(lineno, (yyvsp[-3].cattr), new NamedType((yyvsp[-2].strval)),
                        (yyvsp[-1].typelist), (yyvsp[0].mbrlist));
    }
#line 3382 "y.tab.c" /* yacc.c:1646  */
    break;

    case 186:
#line 681 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.chare) =
          new Array(lineno, 0, (yyvsp[-3].ntype), new NamedType((yyvsp[-2].strval)),
                    (yyvsp[-1].typelist), (yyvsp[0].mbrlist));
    }
#line 3388 "y.tab.c" /* yacc.c:1646  */
    break;

    case 187:
#line 685 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.message) = new Message(lineno, new NamedType((yyvsp[-1].strval)));
    }
#line 3394 "y.tab.c" /* yacc.c:1646  */
    break;

    case 188:
#line 687 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.message) =
          new Message(lineno, new NamedType((yyvsp[-4].strval)), (yyvsp[-2].mvlist));
    }
#line 3400 "y.tab.c" /* yacc.c:1646  */
    break;

    case 189:
#line 691 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.type) = 0;
    }
#line 3406 "y.tab.c" /* yacc.c:1646  */
    break;

    case 190:
#line 693 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.type) = (yyvsp[0].type);
    }
#line 3412 "y.tab.c" /* yacc.c:1646  */
    break;

    case 191:
#line 697 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.strval) = 0;
    }
#line 3418 "y.tab.c" /* yacc.c:1646  */
    break;

    case 192:
#line 699 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.strval) = (yyvsp[0].strval);
    }
#line 3424 "y.tab.c" /* yacc.c:1646  */
    break;

    case 193:
#line 701 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.strval) = (yyvsp[0].strval);
    }
#line 3430 "y.tab.c" /* yacc.c:1646  */
    break;

    case 194:
#line 703 "xi-grammar.y" /* yacc.c:1646  */
    {
      XStr typeStr;
      (yyvsp[0].ntype)->print(typeStr);
      char* tmp = strdup(typeStr.get_string());
      (yyval.strval) = tmp;
    }
#line 3441 "y.tab.c" /* yacc.c:1646  */
    break;

    case 195:
#line 712 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.tvar) = new TType(new NamedType((yyvsp[-1].strval)), (yyvsp[0].type));
    }
#line 3447 "y.tab.c" /* yacc.c:1646  */
    break;

    case 196:
#line 714 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.tvar) = new TFunc((yyvsp[-1].ftype), (yyvsp[0].strval));
    }
#line 3453 "y.tab.c" /* yacc.c:1646  */
    break;

    case 197:
#line 716 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.tvar) = new TName((yyvsp[-2].type), (yyvsp[-1].strval), (yyvsp[0].strval));
    }
#line 3459 "y.tab.c" /* yacc.c:1646  */
    break;

    case 198:
#line 720 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.tvarlist) = new TVarList((yyvsp[0].tvar));
    }
#line 3465 "y.tab.c" /* yacc.c:1646  */
    break;

    case 199:
#line 722 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.tvarlist) = new TVarList((yyvsp[-2].tvar), (yyvsp[0].tvarlist));
    }
#line 3471 "y.tab.c" /* yacc.c:1646  */
    break;

    case 200:
#line 726 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.tvarlist) = (yyvsp[-1].tvarlist);
    }
#line 3477 "y.tab.c" /* yacc.c:1646  */
    break;

    case 201:
#line 730 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare));
      (yyvsp[0].chare)->setTemplate((yyval.templat));
    }
#line 3483 "y.tab.c" /* yacc.c:1646  */
    break;

    case 202:
#line 732 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare));
      (yyvsp[0].chare)->setTemplate((yyval.templat));
    }
#line 3489 "y.tab.c" /* yacc.c:1646  */
    break;

    case 203:
#line 734 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare));
      (yyvsp[0].chare)->setTemplate((yyval.templat));
    }
#line 3495 "y.tab.c" /* yacc.c:1646  */
    break;

    case 204:
#line 736 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare));
      (yyvsp[0].chare)->setTemplate((yyval.templat));
    }
#line 3501 "y.tab.c" /* yacc.c:1646  */
    break;

    case 205:
#line 738 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].message));
      (yyvsp[0].message)->setTemplate((yyval.templat));
    }
#line 3507 "y.tab.c" /* yacc.c:1646  */
    break;

    case 206:
#line 742 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.mbrlist) = 0;
    }
#line 3513 "y.tab.c" /* yacc.c:1646  */
    break;

    case 207:
#line 744 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.mbrlist) = (yyvsp[-2].mbrlist);
    }
#line 3519 "y.tab.c" /* yacc.c:1646  */
    break;

    case 208:
#line 748 "xi-grammar.y" /* yacc.c:1646  */
    {
      if (!connectEntries.empty()) {
        (yyval.mbrlist) = new AstChildren<Member>(connectEntries);
      } else {
        (yyval.mbrlist) = 0;
      }
    }
#line 3531 "y.tab.c" /* yacc.c:1646  */
    break;

    case 209:
#line 756 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.mbrlist) =
          new AstChildren<Member>(-1, (yyvsp[-1].member), (yyvsp[0].mbrlist));
    }
#line 3537 "y.tab.c" /* yacc.c:1646  */
    break;

    case 210:
#line 760 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.member) = (yyvsp[0].readonly);
    }
#line 3543 "y.tab.c" /* yacc.c:1646  */
    break;

    case 211:
#line 762 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.member) = (yyvsp[0].readonly);
    }
#line 3549 "y.tab.c" /* yacc.c:1646  */
    break;

    case 213:
#line 765 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.member) = (yyvsp[0].member);
    }
#line 3555 "y.tab.c" /* yacc.c:1646  */
    break;

    case 214:
#line 767 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.member) = (yyvsp[0].pupable);
    }
#line 3561 "y.tab.c" /* yacc.c:1646  */
    break;

    case 215:
#line 769 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.member) = (yyvsp[0].includeFile);
    }
#line 3567 "y.tab.c" /* yacc.c:1646  */
    break;

    case 216:
#line 771 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.member) = new ClassDeclaration(lineno, (yyvsp[0].strval));
    }
#line 3573 "y.tab.c" /* yacc.c:1646  */
    break;

    case 217:
#line 775 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.member) = new InitCall(lineno, (yyvsp[0].strval), 1);
    }
#line 3579 "y.tab.c" /* yacc.c:1646  */
    break;

    case 218:
#line 777 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.member) = new InitCall(lineno, (yyvsp[-3].strval), 1);
    }
#line 3585 "y.tab.c" /* yacc.c:1646  */
    break;

    case 219:
#line 779 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.member) = new InitCall(lineno,
                                    strdup((std::string((yyvsp[-6].strval)) + '<' +
                                            ((yyvsp[-4].tparlist))->to_string() + '>')
                                               .c_str()),
                                    1);
    }
#line 3595 "y.tab.c" /* yacc.c:1646  */
    break;

    case 220:
#line 785 "xi-grammar.y" /* yacc.c:1646  */
    {
      WARNING("deprecated use of initcall. Use initnode or initproc instead",
              (yylsp[-2]).first_column, (yylsp[-2]).last_column, (yylsp[-2]).first_line);
      (yyval.member) = new InitCall(lineno, (yyvsp[0].strval), 1);
    }
#line 3605 "y.tab.c" /* yacc.c:1646  */
    break;

    case 221:
#line 791 "xi-grammar.y" /* yacc.c:1646  */
    {
      WARNING("deprecated use of initcall. Use initnode or initproc instead",
              (yylsp[-5]).first_column, (yylsp[-5]).last_column, (yylsp[-5]).first_line);
      (yyval.member) = new InitCall(lineno, (yyvsp[-3].strval), 1);
    }
#line 3615 "y.tab.c" /* yacc.c:1646  */
    break;

    case 222:
#line 800 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.member) = new InitCall(lineno, (yyvsp[0].strval), 0);
    }
#line 3621 "y.tab.c" /* yacc.c:1646  */
    break;

    case 223:
#line 802 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.member) = new InitCall(lineno, (yyvsp[-3].strval), 0);
    }
#line 3627 "y.tab.c" /* yacc.c:1646  */
    break;

    case 224:
#line 804 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.member) = new InitCall(lineno,
                                    strdup((std::string((yyvsp[-6].strval)) + '<' +
                                            ((yyvsp[-4].tparlist))->to_string() + '>')
                                               .c_str()),
                                    0);
    }
#line 3637 "y.tab.c" /* yacc.c:1646  */
    break;

    case 225:
#line 810 "xi-grammar.y" /* yacc.c:1646  */
    {
      InitCall* rtn = new InitCall(lineno, (yyvsp[-3].strval), 0);
      rtn->setAccel();
      (yyval.member) = rtn;
    }
#line 3647 "y.tab.c" /* yacc.c:1646  */
    break;

    case 226:
#line 818 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.pupable) = new PUPableClass(lineno, (yyvsp[0].ntype), 0);
    }
#line 3653 "y.tab.c" /* yacc.c:1646  */
    break;

    case 227:
#line 820 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.pupable) = new PUPableClass(lineno, (yyvsp[-2].ntype), (yyvsp[0].pupable));
    }
#line 3659 "y.tab.c" /* yacc.c:1646  */
    break;

    case 228:
#line 823 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.includeFile) = new IncludeFile(lineno, (yyvsp[0].strval));
    }
#line 3665 "y.tab.c" /* yacc.c:1646  */
    break;

    case 229:
#line 827 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.member) = (yyvsp[0].member);
    }
#line 3671 "y.tab.c" /* yacc.c:1646  */
    break;

    case 230:
#line 831 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.member) = (yyvsp[0].entry);
    }
#line 3677 "y.tab.c" /* yacc.c:1646  */
    break;

    case 231:
#line 833 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyvsp[0].entry)->tspec = (yyvsp[-1].tvarlist);
      (yyval.member) = (yyvsp[0].entry);
    }
#line 3686 "y.tab.c" /* yacc.c:1646  */
    break;

    case 232:
#line 838 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.member) = (yyvsp[-1].member);
    }
#line 3692 "y.tab.c" /* yacc.c:1646  */
    break;

    case 233:
#line 840 "xi-grammar.y" /* yacc.c:1646  */
    {
      ERROR("invalid SDAG member", (yyloc).first_column, (yyloc).last_column);
      YYABORT;
    }
#line 3702 "y.tab.c" /* yacc.c:1646  */
    break;

    case 234:
#line 848 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.member) = 0;
    }
#line 3708 "y.tab.c" /* yacc.c:1646  */
    break;

    case 235:
#line 850 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.member) = 0;
    }
#line 3714 "y.tab.c" /* yacc.c:1646  */
    break;

    case 236:
#line 852 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.member) = 0;
    }
#line 3720 "y.tab.c" /* yacc.c:1646  */
    break;

    case 237:
#line 854 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.member) = 0;
    }
#line 3726 "y.tab.c" /* yacc.c:1646  */
    break;

    case 238:
#line 856 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.member) = 0;
    }
#line 3732 "y.tab.c" /* yacc.c:1646  */
    break;

    case 239:
#line 858 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.member) = 0;
    }
#line 3738 "y.tab.c" /* yacc.c:1646  */
    break;

    case 240:
#line 860 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.member) = 0;
    }
#line 3744 "y.tab.c" /* yacc.c:1646  */
    break;

    case 241:
#line 862 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.member) = 0;
    }
#line 3750 "y.tab.c" /* yacc.c:1646  */
    break;

    case 242:
#line 864 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.member) = 0;
    }
#line 3756 "y.tab.c" /* yacc.c:1646  */
    break;

    case 243:
#line 866 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.member) = 0;
    }
#line 3762 "y.tab.c" /* yacc.c:1646  */
    break;

    case 244:
#line 868 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.member) = 0;
    }
#line 3768 "y.tab.c" /* yacc.c:1646  */
    break;

    case 245:
#line 871 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.entry) =
          new Entry(lineno, (yyvsp[-5].intval), (yyvsp[-4].type), (yyvsp[-3].strval),
                    (yyvsp[-2].plist), (yyvsp[-1].val), (yyvsp[0].sentry),
                    (const char*)NULL, (yylsp[-6]).first_line, (yyloc).last_line);
      if ((yyvsp[0].sentry) != 0) {
        (yyvsp[0].sentry)->con1 = new SdagConstruct(SIDENT, (yyvsp[-3].strval));
        (yyvsp[0].sentry)->setEntry((yyval.entry));
        (yyvsp[0].sentry)->param = new ParamList((yyvsp[-2].plist));
      }
      firstRdma = true;
    }
#line 3782 "y.tab.c" /* yacc.c:1646  */
    break;

    case 246:
#line 881 "xi-grammar.y" /* yacc.c:1646  */
    {
      Entry* e = new Entry(lineno, (yyvsp[-3].intval), 0, (yyvsp[-2].strval),
                           (yyvsp[-1].plist), 0, (yyvsp[0].sentry), (const char*)NULL,
                           (yylsp[-4]).first_line, (yyloc).last_line);
      if ((yyvsp[0].sentry) != 0) {
        (yyvsp[0].sentry)->con1 = new SdagConstruct(SIDENT, (yyvsp[-2].strval));
        (yyvsp[0].sentry)->setEntry((yyval.entry));
        (yyvsp[0].sentry)->param = new ParamList((yyvsp[-1].plist));
      }
      firstRdma = true;
      if (e->param && e->param->isCkMigMsgPtr()) {
        WARNING("CkMigrateMsg chare constructor is taken for granted",
                (yyloc).first_column, (yyloc).last_column);
        (yyval.entry) = NULL;
      } else {
        (yyval.entry) = e;
      }
    }
#line 3803 "y.tab.c" /* yacc.c:1646  */
    break;

    case 247:
#line 898 "xi-grammar.y" /* yacc.c:1646  */
    {
      int attribs = SACCEL;
      const char* name = (yyvsp[-7].strval);
      ParamList* paramList = (yyvsp[-6].plist);
      ParamList* accelParamList = (yyvsp[-5].plist);
      XStr* codeBody = new XStr((yyvsp[-3].strval));
      const char* callbackName = (yyvsp[-1].strval);

      (yyval.entry) =
          new Entry(lineno, attribs, new BuiltinType("void"), name, paramList, 0, 0, 0);
      (yyval.entry)->setAccelParam(accelParamList);
      (yyval.entry)->setAccelCodeBody(codeBody);
      (yyval.entry)->setAccelCallbackName(new XStr(callbackName));
      firstRdma = true;
    }
#line 3822 "y.tab.c" /* yacc.c:1646  */
    break;

    case 248:
#line 915 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.accelBlock) = new AccelBlock(lineno, new XStr((yyvsp[-2].strval)));
    }
#line 3828 "y.tab.c" /* yacc.c:1646  */
    break;

    case 249:
#line 917 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.accelBlock) = new AccelBlock(lineno, NULL);
    }
#line 3834 "y.tab.c" /* yacc.c:1646  */
    break;

    case 250:
#line 921 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.type) = (yyvsp[0].type);
    }
#line 3840 "y.tab.c" /* yacc.c:1646  */
    break;

    case 251:
#line 925 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.intval) = 0;
    }
#line 3846 "y.tab.c" /* yacc.c:1646  */
    break;

    case 252:
#line 927 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.intval) = (yyvsp[-1].intval);
    }
#line 3852 "y.tab.c" /* yacc.c:1646  */
    break;

    case 253:
#line 929 "xi-grammar.y" /* yacc.c:1646  */
    {
      ERROR("invalid entry method attribute list", (yyloc).first_column,
            (yyloc).last_column);
      YYABORT;
    }
#line 3861 "y.tab.c" /* yacc.c:1646  */
    break;

    case 254:
#line 936 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.intval) = (yyvsp[0].intval);
    }
#line 3867 "y.tab.c" /* yacc.c:1646  */
    break;

    case 255:
#line 938 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.intval) = (yyvsp[-2].intval) | (yyvsp[0].intval);
    }
#line 3873 "y.tab.c" /* yacc.c:1646  */
    break;

    case 256:
#line 942 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.intval) = STHREADED;
    }
#line 3879 "y.tab.c" /* yacc.c:1646  */
    break;

    case 257:
#line 944 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.intval) = SSYNC;
    }
#line 3885 "y.tab.c" /* yacc.c:1646  */
    break;

    case 258:
#line 946 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.intval) = SIGET;
    }
#line 3891 "y.tab.c" /* yacc.c:1646  */
    break;

    case 259:
#line 948 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.intval) = SLOCKED;
    }
#line 3897 "y.tab.c" /* yacc.c:1646  */
    break;

    case 260:
#line 950 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.intval) = SCREATEHERE;
    }
#line 3903 "y.tab.c" /* yacc.c:1646  */
    break;

    case 261:
#line 952 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.intval) = SCREATEHOME;
    }
#line 3909 "y.tab.c" /* yacc.c:1646  */
    break;

    case 262:
#line 954 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.intval) = SNOKEEP;
    }
#line 3915 "y.tab.c" /* yacc.c:1646  */
    break;

    case 263:
#line 956 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.intval) = SNOTRACE;
    }
#line 3921 "y.tab.c" /* yacc.c:1646  */
    break;

    case 264:
#line 958 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.intval) = SAPPWORK;
    }
#line 3927 "y.tab.c" /* yacc.c:1646  */
    break;

    case 265:
#line 960 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.intval) = SIMMEDIATE;
    }
#line 3933 "y.tab.c" /* yacc.c:1646  */
    break;

    case 266:
#line 962 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.intval) = SSKIPSCHED;
    }
#line 3939 "y.tab.c" /* yacc.c:1646  */
    break;

    case 267:
#line 964 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.intval) = SINLINE;
    }
#line 3945 "y.tab.c" /* yacc.c:1646  */
    break;

    case 268:
#line 966 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.intval) = SLOCAL;
    }
#line 3951 "y.tab.c" /* yacc.c:1646  */
    break;

    case 269:
#line 968 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.intval) = SPYTHON;
    }
#line 3957 "y.tab.c" /* yacc.c:1646  */
    break;

    case 270:
#line 970 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.intval) = SMEM;
    }
#line 3963 "y.tab.c" /* yacc.c:1646  */
    break;

    case 271:
#line 972 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.intval) = SREDUCE;
    }
#line 3969 "y.tab.c" /* yacc.c:1646  */
    break;

    case 272:
#line 974 "xi-grammar.y" /* yacc.c:1646  */
    {
#ifdef CMK_USING_XLC
      WARNING(
          "a known bug in xl compilers (PMR 18366,122,000) currently breaks "
          "aggregate entry methods.\n"
          "Until a fix is released, this tag will be ignored on those compilers.",
          (yylsp[0]).first_column, (yylsp[0]).last_column, (yylsp[0]).first_line);
      (yyval.intval) = 0;
#else
      (yyval.intval) = SAGGREGATE;
#endif
    }
#line 3985 "y.tab.c" /* yacc.c:1646  */
    break;

    case 273:
#line 986 "xi-grammar.y" /* yacc.c:1646  */
    {
      ERROR("invalid entry method attribute", (yylsp[0]).first_column,
            (yylsp[0]).last_column);
      yyclearin;
      yyerrok;
    }
#line 3996 "y.tab.c" /* yacc.c:1646  */
    break;

    case 274:
#line 995 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.val) = new Value((yyvsp[0].strval));
    }
#line 4002 "y.tab.c" /* yacc.c:1646  */
    break;

    case 275:
#line 997 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.val) = new Value((yyvsp[0].strval));
    }
#line 4008 "y.tab.c" /* yacc.c:1646  */
    break;

    case 276:
#line 999 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.val) = new Value((yyvsp[0].strval));
    }
#line 4014 "y.tab.c" /* yacc.c:1646  */
    break;

    case 277:
#line 1003 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.strval) = "";
    }
#line 4020 "y.tab.c" /* yacc.c:1646  */
    break;

    case 278:
#line 1005 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.strval) = (yyvsp[0].strval);
    }
#line 4026 "y.tab.c" /* yacc.c:1646  */
    break;

    case 279:
#line 1007 "xi-grammar.y" /* yacc.c:1646  */
    {                     /*Returned only when in_bracket*/
      char* tmp = new char[strlen((yyvsp[-2].strval)) + strlen((yyvsp[0].strval)) + 3];
      sprintf(tmp, "%s, %s", (yyvsp[-2].strval), (yyvsp[0].strval));
      (yyval.strval) = tmp;
    }
#line 4036 "y.tab.c" /* yacc.c:1646  */
    break;

    case 280:
#line 1015 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.strval) = "";
    }
#line 4042 "y.tab.c" /* yacc.c:1646  */
    break;

    case 281:
#line 1017 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.strval) = (yyvsp[0].strval);
    }
#line 4048 "y.tab.c" /* yacc.c:1646  */
    break;

    case 282:
#line 1019 "xi-grammar.y" /* yacc.c:1646  */
    {                     /*Returned only when in_bracket*/
      char* tmp = new char[strlen((yyvsp[-4].strval)) + strlen((yyvsp[-2].strval)) +
                           strlen((yyvsp[0].strval)) + 3];
      sprintf(tmp, "%s[%s]%s", (yyvsp[-4].strval), (yyvsp[-2].strval), (yyvsp[0].strval));
      (yyval.strval) = tmp;
    }
#line 4058 "y.tab.c" /* yacc.c:1646  */
    break;

    case 283:
#line 1025 "xi-grammar.y" /* yacc.c:1646  */
    {                     /*Returned only when in_braces*/
      char* tmp = new char[strlen((yyvsp[-4].strval)) + strlen((yyvsp[-2].strval)) +
                           strlen((yyvsp[0].strval)) + 3];
      sprintf(tmp, "%s{%s}%s", (yyvsp[-4].strval), (yyvsp[-2].strval), (yyvsp[0].strval));
      (yyval.strval) = tmp;
    }
#line 4068 "y.tab.c" /* yacc.c:1646  */
    break;

    case 284:
#line 1031 "xi-grammar.y" /* yacc.c:1646  */
    {                     /*Returned only when in_braces*/
      char* tmp = new char[strlen((yyvsp[-4].strval)) + strlen((yyvsp[-2].strval)) +
                           strlen((yyvsp[0].strval)) + 3];
      sprintf(tmp, "%s(%s)%s", (yyvsp[-4].strval), (yyvsp[-2].strval), (yyvsp[0].strval));
      (yyval.strval) = tmp;
    }
#line 4078 "y.tab.c" /* yacc.c:1646  */
    break;

    case 285:
#line 1037 "xi-grammar.y" /* yacc.c:1646  */
    {                     /*Returned only when in_braces*/
      char* tmp = new char[strlen((yyvsp[-2].strval)) + strlen((yyvsp[0].strval)) + 3];
      sprintf(tmp, "(%s)%s", (yyvsp[-2].strval), (yyvsp[0].strval));
      (yyval.strval) = tmp;
    }
#line 4088 "y.tab.c" /* yacc.c:1646  */
    break;

    case 286:
#line 1045 "xi-grammar.y" /* yacc.c:1646  */
    {                     /*Start grabbing CPROGRAM segments*/
      in_bracket = 1;
      (yyval.pname) = new Parameter(lineno, (yyvsp[-2].type), (yyvsp[-1].strval));
    }
#line 4097 "y.tab.c" /* yacc.c:1646  */
    break;

    case 287:
#line 1052 "xi-grammar.y" /* yacc.c:1646  */
    {
      /*Start grabbing CPROGRAM segments*/
      in_braces = 1;
      (yyval.intval) = 0;
    }
#line 4107 "y.tab.c" /* yacc.c:1646  */
    break;

    case 288:
#line 1060 "xi-grammar.y" /* yacc.c:1646  */
    {
      in_braces = 0;
      (yyval.intval) = 0;
    }
#line 4116 "y.tab.c" /* yacc.c:1646  */
    break;

    case 289:
#line 1067 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.pname) = new Parameter(lineno, (yyvsp[0].type));
    }
#line 4122 "y.tab.c" /* yacc.c:1646  */
    break;

    case 290:
#line 1069 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.pname) = new Parameter(lineno, (yyvsp[-2].type), (yyvsp[-1].strval));
      (yyval.pname)->setConditional((yyvsp[0].intval));
    }
#line 4128 "y.tab.c" /* yacc.c:1646  */
    break;

    case 291:
#line 1071 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.pname) =
          new Parameter(lineno, (yyvsp[-3].type), (yyvsp[-2].strval), 0, (yyvsp[0].val));
    }
#line 4134 "y.tab.c" /* yacc.c:1646  */
    break;

    case 292:
#line 1073 "xi-grammar.y" /* yacc.c:1646  */
    {                     /*Stop grabbing CPROGRAM segments*/
      in_bracket = 0;
      (yyval.pname) = new Parameter(lineno, (yyvsp[-2].pname)->getType(),
                                    (yyvsp[-2].pname)->getName(), (yyvsp[-1].strval));
    }
#line 4143 "y.tab.c" /* yacc.c:1646  */
    break;

    case 293:
#line 1078 "xi-grammar.y" /* yacc.c:1646  */
    {                     /*Stop grabbing CPROGRAM segments*/
      in_bracket = 0;
      (yyval.pname) = new Parameter(lineno, (yyvsp[-2].pname)->getType(),
                                    (yyvsp[-2].pname)->getName(), (yyvsp[-1].strval));
      (yyval.pname)->setRdma(true);
      if (firstRdma) {
        (yyval.pname)->setFirstRdma(true);
        firstRdma = false;
      }
    }
#line 4157 "y.tab.c" /* yacc.c:1646  */
    break;

    case 294:
#line 1089 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_READONLY;
    }
#line 4163 "y.tab.c" /* yacc.c:1646  */
    break;

    case 295:
#line 1090 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_READWRITE;
    }
#line 4169 "y.tab.c" /* yacc.c:1646  */
    break;

    case 296:
#line 1091 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_WRITEONLY;
    }
#line 4175 "y.tab.c" /* yacc.c:1646  */
    break;

    case 297:
#line 1094 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.xstrptr) = new XStr((yyvsp[0].strval));
    }
#line 4181 "y.tab.c" /* yacc.c:1646  */
    break;

    case 298:
#line 1095 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.xstrptr) = new XStr("");
      *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "->" << (yyvsp[0].strval);
    }
#line 4187 "y.tab.c" /* yacc.c:1646  */
    break;

    case 299:
#line 1096 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.xstrptr) = new XStr("");
      *((yyval.xstrptr)) << *((yyvsp[-2].xstrptr)) << "." << (yyvsp[0].strval);
    }
#line 4193 "y.tab.c" /* yacc.c:1646  */
    break;

    case 300:
#line 1098 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.xstrptr) = new XStr("");
      *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "[" << *((yyvsp[-1].xstrptr))
                         << "]";
      delete (yyvsp[-3].xstrptr);
      delete (yyvsp[-1].xstrptr);
    }
#line 4204 "y.tab.c" /* yacc.c:1646  */
    break;

    case 301:
#line 1105 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.xstrptr) = new XStr("");
      *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "[" << (yyvsp[-1].strval) << "]";
      delete (yyvsp[-3].xstrptr);
    }
#line 4214 "y.tab.c" /* yacc.c:1646  */
    break;

    case 302:
#line 1111 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.xstrptr) = new XStr("");
      *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "(" << *((yyvsp[-1].xstrptr))
                         << ")";
      delete (yyvsp[-3].xstrptr);
      delete (yyvsp[-1].xstrptr);
    }
#line 4225 "y.tab.c" /* yacc.c:1646  */
    break;

    case 303:
#line 1120 "xi-grammar.y" /* yacc.c:1646  */
    {
      in_bracket = 0;
      (yyval.pname) = new Parameter(lineno, (yyvsp[-2].pname)->getType(),
                                    (yyvsp[-2].pname)->getName(), (yyvsp[-1].strval));
    }
#line 4234 "y.tab.c" /* yacc.c:1646  */
    break;

    case 304:
#line 1127 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.pname) = new Parameter(lineno, (yyvsp[-4].type), (yyvsp[-3].strval));
      (yyval.pname)->setAccelInstName((yyvsp[-1].xstrptr));
      (yyval.pname)->setAccelBufferType((yyvsp[-6].intval));
    }
#line 4244 "y.tab.c" /* yacc.c:1646  */
    break;

    case 305:
#line 1133 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.pname) = new Parameter(lineno, (yyvsp[-4].type), (yyvsp[-3].strval));
      (yyval.pname)->setAccelInstName((yyvsp[-1].xstrptr));
      (yyval.pname)->setAccelBufferType(Parameter::ACCEL_BUFFER_TYPE_READWRITE);
    }
#line 4254 "y.tab.c" /* yacc.c:1646  */
    break;

    case 306:
#line 1139 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.pname) = (yyvsp[-3].pname);
      (yyval.pname)->setAccelInstName((yyvsp[-1].xstrptr));
      (yyval.pname)->setAccelBufferType((yyvsp[-5].intval));
    }
#line 4264 "y.tab.c" /* yacc.c:1646  */
    break;

    case 307:
#line 1147 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.plist) = new ParamList((yyvsp[0].pname));
    }
#line 4270 "y.tab.c" /* yacc.c:1646  */
    break;

    case 308:
#line 1149 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.plist) = new ParamList((yyvsp[-2].pname), (yyvsp[0].plist));
    }
#line 4276 "y.tab.c" /* yacc.c:1646  */
    break;

    case 309:
#line 1153 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.plist) = new ParamList((yyvsp[0].pname));
    }
#line 4282 "y.tab.c" /* yacc.c:1646  */
    break;

    case 310:
#line 1155 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.plist) = new ParamList((yyvsp[-2].pname), (yyvsp[0].plist));
    }
#line 4288 "y.tab.c" /* yacc.c:1646  */
    break;

    case 311:
#line 1159 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.plist) = (yyvsp[-1].plist);
    }
#line 4294 "y.tab.c" /* yacc.c:1646  */
    break;

    case 312:
#line 1161 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.plist) = new ParamList(new Parameter(0, new BuiltinType("void")));
    }
#line 4300 "y.tab.c" /* yacc.c:1646  */
    break;

    case 313:
#line 1165 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.plist) = (yyvsp[-1].plist);
    }
#line 4306 "y.tab.c" /* yacc.c:1646  */
    break;

    case 314:
#line 1167 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.plist) = 0;
    }
#line 4312 "y.tab.c" /* yacc.c:1646  */
    break;

    case 315:
#line 1171 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.val) = 0;
    }
#line 4318 "y.tab.c" /* yacc.c:1646  */
    break;

    case 316:
#line 1173 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.val) = new Value((yyvsp[0].strval));
    }
#line 4324 "y.tab.c" /* yacc.c:1646  */
    break;

    case 317:
#line 1177 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.sentry) = 0;
    }
#line 4330 "y.tab.c" /* yacc.c:1646  */
    break;

    case 318:
#line 1179 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.sentry) = new SdagEntryConstruct((yyvsp[0].sc));
    }
#line 4336 "y.tab.c" /* yacc.c:1646  */
    break;

    case 319:
#line 1181 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.sentry) = new SdagEntryConstruct((yyvsp[-2].slist));
    }
#line 4342 "y.tab.c" /* yacc.c:1646  */
    break;

    case 320:
#line 1185 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.slist) = new SListConstruct((yyvsp[0].sc));
    }
#line 4348 "y.tab.c" /* yacc.c:1646  */
    break;

    case 321:
#line 1187 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.slist) = new SListConstruct((yyvsp[-1].sc), (yyvsp[0].slist));
    }
#line 4354 "y.tab.c" /* yacc.c:1646  */
    break;

    case 322:
#line 1191 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.olist) = new OListConstruct((yyvsp[0].sc));
    }
#line 4360 "y.tab.c" /* yacc.c:1646  */
    break;

    case 323:
#line 1193 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.olist) = new OListConstruct((yyvsp[-1].sc), (yyvsp[0].slist));
    }
#line 4366 "y.tab.c" /* yacc.c:1646  */
    break;

    case 324:
#line 1197 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.clist) = new CaseListConstruct((yyvsp[0].when));
    }
#line 4372 "y.tab.c" /* yacc.c:1646  */
    break;

    case 325:
#line 1199 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.clist) = new CaseListConstruct((yyvsp[-1].when), (yyvsp[0].clist));
    }
#line 4378 "y.tab.c" /* yacc.c:1646  */
    break;

    case 326:
#line 1201 "xi-grammar.y" /* yacc.c:1646  */
    {
      ERROR("case blocks can only contain when clauses", (yylsp[0]).first_column,
            (yylsp[0]).last_column);
      (yyval.clist) = 0;
    }
#line 4388 "y.tab.c" /* yacc.c:1646  */
    break;

    case 327:
#line 1209 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.strval) = (yyvsp[0].strval);
    }
#line 4394 "y.tab.c" /* yacc.c:1646  */
    break;

    case 328:
#line 1211 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.strval) = 0;
    }
#line 4400 "y.tab.c" /* yacc.c:1646  */
    break;

    case 329:
#line 1215 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.when) = new WhenConstruct((yyvsp[-2].entrylist), 0);
    }
#line 4406 "y.tab.c" /* yacc.c:1646  */
    break;

    case 330:
#line 1217 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.when) = new WhenConstruct((yyvsp[-1].entrylist), (yyvsp[0].sc));
    }
#line 4412 "y.tab.c" /* yacc.c:1646  */
    break;

    case 331:
#line 1219 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.when) = new WhenConstruct((yyvsp[-3].entrylist), (yyvsp[-1].slist));
    }
#line 4418 "y.tab.c" /* yacc.c:1646  */
    break;

    case 332:
#line 1223 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.when) = 0;
    }
#line 4424 "y.tab.c" /* yacc.c:1646  */
    break;

    case 333:
#line 1225 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.when) = 0;
    }
#line 4430 "y.tab.c" /* yacc.c:1646  */
    break;

    case 334:
#line 1227 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.when) = 0;
    }
#line 4436 "y.tab.c" /* yacc.c:1646  */
    break;

    case 335:
#line 1229 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.when) = 0;
    }
#line 4442 "y.tab.c" /* yacc.c:1646  */
    break;

    case 336:
#line 1231 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.when) = 0;
    }
#line 4448 "y.tab.c" /* yacc.c:1646  */
    break;

    case 337:
#line 1233 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.when) = 0;
    }
#line 4454 "y.tab.c" /* yacc.c:1646  */
    break;

    case 338:
#line 1235 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.when) = 0;
    }
#line 4460 "y.tab.c" /* yacc.c:1646  */
    break;

    case 339:
#line 1237 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.when) = 0;
    }
#line 4466 "y.tab.c" /* yacc.c:1646  */
    break;

    case 340:
#line 1239 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.when) = 0;
    }
#line 4472 "y.tab.c" /* yacc.c:1646  */
    break;

    case 341:
#line 1241 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.when) = 0;
    }
#line 4478 "y.tab.c" /* yacc.c:1646  */
    break;

    case 342:
#line 1243 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.when) = 0;
    }
#line 4484 "y.tab.c" /* yacc.c:1646  */
    break;

    case 343:
#line 1245 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.when) = 0;
    }
#line 4490 "y.tab.c" /* yacc.c:1646  */
    break;

    case 344:
#line 1249 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.sc) = new SerialConstruct((yyvsp[-2].strval), (yyvsp[-4].strval),
                                       (yylsp[-3]).first_line);
    }
#line 4496 "y.tab.c" /* yacc.c:1646  */
    break;

    case 345:
#line 1251 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.sc) = new OverlapConstruct((yyvsp[-1].olist));
    }
#line 4502 "y.tab.c" /* yacc.c:1646  */
    break;

    case 346:
#line 1253 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.sc) = (yyvsp[0].when);
    }
#line 4508 "y.tab.c" /* yacc.c:1646  */
    break;

    case 347:
#line 1255 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.sc) = new CaseConstruct((yyvsp[-1].clist));
    }
#line 4514 "y.tab.c" /* yacc.c:1646  */
    break;

    case 348:
#line 1257 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.sc) = new ForConstruct((yyvsp[-8].intexpr), (yyvsp[-6].intexpr),
                                    (yyvsp[-4].intexpr), (yyvsp[-1].slist));
    }
#line 4520 "y.tab.c" /* yacc.c:1646  */
    break;

    case 349:
#line 1259 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.sc) = new ForConstruct((yyvsp[-6].intexpr), (yyvsp[-4].intexpr),
                                    (yyvsp[-2].intexpr), (yyvsp[0].sc));
    }
#line 4526 "y.tab.c" /* yacc.c:1646  */
    break;

    case 350:
#line 1261 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.sc) = new ForallConstruct(new SdagConstruct(SIDENT, (yyvsp[-9].strval)),
                                       (yyvsp[-6].intexpr), (yyvsp[-4].intexpr),
                                       (yyvsp[-2].intexpr), (yyvsp[0].sc));
    }
#line 4533 "y.tab.c" /* yacc.c:1646  */
    break;

    case 351:
#line 1264 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.sc) = new ForallConstruct(new SdagConstruct(SIDENT, (yyvsp[-11].strval)),
                                       (yyvsp[-8].intexpr), (yyvsp[-6].intexpr),
                                       (yyvsp[-4].intexpr), (yyvsp[-1].slist));
    }
#line 4540 "y.tab.c" /* yacc.c:1646  */
    break;

    case 352:
#line 1267 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.sc) = new IfConstruct((yyvsp[-3].intexpr), (yyvsp[-1].sc), (yyvsp[0].sc));
    }
#line 4546 "y.tab.c" /* yacc.c:1646  */
    break;

    case 353:
#line 1269 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.sc) = new IfConstruct((yyvsp[-5].intexpr), (yyvsp[-2].slist), (yyvsp[0].sc));
    }
#line 4552 "y.tab.c" /* yacc.c:1646  */
    break;

    case 354:
#line 1271 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.sc) = new WhileConstruct((yyvsp[-2].intexpr), (yyvsp[0].sc));
    }
#line 4558 "y.tab.c" /* yacc.c:1646  */
    break;

    case 355:
#line 1273 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.sc) = new WhileConstruct((yyvsp[-4].intexpr), (yyvsp[-1].slist));
    }
#line 4564 "y.tab.c" /* yacc.c:1646  */
    break;

    case 356:
#line 1275 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.sc) = new SerialConstruct((yyvsp[-2].strval), NULL, (yyloc).first_line);
    }
#line 4570 "y.tab.c" /* yacc.c:1646  */
    break;

    case 357:
#line 1277 "xi-grammar.y" /* yacc.c:1646  */
    {
      ERROR(
          "unknown SDAG construct or malformed entry method declaration.\n"
          "You may have forgotten to terminate a previous entry method declaration with a"
          " semicolon or forgotten to mark a block of sequential SDAG code as 'serial'",
          (yyloc).first_column, (yyloc).last_column);
      YYABORT;
    }
#line 4582 "y.tab.c" /* yacc.c:1646  */
    break;

    case 358:
#line 1287 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.sc) = 0;
    }
#line 4588 "y.tab.c" /* yacc.c:1646  */
    break;

    case 359:
#line 1289 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.sc) = new ElseConstruct((yyvsp[0].sc));
    }
#line 4594 "y.tab.c" /* yacc.c:1646  */
    break;

    case 360:
#line 1291 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.sc) = new ElseConstruct((yyvsp[-1].slist));
    }
#line 4600 "y.tab.c" /* yacc.c:1646  */
    break;

    case 361:
#line 1295 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.intexpr) = new IntExprConstruct((yyvsp[0].strval));
    }
#line 4606 "y.tab.c" /* yacc.c:1646  */
    break;

    case 362:
#line 1299 "xi-grammar.y" /* yacc.c:1646  */
    {
      in_int_expr = 0;
      (yyval.intval) = 0;
    }
#line 4612 "y.tab.c" /* yacc.c:1646  */
    break;

    case 363:
#line 1303 "xi-grammar.y" /* yacc.c:1646  */
    {
      in_int_expr = 1;
      (yyval.intval) = 0;
    }
#line 4618 "y.tab.c" /* yacc.c:1646  */
    break;

    case 364:
#line 1307 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.entry) = new Entry(lineno, 0, 0, (yyvsp[-1].strval), (yyvsp[0].plist), 0, 0,
                                0, (yyloc).first_line, (yyloc).last_line);
      firstRdma = true;
    }
#line 4627 "y.tab.c" /* yacc.c:1646  */
    break;

    case 365:
#line 1312 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.entry) =
          new Entry(lineno, 0, 0, (yyvsp[-4].strval), (yyvsp[0].plist), 0, 0,
                    (yyvsp[-2].strval), (yyloc).first_line, (yyloc).last_line);
      firstRdma = true;
    }
#line 4636 "y.tab.c" /* yacc.c:1646  */
    break;

    case 366:
#line 1319 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.entrylist) = new EntryList((yyvsp[0].entry));
    }
#line 4642 "y.tab.c" /* yacc.c:1646  */
    break;

    case 367:
#line 1321 "xi-grammar.y" /* yacc.c:1646  */
    {
      (yyval.entrylist) = new EntryList((yyvsp[-2].entry), (yyvsp[0].entrylist));
    }
#line 4648 "y.tab.c" /* yacc.c:1646  */
    break;

    case 368:
#line 1325 "xi-grammar.y" /* yacc.c:1646  */
    {
      in_bracket = 1;
    }
#line 4654 "y.tab.c" /* yacc.c:1646  */
    break;

    case 369:
#line 1328 "xi-grammar.y" /* yacc.c:1646  */
    {
      in_bracket = 0;
    }
#line 4660 "y.tab.c" /* yacc.c:1646  */
    break;

    case 370:
#line 1332 "xi-grammar.y" /* yacc.c:1646  */
    {
      if (!macroDefined((yyvsp[0].strval), 1)) in_comment = 1;
    }
#line 4666 "y.tab.c" /* yacc.c:1646  */
    break;

    case 371:
#line 1336 "xi-grammar.y" /* yacc.c:1646  */
    {
      if (!macroDefined((yyvsp[0].strval), 0)) in_comment = 1;
    }
#line 4672 "y.tab.c" /* yacc.c:1646  */
    break;

#line 4676 "y.tab.c" /* yacc.c:1646  */
    default:
      break;
  }
  /* User semantic actions sometimes alter yychar, and that requires
     that yytoken be updated with the new translation.  We take the
     approach of translating immediately before every use of yytoken.
     One alternative is translating here after every semantic action,
     but that translation would be missed if the semantic action invokes
     YYABORT, YYACCEPT, or YYERROR immediately after altering yychar or
     if it invokes YYBACKUP.  In the case of YYABORT or YYACCEPT, an
     incorrect destructor might then be invoked immediately.  In the
     case of YYERROR or YYBACKUP, subsequent parser actions might lead
     to an incorrect destructor call or verbose syntax error message
     before the lookahead is translated.  */
  YY_SYMBOL_PRINT("-> $$ =", yyr1[yyn], &yyval, &yyloc);

  YYPOPSTACK(yylen);
  yylen = 0;
  YY_STACK_PRINT(yyss, yyssp);

  *++yyvsp = yyval;
  *++yylsp = yyloc;

  /* Now 'shift' the result of the reduction.  Determine what state
     that goes to, based on the state we popped back to and the rule
     number reduced by.  */

  yyn = yyr1[yyn];

  yystate = yypgoto[yyn - YYNTOKENS] + *yyssp;
  if (0 <= yystate && yystate <= YYLAST && yycheck[yystate] == *yyssp)
    yystate = yytable[yystate];
  else
    yystate = yydefgoto[yyn - YYNTOKENS];

  goto yynewstate;

/*--------------------------------------.
| yyerrlab -- here on detecting error.  |
`--------------------------------------*/
yyerrlab:
  /* Make sure we have latest lookahead translation.  See comments at
     user semantic actions for why this is necessary.  */
  yytoken = yychar == YYEMPTY ? YYEMPTY : YYTRANSLATE(yychar);

  /* If not already recovering from an error, report this error.  */
  if (!yyerrstatus) {
    ++yynerrs;
#if !YYERROR_VERBOSE
    yyerror(YY_("syntax error"));
#else
#define YYSYNTAX_ERROR yysyntax_error(&yymsg_alloc, &yymsg, yyssp, yytoken)
    {
      char const* yymsgp = YY_("syntax error");
      int yysyntax_error_status;
      yysyntax_error_status = YYSYNTAX_ERROR;
      if (yysyntax_error_status == 0)
        yymsgp = yymsg;
      else if (yysyntax_error_status == 1) {
        if (yymsg != yymsgbuf) YYSTACK_FREE(yymsg);
        yymsg = (char*)YYSTACK_ALLOC(yymsg_alloc);
        if (!yymsg) {
          yymsg = yymsgbuf;
          yymsg_alloc = sizeof yymsgbuf;
          yysyntax_error_status = 2;
        } else {
          yysyntax_error_status = YYSYNTAX_ERROR;
          yymsgp = yymsg;
        }
      }
      yyerror(yymsgp);
      if (yysyntax_error_status == 2) goto yyexhaustedlab;
    }
#undef YYSYNTAX_ERROR
#endif
  }

  yyerror_range[1] = yylloc;

  if (yyerrstatus == 3) {
    /* If just tried and failed to reuse lookahead token after an
       error, discard it.  */

    if (yychar <= YYEOF) {
      /* Return failure if at end of input.  */
      if (yychar == YYEOF) YYABORT;
    } else {
      yydestruct("Error: discarding", yytoken, &yylval, &yylloc);
      yychar = YYEMPTY;
    }
  }

  /* Else will try to reuse lookahead token after shifting the error
     token.  */
  goto yyerrlab1;

/*---------------------------------------------------.
| yyerrorlab -- error raised explicitly by YYERROR.  |
`---------------------------------------------------*/
yyerrorlab:

  /* Pacify compilers like GCC when the user code never invokes
     YYERROR and the label yyerrorlab therefore never appears in user
     code.  */
  if (/*CONSTCOND*/ 0) goto yyerrorlab;

  yyerror_range[1] = yylsp[1 - yylen];
  /* Do not reclaim the symbols of the rule whose action triggered
     this YYERROR.  */
  YYPOPSTACK(yylen);
  yylen = 0;
  YY_STACK_PRINT(yyss, yyssp);
  yystate = *yyssp;
  goto yyerrlab1;

/*-------------------------------------------------------------.
| yyerrlab1 -- common code for both syntax error and YYERROR.  |
`-------------------------------------------------------------*/
yyerrlab1:
  yyerrstatus = 3; /* Each real token shifted decrements this.  */

  for (;;) {
    yyn = yypact[yystate];
    if (!yypact_value_is_default(yyn)) {
      yyn += YYTERROR;
      if (0 <= yyn && yyn <= YYLAST && yycheck[yyn] == YYTERROR) {
        yyn = yytable[yyn];
        if (0 < yyn) break;
      }
    }

    /* Pop the current state because it cannot handle the error token.  */
    if (yyssp == yyss) YYABORT;

    yyerror_range[1] = *yylsp;
    yydestruct("Error: popping", yystos[yystate], yyvsp, yylsp);
    YYPOPSTACK(1);
    yystate = *yyssp;
    YY_STACK_PRINT(yyss, yyssp);
  }

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END

  yyerror_range[2] = yylloc;
  /* Using YYLLOC is tempting, but would change the location of
     the lookahead.  YYLOC is available though.  */
  YYLLOC_DEFAULT(yyloc, yyerror_range, 2);
  *++yylsp = yyloc;

  /* Shift the error token.  */
  YY_SYMBOL_PRINT("Shifting", yystos[yyn], yyvsp, yylsp);

  yystate = yyn;
  goto yynewstate;

/*-------------------------------------.
| yyacceptlab -- YYACCEPT comes here.  |
`-------------------------------------*/
yyacceptlab:
  yyresult = 0;
  goto yyreturn;

/*-----------------------------------.
| yyabortlab -- YYABORT comes here.  |
`-----------------------------------*/
yyabortlab:
  yyresult = 1;
  goto yyreturn;

#if !defined yyoverflow || YYERROR_VERBOSE
/*-------------------------------------------------.
| yyexhaustedlab -- memory exhaustion comes here.  |
`-------------------------------------------------*/
yyexhaustedlab:
  yyerror(YY_("memory exhausted"));
  yyresult = 2;
/* Fall through.  */
#endif

yyreturn:
  if (yychar != YYEMPTY) {
    /* Make sure we have latest lookahead translation.  See comments at
       user semantic actions for why this is necessary.  */
    yytoken = YYTRANSLATE(yychar);
    yydestruct("Cleanup: discarding lookahead", yytoken, &yylval, &yylloc);
  }
  /* Do not reclaim the symbols of the rule whose action triggered
     this YYABORT or YYACCEPT.  */
  YYPOPSTACK(yylen);
  YY_STACK_PRINT(yyss, yyssp);
  while (yyssp != yyss) {
    yydestruct("Cleanup: popping", yystos[*yyssp], yyvsp, yylsp);
    YYPOPSTACK(1);
  }
#ifndef yyoverflow
  if (yyss != yyssa) YYSTACK_FREE(yyss);
#endif
#if YYERROR_VERBOSE
  if (yymsg != yymsgbuf) YYSTACK_FREE(yymsg);
#endif
  return yyresult;
}
#line 1339 "xi-grammar.y" /* yacc.c:1906  */

void yyerror(const char* s) {
  fprintf(stderr,
          "[PARSE-ERROR] Unexpected/missing token at line %d. Current token being "
          "parsed: '%s'.\n",
          lineno, yytext);
}

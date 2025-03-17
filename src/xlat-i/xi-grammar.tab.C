/* A Bison parser, made by GNU Bison 3.8.2.  */

/* Bison implementation for Yacc-like parsers in C

   Copyright (C) 1984, 1989-1990, 2000-2015, 2018-2021 Free Software Foundation,
   Inc.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <https://www.gnu.org/licenses/>.  */

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

/* DO NOT RELY ON FEATURES THAT ARE NOT DOCUMENTED in the manual,
   especially those whose name start with YY_ or yy_.  They are
   private implementation details that can be changed or removed.  */

/* All symbols defined below should begin with yy or YY, to avoid
   infringing on user name space.  This should be done even for local
   variables, as they might otherwise be expanded by user macros.
   There are some unavoidable exceptions within include files to
   define necessary library symbols; they are noted "INFRINGES ON
   USER NAME SPACE" below.  */

/* Identify Bison output, and Bison version.  */
#define YYBISON 30802

/* Bison version string.  */
#define YYBISON_VERSION "3.8.2"

/* Skeleton name.  */
#define YYSKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define YYPURE 0

/* Push parsers.  */
#define YYPUSH 0

/* Pull parsers.  */
#define YYPULL 1




/* First part of user prologue.  */
#line 2 "xi-grammar.y"

#include <iostream>
#include <string>
#include <string.h>
#include "xi-symbol.h"
#include "sdag/constructs/Constructs.h"
#include "EToken.h"
#include "xi-Chare.h"

// Has to be a macro since YYABORT can only be used within rule actions.
#define ERROR(...) \
  if (xi::num_errors++ == xi::MAX_NUM_ERRORS) { \
    YYABORT;                                    \
  } else {                                      \
    xi::pretty_msg("error", __VA_ARGS__);       \
  }

#define WARNING(...) \
  if (enable_warnings) {                    \
    xi::pretty_msg("warning", __VA_ARGS__); \
  }

using namespace xi;
extern int yylex (void) ;
extern unsigned char in_comment;
extern unsigned int lineno;
extern int in_bracket,in_braces,in_int_expr;
extern std::list<Entry *> connectEntries;
extern char* yytext;
AstChildren<Module> *modlist;

void yyerror(const char *);

namespace xi {

const int MAX_NUM_ERRORS = 10;
int num_errors = 0;
bool firstRdma = true;
bool firstDeviceRdma = true;

bool enable_warnings = true;

extern int macroDefined(const char *str, int istrue);
extern const char *python_doc;
extern char *fname;
void splitScopedName(const char* name, const char** scope, const char** basename);
void ReservedWord(int token, int fCol, int lCol);
}

#line 121 "y.tab.c"

# ifndef YY_CAST
#  ifdef __cplusplus
#   define YY_CAST(Type, Val) static_cast<Type> (Val)
#   define YY_REINTERPRET_CAST(Type, Val) reinterpret_cast<Type> (Val)
#  else
#   define YY_CAST(Type, Val) ((Type) (Val))
#   define YY_REINTERPRET_CAST(Type, Val) ((Type) (Val))
#  endif
# endif
# ifndef YY_NULLPTR
#  if defined __cplusplus
#   if 201103L <= __cplusplus
#    define YY_NULLPTR nullptr
#   else
#    define YY_NULLPTR 0
#   endif
#  else
#   define YY_NULLPTR ((void*)0)
#  endif
# endif

/* Use api.header.include to #include this header
   instead of duplicating it here.  */
#ifndef YY_YY_Y_TAB_H_INCLUDED
# define YY_YY_Y_TAB_H_INCLUDED
/* Debug traces.  */
#ifndef YYDEBUG
# define YYDEBUG 0
#endif
#if YYDEBUG
extern int yydebug;
#endif

/* Token kinds.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
  enum yytokentype
  {
    YYEMPTY = -2,
    YYEOF = 0,                     /* "end of file"  */
    YYerror = 256,                 /* error  */
    YYUNDEF = 257,                 /* "invalid token"  */
    MODULE = 258,                  /* MODULE  */
    MAINMODULE = 259,              /* MAINMODULE  */
    EXTERN = 260,                  /* EXTERN  */
    READONLY = 261,                /* READONLY  */
    INITCALL = 262,                /* INITCALL  */
    INITNODE = 263,                /* INITNODE  */
    INITPROC = 264,                /* INITPROC  */
    PUPABLE = 265,                 /* PUPABLE  */
    CHARE = 266,                   /* CHARE  */
    MAINCHARE = 267,               /* MAINCHARE  */
    GROUP = 268,                   /* GROUP  */
    NODEGROUP = 269,               /* NODEGROUP  */
    ARRAY = 270,                   /* ARRAY  */
    MESSAGE = 271,                 /* MESSAGE  */
    CONDITIONAL = 272,             /* CONDITIONAL  */
    CLASS = 273,                   /* CLASS  */
    INCLUDE = 274,                 /* INCLUDE  */
    STACKSIZE = 275,               /* STACKSIZE  */
    THREADED = 276,                /* THREADED  */
    TEMPLATE = 277,                /* TEMPLATE  */
    WHENIDLE = 278,                /* WHENIDLE  */
    SYNC = 279,                    /* SYNC  */
    IGET = 280,                    /* IGET  */
    EXCLUSIVE = 281,               /* EXCLUSIVE  */
    IMMEDIATE = 282,               /* IMMEDIATE  */
    SKIPSCHED = 283,               /* SKIPSCHED  */
    INLINE = 284,                  /* INLINE  */
    VIRTUAL = 285,                 /* VIRTUAL  */
    MIGRATABLE = 286,              /* MIGRATABLE  */
    AGGREGATE = 287,               /* AGGREGATE  */
    CREATEHERE = 288,              /* CREATEHERE  */
    CREATEHOME = 289,              /* CREATEHOME  */
    NOKEEP = 290,                  /* NOKEEP  */
    NOTRACE = 291,                 /* NOTRACE  */
    APPWORK = 292,                 /* APPWORK  */
    VOID = 293,                    /* VOID  */
    CONST = 294,                   /* CONST  */
    NOCOPY = 295,                  /* NOCOPY  */
    NOCOPYPOST = 296,              /* NOCOPYPOST  */
    NOCOPYDEVICE = 297,            /* NOCOPYDEVICE  */
    PACKED = 298,                  /* PACKED  */
    VARSIZE = 299,                 /* VARSIZE  */
    ENTRY = 300,                   /* ENTRY  */
    FOR = 301,                     /* FOR  */
    FORALL = 302,                  /* FORALL  */
    WHILE = 303,                   /* WHILE  */
    WHEN = 304,                    /* WHEN  */
    OVERLAP = 305,                 /* OVERLAP  */
    SERIAL = 306,                  /* SERIAL  */
    IF = 307,                      /* IF  */
    ELSE = 308,                    /* ELSE  */
    PYTHON = 309,                  /* PYTHON  */
    LOCAL = 310,                   /* LOCAL  */
    NAMESPACE = 311,               /* NAMESPACE  */
    USING = 312,                   /* USING  */
    IDENT = 313,                   /* IDENT  */
    NUMBER = 314,                  /* NUMBER  */
    LITERAL = 315,                 /* LITERAL  */
    CPROGRAM = 316,                /* CPROGRAM  */
    HASHIF = 317,                  /* HASHIF  */
    HASHIFDEF = 318,               /* HASHIFDEF  */
    INT = 319,                     /* INT  */
    LONG = 320,                    /* LONG  */
    SHORT = 321,                   /* SHORT  */
    CHAR = 322,                    /* CHAR  */
    FLOAT = 323,                   /* FLOAT  */
    DOUBLE = 324,                  /* DOUBLE  */
    UNSIGNED = 325,                /* UNSIGNED  */
    ACCEL = 326,                   /* ACCEL  */
    READWRITE = 327,               /* READWRITE  */
    WRITEONLY = 328,               /* WRITEONLY  */
    ACCELBLOCK = 329,              /* ACCELBLOCK  */
    MEMCRITICAL = 330,             /* MEMCRITICAL  */
    REDUCTIONTARGET = 331,         /* REDUCTIONTARGET  */
    CASE = 332,                    /* CASE  */
    TYPENAME = 333                 /* TYPENAME  */
  };
  typedef enum yytokentype yytoken_kind_t;
#endif
/* Token kinds.  */
#define YYEMPTY -2
#define YYEOF 0
#define YYerror 256
#define YYUNDEF 257
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
#define WHENIDLE 278
#define SYNC 279
#define IGET 280
#define EXCLUSIVE 281
#define IMMEDIATE 282
#define SKIPSCHED 283
#define INLINE 284
#define VIRTUAL 285
#define MIGRATABLE 286
#define AGGREGATE 287
#define CREATEHERE 288
#define CREATEHOME 289
#define NOKEEP 290
#define NOTRACE 291
#define APPWORK 292
#define VOID 293
#define CONST 294
#define NOCOPY 295
#define NOCOPYPOST 296
#define NOCOPYDEVICE 297
#define PACKED 298
#define VARSIZE 299
#define ENTRY 300
#define FOR 301
#define FORALL 302
#define WHILE 303
#define WHEN 304
#define OVERLAP 305
#define SERIAL 306
#define IF 307
#define ELSE 308
#define PYTHON 309
#define LOCAL 310
#define NAMESPACE 311
#define USING 312
#define IDENT 313
#define NUMBER 314
#define LITERAL 315
#define CPROGRAM 316
#define HASHIF 317
#define HASHIFDEF 318
#define INT 319
#define LONG 320
#define SHORT 321
#define CHAR 322
#define FLOAT 323
#define DOUBLE 324
#define UNSIGNED 325
#define ACCEL 326
#define READWRITE 327
#define WRITEONLY 328
#define ACCELBLOCK 329
#define MEMCRITICAL 330
#define REDUCTIONTARGET 331
#define CASE 332
#define TYPENAME 333

/* Value type.  */
#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
union YYSTYPE
{
#line 54 "xi-grammar.y"

  Attribute *attr;
  Attribute::Argument *attrarg;
  AstChildren<Module> *modlist;
  Module *module;
  ConstructList *conslist;
  Construct *construct;
  TParam *tparam;
  TParamList *tparlist;
  Type *type;
  PtrType *ptype;
  NamedType *ntype;
  FuncType *ftype;
  Readonly *readonly;
  Message *message;
  Chare *chare;
  Entry *entry;
  EntryList *entrylist;
  Parameter *pname;
  ParamList *plist;
  Template *templat;
  TypeList *typelist;
  AstChildren<Member> *mbrlist;
  Member *member;
  TVar *tvar;
  TVarList *tvarlist;
  Value *val;
  ValueList *vallist;
  MsgVar *mv;
  MsgVarList *mvlist;
  PUPableClass *pupable;
  IncludeFile *includeFile;
  const char *strval;
  int intval;
  unsigned int cattr; // actually Chare::attrib_t, but referring to that creates nasty #include issues
  SdagConstruct *sc;
  IntExprConstruct *intexpr;
  WhenConstruct *when;
  SListConstruct *slist;
  CaseListConstruct *clist;
  OListConstruct *olist;
  SdagEntryConstruct *sentry;
  XStr* xstrptr;
  AccelBlock* accelBlock;

#line 376 "y.tab.c"

};
typedef union YYSTYPE YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define YYSTYPE_IS_DECLARED 1
#endif

/* Location type.  */
#if ! defined YYLTYPE && ! defined YYLTYPE_IS_DECLARED
typedef struct YYLTYPE YYLTYPE;
struct YYLTYPE
{
  int first_line;
  int first_column;
  int last_line;
  int last_column;
};
# define YYLTYPE_IS_DECLARED 1
# define YYLTYPE_IS_TRIVIAL 1
#endif


extern YYSTYPE yylval;
extern YYLTYPE yylloc;

int yyparse (void);


#endif /* !YY_YY_Y_TAB_H_INCLUDED  */
/* Symbol kind.  */
enum yysymbol_kind_t
{
  YYSYMBOL_YYEMPTY = -2,
  YYSYMBOL_YYEOF = 0,                      /* "end of file"  */
  YYSYMBOL_YYerror = 1,                    /* error  */
  YYSYMBOL_YYUNDEF = 2,                    /* "invalid token"  */
  YYSYMBOL_MODULE = 3,                     /* MODULE  */
  YYSYMBOL_MAINMODULE = 4,                 /* MAINMODULE  */
  YYSYMBOL_EXTERN = 5,                     /* EXTERN  */
  YYSYMBOL_READONLY = 6,                   /* READONLY  */
  YYSYMBOL_INITCALL = 7,                   /* INITCALL  */
  YYSYMBOL_INITNODE = 8,                   /* INITNODE  */
  YYSYMBOL_INITPROC = 9,                   /* INITPROC  */
  YYSYMBOL_PUPABLE = 10,                   /* PUPABLE  */
  YYSYMBOL_CHARE = 11,                     /* CHARE  */
  YYSYMBOL_MAINCHARE = 12,                 /* MAINCHARE  */
  YYSYMBOL_GROUP = 13,                     /* GROUP  */
  YYSYMBOL_NODEGROUP = 14,                 /* NODEGROUP  */
  YYSYMBOL_ARRAY = 15,                     /* ARRAY  */
  YYSYMBOL_MESSAGE = 16,                   /* MESSAGE  */
  YYSYMBOL_CONDITIONAL = 17,               /* CONDITIONAL  */
  YYSYMBOL_CLASS = 18,                     /* CLASS  */
  YYSYMBOL_INCLUDE = 19,                   /* INCLUDE  */
  YYSYMBOL_STACKSIZE = 20,                 /* STACKSIZE  */
  YYSYMBOL_THREADED = 21,                  /* THREADED  */
  YYSYMBOL_TEMPLATE = 22,                  /* TEMPLATE  */
  YYSYMBOL_WHENIDLE = 23,                  /* WHENIDLE  */
  YYSYMBOL_SYNC = 24,                      /* SYNC  */
  YYSYMBOL_IGET = 25,                      /* IGET  */
  YYSYMBOL_EXCLUSIVE = 26,                 /* EXCLUSIVE  */
  YYSYMBOL_IMMEDIATE = 27,                 /* IMMEDIATE  */
  YYSYMBOL_SKIPSCHED = 28,                 /* SKIPSCHED  */
  YYSYMBOL_INLINE = 29,                    /* INLINE  */
  YYSYMBOL_VIRTUAL = 30,                   /* VIRTUAL  */
  YYSYMBOL_MIGRATABLE = 31,                /* MIGRATABLE  */
  YYSYMBOL_AGGREGATE = 32,                 /* AGGREGATE  */
  YYSYMBOL_CREATEHERE = 33,                /* CREATEHERE  */
  YYSYMBOL_CREATEHOME = 34,                /* CREATEHOME  */
  YYSYMBOL_NOKEEP = 35,                    /* NOKEEP  */
  YYSYMBOL_NOTRACE = 36,                   /* NOTRACE  */
  YYSYMBOL_APPWORK = 37,                   /* APPWORK  */
  YYSYMBOL_VOID = 38,                      /* VOID  */
  YYSYMBOL_CONST = 39,                     /* CONST  */
  YYSYMBOL_NOCOPY = 40,                    /* NOCOPY  */
  YYSYMBOL_NOCOPYPOST = 41,                /* NOCOPYPOST  */
  YYSYMBOL_NOCOPYDEVICE = 42,              /* NOCOPYDEVICE  */
  YYSYMBOL_PACKED = 43,                    /* PACKED  */
  YYSYMBOL_VARSIZE = 44,                   /* VARSIZE  */
  YYSYMBOL_ENTRY = 45,                     /* ENTRY  */
  YYSYMBOL_FOR = 46,                       /* FOR  */
  YYSYMBOL_FORALL = 47,                    /* FORALL  */
  YYSYMBOL_WHILE = 48,                     /* WHILE  */
  YYSYMBOL_WHEN = 49,                      /* WHEN  */
  YYSYMBOL_OVERLAP = 50,                   /* OVERLAP  */
  YYSYMBOL_SERIAL = 51,                    /* SERIAL  */
  YYSYMBOL_IF = 52,                        /* IF  */
  YYSYMBOL_ELSE = 53,                      /* ELSE  */
  YYSYMBOL_PYTHON = 54,                    /* PYTHON  */
  YYSYMBOL_LOCAL = 55,                     /* LOCAL  */
  YYSYMBOL_NAMESPACE = 56,                 /* NAMESPACE  */
  YYSYMBOL_USING = 57,                     /* USING  */
  YYSYMBOL_IDENT = 58,                     /* IDENT  */
  YYSYMBOL_NUMBER = 59,                    /* NUMBER  */
  YYSYMBOL_LITERAL = 60,                   /* LITERAL  */
  YYSYMBOL_CPROGRAM = 61,                  /* CPROGRAM  */
  YYSYMBOL_HASHIF = 62,                    /* HASHIF  */
  YYSYMBOL_HASHIFDEF = 63,                 /* HASHIFDEF  */
  YYSYMBOL_INT = 64,                       /* INT  */
  YYSYMBOL_LONG = 65,                      /* LONG  */
  YYSYMBOL_SHORT = 66,                     /* SHORT  */
  YYSYMBOL_CHAR = 67,                      /* CHAR  */
  YYSYMBOL_FLOAT = 68,                     /* FLOAT  */
  YYSYMBOL_DOUBLE = 69,                    /* DOUBLE  */
  YYSYMBOL_UNSIGNED = 70,                  /* UNSIGNED  */
  YYSYMBOL_ACCEL = 71,                     /* ACCEL  */
  YYSYMBOL_READWRITE = 72,                 /* READWRITE  */
  YYSYMBOL_WRITEONLY = 73,                 /* WRITEONLY  */
  YYSYMBOL_ACCELBLOCK = 74,                /* ACCELBLOCK  */
  YYSYMBOL_MEMCRITICAL = 75,               /* MEMCRITICAL  */
  YYSYMBOL_REDUCTIONTARGET = 76,           /* REDUCTIONTARGET  */
  YYSYMBOL_CASE = 77,                      /* CASE  */
  YYSYMBOL_TYPENAME = 78,                  /* TYPENAME  */
  YYSYMBOL_79_ = 79,                       /* ';'  */
  YYSYMBOL_80_ = 80,                       /* ':'  */
  YYSYMBOL_81_ = 81,                       /* '{'  */
  YYSYMBOL_82_ = 82,                       /* '}'  */
  YYSYMBOL_83_ = 83,                       /* ','  */
  YYSYMBOL_84_ = 84,                       /* '<'  */
  YYSYMBOL_85_ = 85,                       /* '>'  */
  YYSYMBOL_86_ = 86,                       /* '*'  */
  YYSYMBOL_87_ = 87,                       /* '('  */
  YYSYMBOL_88_ = 88,                       /* ')'  */
  YYSYMBOL_89_ = 89,                       /* '&'  */
  YYSYMBOL_90_ = 90,                       /* '.'  */
  YYSYMBOL_91_ = 91,                       /* '['  */
  YYSYMBOL_92_ = 92,                       /* ']'  */
  YYSYMBOL_93_ = 93,                       /* '='  */
  YYSYMBOL_94_ = 94,                       /* '-'  */
  YYSYMBOL_YYACCEPT = 95,                  /* $accept  */
  YYSYMBOL_File = 96,                      /* File  */
  YYSYMBOL_ModuleEList = 97,               /* ModuleEList  */
  YYSYMBOL_OptExtern = 98,                 /* OptExtern  */
  YYSYMBOL_OneOrMoreSemiColon = 99,        /* OneOrMoreSemiColon  */
  YYSYMBOL_OptSemiColon = 100,             /* OptSemiColon  */
  YYSYMBOL_Name = 101,                     /* Name  */
  YYSYMBOL_QualName = 102,                 /* QualName  */
  YYSYMBOL_Module = 103,                   /* Module  */
  YYSYMBOL_ConstructEList = 104,           /* ConstructEList  */
  YYSYMBOL_ConstructList = 105,            /* ConstructList  */
  YYSYMBOL_ConstructSemi = 106,            /* ConstructSemi  */
  YYSYMBOL_Construct = 107,                /* Construct  */
  YYSYMBOL_TParam = 108,                   /* TParam  */
  YYSYMBOL_TParamList = 109,               /* TParamList  */
  YYSYMBOL_TParamEList = 110,              /* TParamEList  */
  YYSYMBOL_OptTParams = 111,               /* OptTParams  */
  YYSYMBOL_BuiltinType = 112,              /* BuiltinType  */
  YYSYMBOL_NamedType = 113,                /* NamedType  */
  YYSYMBOL_QualNamedType = 114,            /* QualNamedType  */
  YYSYMBOL_SimpleType = 115,               /* SimpleType  */
  YYSYMBOL_OnePtrType = 116,               /* OnePtrType  */
  YYSYMBOL_PtrType = 117,                  /* PtrType  */
  YYSYMBOL_FuncType = 118,                 /* FuncType  */
  YYSYMBOL_BaseType = 119,                 /* BaseType  */
  YYSYMBOL_BaseDataType = 120,             /* BaseDataType  */
  YYSYMBOL_RestrictedType = 121,           /* RestrictedType  */
  YYSYMBOL_Type = 122,                     /* Type  */
  YYSYMBOL_ArrayDim = 123,                 /* ArrayDim  */
  YYSYMBOL_Dim = 124,                      /* Dim  */
  YYSYMBOL_DimList = 125,                  /* DimList  */
  YYSYMBOL_Readonly = 126,                 /* Readonly  */
  YYSYMBOL_ReadonlyMsg = 127,              /* ReadonlyMsg  */
  YYSYMBOL_OptVoid = 128,                  /* OptVoid  */
  YYSYMBOL_MAttribs = 129,                 /* MAttribs  */
  YYSYMBOL_MAttribList = 130,              /* MAttribList  */
  YYSYMBOL_MAttrib = 131,                  /* MAttrib  */
  YYSYMBOL_CAttribs = 132,                 /* CAttribs  */
  YYSYMBOL_CAttribList = 133,              /* CAttribList  */
  YYSYMBOL_PythonOptions = 134,            /* PythonOptions  */
  YYSYMBOL_ArrayAttrib = 135,              /* ArrayAttrib  */
  YYSYMBOL_ArrayAttribs = 136,             /* ArrayAttribs  */
  YYSYMBOL_ArrayAttribList = 137,          /* ArrayAttribList  */
  YYSYMBOL_CAttrib = 138,                  /* CAttrib  */
  YYSYMBOL_OptConditional = 139,           /* OptConditional  */
  YYSYMBOL_MsgArray = 140,                 /* MsgArray  */
  YYSYMBOL_Var = 141,                      /* Var  */
  YYSYMBOL_VarList = 142,                  /* VarList  */
  YYSYMBOL_Message = 143,                  /* Message  */
  YYSYMBOL_OptBaseList = 144,              /* OptBaseList  */
  YYSYMBOL_BaseList = 145,                 /* BaseList  */
  YYSYMBOL_Chare = 146,                    /* Chare  */
  YYSYMBOL_Group = 147,                    /* Group  */
  YYSYMBOL_NodeGroup = 148,                /* NodeGroup  */
  YYSYMBOL_ArrayIndexType = 149,           /* ArrayIndexType  */
  YYSYMBOL_Array = 150,                    /* Array  */
  YYSYMBOL_TChare = 151,                   /* TChare  */
  YYSYMBOL_TGroup = 152,                   /* TGroup  */
  YYSYMBOL_TNodeGroup = 153,               /* TNodeGroup  */
  YYSYMBOL_TArray = 154,                   /* TArray  */
  YYSYMBOL_TMessage = 155,                 /* TMessage  */
  YYSYMBOL_OptTypeInit = 156,              /* OptTypeInit  */
  YYSYMBOL_OptNameInit = 157,              /* OptNameInit  */
  YYSYMBOL_TVar = 158,                     /* TVar  */
  YYSYMBOL_TVarList = 159,                 /* TVarList  */
  YYSYMBOL_TemplateSpec = 160,             /* TemplateSpec  */
  YYSYMBOL_Template = 161,                 /* Template  */
  YYSYMBOL_MemberEList = 162,              /* MemberEList  */
  YYSYMBOL_MemberList = 163,               /* MemberList  */
  YYSYMBOL_NonEntryMember = 164,           /* NonEntryMember  */
  YYSYMBOL_InitNode = 165,                 /* InitNode  */
  YYSYMBOL_InitProc = 166,                 /* InitProc  */
  YYSYMBOL_PUPableClass = 167,             /* PUPableClass  */
  YYSYMBOL_IncludeFile = 168,              /* IncludeFile  */
  YYSYMBOL_Member = 169,                   /* Member  */
  YYSYMBOL_MemberBody = 170,               /* MemberBody  */
  YYSYMBOL_UnexpectedToken = 171,          /* UnexpectedToken  */
  YYSYMBOL_Entry = 172,                    /* Entry  */
  YYSYMBOL_AccelBlock = 173,               /* AccelBlock  */
  YYSYMBOL_EReturn = 174,                  /* EReturn  */
  YYSYMBOL_EAttribs = 175,                 /* EAttribs  */
  YYSYMBOL_AttributeArg = 176,             /* AttributeArg  */
  YYSYMBOL_AttributeArgList = 177,         /* AttributeArgList  */
  YYSYMBOL_EAttribList = 178,              /* EAttribList  */
  YYSYMBOL_EAttrib = 179,                  /* EAttrib  */
  YYSYMBOL_DefaultParameter = 180,         /* DefaultParameter  */
  YYSYMBOL_CPROGRAM_List = 181,            /* CPROGRAM_List  */
  YYSYMBOL_CCode = 182,                    /* CCode  */
  YYSYMBOL_ParamBracketStart = 183,        /* ParamBracketStart  */
  YYSYMBOL_ParamBraceStart = 184,          /* ParamBraceStart  */
  YYSYMBOL_ParamBraceEnd = 185,            /* ParamBraceEnd  */
  YYSYMBOL_Parameter = 186,                /* Parameter  */
  YYSYMBOL_AccelBufferType = 187,          /* AccelBufferType  */
  YYSYMBOL_AccelInstName = 188,            /* AccelInstName  */
  YYSYMBOL_AccelArrayParam = 189,          /* AccelArrayParam  */
  YYSYMBOL_AccelParameter = 190,           /* AccelParameter  */
  YYSYMBOL_ParamList = 191,                /* ParamList  */
  YYSYMBOL_AccelParamList = 192,           /* AccelParamList  */
  YYSYMBOL_EParameters = 193,              /* EParameters  */
  YYSYMBOL_AccelEParameters = 194,         /* AccelEParameters  */
  YYSYMBOL_OptStackSize = 195,             /* OptStackSize  */
  YYSYMBOL_OptSdagCode = 196,              /* OptSdagCode  */
  YYSYMBOL_Slist = 197,                    /* Slist  */
  YYSYMBOL_Olist = 198,                    /* Olist  */
  YYSYMBOL_CaseList = 199,                 /* CaseList  */
  YYSYMBOL_OptTraceName = 200,             /* OptTraceName  */
  YYSYMBOL_WhenConstruct = 201,            /* WhenConstruct  */
  YYSYMBOL_NonWhenConstruct = 202,         /* NonWhenConstruct  */
  YYSYMBOL_SingleConstruct = 203,          /* SingleConstruct  */
  YYSYMBOL_HasElse = 204,                  /* HasElse  */
  YYSYMBOL_IntExpr = 205,                  /* IntExpr  */
  YYSYMBOL_EndIntExpr = 206,               /* EndIntExpr  */
  YYSYMBOL_StartIntExpr = 207,             /* StartIntExpr  */
  YYSYMBOL_SEntry = 208,                   /* SEntry  */
  YYSYMBOL_SEntryList = 209,               /* SEntryList  */
  YYSYMBOL_SParamBracketStart = 210,       /* SParamBracketStart  */
  YYSYMBOL_SParamBracketEnd = 211,         /* SParamBracketEnd  */
  YYSYMBOL_HashIFComment = 212,            /* HashIFComment  */
  YYSYMBOL_HashIFDefComment = 213          /* HashIFDefComment  */
};
typedef enum yysymbol_kind_t yysymbol_kind_t;




#ifdef short
# undef short
#endif

/* On compilers that do not define __PTRDIFF_MAX__ etc., make sure
   <limits.h> and (if available) <stdint.h> are included
   so that the code can choose integer types of a good width.  */

#ifndef __PTRDIFF_MAX__
# include <limits.h> /* INFRINGES ON USER NAME SPACE */
# if defined __STDC_VERSION__ && 199901 <= __STDC_VERSION__
#  include <stdint.h> /* INFRINGES ON USER NAME SPACE */
#  define YY_STDINT_H
# endif
#endif

/* Narrow types that promote to a signed type and that can represent a
   signed or unsigned integer of at least N bits.  In tables they can
   save space and decrease cache pressure.  Promoting to a signed type
   helps avoid bugs in integer arithmetic.  */

#ifdef __INT_LEAST8_MAX__
typedef __INT_LEAST8_TYPE__ yytype_int8;
#elif defined YY_STDINT_H
typedef int_least8_t yytype_int8;
#else
typedef signed char yytype_int8;
#endif

#ifdef __INT_LEAST16_MAX__
typedef __INT_LEAST16_TYPE__ yytype_int16;
#elif defined YY_STDINT_H
typedef int_least16_t yytype_int16;
#else
typedef short yytype_int16;
#endif

/* Work around bug in HP-UX 11.23, which defines these macros
   incorrectly for preprocessor constants.  This workaround can likely
   be removed in 2023, as HPE has promised support for HP-UX 11.23
   (aka HP-UX 11i v2) only through the end of 2022; see Table 2 of
   <https://h20195.www2.hpe.com/V2/getpdf.aspx/4AA4-7673ENW.pdf>.  */
#ifdef __hpux
# undef UINT_LEAST8_MAX
# undef UINT_LEAST16_MAX
# define UINT_LEAST8_MAX 255
# define UINT_LEAST16_MAX 65535
#endif

#if defined __UINT_LEAST8_MAX__ && __UINT_LEAST8_MAX__ <= __INT_MAX__
typedef __UINT_LEAST8_TYPE__ yytype_uint8;
#elif (!defined __UINT_LEAST8_MAX__ && defined YY_STDINT_H \
       && UINT_LEAST8_MAX <= INT_MAX)
typedef uint_least8_t yytype_uint8;
#elif !defined __UINT_LEAST8_MAX__ && UCHAR_MAX <= INT_MAX
typedef unsigned char yytype_uint8;
#else
typedef short yytype_uint8;
#endif

#if defined __UINT_LEAST16_MAX__ && __UINT_LEAST16_MAX__ <= __INT_MAX__
typedef __UINT_LEAST16_TYPE__ yytype_uint16;
#elif (!defined __UINT_LEAST16_MAX__ && defined YY_STDINT_H \
       && UINT_LEAST16_MAX <= INT_MAX)
typedef uint_least16_t yytype_uint16;
#elif !defined __UINT_LEAST16_MAX__ && USHRT_MAX <= INT_MAX
typedef unsigned short yytype_uint16;
#else
typedef int yytype_uint16;
#endif

#ifndef YYPTRDIFF_T
# if defined __PTRDIFF_TYPE__ && defined __PTRDIFF_MAX__
#  define YYPTRDIFF_T __PTRDIFF_TYPE__
#  define YYPTRDIFF_MAXIMUM __PTRDIFF_MAX__
# elif defined PTRDIFF_MAX
#  ifndef ptrdiff_t
#   include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  endif
#  define YYPTRDIFF_T ptrdiff_t
#  define YYPTRDIFF_MAXIMUM PTRDIFF_MAX
# else
#  define YYPTRDIFF_T long
#  define YYPTRDIFF_MAXIMUM LONG_MAX
# endif
#endif

#ifndef YYSIZE_T
# ifdef __SIZE_TYPE__
#  define YYSIZE_T __SIZE_TYPE__
# elif defined size_t
#  define YYSIZE_T size_t
# elif defined __STDC_VERSION__ && 199901 <= __STDC_VERSION__
#  include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  define YYSIZE_T size_t
# else
#  define YYSIZE_T unsigned
# endif
#endif

#define YYSIZE_MAXIMUM                                  \
  YY_CAST (YYPTRDIFF_T,                                 \
           (YYPTRDIFF_MAXIMUM < YY_CAST (YYSIZE_T, -1)  \
            ? YYPTRDIFF_MAXIMUM                         \
            : YY_CAST (YYSIZE_T, -1)))

#define YYSIZEOF(X) YY_CAST (YYPTRDIFF_T, sizeof (X))


/* Stored state numbers (used for stacks). */
typedef yytype_int16 yy_state_t;

/* State numbers in computations.  */
typedef int yy_state_fast_t;

#ifndef YY_
# if defined YYENABLE_NLS && YYENABLE_NLS
#  if ENABLE_NLS
#   include <libintl.h> /* INFRINGES ON USER NAME SPACE */
#   define YY_(Msgid) dgettext ("bison-runtime", Msgid)
#  endif
# endif
# ifndef YY_
#  define YY_(Msgid) Msgid
# endif
#endif


#ifndef YY_ATTRIBUTE_PURE
# if defined __GNUC__ && 2 < __GNUC__ + (96 <= __GNUC_MINOR__)
#  define YY_ATTRIBUTE_PURE __attribute__ ((__pure__))
# else
#  define YY_ATTRIBUTE_PURE
# endif
#endif

#ifndef YY_ATTRIBUTE_UNUSED
# if defined __GNUC__ && 2 < __GNUC__ + (7 <= __GNUC_MINOR__)
#  define YY_ATTRIBUTE_UNUSED __attribute__ ((__unused__))
# else
#  define YY_ATTRIBUTE_UNUSED
# endif
#endif

/* Suppress unused-variable warnings by "using" E.  */
#if ! defined lint || defined __GNUC__
# define YY_USE(E) ((void) (E))
#else
# define YY_USE(E) /* empty */
#endif

/* Suppress an incorrect diagnostic about yylval being uninitialized.  */
#if defined __GNUC__ && ! defined __ICC && 406 <= __GNUC__ * 100 + __GNUC_MINOR__
# if __GNUC__ * 100 + __GNUC_MINOR__ < 407
#  define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN                           \
    _Pragma ("GCC diagnostic push")                                     \
    _Pragma ("GCC diagnostic ignored \"-Wuninitialized\"")
# else
#  define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN                           \
    _Pragma ("GCC diagnostic push")                                     \
    _Pragma ("GCC diagnostic ignored \"-Wuninitialized\"")              \
    _Pragma ("GCC diagnostic ignored \"-Wmaybe-uninitialized\"")
# endif
# define YY_IGNORE_MAYBE_UNINITIALIZED_END      \
    _Pragma ("GCC diagnostic pop")
#else
# define YY_INITIAL_VALUE(Value) Value
#endif
#ifndef YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_END
#endif
#ifndef YY_INITIAL_VALUE
# define YY_INITIAL_VALUE(Value) /* Nothing. */
#endif

#if defined __cplusplus && defined __GNUC__ && ! defined __ICC && 6 <= __GNUC__
# define YY_IGNORE_USELESS_CAST_BEGIN                          \
    _Pragma ("GCC diagnostic push")                            \
    _Pragma ("GCC diagnostic ignored \"-Wuseless-cast\"")
# define YY_IGNORE_USELESS_CAST_END            \
    _Pragma ("GCC diagnostic pop")
#endif
#ifndef YY_IGNORE_USELESS_CAST_BEGIN
# define YY_IGNORE_USELESS_CAST_BEGIN
# define YY_IGNORE_USELESS_CAST_END
#endif


#define YY_ASSERT(E) ((void) (0 && (E)))

#if !defined yyoverflow

/* The parser invokes alloca or malloc; define the necessary symbols.  */

# ifdef YYSTACK_USE_ALLOCA
#  if YYSTACK_USE_ALLOCA
#   ifdef __GNUC__
#    define YYSTACK_ALLOC __builtin_alloca
#   elif defined __BUILTIN_VA_ARG_INCR
#    include <alloca.h> /* INFRINGES ON USER NAME SPACE */
#   elif defined _AIX
#    define YYSTACK_ALLOC __alloca
#   elif defined _MSC_VER
#    include <malloc.h> /* INFRINGES ON USER NAME SPACE */
#    define alloca _alloca
#   else
#    define YYSTACK_ALLOC alloca
#    if ! defined _ALLOCA_H && ! defined EXIT_SUCCESS
#     include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
      /* Use EXIT_SUCCESS as a witness for stdlib.h.  */
#     ifndef EXIT_SUCCESS
#      define EXIT_SUCCESS 0
#     endif
#    endif
#   endif
#  endif
# endif

# ifdef YYSTACK_ALLOC
   /* Pacify GCC's 'empty if-body' warning.  */
#  define YYSTACK_FREE(Ptr) do { /* empty */; } while (0)
#  ifndef YYSTACK_ALLOC_MAXIMUM
    /* The OS might guarantee only one guard page at the bottom of the stack,
       and a page size can be as small as 4096 bytes.  So we cannot safely
       invoke alloca (N) if N exceeds 4096.  Use a slightly smaller number
       to allow for a few compiler-allocated temporary stack slots.  */
#   define YYSTACK_ALLOC_MAXIMUM 4032 /* reasonable circa 2006 */
#  endif
# else
#  define YYSTACK_ALLOC YYMALLOC
#  define YYSTACK_FREE YYFREE
#  ifndef YYSTACK_ALLOC_MAXIMUM
#   define YYSTACK_ALLOC_MAXIMUM YYSIZE_MAXIMUM
#  endif
#  if (defined __cplusplus && ! defined EXIT_SUCCESS \
       && ! ((defined YYMALLOC || defined malloc) \
             && (defined YYFREE || defined free)))
#   include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#   ifndef EXIT_SUCCESS
#    define EXIT_SUCCESS 0
#   endif
#  endif
#  ifndef YYMALLOC
#   define YYMALLOC malloc
#   if ! defined malloc && ! defined EXIT_SUCCESS
void *malloc (YYSIZE_T); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
#  ifndef YYFREE
#   define YYFREE free
#   if ! defined free && ! defined EXIT_SUCCESS
void free (void *); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
# endif
#endif /* !defined yyoverflow */

#if (! defined yyoverflow \
     && (! defined __cplusplus \
         || (defined YYLTYPE_IS_TRIVIAL && YYLTYPE_IS_TRIVIAL \
             && defined YYSTYPE_IS_TRIVIAL && YYSTYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union yyalloc
{
  yy_state_t yyss_alloc;
  YYSTYPE yyvs_alloc;
  YYLTYPE yyls_alloc;
};

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (YYSIZEOF (union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (YYSIZEOF (yy_state_t) + YYSIZEOF (YYSTYPE) \
             + YYSIZEOF (YYLTYPE)) \
      + 2 * YYSTACK_GAP_MAXIMUM)

# define YYCOPY_NEEDED 1

/* Relocate STACK from its old location to the new one.  The
   local variables YYSIZE and YYSTACKSIZE give the old and new number of
   elements in the stack, and YYPTR gives the new location of the
   stack.  Advance YYPTR to a properly aligned location for the next
   stack.  */
# define YYSTACK_RELOCATE(Stack_alloc, Stack)                           \
    do                                                                  \
      {                                                                 \
        YYPTRDIFF_T yynewbytes;                                         \
        YYCOPY (&yyptr->Stack_alloc, Stack, yysize);                    \
        Stack = &yyptr->Stack_alloc;                                    \
        yynewbytes = yystacksize * YYSIZEOF (*Stack) + YYSTACK_GAP_MAXIMUM; \
        yyptr += yynewbytes / YYSIZEOF (*yyptr);                        \
      }                                                                 \
    while (0)

#endif

#if defined YYCOPY_NEEDED && YYCOPY_NEEDED
/* Copy COUNT objects from SRC to DST.  The source and destination do
   not overlap.  */
# ifndef YYCOPY
#  if defined __GNUC__ && 1 < __GNUC__
#   define YYCOPY(Dst, Src, Count) \
      __builtin_memcpy (Dst, Src, YY_CAST (YYSIZE_T, (Count)) * sizeof (*(Src)))
#  else
#   define YYCOPY(Dst, Src, Count)              \
      do                                        \
        {                                       \
          YYPTRDIFF_T yyi;                      \
          for (yyi = 0; yyi < (Count); yyi++)   \
            (Dst)[yyi] = (Src)[yyi];            \
        }                                       \
      while (0)
#  endif
# endif
#endif /* !YYCOPY_NEEDED */

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL  59
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   1523

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  95
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  119
/* YYNRULES -- Number of rules.  */
#define YYNRULES  400
/* YYNSTATES -- Number of states.  */
#define YYNSTATES  793

/* YYMAXUTOK -- Last valid token kind.  */
#define YYMAXUTOK   333


/* YYTRANSLATE(TOKEN-NUM) -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex, with out-of-bounds checking.  */
#define YYTRANSLATE(YYX)                                \
  (0 <= (YYX) && (YYX) <= YYMAXUTOK                     \
   ? YY_CAST (yysymbol_kind_t, yytranslate[YYX])        \
   : YYSYMBOL_YYUNDEF)

/* YYTRANSLATE[TOKEN-NUM] -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex.  */
static const yytype_int8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    89,     2,
      87,    88,    86,     2,    83,    94,    90,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    80,    79,
      84,    93,    85,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,    91,     2,    92,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,    81,     2,    82,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     1,     2,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    37,    38,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    54,
      55,    56,    57,    58,    59,    60,    61,    62,    63,    64,
      65,    66,    67,    68,    69,    70,    71,    72,    73,    74,
      75,    76,    77,    78
};

#if YYDEBUG
/* YYRLINE[YYN] -- Source line where rule number YYN was defined.  */
static const yytype_int16 yyrline[] =
{
       0,   203,   203,   208,   211,   216,   217,   221,   223,   228,
     229,   234,   236,   237,   238,   240,   241,   242,   244,   245,
     246,   247,   248,   252,   253,   254,   255,   256,   257,   258,
     259,   260,   261,   262,   263,   264,   265,   266,   267,   268,
     269,   270,   271,   272,   275,   276,   277,   278,   279,   280,
     281,   282,   283,   284,   285,   287,   289,   290,   293,   294,
     295,   296,   300,   302,   309,   317,   321,   328,   330,   335,
     336,   340,   342,   344,   346,   348,   362,   364,   366,   368,
     374,   376,   378,   380,   382,   384,   386,   388,   390,   392,
     400,   402,   404,   408,   410,   415,   416,   421,   422,   426,
     428,   430,   432,   434,   436,   438,   440,   442,   444,   446,
     448,   450,   452,   454,   456,   458,   460,   462,   464,   468,
     469,   474,   482,   484,   488,   492,   494,   498,   502,   504,
     506,   508,   510,   512,   516,   518,   520,   522,   524,   528,
     530,   532,   534,   536,   538,   542,   544,   546,   548,   550,
     552,   556,   560,   565,   566,   570,   574,   579,   580,   585,
     586,   596,   598,   602,   604,   609,   610,   614,   616,   621,
     622,   626,   631,   632,   636,   638,   642,   644,   649,   650,
     654,   655,   658,   662,   664,   668,   670,   672,   677,   678,
     682,   684,   688,   690,   694,   698,   702,   708,   712,   714,
     718,   720,   724,   728,   732,   736,   738,   743,   744,   749,
     750,   752,   754,   763,   765,   767,   769,   771,   773,   777,
     779,   783,   787,   789,   791,   793,   795,   799,   801,   806,
     813,   817,   819,   821,   822,   824,   826,   828,   832,   834,
     836,   842,   848,   857,   859,   861,   867,   875,   877,   880,
     884,   888,   890,   895,   897,   905,   907,   909,   911,   913,
     915,   917,   919,   921,   923,   925,   928,   939,   957,   975,
     977,   981,   986,   987,   989,   996,  1000,  1001,  1005,  1006,
    1007,  1008,  1011,  1013,  1015,  1017,  1019,  1021,  1023,  1025,
    1027,  1029,  1031,  1033,  1035,  1037,  1039,  1041,  1043,  1045,
    1049,  1058,  1060,  1062,  1067,  1068,  1070,  1080,  1081,  1083,
    1090,  1097,  1104,  1113,  1120,  1128,  1135,  1137,  1139,  1141,
    1146,  1156,  1166,  1178,  1179,  1180,  1183,  1184,  1185,  1186,
    1193,  1199,  1208,  1215,  1221,  1227,  1235,  1237,  1241,  1243,
    1247,  1249,  1253,  1255,  1260,  1261,  1265,  1267,  1269,  1273,
    1275,  1279,  1281,  1285,  1287,  1289,  1297,  1300,  1303,  1305,
    1307,  1311,  1313,  1315,  1317,  1319,  1321,  1323,  1325,  1327,
    1329,  1331,  1333,  1337,  1339,  1341,  1343,  1345,  1347,  1349,
    1352,  1355,  1357,  1359,  1361,  1363,  1365,  1376,  1377,  1379,
    1383,  1387,  1391,  1395,  1401,  1409,  1411,  1415,  1418,  1422,
    1426
};
#endif

/** Accessing symbol of state STATE.  */
#define YY_ACCESSING_SYMBOL(State) YY_CAST (yysymbol_kind_t, yystos[State])

#if YYDEBUG || 0
/* The user-facing name of the symbol whose (internal) number is
   YYSYMBOL.  No bounds checking.  */
static const char *yysymbol_name (yysymbol_kind_t yysymbol) YY_ATTRIBUTE_UNUSED;

/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "\"end of file\"", "error", "\"invalid token\"", "MODULE", "MAINMODULE",
  "EXTERN", "READONLY", "INITCALL", "INITNODE", "INITPROC", "PUPABLE",
  "CHARE", "MAINCHARE", "GROUP", "NODEGROUP", "ARRAY", "MESSAGE",
  "CONDITIONAL", "CLASS", "INCLUDE", "STACKSIZE", "THREADED", "TEMPLATE",
  "WHENIDLE", "SYNC", "IGET", "EXCLUSIVE", "IMMEDIATE", "SKIPSCHED",
  "INLINE", "VIRTUAL", "MIGRATABLE", "AGGREGATE", "CREATEHERE",
  "CREATEHOME", "NOKEEP", "NOTRACE", "APPWORK", "VOID", "CONST", "NOCOPY",
  "NOCOPYPOST", "NOCOPYDEVICE", "PACKED", "VARSIZE", "ENTRY", "FOR",
  "FORALL", "WHILE", "WHEN", "OVERLAP", "SERIAL", "IF", "ELSE", "PYTHON",
  "LOCAL", "NAMESPACE", "USING", "IDENT", "NUMBER", "LITERAL", "CPROGRAM",
  "HASHIF", "HASHIFDEF", "INT", "LONG", "SHORT", "CHAR", "FLOAT", "DOUBLE",
  "UNSIGNED", "ACCEL", "READWRITE", "WRITEONLY", "ACCELBLOCK",
  "MEMCRITICAL", "REDUCTIONTARGET", "CASE", "TYPENAME", "';'", "':'",
  "'{'", "'}'", "','", "'<'", "'>'", "'*'", "'('", "')'", "'&'", "'.'",
  "'['", "']'", "'='", "'-'", "$accept", "File", "ModuleEList",
  "OptExtern", "OneOrMoreSemiColon", "OptSemiColon", "Name", "QualName",
  "Module", "ConstructEList", "ConstructList", "ConstructSemi",
  "Construct", "TParam", "TParamList", "TParamEList", "OptTParams",
  "BuiltinType", "NamedType", "QualNamedType", "SimpleType", "OnePtrType",
  "PtrType", "FuncType", "BaseType", "BaseDataType", "RestrictedType",
  "Type", "ArrayDim", "Dim", "DimList", "Readonly", "ReadonlyMsg",
  "OptVoid", "MAttribs", "MAttribList", "MAttrib", "CAttribs",
  "CAttribList", "PythonOptions", "ArrayAttrib", "ArrayAttribs",
  "ArrayAttribList", "CAttrib", "OptConditional", "MsgArray", "Var",
  "VarList", "Message", "OptBaseList", "BaseList", "Chare", "Group",
  "NodeGroup", "ArrayIndexType", "Array", "TChare", "TGroup", "TNodeGroup",
  "TArray", "TMessage", "OptTypeInit", "OptNameInit", "TVar", "TVarList",
  "TemplateSpec", "Template", "MemberEList", "MemberList",
  "NonEntryMember", "InitNode", "InitProc", "PUPableClass", "IncludeFile",
  "Member", "MemberBody", "UnexpectedToken", "Entry", "AccelBlock",
  "EReturn", "EAttribs", "AttributeArg", "AttributeArgList", "EAttribList",
  "EAttrib", "DefaultParameter", "CPROGRAM_List", "CCode",
  "ParamBracketStart", "ParamBraceStart", "ParamBraceEnd", "Parameter",
  "AccelBufferType", "AccelInstName", "AccelArrayParam", "AccelParameter",
  "ParamList", "AccelParamList", "EParameters", "AccelEParameters",
  "OptStackSize", "OptSdagCode", "Slist", "Olist", "CaseList",
  "OptTraceName", "WhenConstruct", "NonWhenConstruct", "SingleConstruct",
  "HasElse", "IntExpr", "EndIntExpr", "StartIntExpr", "SEntry",
  "SEntryList", "SParamBracketStart", "SParamBracketEnd", "HashIFComment",
  "HashIFDefComment", YY_NULLPTR
};

static const char *
yysymbol_name (yysymbol_kind_t yysymbol)
{
  return yytname[yysymbol];
}
#endif

#define YYPACT_NINF (-601)

#define yypact_value_is_default(Yyn) \
  ((Yyn) == YYPACT_NINF)

#define YYTABLE_NINF (-352)

#define yytable_value_is_error(Yyn) \
  0

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
static const yytype_int16 yypact[] =
{
     197,  1300,  1300,    61,  -601,   197,  -601,  -601,  -601,  -601,
    -601,  -601,  -601,  -601,  -601,  -601,  -601,  -601,  -601,  -601,
    -601,  -601,  -601,  -601,  -601,  -601,  -601,  -601,  -601,  -601,
    -601,  -601,  -601,  -601,  -601,  -601,  -601,  -601,  -601,  -601,
    -601,  -601,  -601,  -601,  -601,  -601,  -601,  -601,  -601,  -601,
    -601,  -601,  -601,  -601,  -601,  -601,  -601,   162,   162,  -601,
    -601,  -601,   913,    -3,  -601,  -601,  -601,    62,  1300,   222,
    1300,  1300,   207,  1054,    50,   992,   913,  -601,  -601,  -601,
    -601,   206,    70,    96,  -601,    91,  -601,  -601,  -601,    -3,
     -24,  1343,   149,   149,     6,   -18,   118,   118,   118,   118,
     121,   155,  1300,   134,   167,   913,  -601,  -601,  -601,  -601,
    -601,  -601,  -601,  -601,   529,  -601,  -601,  -601,  -601,   175,
    -601,  -601,  -601,  -601,  -601,  -601,  -601,  -601,  -601,  -601,
    -601,    -3,  -601,  -601,  -601,  1188,  1427,   913,    91,   185,
      77,   -24,   186,   329,  -601,  1445,  -601,   198,   217,  -601,
    -601,  -601,   324,    96,   168,  -601,  -601,   209,   216,   234,
    -601,    51,    96,  -601,    96,    96,   246,    96,   250,  -601,
      23,  1300,  1300,  1300,  1300,   103,   252,   270,   176,  1300,
    -601,  -601,  -601,   655,   301,   118,   118,   118,   118,   252,
     155,  -601,  -601,  -601,  -601,  -601,    -3,  -601,  -601,  -601,
    -601,  -601,  -601,  -601,  -601,  -601,  -601,  -601,  -601,  -601,
    -601,  -601,  -601,   341,  -601,  -601,  -601,   311,   200,  1427,
     209,   216,   234,    34,  -601,   -18,   327,    64,   -24,   351,
     -24,   325,  -601,   175,   328,    -9,  -601,   360,  -601,  -601,
    -601,   287,   361,  -601,   168,  1412,  -601,  -601,  -601,  -601,
    -601,   344,   296,   342,   -33,   -30,   111,   339,   336,   -18,
    -601,  -601,   347,   359,   364,   363,   363,   363,   363,  -601,
    1300,   352,   362,   378,   107,  1300,   420,  1300,  -601,  -601,
     388,   398,   403,   825,    -6,    89,  1300,   402,   404,   175,
    1300,  1300,  1300,  1300,  1300,  1300,  -601,  -601,  -601,  1188,
    1300,   451,  -601,   332,   406,  1300,  -601,  -601,  -601,   422,
     424,   421,   413,   -24,    -3,    96,  -601,  -601,   444,  -601,
    -601,  -601,  -601,   429,  -601,   426,  -601,  1300,   423,   425,
     435,  -601,   437,  -601,   -24,   149,  1412,   149,   149,  1412,
     149,  -601,  -601,    23,  -601,   -18,   224,   224,   224,   224,
     434,  -601,   420,  -601,   363,   363,  -601,   176,    22,   439,
     440,   157,   442,   145,  -601,   443,   655,  -601,  -601,   363,
     363,   363,   363,   363,   299,  -601,   458,   431,   461,   460,
     464,   465,   364,   -24,   351,   -24,   -24,  -601,   -33,  -601,
    1412,  -601,   463,   466,   467,  -601,  -601,   469,  -601,   470,
     474,   475,    96,   480,   481,  -601,   485,  -601,   428,    -3,
    -601,  -601,  -601,  -601,  -601,  -601,   224,   224,  -601,  -601,
    -601,  1445,    26,   488,   482,  1445,  -601,  -601,   494,  -601,
    -601,  -601,  -601,  -601,   224,   224,   224,   224,   224,   554,
      -3,   526,  1300,   503,   497,   498,  -601,   502,  -601,  -601,
    -601,  -601,  -601,  -601,   505,   500,  -601,  -601,  -601,  -601,
     508,  -601,    55,   509,  -601,   -18,  -601,   744,   553,   517,
     175,   428,  -601,  -601,  -601,  -601,  1300,  -601,  -601,  1300,
    -601,   544,  -601,  -601,  -601,  -601,  -601,   521,  -601,  -601,
    1188,   514,  -601,  1376,  -601,  1391,  -601,   149,   149,   149,
    -601,  1132,  1074,  -601,   175,    -3,  -601,   515,   440,   440,
     175,  -601,  -601,  1445,  1445,  1445,  -601,  1300,   -24,   522,
     524,   525,   527,   528,   531,   518,   534,   502,  1300,  -601,
     533,   175,  -601,  -601,    -3,  1300,   -24,   -24,   -24,    19,
     535,  1391,  -601,  -601,  -601,  -601,  -601,   579,   530,   502,
    -601,    -3,   532,   536,   537,   539,  -601,   306,  -601,  -601,
    -601,  1300,  -601,   546,   543,   546,   560,   541,   566,   546,
     555,   323,    -3,   -24,  -601,  -601,  -601,   617,  -601,  -601,
    -601,  -601,  -601,    91,  -601,   502,  -601,   -24,   580,   -24,
     240,   563,   593,   607,  -601,   567,   -24,   417,   565,   260,
     186,   556,   530,   559,  -601,   581,   569,   574,  -601,   -24,
     560,   410,  -601,   582,   471,   -24,   574,   546,   575,   546,
     584,   566,   546,   586,   -24,   587,   417,  -601,   175,  -601,
     175,   613,  -601,   290,   567,   -24,   546,  -601,   631,   469,
    -601,  -601,   603,  -601,  -601,   186,   651,   -24,   628,   -24,
     607,   567,   -24,   417,   186,  -601,  -601,  -601,  -601,  -601,
    -601,  -601,  -601,  -601,  1300,   595,   604,   597,   -24,   611,
     -24,   323,  -601,   502,  -601,   175,   323,   638,   625,   600,
     574,   614,   -24,   574,   623,   175,   626,  1445,  1326,  -601,
     186,   -24,   629,   632,  -601,  -601,   633,   890,  -601,   -24,
     546,   897,  -601,   186,   904,  -601,  -601,  1300,  1300,   -24,
     634,  -601,  1300,   574,   -24,  -601,   638,   323,  -601,   637,
     -24,   323,  -601,   175,   323,   638,  -601,   245,    86,   635,
    1300,   175,   911,   643,  -601,   648,   -24,   654,   649,  -601,
     653,  -601,  -601,  1300,  1300,  1225,   652,  1300,  -601,   259,
      -3,   323,  -601,   -24,  -601,   574,   -24,  -601,   638,   412,
    -601,   644,   231,  1300,   392,  -601,   656,   574,   963,   657,
    -601,  -601,  -601,  -601,  -601,  -601,  -601,   971,   323,  -601,
     -24,   323,  -601,   659,   574,   668,  -601,   978,  -601,   323,
    -601,   672,  -601
};

/* YYDEFACT[STATE-NUM] -- Default reduction number in state STATE-NUM.
   Performed when YYTABLE does not specify something else to do.  Zero
   means the default is an error.  */
static const yytype_int16 yydefact[] =
{
       3,     0,     0,     0,     2,     3,    12,    13,    14,    15,
      16,    17,    18,    19,    20,    21,    22,    23,    24,    25,
      26,    27,    28,    29,    30,    31,    32,    36,    37,    38,
      39,    40,    41,    42,    43,    33,    34,    35,    44,    45,
      46,    47,    48,    49,    50,    51,    52,    53,    54,    55,
      56,    11,    57,    58,    59,    60,    61,     0,     0,     1,
       4,     7,     0,    67,    65,    66,    89,     6,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    88,    86,    87,
       8,     0,     0,     0,    62,    72,   399,   400,   314,   270,
     307,     0,   157,   157,   157,     0,   165,   165,   165,   165,
       0,   159,     0,     0,     0,     0,    80,   231,   232,    74,
      81,    82,    83,    84,     0,    85,    73,   234,   233,     9,
     265,   257,   258,   259,   260,   261,   263,   264,   262,   255,
     256,    78,    79,    70,   274,     0,     0,     0,    71,     0,
     308,   307,     0,     0,   118,     0,    99,   100,   102,   104,
     115,   116,     0,     0,    97,   122,   123,   128,   129,   130,
     131,   150,     0,   158,     0,     0,     0,     0,   247,   235,
       0,     0,     0,     0,     0,     0,     0,   172,     0,     0,
     237,   249,   236,     0,     0,   165,   165,   165,   165,     0,
     159,   222,   223,   224,   225,   226,    10,    68,   300,   282,
     283,   284,   285,   286,   292,   293,   294,   299,   287,   288,
     289,   290,   291,   169,   295,   297,   298,     0,   278,     0,
     134,   135,   136,   144,   271,     0,     0,     0,   307,   304,
     307,     0,   315,     0,     0,   132,   101,   113,   117,   103,
     105,   106,   110,   112,    97,    95,   120,   124,   125,   126,
     133,     0,   149,     0,   153,   241,   238,     0,   243,     0,
     176,   177,     0,   167,    97,   188,   188,   188,   188,   171,
       0,     0,   174,     0,     0,     0,     0,     0,   163,   164,
       0,   161,   185,     0,     0,   131,     0,   219,     0,     9,
       0,     0,     0,     0,     0,     0,   170,   296,   273,     0,
       0,   137,   138,   143,     0,     0,    77,    64,    63,     0,
     305,     0,     0,   307,   269,     0,   114,   107,   108,   111,
     121,    91,    92,    93,    96,     0,    90,     0,   148,     0,
       0,   397,   153,   155,   307,   157,     0,   157,   157,     0,
     157,   248,   166,     0,   119,     0,     0,     0,     0,     0,
       0,   197,     0,   173,   188,   188,   160,     0,   178,     0,
     207,    62,     0,     0,   217,   209,     0,   221,    76,   188,
     188,   188,   188,   188,     0,   280,     0,   276,     0,   142,
       0,     0,    97,   307,   304,   307,   307,   312,   153,   109,
       0,    98,     0,     0,     0,   147,   154,     0,   151,     0,
       0,     0,     0,     0,     0,   168,   190,   189,     0,   227,
     192,   193,   194,   195,   196,   175,     0,     0,   162,   179,
     186,     0,   178,     0,     0,     0,   215,   216,     0,   210,
     211,   212,   218,   220,     0,     0,     0,     0,     0,   178,
     205,     0,     0,   279,     0,     0,   141,     0,   310,   306,
     311,   309,   156,    94,     0,     0,   146,   398,   152,   242,
       0,   239,     0,     0,   244,     0,   254,     0,     0,     0,
       0,     0,   250,   251,   198,   199,     0,   184,   187,     0,
     208,     0,   200,   201,   202,   203,   204,     0,   275,   277,
       0,     0,   140,     0,    75,     0,   145,   157,   157,   157,
     191,     0,     0,   252,     9,   253,   230,   180,   207,   207,
       0,   281,   139,     0,     0,     0,   341,   316,   307,   336,
       0,     0,     0,     0,     0,     0,    62,     0,     0,   228,
       0,     0,   213,   214,   206,     0,   307,   307,   307,   178,
       0,     0,   340,   127,   240,   246,   245,     0,     0,     0,
     181,   182,     0,     0,     0,     0,   313,     0,   317,   319,
     337,     0,   386,     0,     0,     0,     0,     0,   357,     0,
       0,     0,   346,   307,   267,   375,   347,   344,   320,   321,
     322,   302,   301,   303,   318,     0,   392,   307,     0,   307,
       0,   395,     0,     0,   356,     0,   307,     0,     0,     0,
       0,     0,     0,     0,   390,     0,     0,     0,   393,   307,
       0,     0,   359,     0,     0,   307,     0,     0,     0,     0,
       0,   357,     0,     0,   307,     0,   353,   355,     9,   350,
       9,     0,   266,     0,     0,   307,     0,   391,     0,     0,
     396,   358,     0,   374,   352,     0,     0,   307,     0,   307,
       0,     0,   307,     0,     0,   376,   354,   348,   385,   345,
     323,   324,   325,   343,     0,     0,   338,     0,   307,     0,
     307,     0,   383,     0,   360,     9,     0,   387,     0,     0,
       0,     0,   307,     0,     0,     9,     0,     0,     0,   342,
       0,   307,     0,     0,   394,   373,     0,     0,   381,   307,
       0,     0,   362,     0,     0,   363,   372,     0,     0,   307,
       0,   339,     0,     0,   307,   384,   387,     0,   388,     0,
     307,     0,   370,     9,     0,   387,   326,     0,     0,     0,
       0,     0,     0,     0,   382,     0,   307,     0,     0,   361,
       0,   368,   334,     0,     0,     0,     0,     0,   332,     0,
     268,     0,   378,   307,   389,     0,   307,   371,   387,     0,
     328,     0,     0,     0,     0,   335,     0,     0,     0,     0,
     369,   331,   330,   329,   327,   333,   377,     0,     0,   365,
     307,     0,   379,     0,     0,     0,   364,     0,   380,     0,
     366,     0,   367
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -601,  -601,   712,  -601,   -55,  -283,    -1,   -62,   666,   702,
     -17,  -601,  -601,  -601,  -221,  -601,  -219,  -601,  -141,   -86,
    -125,  -126,  -121,  -164,   616,   557,  -601,   -87,  -601,  -601,
    -267,  -601,  -601,   -80,   608,   446,  -601,   128,   457,  -601,
    -601,   627,   453,  -601,   267,  -601,  -601,  -354,  -601,  -140,
     358,  -601,  -601,  -601,   -29,  -601,  -601,  -601,  -601,  -601,
    -601,  -319,   452,  -601,   441,   743,  -601,  -199,   353,   752,
    -601,  -601,   568,  -601,  -601,  -601,  -601,   373,  -601,   340,
     376,  -601,   384,  -291,  -601,  -601,   447,   -85,  -491,   -60,
    -559,  -601,  -601,  -537,  -601,  -601,  -442,   169,  -498,  -601,
    -601,   261,  -565,   214,  -578,   258,  -571,  -601,  -521,  -579,
    -561,  -600,  -503,  -601,   271,   294,   247,  -601,  -601
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
       0,     3,     4,    73,   409,   197,   264,   154,     5,    64,
      74,    75,    76,   323,   324,   325,   246,   155,   265,   156,
     157,   158,   159,   160,   161,   223,   224,   326,   397,   332,
     333,   107,   108,   164,   179,   280,   281,   171,   262,   297,
     272,   176,   273,   263,   421,   531,   422,   423,   109,   346,
     407,   110,   111,   112,   177,   113,   191,   192,   193,   194,
     195,   426,   364,   287,   288,   468,   115,   410,   469,   470,
     117,   118,   169,   182,   471,   472,   132,   473,    77,   225,
     136,   377,   378,   217,   218,   584,   311,   604,   518,   573,
     233,   519,   665,   727,   710,   666,   520,   667,   494,   634,
     602,   574,   598,   613,   625,   595,   575,   627,   599,   698,
     605,   638,   587,   591,   592,   334,   458,    78,    79
};

/* YYTABLE[YYPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule whose
   number is the opposite.  If YYTABLE_NINF, syntax error.  */
static const yytype_int16 yytable[] =
{
      57,    58,    63,    63,   162,   142,   368,    85,   375,   168,
     221,   220,    90,   165,   167,   222,   646,    89,   234,   285,
     131,   138,   536,   537,   538,   320,   626,   576,   607,   548,
     250,   266,   267,   268,   629,   616,   419,   140,   282,   419,
      84,   630,   427,   419,   163,   344,   642,   139,   656,   644,
     139,   577,   361,   521,   260,   626,   231,   335,   331,   133,
     153,    59,   589,   141,   196,   396,   596,    82,   477,    86,
      87,   612,   614,   302,   669,   684,    80,   261,   251,   307,
     701,   576,   626,   704,   362,   487,   675,   603,   184,   271,
     250,   244,   608,   221,   220,   685,   286,   166,   222,   560,
     254,   180,   255,   256,   420,   258,   693,    81,  -183,   692,
     556,   696,   557,   732,   647,   400,   649,   672,   403,   652,
     226,   452,   308,   303,   304,   677,   347,   348,   349,   614,
     713,   712,   119,   670,   354,   139,   355,   734,   251,   305,
     252,   253,   498,   309,   723,   312,   741,   275,   411,   412,
     413,   137,   735,   733,    84,   768,   738,   269,   228,   740,
     294,    84,   270,   447,   229,    84,   270,   777,   230,   453,
     747,   139,  -209,   168,  -209,   694,   718,   556,   314,   770,
     722,   153,   363,   725,   787,   153,   766,   163,   271,   532,
     533,   139,   767,   749,   181,   336,   709,   720,   337,   511,
       1,     2,   285,    84,   429,   430,   759,   134,   762,   170,
     764,   752,   175,   783,   416,   417,   785,   474,   475,   278,
     279,   529,   244,   153,   791,   172,   173,   174,   387,   434,
     435,   436,   437,   438,   196,   482,   483,   484,   485,   486,
    -207,    61,  -207,    62,  -272,  -272,   178,   779,   139,   398,
     425,   183,   245,   388,    61,   399,   782,   401,   402,   406,
     404,   562,   236,   237,  -272,   227,   790,   238,   232,   350,
    -272,  -272,  -272,  -272,  -272,  -272,  -272,   431,    83,   286,
      84,   239,   360,   299,  -272,   365,    61,   300,    88,   369,
     370,   371,   372,   373,   374,   247,   660,   135,   448,   376,
     450,   451,   248,    61,   382,   408,   563,   564,   565,   566,
     567,   568,   569,   290,   291,   292,   293,   257,   743,   440,
     249,   744,   745,   773,   562,   746,   392,   493,   144,   145,
     742,   331,   743,   259,   476,   744,   745,   570,   480,   746,
     462,    88,  -349,   274,   765,   657,   743,   658,    84,   744,
     745,   317,   318,   746,   146,   147,   148,   149,   150,   151,
     152,   276,   661,   662,    84,   581,   582,   144,   153,   563,
     564,   565,   566,   567,   568,   569,   221,   220,    61,   406,
     439,   222,   663,   289,  -314,   328,   329,    84,   240,   241,
     242,   243,   695,   146,   147,   148,   149,   150,   151,   152,
     570,   296,   706,   298,    88,  -314,   517,   153,   517,   306,
    -314,   562,   310,   313,   315,   505,   139,   522,   523,   524,
     339,   379,   380,   340,   316,   319,   535,   535,   535,   466,
     327,   338,   330,   540,    91,    92,    93,    94,    95,   342,
     739,   376,   343,   345,   351,   352,   102,   103,   245,   196,
     104,   553,   554,   555,   517,   534,   563,   564,   565,   566,
     567,   568,   569,   617,   618,   619,   566,   620,   621,   622,
     353,  -314,   562,   467,   269,   507,   551,   775,   508,   743,
     356,   357,   744,   745,   358,   366,   746,   570,   600,   367,
     302,    88,   641,   572,   623,   583,   381,  -314,    88,   743,
     771,   527,   744,   745,   383,   386,   746,   384,   389,   385,
    -229,   391,   390,   393,   442,   394,   539,   563,   564,   565,
     566,   567,   568,   569,   639,   395,   414,   549,   331,   424,
     645,   562,   428,   425,   552,   615,   363,   624,   441,   654,
     185,   186,   187,   188,   189,   190,   664,   572,   570,   443,
     444,   454,    88,  -351,   445,   446,   455,   456,   459,   460,
     585,   457,   678,   461,   680,   463,   624,   683,   465,   464,
     478,   419,   479,   196,   668,   196,   563,   564,   565,   566,
     567,   568,   569,   690,   481,   488,   490,   491,   492,   493,
     496,   682,   495,   624,   562,   497,   499,   703,   467,   504,
     708,   664,   509,   510,   512,   541,   530,   570,   562,    61,
     547,   571,   542,   543,   719,   544,   545,   561,   590,   546,
     196,   -11,   593,   556,   729,   550,   594,   559,   578,   579,
     196,   580,   562,   586,   588,   737,   597,   601,   606,   563,
     564,   565,   566,   567,   568,   569,   610,   628,    88,   631,
     633,   755,   562,   563,   564,   565,   566,   567,   568,   569,
     635,   636,   637,   686,   643,   650,   648,   653,   196,   655,
     570,   769,   659,   283,   611,   687,   750,   563,   564,   565,
     566,   567,   568,   569,   570,   674,   679,   688,    88,   689,
     691,   697,   700,   144,   145,   784,   702,   563,   564,   565,
     566,   567,   568,   569,   699,   705,   726,   728,   570,   714,
     707,   731,   671,    84,   715,   716,   736,    60,   730,   146,
     147,   148,   149,   150,   151,   152,   753,   748,   570,   726,
     754,   757,   676,   284,   756,   758,   772,   763,   776,   106,
     780,   786,   726,   760,   726,   134,   726,  -272,  -272,  -272,
     788,  -272,  -272,  -272,   792,  -272,  -272,  -272,  -272,  -272,
      65,   235,   774,  -272,  -272,  -272,  -272,  -272,  -272,  -272,
    -272,  -272,  -272,  -272,  -272,  -272,   301,  -272,  -272,  -272,
    -272,  -272,  -272,  -272,  -272,  -272,  -272,  -272,  -272,  -272,
    -272,  -272,  -272,  -272,  -272,  -272,  -272,  -272,   295,  -272,
     405,  -272,  -272,   418,   277,   415,   558,   433,  -272,  -272,
    -272,  -272,  -272,  -272,  -272,  -272,   114,   432,  -272,  -272,
    -272,  -272,  -272,   500,   506,   116,   489,   341,     6,     7,
       8,   449,     9,    10,    11,   501,    12,    13,    14,    15,
      16,   503,   528,   502,    17,    18,    19,    20,    21,    22,
      23,    24,    25,    26,    27,    28,    29,   711,    30,    31,
      32,    33,    34,   632,   681,    35,    36,    37,    38,    39,
      40,    41,    42,    43,    44,    45,    46,    47,    48,   651,
      49,   640,    50,    51,   609,     0,   673,     0,     0,     0,
       0,   562,     0,     0,     0,     0,    52,     0,   562,    53,
      54,    55,    56,     0,     0,   562,     0,     0,     0,     0,
       0,     0,   562,     0,    66,   359,    -5,    -5,    67,    -5,
      -5,    -5,    -5,    -5,    -5,    -5,    -5,    -5,    -5,    -5,
       0,    -5,    -5,     0,     0,    -5,   563,   564,   565,   566,
     567,   568,   569,   563,   564,   565,   566,   567,   568,   569,
     563,   564,   565,   566,   567,   568,   569,   563,   564,   565,
     566,   567,   568,   569,   562,     0,     0,   570,     0,    68,
      69,   717,   562,     0,   570,    70,    71,     0,   721,   562,
       0,   570,     0,     0,     0,   724,     0,    72,   570,     0,
       0,     0,   751,     0,    -5,   -69,     0,     0,   120,   121,
     122,   123,     0,   124,   125,   126,   127,   128,     0,   563,
     564,   565,   566,   567,   568,   569,     0,   563,   564,   565,
     566,   567,   568,   569,   563,   564,   565,   566,   567,   568,
     569,     0,     0,     0,     0,     0,     0,   129,     0,     0,
     570,     0,     0,     0,   778,     0,     0,     0,   570,     0,
       0,     0,   781,     0,     0,   570,     0,     1,     2,   789,
      91,    92,    93,    94,    95,    96,    97,    98,    99,   100,
     101,    61,   102,   103,   130,     0,   104,     6,     7,     8,
       0,     9,    10,    11,     0,    12,    13,    14,    15,    16,
       0,     0,     0,    17,    18,    19,    20,    21,    22,    23,
      24,    25,    26,    27,    28,    29,     0,    30,    31,    32,
      33,    34,   144,   219,    35,    36,    37,    38,    39,    40,
      41,    42,    43,    44,    45,    46,    47,    48,     0,    49,
       0,    50,   526,   198,     0,   105,     0,     0,   146,   147,
     148,   149,   150,   151,   152,    52,     0,     0,    53,    54,
      55,    56,   153,   199,     0,   200,   201,   202,   203,   204,
     205,   206,     0,     0,   207,   208,   209,   210,   211,   212,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   213,   214,     0,   198,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   525,     0,     0,     0,   215,   216,   199,
       0,   200,   201,   202,   203,   204,   205,   206,     0,     0,
     207,   208,   209,   210,   211,   212,     0,     0,     6,     7,
       8,     0,     9,    10,    11,     0,    12,    13,    14,    15,
      16,     0,   213,   214,    17,    18,    19,    20,    21,    22,
      23,    24,    25,    26,    27,    28,    29,     0,    30,    31,
      32,    33,    34,   215,   216,    35,    36,    37,    38,    39,
      40,    41,    42,    43,    44,    45,    46,    47,    48,     0,
      49,     0,    50,    51,   761,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    52,     0,     0,    53,
      54,    55,    56,     6,     7,     8,     0,     9,    10,    11,
       0,    12,    13,    14,    15,    16,     0,     0,     0,    17,
      18,    19,    20,    21,    22,    23,    24,    25,    26,    27,
      28,    29,   660,    30,    31,    32,    33,    34,     0,     0,
      35,    36,    37,    38,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,     0,    49,     0,    50,    51,   143,
       0,     0,     0,     0,   144,   145,     0,     0,     0,     0,
       0,    52,     0,     0,    53,    54,    55,    56,     0,     0,
       0,   144,   145,     0,    84,     0,     0,     0,     0,     0,
     146,   147,   148,   149,   150,   151,   152,     0,   661,   662,
       0,    84,     0,     0,   153,     0,     0,   146,   147,   148,
     149,   150,   151,   152,   144,   145,   513,   514,   515,     0,
       0,   153,     0,     0,     0,     0,     0,     0,     0,   144,
     145,   513,   514,   515,    84,     0,     0,     0,     0,     0,
     146,   147,   148,   149,   150,   151,   152,     0,     0,    84,
     144,   145,     0,     0,   153,   146,   147,   148,   149,   150,
     151,   152,     0,     0,   516,   144,   219,     0,     0,   153,
      84,   321,   322,     0,     0,     0,   146,   147,   148,   149,
     150,   151,   152,   144,   145,    84,     0,     0,     0,     0,
     153,   146,   147,   148,   149,   150,   151,   152,     0,     0,
       0,     0,     0,    84,     0,   153,     0,     0,     0,   146,
     147,   148,   149,   150,   151,   152,     0,     0,     0,     0,
       0,     0,     0,   153
};

static const yytype_int16 yycheck[] =
{
       1,     2,    57,    58,    91,    90,   289,    69,   299,    95,
     136,   136,    72,    93,    94,   136,   616,    72,   143,   183,
      75,    83,   513,   514,   515,   244,   597,   548,   589,   527,
      39,   172,   173,   174,   599,   596,    17,    61,   179,    17,
      58,   600,   361,    17,    38,   264,   611,    80,   626,   614,
      80,   549,    58,   495,    31,   626,   141,    87,    91,    76,
      78,     0,   565,    87,   119,   332,   569,    68,   422,    70,
      71,   592,   593,    39,   635,   653,    79,    54,    87,    15,
     680,   602,   653,   683,    90,   439,   645,   585,   105,   175,
      39,   153,   590,   219,   219,   654,   183,    91,   219,   541,
     162,   102,   164,   165,    82,   167,   671,    45,    82,   670,
      91,   676,    93,   713,   617,   336,   619,   638,   339,   622,
     137,   388,    58,    89,    90,   646,   266,   267,   268,   650,
     691,   690,    82,   636,   275,    80,   277,   716,    87,   225,
      89,    90,    87,   228,   703,   230,   725,   176,   347,   348,
     349,    81,   717,   714,    58,   755,   721,    54,    81,   724,
     189,    58,    59,   382,    87,    58,    59,   767,    91,   390,
      84,    80,    83,   259,    85,   673,   697,    91,   233,   758,
     701,    78,    93,   704,   784,    78,   751,    38,   274,   508,
     509,    80,   753,   730,    60,    84,   687,   700,    87,   490,
       3,     4,   366,    58,    59,    60,   743,     1,   745,    91,
     747,   732,    91,   778,   354,   355,   781,   416,   417,    43,
      44,   504,   284,    78,   789,    97,    98,    99,   313,   369,
     370,   371,   372,   373,   289,   434,   435,   436,   437,   438,
      83,    79,    85,    81,    38,    39,    91,   768,    80,   334,
      93,    84,    84,   315,    79,   335,   777,   337,   338,   345,
     340,     1,    64,    65,    58,    80,   787,    69,    82,   270,
      64,    65,    66,    67,    68,    69,    70,   363,    56,   366,
      58,    64,   283,    83,    78,   286,    79,    87,    81,   290,
     291,   292,   293,   294,   295,    86,     6,    91,   383,   300,
     385,   386,    86,    79,   305,    81,    46,    47,    48,    49,
      50,    51,    52,   185,   186,   187,   188,    71,    87,   374,
      86,    90,    91,    92,     1,    94,   327,    87,    38,    39,
      85,    91,    87,    83,   421,    90,    91,    77,   425,    94,
     402,    81,    82,    91,    85,   628,    87,   630,    58,    90,
      91,    64,    65,    94,    64,    65,    66,    67,    68,    69,
      70,    91,    72,    73,    58,    59,    60,    38,    78,    46,
      47,    48,    49,    50,    51,    52,   502,   502,    79,   465,
      81,   502,    92,    82,    61,    89,    90,    58,    64,    65,
      66,    67,   675,    64,    65,    66,    67,    68,    69,    70,
      77,    60,   685,    92,    81,    82,   493,    78,   495,    82,
      87,     1,    61,    88,    86,   470,    80,   497,   498,   499,
      84,    89,    90,    87,    64,    64,   513,   514,   515,     1,
      86,    92,    90,   518,     6,     7,     8,     9,    10,    92,
     723,   442,    83,    80,    92,    83,    18,    19,    84,   504,
      22,   536,   537,   538,   541,   510,    46,    47,    48,    49,
      50,    51,    52,    46,    47,    48,    49,    50,    51,    52,
      92,    61,     1,    45,    54,   476,   531,    85,   479,    87,
      92,    83,    90,    91,    81,    83,    94,    77,   573,    85,
      39,    81,    82,   548,    77,   557,    90,    87,    81,    87,
      88,   502,    90,    91,    82,    92,    94,    83,    64,    88,
      82,    85,    83,    90,    83,    90,   517,    46,    47,    48,
      49,    50,    51,    52,   609,    90,    92,   528,    91,    90,
     615,     1,    90,    93,   535,   595,    93,   597,    80,   624,
      11,    12,    13,    14,    15,    16,   633,   602,    77,    88,
      90,    88,    81,    82,    90,    90,    90,    90,    88,    85,
     561,    92,   647,    88,   649,    85,   626,   652,    83,    88,
      82,    17,    90,   628,   634,   630,    46,    47,    48,    49,
      50,    51,    52,   668,    90,    59,    83,    90,    90,    87,
      90,   651,    87,   653,     1,    87,    87,   682,    45,    82,
     687,   688,    58,    82,    90,    83,    91,    77,     1,    79,
      92,    81,    88,    88,   699,    88,    88,    38,    58,    88,
     675,    87,    81,    91,   709,    92,    60,    92,    92,    92,
     685,    92,     1,    87,    91,   720,    81,    20,    58,    46,
      47,    48,    49,    50,    51,    52,    83,    82,    81,    93,
      91,   736,     1,    46,    47,    48,    49,    50,    51,    52,
      79,    92,    88,   664,    82,    81,    91,    81,   723,    82,
      77,   756,    59,    18,    81,    80,   731,    46,    47,    48,
      49,    50,    51,    52,    77,    82,    58,    83,    81,    92,
      79,    53,    92,    38,    39,   780,    82,    46,    47,    48,
      49,    50,    51,    52,    79,    82,   707,   708,    77,    80,
      84,   712,    81,    58,    82,    82,    79,     5,    84,    64,
      65,    66,    67,    68,    69,    70,    83,    92,    77,   730,
      82,    82,    81,    78,    80,    82,    92,    85,    82,    73,
      83,    82,   743,   744,   745,     1,   747,     3,     4,     5,
      82,     7,     8,     9,    82,    11,    12,    13,    14,    15,
      58,   145,   763,    19,    20,    21,    22,    23,    24,    25,
      26,    27,    28,    29,    30,    31,   219,    33,    34,    35,
      36,    37,    38,    39,    40,    41,    42,    43,    44,    45,
      46,    47,    48,    49,    50,    51,    52,    53,   190,    55,
     343,    57,    58,   357,   177,   352,   539,   366,    64,    65,
      66,    67,    68,    69,    70,    71,    73,   365,    74,    75,
      76,    77,    78,   465,   471,    73,   442,   259,     3,     4,
       5,   384,     7,     8,     9,    91,    11,    12,    13,    14,
      15,   468,   502,   467,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,   688,    33,    34,
      35,    36,    37,   602,   650,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    51,    52,    53,   621,
      55,   610,    57,    58,   590,    -1,   639,    -1,    -1,    -1,
      -1,     1,    -1,    -1,    -1,    -1,    71,    -1,     1,    74,
      75,    76,    77,    -1,    -1,     1,    -1,    -1,    -1,    -1,
      -1,    -1,     1,    -1,     1,    90,     3,     4,     5,     6,
       7,     8,     9,    10,    11,    12,    13,    14,    15,    16,
      -1,    18,    19,    -1,    -1,    22,    46,    47,    48,    49,
      50,    51,    52,    46,    47,    48,    49,    50,    51,    52,
      46,    47,    48,    49,    50,    51,    52,    46,    47,    48,
      49,    50,    51,    52,     1,    -1,    -1,    77,    -1,    56,
      57,    81,     1,    -1,    77,    62,    63,    -1,    81,     1,
      -1,    77,    -1,    -1,    -1,    81,    -1,    74,    77,    -1,
      -1,    -1,    81,    -1,    81,    82,    -1,    -1,     6,     7,
       8,     9,    -1,    11,    12,    13,    14,    15,    -1,    46,
      47,    48,    49,    50,    51,    52,    -1,    46,    47,    48,
      49,    50,    51,    52,    46,    47,    48,    49,    50,    51,
      52,    -1,    -1,    -1,    -1,    -1,    -1,    45,    -1,    -1,
      77,    -1,    -1,    -1,    81,    -1,    -1,    -1,    77,    -1,
      -1,    -1,    81,    -1,    -1,    77,    -1,     3,     4,    81,
       6,     7,     8,     9,    10,    11,    12,    13,    14,    15,
      16,    79,    18,    19,    82,    -1,    22,     3,     4,     5,
      -1,     7,     8,     9,    -1,    11,    12,    13,    14,    15,
      -1,    -1,    -1,    19,    20,    21,    22,    23,    24,    25,
      26,    27,    28,    29,    30,    31,    -1,    33,    34,    35,
      36,    37,    38,    39,    40,    41,    42,    43,    44,    45,
      46,    47,    48,    49,    50,    51,    52,    53,    -1,    55,
      -1,    57,    58,     1,    -1,    81,    -1,    -1,    64,    65,
      66,    67,    68,    69,    70,    71,    -1,    -1,    74,    75,
      76,    77,    78,    21,    -1,    23,    24,    25,    26,    27,
      28,    29,    -1,    -1,    32,    33,    34,    35,    36,    37,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    54,    55,    -1,     1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    71,    -1,    -1,    -1,    75,    76,    21,
      -1,    23,    24,    25,    26,    27,    28,    29,    -1,    -1,
      32,    33,    34,    35,    36,    37,    -1,    -1,     3,     4,
       5,    -1,     7,     8,     9,    -1,    11,    12,    13,    14,
      15,    -1,    54,    55,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    -1,    33,    34,
      35,    36,    37,    75,    76,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    -1,
      55,    -1,    57,    58,    59,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    71,    -1,    -1,    74,
      75,    76,    77,     3,     4,     5,    -1,     7,     8,     9,
      -1,    11,    12,    13,    14,    15,    -1,    -1,    -1,    19,
      20,    21,    22,    23,    24,    25,    26,    27,    28,    29,
      30,    31,     6,    33,    34,    35,    36,    37,    -1,    -1,
      40,    41,    42,    43,    44,    45,    46,    47,    48,    49,
      50,    51,    52,    53,    -1,    55,    -1,    57,    58,    16,
      -1,    -1,    -1,    -1,    38,    39,    -1,    -1,    -1,    -1,
      -1,    71,    -1,    -1,    74,    75,    76,    77,    -1,    -1,
      -1,    38,    39,    -1,    58,    -1,    -1,    -1,    -1,    -1,
      64,    65,    66,    67,    68,    69,    70,    -1,    72,    73,
      -1,    58,    -1,    -1,    78,    -1,    -1,    64,    65,    66,
      67,    68,    69,    70,    38,    39,    40,    41,    42,    -1,
      -1,    78,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    38,
      39,    40,    41,    42,    58,    -1,    -1,    -1,    -1,    -1,
      64,    65,    66,    67,    68,    69,    70,    -1,    -1,    58,
      38,    39,    -1,    -1,    78,    64,    65,    66,    67,    68,
      69,    70,    -1,    -1,    88,    38,    39,    -1,    -1,    78,
      58,    59,    60,    -1,    -1,    -1,    64,    65,    66,    67,
      68,    69,    70,    38,    39,    58,    -1,    -1,    -1,    -1,
      78,    64,    65,    66,    67,    68,    69,    70,    -1,    -1,
      -1,    -1,    -1,    58,    -1,    78,    -1,    -1,    -1,    64,
      65,    66,    67,    68,    69,    70,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    78
};

/* YYSTOS[STATE-NUM] -- The symbol kind of the accessing symbol of
   state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,     3,     4,    96,    97,   103,     3,     4,     5,     7,
       8,     9,    11,    12,    13,    14,    15,    19,    20,    21,
      22,    23,    24,    25,    26,    27,    28,    29,    30,    31,
      33,    34,    35,    36,    37,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    55,
      57,    58,    71,    74,    75,    76,    77,   101,   101,     0,
      97,    79,    81,    99,   104,   104,     1,     5,    56,    57,
      62,    63,    74,    98,   105,   106,   107,   173,   212,   213,
      79,    45,   101,    56,    58,   102,   101,   101,    81,    99,
     184,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    18,    19,    22,    81,   103,   126,   127,   143,
     146,   147,   148,   150,   160,   161,   164,   165,   166,    82,
       6,     7,     8,     9,    11,    12,    13,    14,    15,    45,
      82,    99,   171,   105,     1,    91,   175,    81,   102,    80,
      61,    87,   182,    16,    38,    39,    64,    65,    66,    67,
      68,    69,    70,    78,   102,   112,   114,   115,   116,   117,
     118,   119,   122,    38,   128,   128,    91,   128,   114,   167,
      91,   132,   132,   132,   132,    91,   136,   149,    91,   129,
     101,    60,   168,    84,   105,    11,    12,    13,    14,    15,
      16,   151,   152,   153,   154,   155,    99,   100,     1,    21,
      23,    24,    25,    26,    27,    28,    29,    32,    33,    34,
      35,    36,    37,    54,    55,    75,    76,   178,   179,    39,
     115,   116,   117,   120,   121,   174,   105,    80,    81,    87,
      91,   182,    82,   185,   115,   119,    64,    65,    69,    64,
      64,    65,    66,    67,   102,    84,   111,    86,    86,    86,
      39,    87,    89,    90,   102,   102,   102,    71,   102,    83,
      31,    54,   133,   138,   101,   113,   113,   113,   113,    54,
      59,   114,   135,   137,    91,   149,    91,   136,    43,    44,
     130,   131,   113,    18,    78,   118,   122,   158,   159,    82,
     132,   132,   132,   132,   149,   129,    60,   134,    92,    83,
      87,   120,    39,    89,    90,   114,    82,    15,    58,   182,
      61,   181,   182,    88,    99,    86,    64,    64,    65,    64,
     111,    59,    60,   108,   109,   110,   122,    86,    89,    90,
      90,    91,   124,   125,   210,    87,    84,    87,    92,    84,
      87,   167,    92,    83,   111,    80,   144,   144,   144,   144,
     101,    92,    83,    92,   113,   113,    92,    83,    81,    90,
     101,    58,    90,    93,   157,   101,    83,    85,   100,   101,
     101,   101,   101,   101,   101,   178,   101,   176,   177,    89,
      90,    90,   101,    82,    83,    88,    92,   182,   102,    64,
      83,    85,   101,    90,    90,    90,   125,   123,   182,   128,
     109,   128,   128,   109,   128,   133,   114,   145,    81,    99,
     162,   162,   162,   162,    92,   137,   144,   144,   130,    17,
      82,   139,   141,   142,    90,    93,   156,   156,    90,    59,
      60,   114,   157,   159,   144,   144,   144,   144,   144,    81,
      99,    80,    83,    88,    90,    90,    90,   111,   182,   181,
     182,   182,   125,   109,    88,    90,    90,    92,   211,    88,
      85,    88,   102,    85,    88,    83,     1,    45,   160,   163,
     164,   169,   170,   172,   162,   162,   122,   142,    82,    90,
     122,    90,   162,   162,   162,   162,   162,   142,    59,   177,
      83,    90,    90,    87,   193,    87,    90,    87,    87,    87,
     145,    91,   175,   172,    82,    99,   163,   101,   101,    58,
      82,   178,    90,    40,    41,    42,    88,   122,   183,   186,
     191,   191,   128,   128,   128,    71,    58,   101,   174,   100,
      91,   140,   156,   156,    99,   122,   183,   183,   183,   101,
     182,    83,    88,    88,    88,    88,    88,    92,   193,   101,
      92,    99,   101,   182,   182,   182,    91,    93,   139,    92,
     191,    38,     1,    46,    47,    48,    49,    50,    51,    52,
      77,    81,    99,   184,   196,   201,   203,   193,    92,    92,
      92,    59,    60,   102,   180,   101,    87,   207,    91,   207,
      58,   208,   209,    81,    60,   200,   207,    81,   197,   203,
     182,    20,   195,   193,   182,   205,    58,   205,   193,   210,
      83,    81,   203,   198,   203,   184,   205,    46,    47,    48,
      50,    51,    52,    77,   184,   199,   201,   202,    82,   197,
     185,    93,   196,    91,   194,    79,    92,    88,   206,   182,
     209,    82,   197,    82,   197,   182,   206,   207,    91,   207,
      81,   200,   207,    81,   182,    82,   199,   100,   100,    59,
       6,    72,    73,    92,   122,   187,   190,   192,   184,   205,
     207,    81,   203,   211,    82,   185,    81,   203,   182,    58,
     182,   198,   184,   182,   199,   185,   101,    80,    83,    92,
     182,    79,   205,   197,   193,   100,   197,    53,   204,    79,
      92,   206,    82,   182,   206,    82,   100,    84,   122,   183,
     189,   192,   185,   205,    80,    82,    82,    81,   203,   182,
     207,    81,   203,   185,    81,   203,   101,   188,   101,   182,
      84,   101,   206,   205,   204,   197,    79,   182,   197,   100,
     197,   204,    85,    87,    90,    91,    94,    84,    92,   188,
      99,    81,   203,    83,    82,   182,    80,    82,    82,   188,
     101,    59,   188,    85,   188,    85,   197,   205,   206,   182,
     204,    88,    92,    92,   101,    85,    82,   206,    81,   203,
      83,    81,   203,   197,   182,   197,    82,   206,    82,    81,
     203,   197,    82
};

/* YYR1[RULE-NUM] -- Symbol kind of the left-hand side of rule RULE-NUM.  */
static const yytype_uint8 yyr1[] =
{
       0,    95,    96,    97,    97,    98,    98,    99,    99,   100,
     100,   101,   101,   101,   101,   101,   101,   101,   101,   101,
     101,   101,   101,   101,   101,   101,   101,   101,   101,   101,
     101,   101,   101,   101,   101,   101,   101,   101,   101,   101,
     101,   101,   101,   101,   101,   101,   101,   101,   101,   101,
     101,   101,   101,   101,   101,   101,   101,   101,   101,   101,
     101,   101,   102,   102,   102,   103,   103,   104,   104,   105,
     105,   106,   106,   106,   106,   106,   107,   107,   107,   107,
     107,   107,   107,   107,   107,   107,   107,   107,   107,   107,
     108,   108,   108,   109,   109,   110,   110,   111,   111,   112,
     112,   112,   112,   112,   112,   112,   112,   112,   112,   112,
     112,   112,   112,   112,   112,   112,   112,   112,   112,   113,
     114,   114,   115,   115,   116,   117,   117,   118,   119,   119,
     119,   119,   119,   119,   120,   120,   120,   120,   120,   121,
     121,   121,   121,   121,   121,   122,   122,   122,   122,   122,
     122,   123,   124,   125,   125,   126,   127,   128,   128,   129,
     129,   130,   130,   131,   131,   132,   132,   133,   133,   134,
     134,   135,   136,   136,   137,   137,   138,   138,   139,   139,
     140,   140,   141,   142,   142,   143,   143,   143,   144,   144,
     145,   145,   146,   146,   147,   148,   149,   149,   150,   150,
     151,   151,   152,   153,   154,   155,   155,   156,   156,   157,
     157,   157,   157,   158,   158,   158,   158,   158,   158,   159,
     159,   160,   161,   161,   161,   161,   161,   162,   162,   163,
     163,   164,   164,   164,   164,   164,   164,   164,   165,   165,
     165,   165,   165,   166,   166,   166,   166,   167,   167,   168,
     169,   170,   170,   170,   170,   171,   171,   171,   171,   171,
     171,   171,   171,   171,   171,   171,   172,   172,   172,   173,
     173,   174,   175,   175,   175,   176,   177,   177,   178,   178,
     178,   178,   179,   179,   179,   179,   179,   179,   179,   179,
     179,   179,   179,   179,   179,   179,   179,   179,   179,   179,
     179,   180,   180,   180,   181,   181,   181,   182,   182,   182,
     182,   182,   182,   183,   184,   185,   186,   186,   186,   186,
     186,   186,   186,   187,   187,   187,   188,   188,   188,   188,
     188,   188,   189,   190,   190,   190,   191,   191,   192,   192,
     193,   193,   194,   194,   195,   195,   196,   196,   196,   197,
     197,   198,   198,   199,   199,   199,   200,   200,   201,   201,
     201,   202,   202,   202,   202,   202,   202,   202,   202,   202,
     202,   202,   202,   203,   203,   203,   203,   203,   203,   203,
     203,   203,   203,   203,   203,   203,   203,   204,   204,   204,
     205,   206,   207,   208,   208,   209,   209,   210,   211,   212,
     213
};

/* YYR2[RULE-NUM] -- Number of symbols on the right-hand side of rule RULE-NUM.  */
static const yytype_int8 yyr2[] =
{
       0,     2,     1,     0,     2,     0,     1,     1,     2,     0,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     4,     4,     3,     3,     1,     4,     0,
       2,     3,     2,     2,     2,     8,     5,     5,     2,     2,
       2,     2,     2,     2,     2,     2,     1,     1,     1,     1,
       1,     1,     1,     1,     3,     0,     1,     0,     3,     1,
       1,     2,     1,     2,     1,     2,     2,     3,     3,     4,
       2,     3,     2,     2,     3,     1,     1,     2,     1,     2,
       2,     3,     1,     1,     2,     2,     2,     8,     1,     1,
       1,     1,     2,     2,     1,     1,     1,     2,     2,     6,
       5,     4,     3,     2,     1,     6,     5,     4,     3,     2,
       1,     1,     3,     0,     2,     4,     6,     0,     1,     0,
       3,     1,     3,     1,     1,     0,     3,     1,     3,     0,
       1,     1,     0,     3,     1,     3,     1,     1,     0,     1,
       0,     2,     5,     1,     2,     3,     5,     6,     0,     2,
       1,     3,     5,     5,     5,     5,     4,     3,     6,     6,
       5,     5,     5,     5,     5,     4,     7,     0,     2,     0,
       2,     2,     2,     6,     6,     3,     3,     2,     3,     1,
       3,     4,     2,     2,     2,     2,     2,     1,     4,     0,
       2,     1,     1,     1,     1,     2,     2,     2,     3,     6,
       9,     3,     6,     3,     6,     9,     9,     1,     3,     1,
       1,     1,     2,     2,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     7,     5,    13,     5,
       2,     1,     0,     3,     1,     3,     1,     3,     1,     4,
       3,     6,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     2,     1,     1,     1,
       1,     1,     1,     1,     0,     1,     3,     0,     1,     5,
       5,     5,     4,     3,     1,     1,     1,     3,     4,     3,
       4,     4,     4,     1,     1,     1,     1,     4,     3,     4,
       4,     4,     3,     7,     5,     6,     1,     3,     1,     3,
       3,     2,     3,     2,     0,     3,     1,     1,     4,     1,
       2,     1,     2,     1,     2,     1,     1,     0,     4,     3,
       5,     6,     4,     4,    11,     9,    12,    14,     6,     8,
       5,     7,     4,     6,     4,     1,     4,    11,     9,    12,
      14,     6,     8,     5,     7,     4,     1,     0,     2,     4,
       1,     1,     1,     2,     5,     1,     3,     1,     1,     2,
       2
};


enum { YYENOMEM = -2 };

#define yyerrok         (yyerrstatus = 0)
#define yyclearin       (yychar = YYEMPTY)

#define YYACCEPT        goto yyacceptlab
#define YYABORT         goto yyabortlab
#define YYERROR         goto yyerrorlab
#define YYNOMEM         goto yyexhaustedlab


#define YYRECOVERING()  (!!yyerrstatus)

#define YYBACKUP(Token, Value)                                    \
  do                                                              \
    if (yychar == YYEMPTY)                                        \
      {                                                           \
        yychar = (Token);                                         \
        yylval = (Value);                                         \
        YYPOPSTACK (yylen);                                       \
        yystate = *yyssp;                                         \
        goto yybackup;                                            \
      }                                                           \
    else                                                          \
      {                                                           \
        yyerror (YY_("syntax error: cannot back up")); \
        YYERROR;                                                  \
      }                                                           \
  while (0)

/* Backward compatibility with an undocumented macro.
   Use YYerror or YYUNDEF. */
#define YYERRCODE YYUNDEF

/* YYLLOC_DEFAULT -- Set CURRENT to span from RHS[1] to RHS[N].
   If N is 0, then set CURRENT to the empty location which ends
   the previous symbol: RHS[0] (always defined).  */

#ifndef YYLLOC_DEFAULT
# define YYLLOC_DEFAULT(Current, Rhs, N)                                \
    do                                                                  \
      if (N)                                                            \
        {                                                               \
          (Current).first_line   = YYRHSLOC (Rhs, 1).first_line;        \
          (Current).first_column = YYRHSLOC (Rhs, 1).first_column;      \
          (Current).last_line    = YYRHSLOC (Rhs, N).last_line;         \
          (Current).last_column  = YYRHSLOC (Rhs, N).last_column;       \
        }                                                               \
      else                                                              \
        {                                                               \
          (Current).first_line   = (Current).last_line   =              \
            YYRHSLOC (Rhs, 0).last_line;                                \
          (Current).first_column = (Current).last_column =              \
            YYRHSLOC (Rhs, 0).last_column;                              \
        }                                                               \
    while (0)
#endif

#define YYRHSLOC(Rhs, K) ((Rhs)[K])


/* Enable debugging if requested.  */
#if YYDEBUG

# ifndef YYFPRINTF
#  include <stdio.h> /* INFRINGES ON USER NAME SPACE */
#  define YYFPRINTF fprintf
# endif

# define YYDPRINTF(Args)                        \
do {                                            \
  if (yydebug)                                  \
    YYFPRINTF Args;                             \
} while (0)


/* YYLOCATION_PRINT -- Print the location on the stream.
   This macro was not mandated originally: define only if we know
   we won't break user code: when these are the locations we know.  */

# ifndef YYLOCATION_PRINT

#  if defined YY_LOCATION_PRINT

   /* Temporary convenience wrapper in case some people defined the
      undocumented and private YY_LOCATION_PRINT macros.  */
#   define YYLOCATION_PRINT(File, Loc)  YY_LOCATION_PRINT(File, *(Loc))

#  elif defined YYLTYPE_IS_TRIVIAL && YYLTYPE_IS_TRIVIAL

/* Print *YYLOCP on YYO.  Private, do not rely on its existence. */

YY_ATTRIBUTE_UNUSED
static int
yy_location_print_ (FILE *yyo, YYLTYPE const * const yylocp)
{
  int res = 0;
  int end_col = 0 != yylocp->last_column ? yylocp->last_column - 1 : 0;
  if (0 <= yylocp->first_line)
    {
      res += YYFPRINTF (yyo, "%d", yylocp->first_line);
      if (0 <= yylocp->first_column)
        res += YYFPRINTF (yyo, ".%d", yylocp->first_column);
    }
  if (0 <= yylocp->last_line)
    {
      if (yylocp->first_line < yylocp->last_line)
        {
          res += YYFPRINTF (yyo, "-%d", yylocp->last_line);
          if (0 <= end_col)
            res += YYFPRINTF (yyo, ".%d", end_col);
        }
      else if (0 <= end_col && yylocp->first_column < end_col)
        res += YYFPRINTF (yyo, "-%d", end_col);
    }
  return res;
}

#   define YYLOCATION_PRINT  yy_location_print_

    /* Temporary convenience wrapper in case some people defined the
       undocumented and private YY_LOCATION_PRINT macros.  */
#   define YY_LOCATION_PRINT(File, Loc)  YYLOCATION_PRINT(File, &(Loc))

#  else

#   define YYLOCATION_PRINT(File, Loc) ((void) 0)
    /* Temporary convenience wrapper in case some people defined the
       undocumented and private YY_LOCATION_PRINT macros.  */
#   define YY_LOCATION_PRINT  YYLOCATION_PRINT

#  endif
# endif /* !defined YYLOCATION_PRINT */


# define YY_SYMBOL_PRINT(Title, Kind, Value, Location)                    \
do {                                                                      \
  if (yydebug)                                                            \
    {                                                                     \
      YYFPRINTF (stderr, "%s ", Title);                                   \
      yy_symbol_print (stderr,                                            \
                  Kind, Value, Location); \
      YYFPRINTF (stderr, "\n");                                           \
    }                                                                     \
} while (0)


/*-----------------------------------.
| Print this symbol's value on YYO.  |
`-----------------------------------*/

static void
yy_symbol_value_print (FILE *yyo,
                       yysymbol_kind_t yykind, YYSTYPE const * const yyvaluep, YYLTYPE const * const yylocationp)
{
  FILE *yyoutput = yyo;
  YY_USE (yyoutput);
  YY_USE (yylocationp);
  if (!yyvaluep)
    return;
  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  YY_USE (yykind);
  YY_IGNORE_MAYBE_UNINITIALIZED_END
}


/*---------------------------.
| Print this symbol on YYO.  |
`---------------------------*/

static void
yy_symbol_print (FILE *yyo,
                 yysymbol_kind_t yykind, YYSTYPE const * const yyvaluep, YYLTYPE const * const yylocationp)
{
  YYFPRINTF (yyo, "%s %s (",
             yykind < YYNTOKENS ? "token" : "nterm", yysymbol_name (yykind));

  YYLOCATION_PRINT (yyo, yylocationp);
  YYFPRINTF (yyo, ": ");
  yy_symbol_value_print (yyo, yykind, yyvaluep, yylocationp);
  YYFPRINTF (yyo, ")");
}

/*------------------------------------------------------------------.
| yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

static void
yy_stack_print (yy_state_t *yybottom, yy_state_t *yytop)
{
  YYFPRINTF (stderr, "Stack now");
  for (; yybottom <= yytop; yybottom++)
    {
      int yybot = *yybottom;
      YYFPRINTF (stderr, " %d", yybot);
    }
  YYFPRINTF (stderr, "\n");
}

# define YY_STACK_PRINT(Bottom, Top)                            \
do {                                                            \
  if (yydebug)                                                  \
    yy_stack_print ((Bottom), (Top));                           \
} while (0)


/*------------------------------------------------.
| Report that the YYRULE is going to be reduced.  |
`------------------------------------------------*/

static void
yy_reduce_print (yy_state_t *yyssp, YYSTYPE *yyvsp, YYLTYPE *yylsp,
                 int yyrule)
{
  int yylno = yyrline[yyrule];
  int yynrhs = yyr2[yyrule];
  int yyi;
  YYFPRINTF (stderr, "Reducing stack by rule %d (line %d):\n",
             yyrule - 1, yylno);
  /* The symbols being reduced.  */
  for (yyi = 0; yyi < yynrhs; yyi++)
    {
      YYFPRINTF (stderr, "   $%d = ", yyi + 1);
      yy_symbol_print (stderr,
                       YY_ACCESSING_SYMBOL (+yyssp[yyi + 1 - yynrhs]),
                       &yyvsp[(yyi + 1) - (yynrhs)],
                       &(yylsp[(yyi + 1) - (yynrhs)]));
      YYFPRINTF (stderr, "\n");
    }
}

# define YY_REDUCE_PRINT(Rule)          \
do {                                    \
  if (yydebug)                          \
    yy_reduce_print (yyssp, yyvsp, yylsp, Rule); \
} while (0)

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int yydebug;
#else /* !YYDEBUG */
# define YYDPRINTF(Args) ((void) 0)
# define YY_SYMBOL_PRINT(Title, Kind, Value, Location)
# define YY_STACK_PRINT(Bottom, Top)
# define YY_REDUCE_PRINT(Rule)
#endif /* !YYDEBUG */


/* YYINITDEPTH -- initial size of the parser's stacks.  */
#ifndef YYINITDEPTH
# define YYINITDEPTH 200
#endif

/* YYMAXDEPTH -- maximum size the stacks can grow to (effective only
   if the built-in stack extension method is used).

   Do not make this value too large; the results are undefined if
   YYSTACK_ALLOC_MAXIMUM < YYSTACK_BYTES (YYMAXDEPTH)
   evaluated with infinite-precision integer arithmetic.  */

#ifndef YYMAXDEPTH
# define YYMAXDEPTH 10000
#endif






/*-----------------------------------------------.
| Release the memory associated to this symbol.  |
`-----------------------------------------------*/

static void
yydestruct (const char *yymsg,
            yysymbol_kind_t yykind, YYSTYPE *yyvaluep, YYLTYPE *yylocationp)
{
  YY_USE (yyvaluep);
  YY_USE (yylocationp);
  if (!yymsg)
    yymsg = "Deleting";
  YY_SYMBOL_PRINT (yymsg, yykind, yyvaluep, yylocationp);

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  YY_USE (yykind);
  YY_IGNORE_MAYBE_UNINITIALIZED_END
}


/* Lookahead token kind.  */
int yychar;

/* The semantic value of the lookahead symbol.  */
YYSTYPE yylval;
/* Location data for the lookahead symbol.  */
YYLTYPE yylloc
# if defined YYLTYPE_IS_TRIVIAL && YYLTYPE_IS_TRIVIAL
  = { 1, 1, 1, 1 }
# endif
;
/* Number of syntax errors so far.  */
int yynerrs;




/*----------.
| yyparse.  |
`----------*/

int
yyparse (void)
{
    yy_state_fast_t yystate = 0;
    /* Number of tokens to shift before error messages enabled.  */
    int yyerrstatus = 0;

    /* Refer to the stacks through separate pointers, to allow yyoverflow
       to reallocate them elsewhere.  */

    /* Their size.  */
    YYPTRDIFF_T yystacksize = YYINITDEPTH;

    /* The state stack: array, bottom, top.  */
    yy_state_t yyssa[YYINITDEPTH];
    yy_state_t *yyss = yyssa;
    yy_state_t *yyssp = yyss;

    /* The semantic value stack: array, bottom, top.  */
    YYSTYPE yyvsa[YYINITDEPTH];
    YYSTYPE *yyvs = yyvsa;
    YYSTYPE *yyvsp = yyvs;

    /* The location stack: array, bottom, top.  */
    YYLTYPE yylsa[YYINITDEPTH];
    YYLTYPE *yyls = yylsa;
    YYLTYPE *yylsp = yyls;

  int yyn;
  /* The return value of yyparse.  */
  int yyresult;
  /* Lookahead symbol kind.  */
  yysymbol_kind_t yytoken = YYSYMBOL_YYEMPTY;
  /* The variables used to return semantic value and location from the
     action routines.  */
  YYSTYPE yyval;
  YYLTYPE yyloc;

  /* The locations where the error started and ended.  */
  YYLTYPE yyerror_range[3];



#define YYPOPSTACK(N)   (yyvsp -= (N), yyssp -= (N), yylsp -= (N))

  /* The number of symbols on the RHS of the reduced rule.
     Keep to zero when no symbol should be popped.  */
  int yylen = 0;

  YYDPRINTF ((stderr, "Starting parse\n"));

  yychar = YYEMPTY; /* Cause a token to be read.  */

  yylsp[0] = yylloc;
  goto yysetstate;


/*------------------------------------------------------------.
| yynewstate -- push a new state, which is found in yystate.  |
`------------------------------------------------------------*/
yynewstate:
  /* In all cases, when you get here, the value and location stacks
     have just been pushed.  So pushing a state here evens the stacks.  */
  yyssp++;


/*--------------------------------------------------------------------.
| yysetstate -- set current state (the top of the stack) to yystate.  |
`--------------------------------------------------------------------*/
yysetstate:
  YYDPRINTF ((stderr, "Entering state %d\n", yystate));
  YY_ASSERT (0 <= yystate && yystate < YYNSTATES);
  YY_IGNORE_USELESS_CAST_BEGIN
  *yyssp = YY_CAST (yy_state_t, yystate);
  YY_IGNORE_USELESS_CAST_END
  YY_STACK_PRINT (yyss, yyssp);

  if (yyss + yystacksize - 1 <= yyssp)
#if !defined yyoverflow && !defined YYSTACK_RELOCATE
    YYNOMEM;
#else
    {
      /* Get the current used size of the three stacks, in elements.  */
      YYPTRDIFF_T yysize = yyssp - yyss + 1;

# if defined yyoverflow
      {
        /* Give user a chance to reallocate the stack.  Use copies of
           these so that the &'s don't force the real ones into
           memory.  */
        yy_state_t *yyss1 = yyss;
        YYSTYPE *yyvs1 = yyvs;
        YYLTYPE *yyls1 = yyls;

        /* Each stack pointer address is followed by the size of the
           data in use in that stack, in bytes.  This used to be a
           conditional around just the two extra args, but that might
           be undefined if yyoverflow is a macro.  */
        yyoverflow (YY_("memory exhausted"),
                    &yyss1, yysize * YYSIZEOF (*yyssp),
                    &yyvs1, yysize * YYSIZEOF (*yyvsp),
                    &yyls1, yysize * YYSIZEOF (*yylsp),
                    &yystacksize);
        yyss = yyss1;
        yyvs = yyvs1;
        yyls = yyls1;
      }
# else /* defined YYSTACK_RELOCATE */
      /* Extend the stack our own way.  */
      if (YYMAXDEPTH <= yystacksize)
        YYNOMEM;
      yystacksize *= 2;
      if (YYMAXDEPTH < yystacksize)
        yystacksize = YYMAXDEPTH;

      {
        yy_state_t *yyss1 = yyss;
        union yyalloc *yyptr =
          YY_CAST (union yyalloc *,
                   YYSTACK_ALLOC (YY_CAST (YYSIZE_T, YYSTACK_BYTES (yystacksize))));
        if (! yyptr)
          YYNOMEM;
        YYSTACK_RELOCATE (yyss_alloc, yyss);
        YYSTACK_RELOCATE (yyvs_alloc, yyvs);
        YYSTACK_RELOCATE (yyls_alloc, yyls);
#  undef YYSTACK_RELOCATE
        if (yyss1 != yyssa)
          YYSTACK_FREE (yyss1);
      }
# endif

      yyssp = yyss + yysize - 1;
      yyvsp = yyvs + yysize - 1;
      yylsp = yyls + yysize - 1;

      YY_IGNORE_USELESS_CAST_BEGIN
      YYDPRINTF ((stderr, "Stack size increased to %ld\n",
                  YY_CAST (long, yystacksize)));
      YY_IGNORE_USELESS_CAST_END

      if (yyss + yystacksize - 1 <= yyssp)
        YYABORT;
    }
#endif /* !defined yyoverflow && !defined YYSTACK_RELOCATE */


  if (yystate == YYFINAL)
    YYACCEPT;

  goto yybackup;


/*-----------.
| yybackup.  |
`-----------*/
yybackup:
  /* Do appropriate processing given the current state.  Read a
     lookahead token if we need one and don't already have one.  */

  /* First try to decide what to do without reference to lookahead token.  */
  yyn = yypact[yystate];
  if (yypact_value_is_default (yyn))
    goto yydefault;

  /* Not known => get a lookahead token if don't already have one.  */

  /* YYCHAR is either empty, or end-of-input, or a valid lookahead.  */
  if (yychar == YYEMPTY)
    {
      YYDPRINTF ((stderr, "Reading a token\n"));
      yychar = yylex ();
    }

  if (yychar <= YYEOF)
    {
      yychar = YYEOF;
      yytoken = YYSYMBOL_YYEOF;
      YYDPRINTF ((stderr, "Now at end of input.\n"));
    }
  else if (yychar == YYerror)
    {
      /* The scanner already issued an error message, process directly
         to error recovery.  But do not keep the error token as
         lookahead, it is too special and may lead us to an endless
         loop in error recovery. */
      yychar = YYUNDEF;
      yytoken = YYSYMBOL_YYerror;
      yyerror_range[1] = yylloc;
      goto yyerrlab1;
    }
  else
    {
      yytoken = YYTRANSLATE (yychar);
      YY_SYMBOL_PRINT ("Next token is", yytoken, &yylval, &yylloc);
    }

  /* If the proper action on seeing token YYTOKEN is to reduce or to
     detect an error, take that action.  */
  yyn += yytoken;
  if (yyn < 0 || YYLAST < yyn || yycheck[yyn] != yytoken)
    goto yydefault;
  yyn = yytable[yyn];
  if (yyn <= 0)
    {
      if (yytable_value_is_error (yyn))
        goto yyerrlab;
      yyn = -yyn;
      goto yyreduce;
    }

  /* Count tokens shifted since error; after three, turn off error
     status.  */
  if (yyerrstatus)
    yyerrstatus--;

  /* Shift the lookahead token.  */
  YY_SYMBOL_PRINT ("Shifting", yytoken, &yylval, &yylloc);
  yystate = yyn;
  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END
  *++yylsp = yylloc;

  /* Discard the shifted token.  */
  yychar = YYEMPTY;
  goto yynewstate;


/*-----------------------------------------------------------.
| yydefault -- do the default action for the current state.  |
`-----------------------------------------------------------*/
yydefault:
  yyn = yydefact[yystate];
  if (yyn == 0)
    goto yyerrlab;
  goto yyreduce;


/*-----------------------------.
| yyreduce -- do a reduction.  |
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
  yyval = yyvsp[1-yylen];

  /* Default location. */
  YYLLOC_DEFAULT (yyloc, (yylsp - yylen), yylen);
  yyerror_range[1] = yyloc;
  YY_REDUCE_PRINT (yyn);
  switch (yyn)
    {
  case 2: /* File: ModuleEList  */
#line 204 "xi-grammar.y"
                { (yyval.modlist) = (yyvsp[0].modlist); modlist = (yyvsp[0].modlist); }
#line 2413 "y.tab.c"
    break;

  case 3: /* ModuleEList: %empty  */
#line 208 "xi-grammar.y"
                { 
		  (yyval.modlist) = 0; 
		}
#line 2421 "y.tab.c"
    break;

  case 4: /* ModuleEList: Module ModuleEList  */
#line 212 "xi-grammar.y"
                { (yyval.modlist) = new AstChildren<Module>(lineno, (yyvsp[-1].module), (yyvsp[0].modlist)); }
#line 2427 "y.tab.c"
    break;

  case 5: /* OptExtern: %empty  */
#line 216 "xi-grammar.y"
                { (yyval.intval) = 0; }
#line 2433 "y.tab.c"
    break;

  case 6: /* OptExtern: EXTERN  */
#line 218 "xi-grammar.y"
                { (yyval.intval) = 1; }
#line 2439 "y.tab.c"
    break;

  case 7: /* OneOrMoreSemiColon: ';'  */
#line 222 "xi-grammar.y"
                { (yyval.intval) = 1; }
#line 2445 "y.tab.c"
    break;

  case 8: /* OneOrMoreSemiColon: OneOrMoreSemiColon ';'  */
#line 224 "xi-grammar.y"
                { (yyval.intval) = 2; }
#line 2451 "y.tab.c"
    break;

  case 9: /* OptSemiColon: %empty  */
#line 228 "xi-grammar.y"
                { (yyval.intval) = 0; }
#line 2457 "y.tab.c"
    break;

  case 10: /* OptSemiColon: OneOrMoreSemiColon  */
#line 230 "xi-grammar.y"
                { (yyval.intval) = 1; }
#line 2463 "y.tab.c"
    break;

  case 11: /* Name: IDENT  */
#line 235 "xi-grammar.y"
                { (yyval.strval) = (yyvsp[0].strval); }
#line 2469 "y.tab.c"
    break;

  case 12: /* Name: MODULE  */
#line 236 "xi-grammar.y"
                         { ReservedWord(MODULE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2475 "y.tab.c"
    break;

  case 13: /* Name: MAINMODULE  */
#line 237 "xi-grammar.y"
                             { ReservedWord(MAINMODULE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2481 "y.tab.c"
    break;

  case 14: /* Name: EXTERN  */
#line 238 "xi-grammar.y"
                         { ReservedWord(EXTERN, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2487 "y.tab.c"
    break;

  case 15: /* Name: INITCALL  */
#line 240 "xi-grammar.y"
                           { ReservedWord(INITCALL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2493 "y.tab.c"
    break;

  case 16: /* Name: INITNODE  */
#line 241 "xi-grammar.y"
                           { ReservedWord(INITNODE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2499 "y.tab.c"
    break;

  case 17: /* Name: INITPROC  */
#line 242 "xi-grammar.y"
                           { ReservedWord(INITPROC, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2505 "y.tab.c"
    break;

  case 18: /* Name: CHARE  */
#line 244 "xi-grammar.y"
                        { ReservedWord(CHARE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2511 "y.tab.c"
    break;

  case 19: /* Name: MAINCHARE  */
#line 245 "xi-grammar.y"
                            { ReservedWord(MAINCHARE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2517 "y.tab.c"
    break;

  case 20: /* Name: GROUP  */
#line 246 "xi-grammar.y"
                        { ReservedWord(GROUP, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2523 "y.tab.c"
    break;

  case 21: /* Name: NODEGROUP  */
#line 247 "xi-grammar.y"
                            { ReservedWord(NODEGROUP, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2529 "y.tab.c"
    break;

  case 22: /* Name: ARRAY  */
#line 248 "xi-grammar.y"
                        { ReservedWord(ARRAY, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2535 "y.tab.c"
    break;

  case 23: /* Name: INCLUDE  */
#line 252 "xi-grammar.y"
                          { ReservedWord(INCLUDE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2541 "y.tab.c"
    break;

  case 24: /* Name: STACKSIZE  */
#line 253 "xi-grammar.y"
                            { ReservedWord(STACKSIZE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2547 "y.tab.c"
    break;

  case 25: /* Name: THREADED  */
#line 254 "xi-grammar.y"
                           { ReservedWord(THREADED, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2553 "y.tab.c"
    break;

  case 26: /* Name: TEMPLATE  */
#line 255 "xi-grammar.y"
                           { ReservedWord(TEMPLATE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2559 "y.tab.c"
    break;

  case 27: /* Name: WHENIDLE  */
#line 256 "xi-grammar.y"
                           { ReservedWord(WHENIDLE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2565 "y.tab.c"
    break;

  case 28: /* Name: SYNC  */
#line 257 "xi-grammar.y"
                       { ReservedWord(SYNC, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2571 "y.tab.c"
    break;

  case 29: /* Name: IGET  */
#line 258 "xi-grammar.y"
                       { ReservedWord(IGET, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2577 "y.tab.c"
    break;

  case 30: /* Name: EXCLUSIVE  */
#line 259 "xi-grammar.y"
                            { ReservedWord(EXCLUSIVE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2583 "y.tab.c"
    break;

  case 31: /* Name: IMMEDIATE  */
#line 260 "xi-grammar.y"
                            { ReservedWord(IMMEDIATE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2589 "y.tab.c"
    break;

  case 32: /* Name: SKIPSCHED  */
#line 261 "xi-grammar.y"
                            { ReservedWord(SKIPSCHED, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2595 "y.tab.c"
    break;

  case 33: /* Name: NOCOPY  */
#line 262 "xi-grammar.y"
                         { ReservedWord(NOCOPY, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2601 "y.tab.c"
    break;

  case 34: /* Name: NOCOPYPOST  */
#line 263 "xi-grammar.y"
                             { ReservedWord(NOCOPYPOST, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2607 "y.tab.c"
    break;

  case 35: /* Name: NOCOPYDEVICE  */
#line 264 "xi-grammar.y"
                               { ReservedWord(NOCOPYDEVICE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2613 "y.tab.c"
    break;

  case 36: /* Name: INLINE  */
#line 265 "xi-grammar.y"
                         { ReservedWord(INLINE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2619 "y.tab.c"
    break;

  case 37: /* Name: VIRTUAL  */
#line 266 "xi-grammar.y"
                          { ReservedWord(VIRTUAL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2625 "y.tab.c"
    break;

  case 38: /* Name: MIGRATABLE  */
#line 267 "xi-grammar.y"
                             { ReservedWord(MIGRATABLE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2631 "y.tab.c"
    break;

  case 39: /* Name: CREATEHERE  */
#line 268 "xi-grammar.y"
                             { ReservedWord(CREATEHERE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2637 "y.tab.c"
    break;

  case 40: /* Name: CREATEHOME  */
#line 269 "xi-grammar.y"
                             { ReservedWord(CREATEHOME, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2643 "y.tab.c"
    break;

  case 41: /* Name: NOKEEP  */
#line 270 "xi-grammar.y"
                         { ReservedWord(NOKEEP, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2649 "y.tab.c"
    break;

  case 42: /* Name: NOTRACE  */
#line 271 "xi-grammar.y"
                          { ReservedWord(NOTRACE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2655 "y.tab.c"
    break;

  case 43: /* Name: APPWORK  */
#line 272 "xi-grammar.y"
                          { ReservedWord(APPWORK, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2661 "y.tab.c"
    break;

  case 44: /* Name: PACKED  */
#line 275 "xi-grammar.y"
                         { ReservedWord(PACKED, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2667 "y.tab.c"
    break;

  case 45: /* Name: VARSIZE  */
#line 276 "xi-grammar.y"
                          { ReservedWord(VARSIZE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2673 "y.tab.c"
    break;

  case 46: /* Name: ENTRY  */
#line 277 "xi-grammar.y"
                        { ReservedWord(ENTRY, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2679 "y.tab.c"
    break;

  case 47: /* Name: FOR  */
#line 278 "xi-grammar.y"
                      { ReservedWord(FOR, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2685 "y.tab.c"
    break;

  case 48: /* Name: FORALL  */
#line 279 "xi-grammar.y"
                         { ReservedWord(FORALL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2691 "y.tab.c"
    break;

  case 49: /* Name: WHILE  */
#line 280 "xi-grammar.y"
                        { ReservedWord(WHILE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2697 "y.tab.c"
    break;

  case 50: /* Name: WHEN  */
#line 281 "xi-grammar.y"
                       { ReservedWord(WHEN, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2703 "y.tab.c"
    break;

  case 51: /* Name: OVERLAP  */
#line 282 "xi-grammar.y"
                          { ReservedWord(OVERLAP, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2709 "y.tab.c"
    break;

  case 52: /* Name: SERIAL  */
#line 283 "xi-grammar.y"
                         { ReservedWord(SERIAL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2715 "y.tab.c"
    break;

  case 53: /* Name: IF  */
#line 284 "xi-grammar.y"
                     { ReservedWord(IF, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2721 "y.tab.c"
    break;

  case 54: /* Name: ELSE  */
#line 285 "xi-grammar.y"
                       { ReservedWord(ELSE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2727 "y.tab.c"
    break;

  case 55: /* Name: LOCAL  */
#line 287 "xi-grammar.y"
                        { ReservedWord(LOCAL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2733 "y.tab.c"
    break;

  case 56: /* Name: USING  */
#line 289 "xi-grammar.y"
                        { ReservedWord(USING, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2739 "y.tab.c"
    break;

  case 57: /* Name: ACCEL  */
#line 290 "xi-grammar.y"
                        { ReservedWord(ACCEL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2745 "y.tab.c"
    break;

  case 58: /* Name: ACCELBLOCK  */
#line 293 "xi-grammar.y"
                             { ReservedWord(ACCELBLOCK, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2751 "y.tab.c"
    break;

  case 59: /* Name: MEMCRITICAL  */
#line 294 "xi-grammar.y"
                              { ReservedWord(MEMCRITICAL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2757 "y.tab.c"
    break;

  case 60: /* Name: REDUCTIONTARGET  */
#line 295 "xi-grammar.y"
                                  { ReservedWord(REDUCTIONTARGET, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2763 "y.tab.c"
    break;

  case 61: /* Name: CASE  */
#line 296 "xi-grammar.y"
                       { ReservedWord(CASE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2769 "y.tab.c"
    break;

  case 62: /* QualName: IDENT  */
#line 301 "xi-grammar.y"
                { (yyval.strval) = (yyvsp[0].strval); }
#line 2775 "y.tab.c"
    break;

  case 63: /* QualName: QualName ':' ':' IDENT  */
#line 303 "xi-grammar.y"
                {
		  int len = strlen((yyvsp[-3].strval))+strlen((yyvsp[0].strval))+3;
		  char *tmp = new char[len];
		  snprintf(tmp,len,"%s::%s", (yyvsp[-3].strval), (yyvsp[0].strval));
		  (yyval.strval) = tmp;
		}
#line 2786 "y.tab.c"
    break;

  case 64: /* QualName: QualName ':' ':' ARRAY  */
#line 310 "xi-grammar.y"
                {
		  int len = strlen((yyvsp[-3].strval))+5+3;
		  char *tmp = new char[len];
		  snprintf(tmp, len, "%s::array", (yyvsp[-3].strval));
		  (yyval.strval) = tmp;
		}
#line 2797 "y.tab.c"
    break;

  case 65: /* Module: MODULE Name ConstructEList  */
#line 318 "xi-grammar.y"
                { 
		    (yyval.module) = new Module(lineno, (yyvsp[-1].strval), (yyvsp[0].conslist)); 
		}
#line 2805 "y.tab.c"
    break;

  case 66: /* Module: MAINMODULE Name ConstructEList  */
#line 322 "xi-grammar.y"
                {  
		    (yyval.module) = new Module(lineno, (yyvsp[-1].strval), (yyvsp[0].conslist)); 
		    (yyval.module)->setMain();
		}
#line 2814 "y.tab.c"
    break;

  case 67: /* ConstructEList: OneOrMoreSemiColon  */
#line 329 "xi-grammar.y"
                { (yyval.conslist) = 0; }
#line 2820 "y.tab.c"
    break;

  case 68: /* ConstructEList: '{' ConstructList '}' OptSemiColon  */
#line 331 "xi-grammar.y"
                { (yyval.conslist) = (yyvsp[-2].conslist); }
#line 2826 "y.tab.c"
    break;

  case 69: /* ConstructList: %empty  */
#line 335 "xi-grammar.y"
                { (yyval.conslist) = 0; }
#line 2832 "y.tab.c"
    break;

  case 70: /* ConstructList: Construct ConstructList  */
#line 337 "xi-grammar.y"
                { (yyval.conslist) = new ConstructList(lineno, (yyvsp[-1].construct), (yyvsp[0].conslist)); }
#line 2838 "y.tab.c"
    break;

  case 71: /* ConstructSemi: USING NAMESPACE QualName  */
#line 341 "xi-grammar.y"
                { (yyval.construct) = new UsingScope((yyvsp[0].strval), false); }
#line 2844 "y.tab.c"
    break;

  case 72: /* ConstructSemi: USING QualName  */
#line 343 "xi-grammar.y"
                { (yyval.construct) = new UsingScope((yyvsp[0].strval), true); }
#line 2850 "y.tab.c"
    break;

  case 73: /* ConstructSemi: OptExtern NonEntryMember  */
#line 345 "xi-grammar.y"
                { (yyvsp[0].member)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].member); }
#line 2856 "y.tab.c"
    break;

  case 74: /* ConstructSemi: OptExtern Message  */
#line 347 "xi-grammar.y"
                { (yyvsp[0].message)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].message); }
#line 2862 "y.tab.c"
    break;

  case 75: /* ConstructSemi: EXTERN ENTRY EAttribs EReturn QualNamedType Name OptTParams EParameters  */
#line 349 "xi-grammar.y"
                {
                  Entry *e = new Entry(lineno, (yyvsp[-5].attr), (yyvsp[-4].type), (yyvsp[-2].strval), (yyvsp[0].plist), 0, 0, 0, (yylsp[-7]).first_line, (yyloc).last_line);
                  int isExtern = 1;
                  e->setExtern(isExtern);
                  e->targs = (yyvsp[-1].tparlist);
                  e->label = new XStr;
                  (yyvsp[-3].ntype)->print(*e->label);
                  (yyval.construct) = e;
                  firstRdma = true;
                  firstDeviceRdma = true;
                }
#line 2878 "y.tab.c"
    break;

  case 76: /* Construct: OptExtern '{' ConstructList '}' OptSemiColon  */
#line 363 "xi-grammar.y"
        { if((yyvsp[-2].conslist)) (yyvsp[-2].conslist)->recurse<int&>((yyvsp[-4].intval), &Construct::setExtern); (yyval.construct) = (yyvsp[-2].conslist); }
#line 2884 "y.tab.c"
    break;

  case 77: /* Construct: NAMESPACE Name '{' ConstructList '}'  */
#line 365 "xi-grammar.y"
        { (yyval.construct) = new Scope((yyvsp[-3].strval), (yyvsp[-1].conslist)); }
#line 2890 "y.tab.c"
    break;

  case 78: /* Construct: ConstructSemi OneOrMoreSemiColon  */
#line 367 "xi-grammar.y"
        { (yyval.construct) = (yyvsp[-1].construct); }
#line 2896 "y.tab.c"
    break;

  case 79: /* Construct: ConstructSemi UnexpectedToken  */
#line 369 "xi-grammar.y"
        {
          ERROR("preceding construct must be semicolon terminated",
                (yyloc).first_column, (yyloc).last_column);
          YYABORT;
        }
#line 2906 "y.tab.c"
    break;

  case 80: /* Construct: OptExtern Module  */
#line 375 "xi-grammar.y"
        { (yyvsp[0].module)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].module); }
#line 2912 "y.tab.c"
    break;

  case 81: /* Construct: OptExtern Chare  */
#line 377 "xi-grammar.y"
        { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2918 "y.tab.c"
    break;

  case 82: /* Construct: OptExtern Group  */
#line 379 "xi-grammar.y"
        { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2924 "y.tab.c"
    break;

  case 83: /* Construct: OptExtern NodeGroup  */
#line 381 "xi-grammar.y"
        { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2930 "y.tab.c"
    break;

  case 84: /* Construct: OptExtern Array  */
#line 383 "xi-grammar.y"
        { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2936 "y.tab.c"
    break;

  case 85: /* Construct: OptExtern Template  */
#line 385 "xi-grammar.y"
        { (yyvsp[0].templat)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].templat); }
#line 2942 "y.tab.c"
    break;

  case 86: /* Construct: HashIFComment  */
#line 387 "xi-grammar.y"
        { (yyval.construct) = NULL; }
#line 2948 "y.tab.c"
    break;

  case 87: /* Construct: HashIFDefComment  */
#line 389 "xi-grammar.y"
        { (yyval.construct) = NULL; }
#line 2954 "y.tab.c"
    break;

  case 88: /* Construct: AccelBlock  */
#line 391 "xi-grammar.y"
        { (yyval.construct) = (yyvsp[0].accelBlock); }
#line 2960 "y.tab.c"
    break;

  case 89: /* Construct: error  */
#line 393 "xi-grammar.y"
        {
          ERROR("invalid construct",
                (yyloc).first_column, (yyloc).last_column);
          YYABORT;
        }
#line 2970 "y.tab.c"
    break;

  case 90: /* TParam: Type  */
#line 401 "xi-grammar.y"
                { (yyval.tparam) = new TParamType((yyvsp[0].type)); }
#line 2976 "y.tab.c"
    break;

  case 91: /* TParam: NUMBER  */
#line 403 "xi-grammar.y"
                { (yyval.tparam) = new TParamVal((yyvsp[0].strval)); }
#line 2982 "y.tab.c"
    break;

  case 92: /* TParam: LITERAL  */
#line 405 "xi-grammar.y"
                { (yyval.tparam) = new TParamVal((yyvsp[0].strval)); }
#line 2988 "y.tab.c"
    break;

  case 93: /* TParamList: TParam  */
#line 409 "xi-grammar.y"
                { (yyval.tparlist) = new TParamList((yyvsp[0].tparam)); }
#line 2994 "y.tab.c"
    break;

  case 94: /* TParamList: TParam ',' TParamList  */
#line 411 "xi-grammar.y"
                { (yyval.tparlist) = new TParamList((yyvsp[-2].tparam), (yyvsp[0].tparlist)); }
#line 3000 "y.tab.c"
    break;

  case 95: /* TParamEList: %empty  */
#line 415 "xi-grammar.y"
                { (yyval.tparlist) = new TParamList(0); }
#line 3006 "y.tab.c"
    break;

  case 96: /* TParamEList: TParamList  */
#line 417 "xi-grammar.y"
                { (yyval.tparlist) = (yyvsp[0].tparlist); }
#line 3012 "y.tab.c"
    break;

  case 97: /* OptTParams: %empty  */
#line 421 "xi-grammar.y"
                { (yyval.tparlist) = 0; }
#line 3018 "y.tab.c"
    break;

  case 98: /* OptTParams: '<' TParamEList '>'  */
#line 423 "xi-grammar.y"
                { (yyval.tparlist) = (yyvsp[-1].tparlist); }
#line 3024 "y.tab.c"
    break;

  case 99: /* BuiltinType: INT  */
#line 427 "xi-grammar.y"
                { (yyval.type) = new BuiltinType("int"); }
#line 3030 "y.tab.c"
    break;

  case 100: /* BuiltinType: LONG  */
#line 429 "xi-grammar.y"
                { (yyval.type) = new BuiltinType("long"); }
#line 3036 "y.tab.c"
    break;

  case 101: /* BuiltinType: LONG INT  */
#line 431 "xi-grammar.y"
                { (yyval.type) = new BuiltinType("long int"); }
#line 3042 "y.tab.c"
    break;

  case 102: /* BuiltinType: SHORT  */
#line 433 "xi-grammar.y"
                { (yyval.type) = new BuiltinType("short"); }
#line 3048 "y.tab.c"
    break;

  case 103: /* BuiltinType: SHORT INT  */
#line 435 "xi-grammar.y"
                { (yyval.type) = new BuiltinType("short int"); }
#line 3054 "y.tab.c"
    break;

  case 104: /* BuiltinType: CHAR  */
#line 437 "xi-grammar.y"
                { (yyval.type) = new BuiltinType("char"); }
#line 3060 "y.tab.c"
    break;

  case 105: /* BuiltinType: UNSIGNED INT  */
#line 439 "xi-grammar.y"
                { (yyval.type) = new BuiltinType("unsigned int"); }
#line 3066 "y.tab.c"
    break;

  case 106: /* BuiltinType: UNSIGNED LONG  */
#line 441 "xi-grammar.y"
                { (yyval.type) = new BuiltinType("unsigned long"); }
#line 3072 "y.tab.c"
    break;

  case 107: /* BuiltinType: UNSIGNED LONG INT  */
#line 443 "xi-grammar.y"
                { (yyval.type) = new BuiltinType("unsigned long int"); }
#line 3078 "y.tab.c"
    break;

  case 108: /* BuiltinType: UNSIGNED LONG LONG  */
#line 445 "xi-grammar.y"
                { (yyval.type) = new BuiltinType("unsigned long long"); }
#line 3084 "y.tab.c"
    break;

  case 109: /* BuiltinType: UNSIGNED LONG LONG INT  */
#line 447 "xi-grammar.y"
                { (yyval.type) = new BuiltinType("unsigned long long int"); }
#line 3090 "y.tab.c"
    break;

  case 110: /* BuiltinType: UNSIGNED SHORT  */
#line 449 "xi-grammar.y"
                { (yyval.type) = new BuiltinType("unsigned short"); }
#line 3096 "y.tab.c"
    break;

  case 111: /* BuiltinType: UNSIGNED SHORT INT  */
#line 451 "xi-grammar.y"
                { (yyval.type) = new BuiltinType("unsigned short int"); }
#line 3102 "y.tab.c"
    break;

  case 112: /* BuiltinType: UNSIGNED CHAR  */
#line 453 "xi-grammar.y"
                { (yyval.type) = new BuiltinType("unsigned char"); }
#line 3108 "y.tab.c"
    break;

  case 113: /* BuiltinType: LONG LONG  */
#line 455 "xi-grammar.y"
                { (yyval.type) = new BuiltinType("long long"); }
#line 3114 "y.tab.c"
    break;

  case 114: /* BuiltinType: LONG LONG INT  */
#line 457 "xi-grammar.y"
                { (yyval.type) = new BuiltinType("long long int"); }
#line 3120 "y.tab.c"
    break;

  case 115: /* BuiltinType: FLOAT  */
#line 459 "xi-grammar.y"
                { (yyval.type) = new BuiltinType("float"); }
#line 3126 "y.tab.c"
    break;

  case 116: /* BuiltinType: DOUBLE  */
#line 461 "xi-grammar.y"
                { (yyval.type) = new BuiltinType("double"); }
#line 3132 "y.tab.c"
    break;

  case 117: /* BuiltinType: LONG DOUBLE  */
#line 463 "xi-grammar.y"
                { (yyval.type) = new BuiltinType("long double"); }
#line 3138 "y.tab.c"
    break;

  case 118: /* BuiltinType: VOID  */
#line 465 "xi-grammar.y"
                { (yyval.type) = new BuiltinType("void"); }
#line 3144 "y.tab.c"
    break;

  case 119: /* NamedType: Name OptTParams  */
#line 468 "xi-grammar.y"
                                  { (yyval.ntype) = new NamedType((yyvsp[-1].strval),(yyvsp[0].tparlist)); }
#line 3150 "y.tab.c"
    break;

  case 120: /* QualNamedType: QualName OptTParams  */
#line 469 "xi-grammar.y"
                                      { 
                    const char* basename, *scope;
                    splitScopedName((yyvsp[-1].strval), &scope, &basename);
                    (yyval.ntype) = new NamedType(basename, (yyvsp[0].tparlist), scope);
                }
#line 3160 "y.tab.c"
    break;

  case 121: /* QualNamedType: TYPENAME QualName OptTParams  */
#line 475 "xi-grammar.y"
                {
			const char* basename, *scope;
			splitScopedName((yyvsp[-1].strval), &scope, &basename);
			(yyval.ntype) = new NamedType(basename, (yyvsp[0].tparlist), scope, true);
		}
#line 3170 "y.tab.c"
    break;

  case 122: /* SimpleType: BuiltinType  */
#line 483 "xi-grammar.y"
                { (yyval.type) = (yyvsp[0].type); }
#line 3176 "y.tab.c"
    break;

  case 123: /* SimpleType: QualNamedType  */
#line 485 "xi-grammar.y"
                { (yyval.type) = (yyvsp[0].ntype); }
#line 3182 "y.tab.c"
    break;

  case 124: /* OnePtrType: SimpleType '*'  */
#line 489 "xi-grammar.y"
                { (yyval.ptype) = new PtrType((yyvsp[-1].type)); }
#line 3188 "y.tab.c"
    break;

  case 125: /* PtrType: OnePtrType '*'  */
#line 493 "xi-grammar.y"
                { (yyvsp[-1].ptype)->indirect(); (yyval.ptype) = (yyvsp[-1].ptype); }
#line 3194 "y.tab.c"
    break;

  case 126: /* PtrType: PtrType '*'  */
#line 495 "xi-grammar.y"
                { (yyvsp[-1].ptype)->indirect(); (yyval.ptype) = (yyvsp[-1].ptype); }
#line 3200 "y.tab.c"
    break;

  case 127: /* FuncType: BaseType '(' '*' Name ')' '(' ParamList ')'  */
#line 499 "xi-grammar.y"
                { (yyval.ftype) = new FuncType((yyvsp[-7].type), (yyvsp[-4].strval), (yyvsp[-1].plist)); }
#line 3206 "y.tab.c"
    break;

  case 128: /* BaseType: SimpleType  */
#line 503 "xi-grammar.y"
                { (yyval.type) = (yyvsp[0].type); }
#line 3212 "y.tab.c"
    break;

  case 129: /* BaseType: OnePtrType  */
#line 505 "xi-grammar.y"
                { (yyval.type) = (yyvsp[0].ptype); }
#line 3218 "y.tab.c"
    break;

  case 130: /* BaseType: PtrType  */
#line 507 "xi-grammar.y"
                { (yyval.type) = (yyvsp[0].ptype); }
#line 3224 "y.tab.c"
    break;

  case 131: /* BaseType: FuncType  */
#line 509 "xi-grammar.y"
                { (yyval.type) = (yyvsp[0].ftype); }
#line 3230 "y.tab.c"
    break;

  case 132: /* BaseType: CONST BaseType  */
#line 511 "xi-grammar.y"
                { (yyval.type) = new ConstType((yyvsp[0].type)); }
#line 3236 "y.tab.c"
    break;

  case 133: /* BaseType: BaseType CONST  */
#line 513 "xi-grammar.y"
                { (yyval.type) = new ConstType((yyvsp[-1].type)); }
#line 3242 "y.tab.c"
    break;

  case 134: /* BaseDataType: SimpleType  */
#line 517 "xi-grammar.y"
                { (yyval.type) = (yyvsp[0].type); }
#line 3248 "y.tab.c"
    break;

  case 135: /* BaseDataType: OnePtrType  */
#line 519 "xi-grammar.y"
                { (yyval.type) = (yyvsp[0].ptype); }
#line 3254 "y.tab.c"
    break;

  case 136: /* BaseDataType: PtrType  */
#line 521 "xi-grammar.y"
                { (yyval.type) = (yyvsp[0].ptype); }
#line 3260 "y.tab.c"
    break;

  case 137: /* BaseDataType: CONST BaseDataType  */
#line 523 "xi-grammar.y"
                { (yyval.type) = new ConstType((yyvsp[0].type)); }
#line 3266 "y.tab.c"
    break;

  case 138: /* BaseDataType: BaseDataType CONST  */
#line 525 "xi-grammar.y"
                { (yyval.type) = new ConstType((yyvsp[-1].type)); }
#line 3272 "y.tab.c"
    break;

  case 139: /* RestrictedType: BaseDataType '&' '&' '.' '.' '.'  */
#line 529 "xi-grammar.y"
                { (yyval.type) = new EllipsisType(new RValueReferenceType((yyvsp[-5].type))); }
#line 3278 "y.tab.c"
    break;

  case 140: /* RestrictedType: BaseDataType '&' '.' '.' '.'  */
#line 531 "xi-grammar.y"
                { (yyval.type) = new EllipsisType(new ReferenceType((yyvsp[-4].type))); }
#line 3284 "y.tab.c"
    break;

  case 141: /* RestrictedType: BaseDataType '.' '.' '.'  */
#line 533 "xi-grammar.y"
                { (yyval.type) = new EllipsisType((yyvsp[-3].type)); }
#line 3290 "y.tab.c"
    break;

  case 142: /* RestrictedType: BaseDataType '&' '&'  */
#line 535 "xi-grammar.y"
                { (yyval.type) = new RValueReferenceType((yyvsp[-2].type)); }
#line 3296 "y.tab.c"
    break;

  case 143: /* RestrictedType: BaseDataType '&'  */
#line 537 "xi-grammar.y"
                { (yyval.type) = new ReferenceType((yyvsp[-1].type)); }
#line 3302 "y.tab.c"
    break;

  case 144: /* RestrictedType: BaseDataType  */
#line 539 "xi-grammar.y"
                { (yyval.type) = (yyvsp[0].type); }
#line 3308 "y.tab.c"
    break;

  case 145: /* Type: BaseType '&' '&' '.' '.' '.'  */
#line 543 "xi-grammar.y"
                { (yyval.type) = new EllipsisType(new RValueReferenceType((yyvsp[-5].type))); }
#line 3314 "y.tab.c"
    break;

  case 146: /* Type: BaseType '&' '.' '.' '.'  */
#line 545 "xi-grammar.y"
                { (yyval.type) = new EllipsisType(new ReferenceType((yyvsp[-4].type))); }
#line 3320 "y.tab.c"
    break;

  case 147: /* Type: BaseType '.' '.' '.'  */
#line 547 "xi-grammar.y"
                { (yyval.type) = new EllipsisType((yyvsp[-3].type)); }
#line 3326 "y.tab.c"
    break;

  case 148: /* Type: BaseType '&' '&'  */
#line 549 "xi-grammar.y"
                { (yyval.type) = new RValueReferenceType((yyvsp[-2].type)); }
#line 3332 "y.tab.c"
    break;

  case 149: /* Type: BaseType '&'  */
#line 551 "xi-grammar.y"
                { (yyval.type) = new ReferenceType((yyvsp[-1].type)); }
#line 3338 "y.tab.c"
    break;

  case 150: /* Type: BaseType  */
#line 553 "xi-grammar.y"
                { (yyval.type) = (yyvsp[0].type); }
#line 3344 "y.tab.c"
    break;

  case 151: /* ArrayDim: CCode  */
#line 557 "xi-grammar.y"
                { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 3350 "y.tab.c"
    break;

  case 152: /* Dim: SParamBracketStart ArrayDim SParamBracketEnd  */
#line 561 "xi-grammar.y"
                { (yyval.val) = (yyvsp[-1].val); }
#line 3356 "y.tab.c"
    break;

  case 153: /* DimList: %empty  */
#line 565 "xi-grammar.y"
                { (yyval.vallist) = 0; }
#line 3362 "y.tab.c"
    break;

  case 154: /* DimList: Dim DimList  */
#line 567 "xi-grammar.y"
                { (yyval.vallist) = new ValueList((yyvsp[-1].val), (yyvsp[0].vallist)); }
#line 3368 "y.tab.c"
    break;

  case 155: /* Readonly: READONLY Type QualName DimList  */
#line 571 "xi-grammar.y"
                { (yyval.readonly) = new Readonly(lineno, (yyvsp[-2].type), (yyvsp[-1].strval), (yyvsp[0].vallist)); }
#line 3374 "y.tab.c"
    break;

  case 156: /* ReadonlyMsg: READONLY MESSAGE SimpleType '*' QualName DimList  */
#line 575 "xi-grammar.y"
                { (yyval.readonly) = new Readonly(lineno, (yyvsp[-3].type), (yyvsp[-1].strval), (yyvsp[0].vallist), 1); }
#line 3380 "y.tab.c"
    break;

  case 157: /* OptVoid: %empty  */
#line 579 "xi-grammar.y"
                { (yyval.intval) = 0;}
#line 3386 "y.tab.c"
    break;

  case 158: /* OptVoid: VOID  */
#line 581 "xi-grammar.y"
                { (yyval.intval) = 0;}
#line 3392 "y.tab.c"
    break;

  case 159: /* MAttribs: %empty  */
#line 585 "xi-grammar.y"
                { (yyval.intval) = 0; }
#line 3398 "y.tab.c"
    break;

  case 160: /* MAttribs: '[' MAttribList ']'  */
#line 587 "xi-grammar.y"
                { 
		  /*
		  printf("Warning: Message attributes are being phased out.\n");
		  printf("Warning: Please remove them from interface files.\n");
		  */
		  (yyval.intval) = (yyvsp[-1].intval); 
		}
#line 3410 "y.tab.c"
    break;

  case 161: /* MAttribList: MAttrib  */
#line 597 "xi-grammar.y"
                { (yyval.intval) = (yyvsp[0].intval); }
#line 3416 "y.tab.c"
    break;

  case 162: /* MAttribList: MAttrib ',' MAttribList  */
#line 599 "xi-grammar.y"
                { (yyval.intval) = (yyvsp[-2].intval) | (yyvsp[0].intval); }
#line 3422 "y.tab.c"
    break;

  case 163: /* MAttrib: PACKED  */
#line 603 "xi-grammar.y"
                { (yyval.intval) = 0; }
#line 3428 "y.tab.c"
    break;

  case 164: /* MAttrib: VARSIZE  */
#line 605 "xi-grammar.y"
                { (yyval.intval) = 0; }
#line 3434 "y.tab.c"
    break;

  case 165: /* CAttribs: %empty  */
#line 609 "xi-grammar.y"
                { (yyval.cattr) = 0; }
#line 3440 "y.tab.c"
    break;

  case 166: /* CAttribs: '[' CAttribList ']'  */
#line 611 "xi-grammar.y"
                { (yyval.cattr) = (yyvsp[-1].cattr); }
#line 3446 "y.tab.c"
    break;

  case 167: /* CAttribList: CAttrib  */
#line 615 "xi-grammar.y"
                { (yyval.cattr) = (yyvsp[0].cattr); }
#line 3452 "y.tab.c"
    break;

  case 168: /* CAttribList: CAttrib ',' CAttribList  */
#line 617 "xi-grammar.y"
                { (yyval.cattr) = (yyvsp[-2].cattr) | (yyvsp[0].cattr); }
#line 3458 "y.tab.c"
    break;

  case 169: /* PythonOptions: %empty  */
#line 621 "xi-grammar.y"
                { python_doc = NULL; (yyval.intval) = 0; }
#line 3464 "y.tab.c"
    break;

  case 170: /* PythonOptions: LITERAL  */
#line 623 "xi-grammar.y"
                { python_doc = (yyvsp[0].strval); (yyval.intval) = 0; }
#line 3470 "y.tab.c"
    break;

  case 171: /* ArrayAttrib: PYTHON  */
#line 627 "xi-grammar.y"
                { (yyval.cattr) = Chare::CPYTHON; }
#line 3476 "y.tab.c"
    break;

  case 172: /* ArrayAttribs: %empty  */
#line 631 "xi-grammar.y"
                { (yyval.cattr) = 0; }
#line 3482 "y.tab.c"
    break;

  case 173: /* ArrayAttribs: '[' ArrayAttribList ']'  */
#line 633 "xi-grammar.y"
                { (yyval.cattr) = (yyvsp[-1].cattr); }
#line 3488 "y.tab.c"
    break;

  case 174: /* ArrayAttribList: ArrayAttrib  */
#line 637 "xi-grammar.y"
                { (yyval.cattr) = (yyvsp[0].cattr); }
#line 3494 "y.tab.c"
    break;

  case 175: /* ArrayAttribList: ArrayAttrib ',' ArrayAttribList  */
#line 639 "xi-grammar.y"
                { (yyval.cattr) = (yyvsp[-2].cattr) | (yyvsp[0].cattr); }
#line 3500 "y.tab.c"
    break;

  case 176: /* CAttrib: MIGRATABLE  */
#line 643 "xi-grammar.y"
                { (yyval.cattr) = Chare::CMIGRATABLE; }
#line 3506 "y.tab.c"
    break;

  case 177: /* CAttrib: PYTHON  */
#line 645 "xi-grammar.y"
                { (yyval.cattr) = Chare::CPYTHON; }
#line 3512 "y.tab.c"
    break;

  case 178: /* OptConditional: %empty  */
#line 649 "xi-grammar.y"
                { (yyval.intval) = 0; }
#line 3518 "y.tab.c"
    break;

  case 179: /* OptConditional: CONDITIONAL  */
#line 651 "xi-grammar.y"
                { (yyval.intval) = 1; }
#line 3524 "y.tab.c"
    break;

  case 180: /* MsgArray: %empty  */
#line 654 "xi-grammar.y"
                { (yyval.intval) = 0; }
#line 3530 "y.tab.c"
    break;

  case 181: /* MsgArray: '[' ']'  */
#line 656 "xi-grammar.y"
                { (yyval.intval) = 1; }
#line 3536 "y.tab.c"
    break;

  case 182: /* Var: OptConditional Type Name MsgArray OneOrMoreSemiColon  */
#line 659 "xi-grammar.y"
                { (yyval.mv) = new MsgVar((yyvsp[-3].type), (yyvsp[-2].strval), (yyvsp[-4].intval), (yyvsp[-1].intval)); }
#line 3542 "y.tab.c"
    break;

  case 183: /* VarList: Var  */
#line 663 "xi-grammar.y"
                { (yyval.mvlist) = new MsgVarList((yyvsp[0].mv)); }
#line 3548 "y.tab.c"
    break;

  case 184: /* VarList: Var VarList  */
#line 665 "xi-grammar.y"
                { (yyval.mvlist) = new MsgVarList((yyvsp[-1].mv), (yyvsp[0].mvlist)); }
#line 3554 "y.tab.c"
    break;

  case 185: /* Message: MESSAGE MAttribs NamedType  */
#line 669 "xi-grammar.y"
                { (yyval.message) = new Message(lineno, (yyvsp[0].ntype)); }
#line 3560 "y.tab.c"
    break;

  case 186: /* Message: MESSAGE MAttribs NamedType '{' '}'  */
#line 671 "xi-grammar.y"
                { (yyval.message) = new Message(lineno, (yyvsp[-2].ntype)); }
#line 3566 "y.tab.c"
    break;

  case 187: /* Message: MESSAGE MAttribs NamedType '{' VarList '}'  */
#line 673 "xi-grammar.y"
                { (yyval.message) = new Message(lineno, (yyvsp[-3].ntype), (yyvsp[-1].mvlist)); }
#line 3572 "y.tab.c"
    break;

  case 188: /* OptBaseList: %empty  */
#line 677 "xi-grammar.y"
                { (yyval.typelist) = 0; }
#line 3578 "y.tab.c"
    break;

  case 189: /* OptBaseList: ':' BaseList  */
#line 679 "xi-grammar.y"
                { (yyval.typelist) = (yyvsp[0].typelist); }
#line 3584 "y.tab.c"
    break;

  case 190: /* BaseList: QualNamedType  */
#line 683 "xi-grammar.y"
                { (yyval.typelist) = new TypeList((yyvsp[0].ntype)); }
#line 3590 "y.tab.c"
    break;

  case 191: /* BaseList: QualNamedType ',' BaseList  */
#line 685 "xi-grammar.y"
                { (yyval.typelist) = new TypeList((yyvsp[-2].ntype), (yyvsp[0].typelist)); }
#line 3596 "y.tab.c"
    break;

  case 192: /* Chare: CHARE CAttribs NamedType OptBaseList MemberEList  */
#line 689 "xi-grammar.y"
                { (yyval.chare) = new Chare(lineno, (yyvsp[-3].cattr)|Chare::CCHARE, (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3602 "y.tab.c"
    break;

  case 193: /* Chare: MAINCHARE CAttribs NamedType OptBaseList MemberEList  */
#line 691 "xi-grammar.y"
                { (yyval.chare) = new MainChare(lineno, (yyvsp[-3].cattr), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3608 "y.tab.c"
    break;

  case 194: /* Group: GROUP CAttribs NamedType OptBaseList MemberEList  */
#line 695 "xi-grammar.y"
                { (yyval.chare) = new Group(lineno, (yyvsp[-3].cattr), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3614 "y.tab.c"
    break;

  case 195: /* NodeGroup: NODEGROUP CAttribs NamedType OptBaseList MemberEList  */
#line 699 "xi-grammar.y"
                { (yyval.chare) = new NodeGroup(lineno, (yyvsp[-3].cattr), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3620 "y.tab.c"
    break;

  case 196: /* ArrayIndexType: '[' NUMBER Name ']'  */
#line 703 "xi-grammar.y"
                {/*Stupid special case for [1D] indices*/
			char *buf=new char[40];
			snprintf(buf,40,"%sD",(yyvsp[-2].strval));
			(yyval.ntype) = new NamedType(buf); 
		}
#line 3630 "y.tab.c"
    break;

  case 197: /* ArrayIndexType: '[' QualNamedType ']'  */
#line 709 "xi-grammar.y"
                { (yyval.ntype) = (yyvsp[-1].ntype); }
#line 3636 "y.tab.c"
    break;

  case 198: /* Array: ARRAY ArrayAttribs ArrayIndexType NamedType OptBaseList MemberEList  */
#line 713 "xi-grammar.y"
                {  (yyval.chare) = new Array(lineno, (yyvsp[-4].cattr), (yyvsp[-3].ntype), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3642 "y.tab.c"
    break;

  case 199: /* Array: ARRAY ArrayIndexType ArrayAttribs NamedType OptBaseList MemberEList  */
#line 715 "xi-grammar.y"
                {  (yyval.chare) = new Array(lineno, (yyvsp[-3].cattr), (yyvsp[-4].ntype), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3648 "y.tab.c"
    break;

  case 200: /* TChare: CHARE CAttribs Name OptBaseList MemberEList  */
#line 719 "xi-grammar.y"
                { (yyval.chare) = new Chare(lineno, (yyvsp[-3].cattr)|Chare::CCHARE, new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist));}
#line 3654 "y.tab.c"
    break;

  case 201: /* TChare: MAINCHARE CAttribs Name OptBaseList MemberEList  */
#line 721 "xi-grammar.y"
                { (yyval.chare) = new MainChare(lineno, (yyvsp[-3].cattr), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3660 "y.tab.c"
    break;

  case 202: /* TGroup: GROUP CAttribs Name OptBaseList MemberEList  */
#line 725 "xi-grammar.y"
                { (yyval.chare) = new Group(lineno, (yyvsp[-3].cattr), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3666 "y.tab.c"
    break;

  case 203: /* TNodeGroup: NODEGROUP CAttribs Name OptBaseList MemberEList  */
#line 729 "xi-grammar.y"
                { (yyval.chare) = new NodeGroup( lineno, (yyvsp[-3].cattr), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3672 "y.tab.c"
    break;

  case 204: /* TArray: ARRAY ArrayIndexType Name OptBaseList MemberEList  */
#line 733 "xi-grammar.y"
                { (yyval.chare) = new Array( lineno, 0, (yyvsp[-3].ntype), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3678 "y.tab.c"
    break;

  case 205: /* TMessage: MESSAGE MAttribs Name OneOrMoreSemiColon  */
#line 737 "xi-grammar.y"
                { (yyval.message) = new Message(lineno, new NamedType((yyvsp[-1].strval))); }
#line 3684 "y.tab.c"
    break;

  case 206: /* TMessage: MESSAGE MAttribs Name '{' VarList '}' OneOrMoreSemiColon  */
#line 739 "xi-grammar.y"
                { (yyval.message) = new Message(lineno, new NamedType((yyvsp[-4].strval)), (yyvsp[-2].mvlist)); }
#line 3690 "y.tab.c"
    break;

  case 207: /* OptTypeInit: %empty  */
#line 743 "xi-grammar.y"
                { (yyval.type) = 0; }
#line 3696 "y.tab.c"
    break;

  case 208: /* OptTypeInit: '=' Type  */
#line 745 "xi-grammar.y"
                { (yyval.type) = (yyvsp[0].type); }
#line 3702 "y.tab.c"
    break;

  case 209: /* OptNameInit: %empty  */
#line 749 "xi-grammar.y"
                { (yyval.strval) = 0; }
#line 3708 "y.tab.c"
    break;

  case 210: /* OptNameInit: '=' NUMBER  */
#line 751 "xi-grammar.y"
                { (yyval.strval) = (yyvsp[0].strval); }
#line 3714 "y.tab.c"
    break;

  case 211: /* OptNameInit: '=' LITERAL  */
#line 753 "xi-grammar.y"
                { (yyval.strval) = (yyvsp[0].strval); }
#line 3720 "y.tab.c"
    break;

  case 212: /* OptNameInit: '=' QualNamedType  */
#line 755 "xi-grammar.y"
                {
		  XStr typeStr;
		  (yyvsp[0].ntype)->print(typeStr);
		  char *tmp = strdup(typeStr.get_string());
		  (yyval.strval) = tmp;
		}
#line 3731 "y.tab.c"
    break;

  case 213: /* TVar: CLASS '.' '.' '.' Name OptTypeInit  */
#line 764 "xi-grammar.y"
                { (yyval.tvar) = new TTypeEllipsis(new NamedEllipsisType((yyvsp[-1].strval)), (yyvsp[0].type)); }
#line 3737 "y.tab.c"
    break;

  case 214: /* TVar: TYPENAME '.' '.' '.' IDENT OptTypeInit  */
#line 766 "xi-grammar.y"
                { (yyval.tvar) = new TTypeEllipsis(new NamedEllipsisType((yyvsp[-1].strval)), (yyvsp[0].type)); }
#line 3743 "y.tab.c"
    break;

  case 215: /* TVar: CLASS Name OptTypeInit  */
#line 768 "xi-grammar.y"
                { (yyval.tvar) = new TType(new NamedType((yyvsp[-1].strval)), (yyvsp[0].type)); }
#line 3749 "y.tab.c"
    break;

  case 216: /* TVar: TYPENAME IDENT OptTypeInit  */
#line 770 "xi-grammar.y"
                { (yyval.tvar) = new TType(new NamedType((yyvsp[-1].strval)), (yyvsp[0].type)); }
#line 3755 "y.tab.c"
    break;

  case 217: /* TVar: FuncType OptNameInit  */
#line 772 "xi-grammar.y"
                { (yyval.tvar) = new TFunc((yyvsp[-1].ftype), (yyvsp[0].strval)); }
#line 3761 "y.tab.c"
    break;

  case 218: /* TVar: Type Name OptNameInit  */
#line 774 "xi-grammar.y"
                { (yyval.tvar) = new TName((yyvsp[-2].type), (yyvsp[-1].strval), (yyvsp[0].strval)); }
#line 3767 "y.tab.c"
    break;

  case 219: /* TVarList: TVar  */
#line 778 "xi-grammar.y"
                { (yyval.tvarlist) = new TVarList((yyvsp[0].tvar)); }
#line 3773 "y.tab.c"
    break;

  case 220: /* TVarList: TVar ',' TVarList  */
#line 780 "xi-grammar.y"
                { (yyval.tvarlist) = new TVarList((yyvsp[-2].tvar), (yyvsp[0].tvarlist)); }
#line 3779 "y.tab.c"
    break;

  case 221: /* TemplateSpec: TEMPLATE '<' TVarList '>'  */
#line 784 "xi-grammar.y"
                { (yyval.tvarlist) = (yyvsp[-1].tvarlist); }
#line 3785 "y.tab.c"
    break;

  case 222: /* Template: TemplateSpec TChare  */
#line 788 "xi-grammar.y"
                { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3791 "y.tab.c"
    break;

  case 223: /* Template: TemplateSpec TGroup  */
#line 790 "xi-grammar.y"
                { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3797 "y.tab.c"
    break;

  case 224: /* Template: TemplateSpec TNodeGroup  */
#line 792 "xi-grammar.y"
                { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3803 "y.tab.c"
    break;

  case 225: /* Template: TemplateSpec TArray  */
#line 794 "xi-grammar.y"
                { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3809 "y.tab.c"
    break;

  case 226: /* Template: TemplateSpec TMessage  */
#line 796 "xi-grammar.y"
                { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].message)); (yyvsp[0].message)->setTemplate((yyval.templat)); }
#line 3815 "y.tab.c"
    break;

  case 227: /* MemberEList: OneOrMoreSemiColon  */
#line 800 "xi-grammar.y"
                { (yyval.mbrlist) = 0; }
#line 3821 "y.tab.c"
    break;

  case 228: /* MemberEList: '{' MemberList '}' OptSemiColon  */
#line 802 "xi-grammar.y"
                { (yyval.mbrlist) = (yyvsp[-2].mbrlist); }
#line 3827 "y.tab.c"
    break;

  case 229: /* MemberList: %empty  */
#line 806 "xi-grammar.y"
                { 
                  if (!connectEntries.empty()) {
                    (yyval.mbrlist) = new AstChildren<Member>(connectEntries);
		  } else {
		    (yyval.mbrlist) = 0; 
                  }
		}
#line 3839 "y.tab.c"
    break;

  case 230: /* MemberList: Member MemberList  */
#line 814 "xi-grammar.y"
                { (yyval.mbrlist) = new AstChildren<Member>(-1, (yyvsp[-1].member), (yyvsp[0].mbrlist)); }
#line 3845 "y.tab.c"
    break;

  case 231: /* NonEntryMember: Readonly  */
#line 818 "xi-grammar.y"
                { (yyval.member) = (yyvsp[0].readonly); }
#line 3851 "y.tab.c"
    break;

  case 232: /* NonEntryMember: ReadonlyMsg  */
#line 820 "xi-grammar.y"
                { (yyval.member) = (yyvsp[0].readonly); }
#line 3857 "y.tab.c"
    break;

  case 234: /* NonEntryMember: InitNode  */
#line 823 "xi-grammar.y"
                { (yyval.member) = (yyvsp[0].member); }
#line 3863 "y.tab.c"
    break;

  case 235: /* NonEntryMember: PUPABLE PUPableClass  */
#line 825 "xi-grammar.y"
                { (yyval.member) = (yyvsp[0].pupable); }
#line 3869 "y.tab.c"
    break;

  case 236: /* NonEntryMember: INCLUDE IncludeFile  */
#line 827 "xi-grammar.y"
                { (yyval.member) = (yyvsp[0].includeFile); }
#line 3875 "y.tab.c"
    break;

  case 237: /* NonEntryMember: CLASS Name  */
#line 829 "xi-grammar.y"
                { (yyval.member) = new ClassDeclaration(lineno,(yyvsp[0].strval)); }
#line 3881 "y.tab.c"
    break;

  case 238: /* InitNode: INITNODE OptVoid QualName  */
#line 833 "xi-grammar.y"
                { (yyval.member) = new InitCall(lineno, (yyvsp[0].strval), 1); }
#line 3887 "y.tab.c"
    break;

  case 239: /* InitNode: INITNODE OptVoid QualName '(' OptVoid ')'  */
#line 835 "xi-grammar.y"
                { (yyval.member) = new InitCall(lineno, (yyvsp[-3].strval), 1); }
#line 3893 "y.tab.c"
    break;

  case 240: /* InitNode: INITNODE OptVoid QualName '<' TParamList '>' '(' OptVoid ')'  */
#line 837 "xi-grammar.y"
                { (yyval.member) = new InitCall(lineno,
				    strdup((std::string((yyvsp[-6].strval)) + '<' +
					    ((yyvsp[-4].tparlist))->to_string() + '>').c_str()),
				    1);
		}
#line 3903 "y.tab.c"
    break;

  case 241: /* InitNode: INITCALL OptVoid QualName  */
#line 843 "xi-grammar.y"
                {
		  WARNING("deprecated use of initcall. Use initnode or initproc instead",
		          (yylsp[-2]).first_column, (yylsp[-2]).last_column, (yylsp[-2]).first_line);
		  (yyval.member) = new InitCall(lineno, (yyvsp[0].strval), 1);
		}
#line 3913 "y.tab.c"
    break;

  case 242: /* InitNode: INITCALL OptVoid QualName '(' OptVoid ')'  */
#line 849 "xi-grammar.y"
                {
		  WARNING("deprecated use of initcall. Use initnode or initproc instead",
		          (yylsp[-5]).first_column, (yylsp[-5]).last_column, (yylsp[-5]).first_line);
		  (yyval.member) = new InitCall(lineno, (yyvsp[-3].strval), 1);
		}
#line 3923 "y.tab.c"
    break;

  case 243: /* InitProc: INITPROC OptVoid QualName  */
#line 858 "xi-grammar.y"
                { (yyval.member) = new InitCall(lineno, (yyvsp[0].strval), 0); }
#line 3929 "y.tab.c"
    break;

  case 244: /* InitProc: INITPROC OptVoid QualName '(' OptVoid ')'  */
#line 860 "xi-grammar.y"
                { (yyval.member) = new InitCall(lineno, (yyvsp[-3].strval), 0); }
#line 3935 "y.tab.c"
    break;

  case 245: /* InitProc: INITPROC OptVoid QualName '<' TParamList '>' '(' OptVoid ')'  */
#line 862 "xi-grammar.y"
                { (yyval.member) = new InitCall(lineno,
				    strdup((std::string((yyvsp[-6].strval)) + '<' +
					    ((yyvsp[-4].tparlist))->to_string() + '>').c_str()),
				    0);
		}
#line 3945 "y.tab.c"
    break;

  case 246: /* InitProc: INITPROC '[' ACCEL ']' OptVoid QualName '(' OptVoid ')'  */
#line 868 "xi-grammar.y"
                {
                  InitCall* rtn = new InitCall(lineno, (yyvsp[-3].strval), 0);
                  rtn->setAccel();
                  (yyval.member) = rtn;
		}
#line 3955 "y.tab.c"
    break;

  case 247: /* PUPableClass: QualNamedType  */
#line 876 "xi-grammar.y"
                { (yyval.pupable) = new PUPableClass(lineno,(yyvsp[0].ntype),0); }
#line 3961 "y.tab.c"
    break;

  case 248: /* PUPableClass: QualNamedType ',' PUPableClass  */
#line 878 "xi-grammar.y"
                { (yyval.pupable) = new PUPableClass(lineno,(yyvsp[-2].ntype),(yyvsp[0].pupable)); }
#line 3967 "y.tab.c"
    break;

  case 249: /* IncludeFile: LITERAL  */
#line 881 "xi-grammar.y"
                { (yyval.includeFile) = new IncludeFile(lineno,(yyvsp[0].strval)); }
#line 3973 "y.tab.c"
    break;

  case 250: /* Member: MemberBody  */
#line 885 "xi-grammar.y"
                { (yyval.member) = (yyvsp[0].member); }
#line 3979 "y.tab.c"
    break;

  case 251: /* MemberBody: Entry  */
#line 889 "xi-grammar.y"
                { (yyval.member) = (yyvsp[0].entry); }
#line 3985 "y.tab.c"
    break;

  case 252: /* MemberBody: TemplateSpec Entry  */
#line 891 "xi-grammar.y"
                {
                  (yyvsp[0].entry)->tspec = (yyvsp[-1].tvarlist);
                  (yyval.member) = (yyvsp[0].entry);
                }
#line 3994 "y.tab.c"
    break;

  case 253: /* MemberBody: NonEntryMember OneOrMoreSemiColon  */
#line 896 "xi-grammar.y"
                { (yyval.member) = (yyvsp[-1].member); }
#line 4000 "y.tab.c"
    break;

  case 254: /* MemberBody: error  */
#line 898 "xi-grammar.y"
        {
          ERROR("invalid SDAG member",
                (yyloc).first_column, (yyloc).last_column);
          YYABORT;
        }
#line 4010 "y.tab.c"
    break;

  case 255: /* UnexpectedToken: ENTRY  */
#line 906 "xi-grammar.y"
                { (yyval.member) = 0; }
#line 4016 "y.tab.c"
    break;

  case 256: /* UnexpectedToken: '}'  */
#line 908 "xi-grammar.y"
                { (yyval.member) = 0; }
#line 4022 "y.tab.c"
    break;

  case 257: /* UnexpectedToken: INITCALL  */
#line 910 "xi-grammar.y"
                { (yyval.member) = 0; }
#line 4028 "y.tab.c"
    break;

  case 258: /* UnexpectedToken: INITNODE  */
#line 912 "xi-grammar.y"
                { (yyval.member) = 0; }
#line 4034 "y.tab.c"
    break;

  case 259: /* UnexpectedToken: INITPROC  */
#line 914 "xi-grammar.y"
                { (yyval.member) = 0; }
#line 4040 "y.tab.c"
    break;

  case 260: /* UnexpectedToken: CHARE  */
#line 916 "xi-grammar.y"
                { (yyval.member) = 0; }
#line 4046 "y.tab.c"
    break;

  case 261: /* UnexpectedToken: MAINCHARE  */
#line 918 "xi-grammar.y"
                { (yyval.member) = 0; }
#line 4052 "y.tab.c"
    break;

  case 262: /* UnexpectedToken: ARRAY  */
#line 920 "xi-grammar.y"
                { (yyval.member) = 0; }
#line 4058 "y.tab.c"
    break;

  case 263: /* UnexpectedToken: GROUP  */
#line 922 "xi-grammar.y"
                { (yyval.member) = 0; }
#line 4064 "y.tab.c"
    break;

  case 264: /* UnexpectedToken: NODEGROUP  */
#line 924 "xi-grammar.y"
                { (yyval.member) = 0; }
#line 4070 "y.tab.c"
    break;

  case 265: /* UnexpectedToken: READONLY  */
#line 926 "xi-grammar.y"
                { (yyval.member) = 0; }
#line 4076 "y.tab.c"
    break;

  case 266: /* Entry: ENTRY EAttribs EReturn Name EParameters OptStackSize OptSdagCode  */
#line 929 "xi-grammar.y"
                { 
                  (yyval.entry) = new Entry(lineno, (yyvsp[-5].attr), (yyvsp[-4].type), (yyvsp[-3].strval), (yyvsp[-2].plist), (yyvsp[-1].val), (yyvsp[0].sentry), (const char *) NULL, (yylsp[-6]).first_line, (yyloc).last_line);
		  if ((yyvsp[0].sentry) != 0) { 
		    (yyvsp[0].sentry)->con1 = new SdagConstruct(SIDENT, (yyvsp[-3].strval));
                    (yyvsp[0].sentry)->setEntry((yyval.entry));
                    (yyvsp[0].sentry)->param = new ParamList((yyvsp[-2].plist));
                  }
                  firstRdma = true;
                  firstDeviceRdma = true;
		}
#line 4091 "y.tab.c"
    break;

  case 267: /* Entry: ENTRY EAttribs Name EParameters OptSdagCode  */
#line 940 "xi-grammar.y"
                { 
                  Entry *e = new Entry(lineno, (yyvsp[-3].attr), 0, (yyvsp[-2].strval), (yyvsp[-1].plist),  0, (yyvsp[0].sentry), (const char *) NULL, (yylsp[-4]).first_line, (yyloc).last_line);
                  if ((yyvsp[0].sentry) != 0) {
		    (yyvsp[0].sentry)->con1 = new SdagConstruct(SIDENT, (yyvsp[-2].strval));
                    (yyvsp[0].sentry)->setEntry((yyval.entry));
                    (yyvsp[0].sentry)->param = new ParamList((yyvsp[-1].plist));
                  }
                  firstRdma = true;
                  firstDeviceRdma = true;
		  if (e->param && e->param->isCkMigMsgPtr()) {
		    WARNING("CkMigrateMsg chare constructor is taken for granted",
		            (yyloc).first_column, (yyloc).last_column);
		    (yyval.entry) = NULL;
		  } else {
		    (yyval.entry) = e;
		  }
		}
#line 4113 "y.tab.c"
    break;

  case 268: /* Entry: ENTRY '[' ACCEL ']' VOID Name EParameters AccelEParameters ParamBraceStart CCode ParamBraceEnd Name OneOrMoreSemiColon  */
#line 958 "xi-grammar.y"
                {
                  Attribute* attribs = new Attribute(SACCEL);
                  const char* name = (yyvsp[-7].strval);
                  ParamList* paramList = (yyvsp[-6].plist);
                  ParamList* accelParamList = (yyvsp[-5].plist);
		  XStr* codeBody = new XStr((yyvsp[-3].strval));
                  const char* callbackName = (yyvsp[-1].strval);

                  (yyval.entry) = new Entry(lineno, attribs, new BuiltinType("void"), name, paramList, 0, 0, 0 );
                  (yyval.entry)->setAccelParam(accelParamList);
                  (yyval.entry)->setAccelCodeBody(codeBody);
                  (yyval.entry)->setAccelCallbackName(new XStr(callbackName));
                  firstRdma = true;
                  firstDeviceRdma = true;
                }
#line 4133 "y.tab.c"
    break;

  case 269: /* AccelBlock: ACCELBLOCK ParamBraceStart CCode ParamBraceEnd OneOrMoreSemiColon  */
#line 976 "xi-grammar.y"
                { (yyval.accelBlock) = new AccelBlock(lineno, new XStr((yyvsp[-2].strval))); }
#line 4139 "y.tab.c"
    break;

  case 270: /* AccelBlock: ACCELBLOCK OneOrMoreSemiColon  */
#line 978 "xi-grammar.y"
                { (yyval.accelBlock) = new AccelBlock(lineno, NULL); }
#line 4145 "y.tab.c"
    break;

  case 271: /* EReturn: RestrictedType  */
#line 982 "xi-grammar.y"
                { (yyval.type) = (yyvsp[0].type); }
#line 4151 "y.tab.c"
    break;

  case 272: /* EAttribs: %empty  */
#line 986 "xi-grammar.y"
                { (yyval.attr) = 0; }
#line 4157 "y.tab.c"
    break;

  case 273: /* EAttribs: '[' EAttribList ']'  */
#line 988 "xi-grammar.y"
                { (yyval.attr) = (yyvsp[-1].attr); }
#line 4163 "y.tab.c"
    break;

  case 274: /* EAttribs: error  */
#line 990 "xi-grammar.y"
                { ERROR("invalid entry method attribute list",
		        (yyloc).first_column, (yyloc).last_column);
		  YYABORT;
		}
#line 4172 "y.tab.c"
    break;

  case 275: /* AttributeArg: Name ':' NUMBER  */
#line 996 "xi-grammar.y"
                      { (yyval.attrarg) = new Attribute::Argument((yyvsp[-2].strval), atoi((yyvsp[0].strval))); }
#line 4178 "y.tab.c"
    break;

  case 276: /* AttributeArgList: AttributeArg  */
#line 1000 "xi-grammar.y"
                                         { (yyval.attrarg) = (yyvsp[0].attrarg); }
#line 4184 "y.tab.c"
    break;

  case 277: /* AttributeArgList: AttributeArg ',' AttributeArgList  */
#line 1001 "xi-grammar.y"
                                         { (yyval.attrarg) = (yyvsp[-2].attrarg); (yyvsp[-2].attrarg)->next = (yyvsp[0].attrarg); }
#line 4190 "y.tab.c"
    break;

  case 278: /* EAttribList: EAttrib  */
#line 1005 "xi-grammar.y"
                                                        { (yyval.attr) = new Attribute((yyvsp[0].intval));           }
#line 4196 "y.tab.c"
    break;

  case 279: /* EAttribList: EAttrib '(' AttributeArgList ')'  */
#line 1006 "xi-grammar.y"
                                                        { (yyval.attr) = new Attribute((yyvsp[-3].intval), (yyvsp[-1].attrarg));       }
#line 4202 "y.tab.c"
    break;

  case 280: /* EAttribList: EAttrib ',' EAttribList  */
#line 1007 "xi-grammar.y"
                                                                    { (yyval.attr) = new Attribute((yyvsp[-2].intval), NULL, (yyvsp[0].attr)); }
#line 4208 "y.tab.c"
    break;

  case 281: /* EAttribList: EAttrib '(' AttributeArgList ')' ',' EAttribList  */
#line 1008 "xi-grammar.y"
                                                                    { (yyval.attr) = new Attribute((yyvsp[-5].intval), (yyvsp[-3].attrarg), (yyvsp[0].attr));   }
#line 4214 "y.tab.c"
    break;

  case 282: /* EAttrib: THREADED  */
#line 1012 "xi-grammar.y"
                { (yyval.intval) = STHREADED; }
#line 4220 "y.tab.c"
    break;

  case 283: /* EAttrib: WHENIDLE  */
#line 1014 "xi-grammar.y"
                { (yyval.intval) = SWHENIDLE; }
#line 4226 "y.tab.c"
    break;

  case 284: /* EAttrib: SYNC  */
#line 1016 "xi-grammar.y"
                { (yyval.intval) = SSYNC; }
#line 4232 "y.tab.c"
    break;

  case 285: /* EAttrib: IGET  */
#line 1018 "xi-grammar.y"
                { (yyval.intval) = SIGET; }
#line 4238 "y.tab.c"
    break;

  case 286: /* EAttrib: EXCLUSIVE  */
#line 1020 "xi-grammar.y"
                { (yyval.intval) = SLOCKED; }
#line 4244 "y.tab.c"
    break;

  case 287: /* EAttrib: CREATEHERE  */
#line 1022 "xi-grammar.y"
                { (yyval.intval) = SCREATEHERE; }
#line 4250 "y.tab.c"
    break;

  case 288: /* EAttrib: CREATEHOME  */
#line 1024 "xi-grammar.y"
                { (yyval.intval) = SCREATEHOME; }
#line 4256 "y.tab.c"
    break;

  case 289: /* EAttrib: NOKEEP  */
#line 1026 "xi-grammar.y"
                { (yyval.intval) = SNOKEEP; }
#line 4262 "y.tab.c"
    break;

  case 290: /* EAttrib: NOTRACE  */
#line 1028 "xi-grammar.y"
                { (yyval.intval) = SNOTRACE; }
#line 4268 "y.tab.c"
    break;

  case 291: /* EAttrib: APPWORK  */
#line 1030 "xi-grammar.y"
                { (yyval.intval) = SAPPWORK; }
#line 4274 "y.tab.c"
    break;

  case 292: /* EAttrib: IMMEDIATE  */
#line 1032 "xi-grammar.y"
                { (yyval.intval) = SIMMEDIATE; }
#line 4280 "y.tab.c"
    break;

  case 293: /* EAttrib: SKIPSCHED  */
#line 1034 "xi-grammar.y"
                { (yyval.intval) = SSKIPSCHED; }
#line 4286 "y.tab.c"
    break;

  case 294: /* EAttrib: INLINE  */
#line 1036 "xi-grammar.y"
                { (yyval.intval) = SINLINE; }
#line 4292 "y.tab.c"
    break;

  case 295: /* EAttrib: LOCAL  */
#line 1038 "xi-grammar.y"
                { (yyval.intval) = SLOCAL; }
#line 4298 "y.tab.c"
    break;

  case 296: /* EAttrib: PYTHON PythonOptions  */
#line 1040 "xi-grammar.y"
                { (yyval.intval) = SPYTHON; }
#line 4304 "y.tab.c"
    break;

  case 297: /* EAttrib: MEMCRITICAL  */
#line 1042 "xi-grammar.y"
                { (yyval.intval) = SMEM; }
#line 4310 "y.tab.c"
    break;

  case 298: /* EAttrib: REDUCTIONTARGET  */
#line 1044 "xi-grammar.y"
                { (yyval.intval) = SREDUCE; }
#line 4316 "y.tab.c"
    break;

  case 299: /* EAttrib: AGGREGATE  */
#line 1046 "xi-grammar.y"
                {
        (yyval.intval) = SAGGREGATE;
    }
#line 4324 "y.tab.c"
    break;

  case 300: /* EAttrib: error  */
#line 1050 "xi-grammar.y"
                {
		  ERROR("invalid entry method attribute",
		        (yylsp[0]).first_column, (yylsp[0]).last_column);
		  yyclearin;
		  yyerrok;
		}
#line 4335 "y.tab.c"
    break;

  case 301: /* DefaultParameter: LITERAL  */
#line 1059 "xi-grammar.y"
                { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 4341 "y.tab.c"
    break;

  case 302: /* DefaultParameter: NUMBER  */
#line 1061 "xi-grammar.y"
                { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 4347 "y.tab.c"
    break;

  case 303: /* DefaultParameter: QualName  */
#line 1063 "xi-grammar.y"
                { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 4353 "y.tab.c"
    break;

  case 304: /* CPROGRAM_List: %empty  */
#line 1067 "xi-grammar.y"
                { (yyval.strval) = ""; }
#line 4359 "y.tab.c"
    break;

  case 305: /* CPROGRAM_List: CPROGRAM  */
#line 1069 "xi-grammar.y"
                { (yyval.strval) = (yyvsp[0].strval); }
#line 4365 "y.tab.c"
    break;

  case 306: /* CPROGRAM_List: CPROGRAM ',' CPROGRAM_List  */
#line 1071 "xi-grammar.y"
                {  /*Returned only when in_bracket*/
			int len = strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3;
			char *tmp = new char[len];
			snprintf(tmp,len,"%s, %s", (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4376 "y.tab.c"
    break;

  case 307: /* CCode: %empty  */
#line 1080 "xi-grammar.y"
                { (yyval.strval) = ""; }
#line 4382 "y.tab.c"
    break;

  case 308: /* CCode: CPROGRAM  */
#line 1082 "xi-grammar.y"
                { (yyval.strval) = (yyvsp[0].strval); }
#line 4388 "y.tab.c"
    break;

  case 309: /* CCode: CPROGRAM '[' CCode ']' CCode  */
#line 1084 "xi-grammar.y"
                {  /*Returned only when in_bracket*/
			int len = strlen((yyvsp[-4].strval))+strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3;
			char *tmp = new char[len];
			snprintf(tmp, len, "%s[%s]%s", (yyvsp[-4].strval), (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4399 "y.tab.c"
    break;

  case 310: /* CCode: CPROGRAM '{' CCode '}' CCode  */
#line 1091 "xi-grammar.y"
                { /*Returned only when in_braces*/
			int len = strlen((yyvsp[-4].strval))+strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3;
			char *tmp = new char[len];
			snprintf(tmp, len, "%s{%s}%s", (yyvsp[-4].strval), (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4410 "y.tab.c"
    break;

  case 311: /* CCode: CPROGRAM '(' CPROGRAM_List ')' CCode  */
#line 1098 "xi-grammar.y"
                { /*Returned only when in_braces*/
			int len = strlen((yyvsp[-4].strval))+strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3;
			char *tmp = new char[len];
			snprintf(tmp, len, "%s(%s)%s", (yyvsp[-4].strval), (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4421 "y.tab.c"
    break;

  case 312: /* CCode: '(' CCode ')' CCode  */
#line 1105 "xi-grammar.y"
                { /*Returned only when in_braces*/
			int len = strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3;
			char *tmp = new char[len];
			snprintf(tmp, len, "(%s)%s", (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4432 "y.tab.c"
    break;

  case 313: /* ParamBracketStart: Type Name '['  */
#line 1114 "xi-grammar.y"
                {  /*Start grabbing CPROGRAM segments*/
			in_bracket=1;
			(yyval.pname) = new Parameter(lineno, (yyvsp[-2].type),(yyvsp[-1].strval));
		}
#line 4441 "y.tab.c"
    break;

  case 314: /* ParamBraceStart: '{'  */
#line 1121 "xi-grammar.y"
                { 
                   /*Start grabbing CPROGRAM segments*/
			in_braces=1;
			(yyval.intval) = 0;
		}
#line 4451 "y.tab.c"
    break;

  case 315: /* ParamBraceEnd: '}'  */
#line 1129 "xi-grammar.y"
                { 
			in_braces=0;
			(yyval.intval) = 0;
		}
#line 4460 "y.tab.c"
    break;

  case 316: /* Parameter: Type  */
#line 1136 "xi-grammar.y"
                { (yyval.pname) = new Parameter(lineno, (yyvsp[0].type));}
#line 4466 "y.tab.c"
    break;

  case 317: /* Parameter: Type Name OptConditional  */
#line 1138 "xi-grammar.y"
                { (yyval.pname) = new Parameter(lineno, (yyvsp[-2].type),(yyvsp[-1].strval)); (yyval.pname)->setConditional((yyvsp[0].intval)); }
#line 4472 "y.tab.c"
    break;

  case 318: /* Parameter: Type Name '=' DefaultParameter  */
#line 1140 "xi-grammar.y"
                { (yyval.pname) = new Parameter(lineno, (yyvsp[-3].type),(yyvsp[-2].strval),0,(yyvsp[0].val));}
#line 4478 "y.tab.c"
    break;

  case 319: /* Parameter: ParamBracketStart CCode ']'  */
#line 1142 "xi-grammar.y"
                { /*Stop grabbing CPROGRAM segments*/
			in_bracket=0;
			(yyval.pname) = new Parameter(lineno, (yyvsp[-2].pname)->getType(), (yyvsp[-2].pname)->getName() ,(yyvsp[-1].strval));
		}
#line 4487 "y.tab.c"
    break;

  case 320: /* Parameter: NOCOPY ParamBracketStart CCode ']'  */
#line 1147 "xi-grammar.y"
                { /*Stop grabbing CPROGRAM segments*/
			in_bracket=0;
			(yyval.pname) = new Parameter(lineno, (yyvsp[-2].pname)->getType(), (yyvsp[-2].pname)->getName() ,(yyvsp[-1].strval));
			(yyval.pname)->setRdma(CMK_ZC_P2P_SEND_MSG);
			if(firstRdma) {
				(yyval.pname)->setFirstRdma(true);
				firstRdma = false;
			}
		}
#line 4501 "y.tab.c"
    break;

  case 321: /* Parameter: NOCOPYPOST ParamBracketStart CCode ']'  */
#line 1157 "xi-grammar.y"
                { /*Stop grabbing CPROGRAM segments*/
			in_bracket=0;
			(yyval.pname) = new Parameter(lineno, (yyvsp[-2].pname)->getType(), (yyvsp[-2].pname)->getName() ,(yyvsp[-1].strval));
			(yyval.pname)->setRdma(CMK_ZC_P2P_RECV_MSG);
			if(firstRdma) {
				(yyval.pname)->setFirstRdma(true);
				firstRdma = false;
			}
		}
#line 4515 "y.tab.c"
    break;

  case 322: /* Parameter: NOCOPYDEVICE ParamBracketStart CCode ']'  */
#line 1167 "xi-grammar.y"
                { /*Stop grabbing CPROGRAM segments*/
			in_bracket=0;
			(yyval.pname) = new Parameter(lineno, (yyvsp[-2].pname)->getType(), (yyvsp[-2].pname)->getName() ,(yyvsp[-1].strval));
			(yyval.pname)->setRdma(CMK_ZC_DEVICE_MSG);
			if (firstDeviceRdma) {
				(yyval.pname)->setFirstDeviceRdma(true);
				firstDeviceRdma = false;
			}
		}
#line 4529 "y.tab.c"
    break;

  case 323: /* AccelBufferType: READONLY  */
#line 1178 "xi-grammar.y"
                            { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_READONLY; }
#line 4535 "y.tab.c"
    break;

  case 324: /* AccelBufferType: READWRITE  */
#line 1179 "xi-grammar.y"
                            { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_READWRITE; }
#line 4541 "y.tab.c"
    break;

  case 325: /* AccelBufferType: WRITEONLY  */
#line 1180 "xi-grammar.y"
                            { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_WRITEONLY; }
#line 4547 "y.tab.c"
    break;

  case 326: /* AccelInstName: Name  */
#line 1183 "xi-grammar.y"
                       { (yyval.xstrptr) = new XStr((yyvsp[0].strval)); }
#line 4553 "y.tab.c"
    break;

  case 327: /* AccelInstName: AccelInstName '-' '>' Name  */
#line 1184 "xi-grammar.y"
                                             { (yyval.xstrptr) = new XStr(""); *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "->" << (yyvsp[0].strval); }
#line 4559 "y.tab.c"
    break;

  case 328: /* AccelInstName: AccelInstName '.' Name  */
#line 1185 "xi-grammar.y"
                                         { (yyval.xstrptr) = new XStr(""); *((yyval.xstrptr)) << *((yyvsp[-2].xstrptr)) << "." << (yyvsp[0].strval); }
#line 4565 "y.tab.c"
    break;

  case 329: /* AccelInstName: AccelInstName '[' AccelInstName ']'  */
#line 1187 "xi-grammar.y"
                {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "[" << *((yyvsp[-1].xstrptr)) << "]";
                  delete (yyvsp[-3].xstrptr);
                  delete (yyvsp[-1].xstrptr);
                }
#line 4576 "y.tab.c"
    break;

  case 330: /* AccelInstName: AccelInstName '[' NUMBER ']'  */
#line 1194 "xi-grammar.y"
                {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "[" << (yyvsp[-1].strval) << "]";
                  delete (yyvsp[-3].xstrptr);
                }
#line 4586 "y.tab.c"
    break;

  case 331: /* AccelInstName: AccelInstName '(' AccelInstName ')'  */
#line 1200 "xi-grammar.y"
                {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "(" << *((yyvsp[-1].xstrptr)) << ")";
                  delete (yyvsp[-3].xstrptr);
                  delete (yyvsp[-1].xstrptr);
                }
#line 4597 "y.tab.c"
    break;

  case 332: /* AccelArrayParam: ParamBracketStart CCode ']'  */
#line 1209 "xi-grammar.y"
                {
                  in_bracket = 0;
                  (yyval.pname) = new Parameter(lineno, (yyvsp[-2].pname)->getType(), (yyvsp[-2].pname)->getName(), (yyvsp[-1].strval));
                }
#line 4606 "y.tab.c"
    break;

  case 333: /* AccelParameter: AccelBufferType ':' Type Name '<' AccelInstName '>'  */
#line 1216 "xi-grammar.y"
                {
                  (yyval.pname) = new Parameter(lineno, (yyvsp[-4].type), (yyvsp[-3].strval));
                  (yyval.pname)->setAccelInstName((yyvsp[-1].xstrptr));
                  (yyval.pname)->setAccelBufferType((yyvsp[-6].intval));
                }
#line 4616 "y.tab.c"
    break;

  case 334: /* AccelParameter: Type Name '<' AccelInstName '>'  */
#line 1222 "xi-grammar.y"
                {
		  (yyval.pname) = new Parameter(lineno, (yyvsp[-4].type), (yyvsp[-3].strval));
                  (yyval.pname)->setAccelInstName((yyvsp[-1].xstrptr));
                  (yyval.pname)->setAccelBufferType(Parameter::ACCEL_BUFFER_TYPE_READWRITE);
		}
#line 4626 "y.tab.c"
    break;

  case 335: /* AccelParameter: AccelBufferType ':' AccelArrayParam '<' AccelInstName '>'  */
#line 1228 "xi-grammar.y"
                {
                  (yyval.pname) = (yyvsp[-3].pname);
                  (yyval.pname)->setAccelInstName((yyvsp[-1].xstrptr));
                  (yyval.pname)->setAccelBufferType((yyvsp[-5].intval));
		}
#line 4636 "y.tab.c"
    break;

  case 336: /* ParamList: Parameter  */
#line 1236 "xi-grammar.y"
                { (yyval.plist) = new ParamList((yyvsp[0].pname)); }
#line 4642 "y.tab.c"
    break;

  case 337: /* ParamList: Parameter ',' ParamList  */
#line 1238 "xi-grammar.y"
                { (yyval.plist) = new ParamList((yyvsp[-2].pname),(yyvsp[0].plist)); }
#line 4648 "y.tab.c"
    break;

  case 338: /* AccelParamList: AccelParameter  */
#line 1242 "xi-grammar.y"
                { (yyval.plist) = new ParamList((yyvsp[0].pname)); }
#line 4654 "y.tab.c"
    break;

  case 339: /* AccelParamList: AccelParameter ',' AccelParamList  */
#line 1244 "xi-grammar.y"
                { (yyval.plist) = new ParamList((yyvsp[-2].pname),(yyvsp[0].plist)); }
#line 4660 "y.tab.c"
    break;

  case 340: /* EParameters: '(' ParamList ')'  */
#line 1248 "xi-grammar.y"
                { (yyval.plist) = (yyvsp[-1].plist); }
#line 4666 "y.tab.c"
    break;

  case 341: /* EParameters: '(' ')'  */
#line 1250 "xi-grammar.y"
                { (yyval.plist) = new ParamList(new Parameter(0, new BuiltinType("void"))); }
#line 4672 "y.tab.c"
    break;

  case 342: /* AccelEParameters: '[' AccelParamList ']'  */
#line 1254 "xi-grammar.y"
                  { (yyval.plist) = (yyvsp[-1].plist); }
#line 4678 "y.tab.c"
    break;

  case 343: /* AccelEParameters: '[' ']'  */
#line 1256 "xi-grammar.y"
                  { (yyval.plist) = 0; }
#line 4684 "y.tab.c"
    break;

  case 344: /* OptStackSize: %empty  */
#line 1260 "xi-grammar.y"
                { (yyval.val) = 0; }
#line 4690 "y.tab.c"
    break;

  case 345: /* OptStackSize: STACKSIZE '=' NUMBER  */
#line 1262 "xi-grammar.y"
                { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 4696 "y.tab.c"
    break;

  case 346: /* OptSdagCode: OneOrMoreSemiColon  */
#line 1266 "xi-grammar.y"
                { (yyval.sentry) = 0; }
#line 4702 "y.tab.c"
    break;

  case 347: /* OptSdagCode: SingleConstruct  */
#line 1268 "xi-grammar.y"
                { (yyval.sentry) = new SdagEntryConstruct((yyvsp[0].sc)); }
#line 4708 "y.tab.c"
    break;

  case 348: /* OptSdagCode: '{' Slist '}' OptSemiColon  */
#line 1270 "xi-grammar.y"
                { (yyval.sentry) = new SdagEntryConstruct((yyvsp[-2].slist)); }
#line 4714 "y.tab.c"
    break;

  case 349: /* Slist: SingleConstruct  */
#line 1274 "xi-grammar.y"
                { (yyval.slist) = new SListConstruct((yyvsp[0].sc)); }
#line 4720 "y.tab.c"
    break;

  case 350: /* Slist: SingleConstruct Slist  */
#line 1276 "xi-grammar.y"
                { (yyval.slist) = new SListConstruct((yyvsp[-1].sc), (yyvsp[0].slist));  }
#line 4726 "y.tab.c"
    break;

  case 351: /* Olist: SingleConstruct  */
#line 1280 "xi-grammar.y"
                { (yyval.olist) = new OListConstruct((yyvsp[0].sc)); }
#line 4732 "y.tab.c"
    break;

  case 352: /* Olist: SingleConstruct Slist  */
#line 1282 "xi-grammar.y"
                { (yyval.olist) = new OListConstruct((yyvsp[-1].sc), (yyvsp[0].slist)); }
#line 4738 "y.tab.c"
    break;

  case 353: /* CaseList: WhenConstruct  */
#line 1286 "xi-grammar.y"
                { (yyval.clist) = new CaseListConstruct((yyvsp[0].when)); }
#line 4744 "y.tab.c"
    break;

  case 354: /* CaseList: WhenConstruct CaseList  */
#line 1288 "xi-grammar.y"
                { (yyval.clist) = new CaseListConstruct((yyvsp[-1].when), (yyvsp[0].clist)); }
#line 4750 "y.tab.c"
    break;

  case 355: /* CaseList: NonWhenConstruct  */
#line 1290 "xi-grammar.y"
                {
		  ERROR("case blocks can only contain when clauses",
		        (yylsp[0]).first_column, (yylsp[0]).last_column);
		  (yyval.clist) = 0;
		}
#line 4760 "y.tab.c"
    break;

  case 356: /* OptTraceName: LITERAL  */
#line 1298 "xi-grammar.y"
                 { (yyval.strval) = (yyvsp[0].strval); }
#line 4766 "y.tab.c"
    break;

  case 357: /* OptTraceName: %empty  */
#line 1300 "xi-grammar.y"
                 { (yyval.strval) = 0; }
#line 4772 "y.tab.c"
    break;

  case 358: /* WhenConstruct: WHEN SEntryList '{' '}'  */
#line 1304 "xi-grammar.y"
                { (yyval.when) = new WhenConstruct((yyvsp[-2].entrylist), 0); }
#line 4778 "y.tab.c"
    break;

  case 359: /* WhenConstruct: WHEN SEntryList SingleConstruct  */
#line 1306 "xi-grammar.y"
                { (yyval.when) = new WhenConstruct((yyvsp[-1].entrylist), (yyvsp[0].sc)); }
#line 4784 "y.tab.c"
    break;

  case 360: /* WhenConstruct: WHEN SEntryList '{' Slist '}'  */
#line 1308 "xi-grammar.y"
                { (yyval.when) = new WhenConstruct((yyvsp[-3].entrylist), (yyvsp[-1].slist)); }
#line 4790 "y.tab.c"
    break;

  case 361: /* NonWhenConstruct: SERIAL OptTraceName ParamBraceStart CCode ParamBraceEnd OptSemiColon  */
#line 1312 "xi-grammar.y"
                { (yyval.when) = 0; }
#line 4796 "y.tab.c"
    break;

  case 362: /* NonWhenConstruct: OVERLAP '{' Olist '}'  */
#line 1314 "xi-grammar.y"
                { (yyval.when) = 0; }
#line 4802 "y.tab.c"
    break;

  case 363: /* NonWhenConstruct: CASE '{' CaseList '}'  */
#line 1316 "xi-grammar.y"
                { (yyval.when) = 0; }
#line 4808 "y.tab.c"
    break;

  case 364: /* NonWhenConstruct: FOR StartIntExpr CCode ';' CCode ';' CCode EndIntExpr '{' Slist '}'  */
#line 1318 "xi-grammar.y"
                { (yyval.when) = 0; }
#line 4814 "y.tab.c"
    break;

  case 365: /* NonWhenConstruct: FOR StartIntExpr CCode ';' CCode ';' CCode EndIntExpr SingleConstruct  */
#line 1320 "xi-grammar.y"
                { (yyval.when) = 0; }
#line 4820 "y.tab.c"
    break;

  case 366: /* NonWhenConstruct: FORALL '[' IDENT ']' StartIntExpr CCode ':' CCode ',' CCode EndIntExpr SingleConstruct  */
#line 1322 "xi-grammar.y"
                { (yyval.when) = 0; }
#line 4826 "y.tab.c"
    break;

  case 367: /* NonWhenConstruct: FORALL '[' IDENT ']' StartIntExpr CCode ':' CCode ',' CCode EndIntExpr '{' Slist '}'  */
#line 1324 "xi-grammar.y"
                { (yyval.when) = 0; }
#line 4832 "y.tab.c"
    break;

  case 368: /* NonWhenConstruct: IF StartIntExpr CCode EndIntExpr SingleConstruct HasElse  */
#line 1326 "xi-grammar.y"
                { (yyval.when) = 0; }
#line 4838 "y.tab.c"
    break;

  case 369: /* NonWhenConstruct: IF StartIntExpr CCode EndIntExpr '{' Slist '}' HasElse  */
#line 1328 "xi-grammar.y"
                { (yyval.when) = 0; }
#line 4844 "y.tab.c"
    break;

  case 370: /* NonWhenConstruct: WHILE StartIntExpr CCode EndIntExpr SingleConstruct  */
#line 1330 "xi-grammar.y"
                { (yyval.when) = 0; }
#line 4850 "y.tab.c"
    break;

  case 371: /* NonWhenConstruct: WHILE StartIntExpr CCode EndIntExpr '{' Slist '}'  */
#line 1332 "xi-grammar.y"
                { (yyval.when) = 0; }
#line 4856 "y.tab.c"
    break;

  case 372: /* NonWhenConstruct: ParamBraceStart CCode ParamBraceEnd OptSemiColon  */
#line 1334 "xi-grammar.y"
                { (yyval.when) = 0; }
#line 4862 "y.tab.c"
    break;

  case 373: /* SingleConstruct: SERIAL OptTraceName ParamBraceStart CCode ParamBraceEnd OptSemiColon  */
#line 1338 "xi-grammar.y"
                { (yyval.sc) = new SerialConstruct((yyvsp[-2].strval), (yyvsp[-4].strval), (yylsp[-3]).first_line); }
#line 4868 "y.tab.c"
    break;

  case 374: /* SingleConstruct: OVERLAP '{' Olist '}'  */
#line 1340 "xi-grammar.y"
                { (yyval.sc) = new OverlapConstruct((yyvsp[-1].olist)); }
#line 4874 "y.tab.c"
    break;

  case 375: /* SingleConstruct: WhenConstruct  */
#line 1342 "xi-grammar.y"
                { (yyval.sc) = (yyvsp[0].when); }
#line 4880 "y.tab.c"
    break;

  case 376: /* SingleConstruct: CASE '{' CaseList '}'  */
#line 1344 "xi-grammar.y"
                { (yyval.sc) = new CaseConstruct((yyvsp[-1].clist)); }
#line 4886 "y.tab.c"
    break;

  case 377: /* SingleConstruct: FOR StartIntExpr IntExpr ';' IntExpr ';' IntExpr EndIntExpr '{' Slist '}'  */
#line 1346 "xi-grammar.y"
                { (yyval.sc) = new ForConstruct((yyvsp[-8].intexpr), (yyvsp[-6].intexpr), (yyvsp[-4].intexpr), (yyvsp[-1].slist)); }
#line 4892 "y.tab.c"
    break;

  case 378: /* SingleConstruct: FOR StartIntExpr IntExpr ';' IntExpr ';' IntExpr EndIntExpr SingleConstruct  */
#line 1348 "xi-grammar.y"
                { (yyval.sc) = new ForConstruct((yyvsp[-6].intexpr), (yyvsp[-4].intexpr), (yyvsp[-2].intexpr), (yyvsp[0].sc)); }
#line 4898 "y.tab.c"
    break;

  case 379: /* SingleConstruct: FORALL '[' IDENT ']' StartIntExpr IntExpr ':' IntExpr ',' IntExpr EndIntExpr SingleConstruct  */
#line 1350 "xi-grammar.y"
                { (yyval.sc) = new ForallConstruct(new SdagConstruct(SIDENT, (yyvsp[-9].strval)), (yyvsp[-6].intexpr),
		             (yyvsp[-4].intexpr), (yyvsp[-2].intexpr), (yyvsp[0].sc)); }
#line 4905 "y.tab.c"
    break;

  case 380: /* SingleConstruct: FORALL '[' IDENT ']' StartIntExpr IntExpr ':' IntExpr ',' IntExpr EndIntExpr '{' Slist '}'  */
#line 1353 "xi-grammar.y"
                { (yyval.sc) = new ForallConstruct(new SdagConstruct(SIDENT, (yyvsp[-11].strval)), (yyvsp[-8].intexpr),
		             (yyvsp[-6].intexpr), (yyvsp[-4].intexpr), (yyvsp[-1].slist)); }
#line 4912 "y.tab.c"
    break;

  case 381: /* SingleConstruct: IF StartIntExpr IntExpr EndIntExpr SingleConstruct HasElse  */
#line 1356 "xi-grammar.y"
                { (yyval.sc) = new IfConstruct((yyvsp[-3].intexpr), (yyvsp[-1].sc), (yyvsp[0].sc)); }
#line 4918 "y.tab.c"
    break;

  case 382: /* SingleConstruct: IF StartIntExpr IntExpr EndIntExpr '{' Slist '}' HasElse  */
#line 1358 "xi-grammar.y"
                { (yyval.sc) = new IfConstruct((yyvsp[-5].intexpr), (yyvsp[-2].slist), (yyvsp[0].sc)); }
#line 4924 "y.tab.c"
    break;

  case 383: /* SingleConstruct: WHILE StartIntExpr IntExpr EndIntExpr SingleConstruct  */
#line 1360 "xi-grammar.y"
                { (yyval.sc) = new WhileConstruct((yyvsp[-2].intexpr), (yyvsp[0].sc)); }
#line 4930 "y.tab.c"
    break;

  case 384: /* SingleConstruct: WHILE StartIntExpr IntExpr EndIntExpr '{' Slist '}'  */
#line 1362 "xi-grammar.y"
                { (yyval.sc) = new WhileConstruct((yyvsp[-4].intexpr), (yyvsp[-1].slist)); }
#line 4936 "y.tab.c"
    break;

  case 385: /* SingleConstruct: ParamBraceStart CCode ParamBraceEnd OptSemiColon  */
#line 1364 "xi-grammar.y"
                { (yyval.sc) = new SerialConstruct((yyvsp[-2].strval), NULL, (yyloc).first_line); }
#line 4942 "y.tab.c"
    break;

  case 386: /* SingleConstruct: error  */
#line 1366 "xi-grammar.y"
                {
		  ERROR("unknown SDAG construct or malformed entry method declaration.\n"
		        "You may have forgotten to terminate a previous entry method declaration with a"
		        " semicolon or forgotten to mark a block of sequential SDAG code as 'serial'",
		        (yyloc).first_column, (yyloc).last_column);
		  YYABORT;
		}
#line 4954 "y.tab.c"
    break;

  case 387: /* HasElse: %empty  */
#line 1376 "xi-grammar.y"
                { (yyval.sc) = 0; }
#line 4960 "y.tab.c"
    break;

  case 388: /* HasElse: ELSE SingleConstruct  */
#line 1378 "xi-grammar.y"
                { (yyval.sc) = new ElseConstruct((yyvsp[0].sc)); }
#line 4966 "y.tab.c"
    break;

  case 389: /* HasElse: ELSE '{' Slist '}'  */
#line 1380 "xi-grammar.y"
                { (yyval.sc) = new ElseConstruct((yyvsp[-1].slist)); }
#line 4972 "y.tab.c"
    break;

  case 390: /* IntExpr: CCode  */
#line 1384 "xi-grammar.y"
                { (yyval.intexpr) = new IntExprConstruct((yyvsp[0].strval)); }
#line 4978 "y.tab.c"
    break;

  case 391: /* EndIntExpr: ')'  */
#line 1388 "xi-grammar.y"
                { in_int_expr = 0; (yyval.intval) = 0; }
#line 4984 "y.tab.c"
    break;

  case 392: /* StartIntExpr: '('  */
#line 1392 "xi-grammar.y"
                { in_int_expr = 1; (yyval.intval) = 0; }
#line 4990 "y.tab.c"
    break;

  case 393: /* SEntry: IDENT EParameters  */
#line 1396 "xi-grammar.y"
                {
		  (yyval.entry) = new Entry(lineno, NULL, 0, (yyvsp[-1].strval), (yyvsp[0].plist), 0, 0, 0, (yyloc).first_line, (yyloc).last_line);
		  firstRdma = true;
		  firstDeviceRdma = true;
		}
#line 5000 "y.tab.c"
    break;

  case 394: /* SEntry: IDENT SParamBracketStart CCode SParamBracketEnd EParameters  */
#line 1402 "xi-grammar.y"
                {
		  (yyval.entry) = new Entry(lineno, NULL, 0, (yyvsp[-4].strval), (yyvsp[0].plist), 0, 0, (yyvsp[-2].strval), (yyloc).first_line, (yyloc).last_line);
		  firstRdma = true;
		  firstDeviceRdma = true;
		}
#line 5010 "y.tab.c"
    break;

  case 395: /* SEntryList: SEntry  */
#line 1410 "xi-grammar.y"
                { (yyval.entrylist) = new EntryList((yyvsp[0].entry)); }
#line 5016 "y.tab.c"
    break;

  case 396: /* SEntryList: SEntry ',' SEntryList  */
#line 1412 "xi-grammar.y"
                { (yyval.entrylist) = new EntryList((yyvsp[-2].entry),(yyvsp[0].entrylist)); }
#line 5022 "y.tab.c"
    break;

  case 397: /* SParamBracketStart: '['  */
#line 1416 "xi-grammar.y"
                   { in_bracket=1; }
#line 5028 "y.tab.c"
    break;

  case 398: /* SParamBracketEnd: ']'  */
#line 1419 "xi-grammar.y"
                   { in_bracket=0; }
#line 5034 "y.tab.c"
    break;

  case 399: /* HashIFComment: HASHIF Name  */
#line 1423 "xi-grammar.y"
                { if (!macroDefined((yyvsp[0].strval), 1)) in_comment = 1; }
#line 5040 "y.tab.c"
    break;

  case 400: /* HashIFDefComment: HASHIFDEF Name  */
#line 1427 "xi-grammar.y"
                { if (!macroDefined((yyvsp[0].strval), 0)) in_comment = 1; }
#line 5046 "y.tab.c"
    break;


#line 5050 "y.tab.c"

      default: break;
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
  YY_SYMBOL_PRINT ("-> $$ =", YY_CAST (yysymbol_kind_t, yyr1[yyn]), &yyval, &yyloc);

  YYPOPSTACK (yylen);
  yylen = 0;

  *++yyvsp = yyval;
  *++yylsp = yyloc;

  /* Now 'shift' the result of the reduction.  Determine what state
     that goes to, based on the state we popped back to and the rule
     number reduced by.  */
  {
    const int yylhs = yyr1[yyn] - YYNTOKENS;
    const int yyi = yypgoto[yylhs] + *yyssp;
    yystate = (0 <= yyi && yyi <= YYLAST && yycheck[yyi] == *yyssp
               ? yytable[yyi]
               : yydefgoto[yylhs]);
  }

  goto yynewstate;


/*--------------------------------------.
| yyerrlab -- here on detecting error.  |
`--------------------------------------*/
yyerrlab:
  /* Make sure we have latest lookahead translation.  See comments at
     user semantic actions for why this is necessary.  */
  yytoken = yychar == YYEMPTY ? YYSYMBOL_YYEMPTY : YYTRANSLATE (yychar);
  /* If not already recovering from an error, report this error.  */
  if (!yyerrstatus)
    {
      ++yynerrs;
      yyerror (YY_("syntax error"));
    }

  yyerror_range[1] = yylloc;
  if (yyerrstatus == 3)
    {
      /* If just tried and failed to reuse lookahead token after an
         error, discard it.  */

      if (yychar <= YYEOF)
        {
          /* Return failure if at end of input.  */
          if (yychar == YYEOF)
            YYABORT;
        }
      else
        {
          yydestruct ("Error: discarding",
                      yytoken, &yylval, &yylloc);
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
  /* Pacify compilers when the user code never invokes YYERROR and the
     label yyerrorlab therefore never appears in user code.  */
  if (0)
    YYERROR;
  ++yynerrs;

  /* Do not reclaim the symbols of the rule whose action triggered
     this YYERROR.  */
  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);
  yystate = *yyssp;
  goto yyerrlab1;


/*-------------------------------------------------------------.
| yyerrlab1 -- common code for both syntax error and YYERROR.  |
`-------------------------------------------------------------*/
yyerrlab1:
  yyerrstatus = 3;      /* Each real token shifted decrements this.  */

  /* Pop stack until we find a state that shifts the error token.  */
  for (;;)
    {
      yyn = yypact[yystate];
      if (!yypact_value_is_default (yyn))
        {
          yyn += YYSYMBOL_YYerror;
          if (0 <= yyn && yyn <= YYLAST && yycheck[yyn] == YYSYMBOL_YYerror)
            {
              yyn = yytable[yyn];
              if (0 < yyn)
                break;
            }
        }

      /* Pop the current state because it cannot handle the error token.  */
      if (yyssp == yyss)
        YYABORT;

      yyerror_range[1] = *yylsp;
      yydestruct ("Error: popping",
                  YY_ACCESSING_SYMBOL (yystate), yyvsp, yylsp);
      YYPOPSTACK (1);
      yystate = *yyssp;
      YY_STACK_PRINT (yyss, yyssp);
    }

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END

  yyerror_range[2] = yylloc;
  ++yylsp;
  YYLLOC_DEFAULT (*yylsp, yyerror_range, 2);

  /* Shift the error token.  */
  YY_SYMBOL_PRINT ("Shifting", YY_ACCESSING_SYMBOL (yyn), yyvsp, yylsp);

  yystate = yyn;
  goto yynewstate;


/*-------------------------------------.
| yyacceptlab -- YYACCEPT comes here.  |
`-------------------------------------*/
yyacceptlab:
  yyresult = 0;
  goto yyreturnlab;


/*-----------------------------------.
| yyabortlab -- YYABORT comes here.  |
`-----------------------------------*/
yyabortlab:
  yyresult = 1;
  goto yyreturnlab;


/*-----------------------------------------------------------.
| yyexhaustedlab -- YYNOMEM (memory exhaustion) comes here.  |
`-----------------------------------------------------------*/
yyexhaustedlab:
  yyerror (YY_("memory exhausted"));
  yyresult = 2;
  goto yyreturnlab;


/*----------------------------------------------------------.
| yyreturnlab -- parsing is finished, clean up and return.  |
`----------------------------------------------------------*/
yyreturnlab:
  if (yychar != YYEMPTY)
    {
      /* Make sure we have latest lookahead translation.  See comments at
         user semantic actions for why this is necessary.  */
      yytoken = YYTRANSLATE (yychar);
      yydestruct ("Cleanup: discarding lookahead",
                  yytoken, &yylval, &yylloc);
    }
  /* Do not reclaim the symbols of the rule whose action triggered
     this YYABORT or YYACCEPT.  */
  YYPOPSTACK (yylen);
  YY_STACK_PRINT (yyss, yyssp);
  while (yyssp != yyss)
    {
      yydestruct ("Cleanup: popping",
                  YY_ACCESSING_SYMBOL (+*yyssp), yyvsp, yylsp);
      YYPOPSTACK (1);
    }
#ifndef yyoverflow
  if (yyss != yyssa)
    YYSTACK_FREE (yyss);
#endif

  return yyresult;
}

#line 1430 "xi-grammar.y"


void yyerror(const char *s) 
{
	fprintf(stderr, "[PARSE-ERROR] Unexpected/missing token at line %d. Current token being parsed: '%s'.\n", lineno, yytext);
}

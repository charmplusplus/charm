/* A Bison parser, made by GNU Bison 3.8.2.  */

/* Bison interface for Yacc-like parsers in C

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

/* DO NOT RELY ON FEATURES THAT ARE NOT DOCUMENTED in the manual,
   especially those whose name start with YY_ or yy_.  They are
   private implementation details that can be changed or removed.  */

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

#line 269 "y.tab.h"

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

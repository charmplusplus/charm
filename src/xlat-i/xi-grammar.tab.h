/* A Bison parser, made by GNU Bison 3.0.4.  */

/* Bison interface for Yacc-like parsers in C

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

#ifndef YY_YY_Y_TAB_H_INCLUDED
# define YY_YY_Y_TAB_H_INCLUDED
/* Debug traces.  */
#ifndef YYDEBUG
# define YYDEBUG 0
#endif
#if YYDEBUG
extern int yydebug;
#endif

/* Token type.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
  enum yytokentype
  {
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
    WHENIDLE = 278,
    SYNC = 279,
    IGET = 280,
    EXCLUSIVE = 281,
    IMMEDIATE = 282,
    SKIPSCHED = 283,
    INLINE = 284,
    VIRTUAL = 285,
    MIGRATABLE = 286,
    AGGREGATE = 287,
    CREATEHERE = 288,
    CREATEHOME = 289,
    NOKEEP = 290,
    NOTRACE = 291,
    APPWORK = 292,
    VOID = 293,
    CONST = 294,
    NOCOPY = 295,
    NOCOPYPOST = 296,
    NOCOPYDEVICE = 297,
    PACKED = 298,
    VARSIZE = 299,
    ENTRY = 300,
    FOR = 301,
    FORALL = 302,
    WHILE = 303,
    WHEN = 304,
    OVERLAP = 305,
    SERIAL = 306,
    IF = 307,
    ELSE = 308,
    PYTHON = 309,
    LOCAL = 310,
    NAMESPACE = 311,
    USING = 312,
    IDENT = 313,
    NUMBER = 314,
    LITERAL = 315,
    CPROGRAM = 316,
    HASHIF = 317,
    HASHIFDEF = 318,
    INT = 319,
    LONG = 320,
    SHORT = 321,
    CHAR = 322,
    FLOAT = 323,
    DOUBLE = 324,
    UNSIGNED = 325,
    ACCEL = 326,
    READWRITE = 327,
    WRITEONLY = 328,
    ACCELBLOCK = 329,
    MEMCRITICAL = 330,
    REDUCTIONTARGET = 331,
    CASE = 332,
    TYPENAME = 333
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
#line 54 "xi-grammar.y" /* yacc.c:1909  */

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

#line 256 "y.tab.h" /* yacc.c:1909  */
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

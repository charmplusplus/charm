/* A Bison parser, made by GNU Bison 1.875.  */

/* Skeleton parser for Yacc-like parsing with Bison,
   Copyright (C) 1984, 1989, 1990, 2000, 2001, 2002 Free Software Foundation, Inc.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2, or (at your option)
   any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 59 Temple Place - Suite 330,
   Boston, MA 02111-1307, USA.  */

/* As a special exception, when this file is copied by Bison into a
   Bison output file, you may use that output file without restriction.
   This special exception was added by the Free Software Foundation
   in version 1.24 of Bison.  */

/* Tokens.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
   /* Put the tokens into the symbol table, so that GDB and other debuggers
      know about them.  */
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
     CLASS = 272,
     INCLUDE = 273,
     STACKSIZE = 274,
     THREADED = 275,
     TEMPLATE = 276,
     SYNC = 277,
     EXCLUSIVE = 278,
     IMMEDIATE = 279,
     SKIPSCHED = 280,
     VIRTUAL = 281,
     MIGRATABLE = 282,
     CREATEHERE = 283,
     CREATEHOME = 284,
     NOKEEP = 285,
     NOTRACE = 286,
     VOID = 287,
     CONST = 288,
     PACKED = 289,
     VARSIZE = 290,
     ENTRY = 291,
     FOR = 292,
     FORALL = 293,
     WHILE = 294,
     WHEN = 295,
     OVERLAP = 296,
     ATOMIC = 297,
     FORWARD = 298,
     IF = 299,
     ELSE = 300,
     CONNECT = 301,
     PUBLISHES = 302,
     IDENT = 303,
     NUMBER = 304,
     LITERAL = 305,
     CPROGRAM = 306,
     HASHIF = 307,
     HASHIFDEF = 308,
     INT = 309,
     LONG = 310,
     SHORT = 311,
     CHAR = 312,
     FLOAT = 313,
     DOUBLE = 314,
     UNSIGNED = 315
   };
#endif
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
#define CLASS 272
#define INCLUDE 273
#define STACKSIZE 274
#define THREADED 275
#define TEMPLATE 276
#define SYNC 277
#define EXCLUSIVE 278
#define IMMEDIATE 279
#define SKIPSCHED 280
#define VIRTUAL 281
#define MIGRATABLE 282
#define CREATEHERE 283
#define CREATEHOME 284
#define NOKEEP 285
#define NOTRACE 286
#define VOID 287
#define CONST 288
#define PACKED 289
#define VARSIZE 290
#define ENTRY 291
#define FOR 292
#define FORALL 293
#define WHILE 294
#define WHEN 295
#define OVERLAP 296
#define ATOMIC 297
#define FORWARD 298
#define IF 299
#define ELSE 300
#define CONNECT 301
#define PUBLISHES 302
#define IDENT 303
#define NUMBER 304
#define LITERAL 305
#define CPROGRAM 306
#define HASHIF 307
#define HASHIFDEF 308
#define INT 309
#define LONG 310
#define SHORT 311
#define CHAR 312
#define FLOAT 313
#define DOUBLE 314
#define UNSIGNED 315




#if ! defined (YYSTYPE) && ! defined (YYSTYPE_IS_DECLARED)
#line 16 "xi-grammar.y"
typedef union YYSTYPE {
  ModuleList *modlist;
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
  MemberList *mbrlist;
  Member *member;
  TVar *tvar;
  TVarList *tvarlist;
  Value *val;
  ValueList *vallist;
  MsgVar *mv;
  MsgVarList *mvlist;
  PUPableClass *pupable;
  IncludeFile *includeFile;
  char *strval;
  int intval;
  Chare::attrib_t cattr;
  SdagConstruct *sc;
} YYSTYPE;
/* Line 1249 of yacc.c.  */
#line 192 "y.tab.h"
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif

extern YYSTYPE yylval;




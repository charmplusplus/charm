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
     VIRTUAL = 280,
     MIGRATABLE = 281,
     CREATEHERE = 282,
     CREATEHOME = 283,
     NOKEEP = 284,
     VOID = 285,
     CONST = 286,
     PACKED = 287,
     VARSIZE = 288,
     ENTRY = 289,
     FOR = 290,
     FORALL = 291,
     WHILE = 292,
     WHEN = 293,
     OVERLAP = 294,
     ATOMIC = 295,
     FORWARD = 296,
     IF = 297,
     ELSE = 298,
     CONNECT = 299,
     PUBLISHES = 300,
     IDENT = 301,
     NUMBER = 302,
     LITERAL = 303,
     CPROGRAM = 304,
     HASHIF = 305,
     HASHIFDEF = 306,
     INT = 307,
     LONG = 308,
     SHORT = 309,
     CHAR = 310,
     FLOAT = 311,
     DOUBLE = 312,
     UNSIGNED = 313
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
#define VIRTUAL 280
#define MIGRATABLE 281
#define CREATEHERE 282
#define CREATEHOME 283
#define NOKEEP 284
#define VOID 285
#define CONST 286
#define PACKED 287
#define VARSIZE 288
#define ENTRY 289
#define FOR 290
#define FORALL 291
#define WHILE 292
#define WHEN 293
#define OVERLAP 294
#define ATOMIC 295
#define FORWARD 296
#define IF 297
#define ELSE 298
#define CONNECT 299
#define PUBLISHES 300
#define IDENT 301
#define NUMBER 302
#define LITERAL 303
#define CPROGRAM 304
#define HASHIF 305
#define HASHIFDEF 306
#define INT 307
#define LONG 308
#define SHORT 309
#define CHAR 310
#define FLOAT 311
#define DOUBLE 312
#define UNSIGNED 313




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
#line 188 "y.tab.h"
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif

extern YYSTYPE yylval;




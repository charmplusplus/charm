#ifndef BISON_Y_TAB_H
# define BISON_Y_TAB_H

#ifndef YYSTYPE
typedef union {
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
} yystype;
# define YYSTYPE yystype
# define YYSTYPE_IS_TRIVIAL 1
#endif
# define	MODULE	257
# define	MAINMODULE	258
# define	EXTERN	259
# define	READONLY	260
# define	INITCALL	261
# define	INITNODE	262
# define	INITPROC	263
# define	PUPABLE	264
# define	CHARE	265
# define	MAINCHARE	266
# define	GROUP	267
# define	NODEGROUP	268
# define	ARRAY	269
# define	MESSAGE	270
# define	CLASS	271
# define	INCLUDE	272
# define	STACKSIZE	273
# define	THREADED	274
# define	TEMPLATE	275
# define	SYNC	276
# define	EXCLUSIVE	277
# define	IMMEDIATE	278
# define	VIRTUAL	279
# define	MIGRATABLE	280
# define	CREATEHERE	281
# define	CREATEHOME	282
# define	NOKEEP	283
# define	VOID	284
# define	CONST	285
# define	PACKED	286
# define	VARSIZE	287
# define	ENTRY	288
# define	FOR	289
# define	FORALL	290
# define	WHILE	291
# define	WHEN	292
# define	OVERLAP	293
# define	ATOMIC	294
# define	FORWARD	295
# define	IF	296
# define	ELSE	297
# define	CONNECT	298
# define	PUBLISHES	299
# define	IDENT	300
# define	NUMBER	301
# define	LITERAL	302
# define	CPROGRAM	303
# define	INT	304
# define	LONG	305
# define	SHORT	306
# define	CHAR	307
# define	FLOAT	308
# define	DOUBLE	309
# define	UNSIGNED	310


extern YYSTYPE yylval;

#endif /* not BISON_Y_TAB_H */

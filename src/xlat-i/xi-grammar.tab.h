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
  char *strval;
  int intval;
  Chare::attrib_t cattr;
  SdagConstruct *sc;
} YYSTYPE;
#define	MODULE	257
#define	MAINMODULE	258
#define	EXTERN	259
#define	READONLY	260
#define	INITCALL	261
#define	INITNODE	262
#define	INITPROC	263
#define	PUPABLE	264
#define	CHARE	265
#define	MAINCHARE	266
#define	GROUP	267
#define	NODEGROUP	268
#define	ARRAY	269
#define	MESSAGE	270
#define	CLASS	271
#define	STACKSIZE	272
#define	THREADED	273
#define	TEMPLATE	274
#define	SYNC	275
#define	EXCLUSIVE	276
#define	IMMEDIATE	277
#define	VIRTUAL	278
#define	MIGRATABLE	279
#define	CREATEHERE	280
#define	CREATEHOME	281
#define	NOKEEP	282
#define	VOID	283
#define	CONST	284
#define	PACKED	285
#define	VARSIZE	286
#define	ENTRY	287
#define	FOR	288
#define	FORALL	289
#define	WHILE	290
#define	WHEN	291
#define	OVERLAP	292
#define	ATOMIC	293
#define	FORWARD	294
#define	IF	295
#define	ELSE	296
#define	CONNECT	297
#define	PUBLISHES	298
#define	IDENT	299
#define	NUMBER	300
#define	LITERAL	301
#define	CPROGRAM	302
#define	INT	303
#define	LONG	304
#define	SHORT	305
#define	CHAR	306
#define	FLOAT	307
#define	DOUBLE	308
#define	UNSIGNED	309


extern YYSTYPE yylval;

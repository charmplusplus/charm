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
#define	PUPABLE	262
#define	CHARE	263
#define	MAINCHARE	264
#define	GROUP	265
#define	NODEGROUP	266
#define	ARRAY	267
#define	MESSAGE	268
#define	CLASS	269
#define	STACKSIZE	270
#define	THREADED	271
#define	TEMPLATE	272
#define	SYNC	273
#define	EXCLUSIVE	274
#define	IMMEDIATE	275
#define	VIRTUAL	276
#define	MIGRATABLE	277
#define	CREATEHERE	278
#define	CREATEHOME	279
#define	NOKEEP	280
#define	VOID	281
#define	CONST	282
#define	PACKED	283
#define	VARSIZE	284
#define	ENTRY	285
#define	FOR	286
#define	FORALL	287
#define	WHILE	288
#define	WHEN	289
#define	OVERLAP	290
#define	ATOMIC	291
#define	FORWARD	292
#define	IF	293
#define	ELSE	294
#define	IDENT	295
#define	NUMBER	296
#define	LITERAL	297
#define	CPROGRAM	298
#define	INT	299
#define	LONG	300
#define	SHORT	301
#define	CHAR	302
#define	FLOAT	303
#define	DOUBLE	304
#define	UNSIGNED	305


extern YYSTYPE yylval;

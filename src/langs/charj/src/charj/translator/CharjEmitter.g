/**
 * ANTLR (v3) Tree Parser for the Charj Language
 */

tree grammar CharjEmitter;

options {
    backtrack = true; 
    memoize = true;
    tokenVocab = Charj;
    ASTLabelType = CharjAST;
    output = template;
}


@header {
package charj.translator;
}


@members {
    SymbolTable symtab = null;

    PackageScope currentPackage = null;
    ClassSymbol currentClass = null;
    MethodSymbol currentMethod = null;
    LocalScope currentLocalScope = null;

    Translator translator_;
    OutputMode mode_;

    private boolean emitCC() { return mode_ == OutputMode.cc; }
    private boolean emitCI() { return mode_ == OutputMode.ci; }
    private boolean emitH() { return mode_ == OutputMode.h; }
    private boolean debug() { return translator_.debug(); }
    private String basename() { return translator_.basename(); }

    /**
     *  Override ANTLR's token mismatch behavior so we throw exceptions early.
     */
    protected void mismatch(IntStream input, int ttype, BitSet follow)
        throws RecognitionException {
        throw new MismatchedTokenException(ttype, input);
    }

    /**
     *  Override ANTLR's set mismatch behavior so we throw exceptions early.
     */
    public Object recoverFromMismatchedSet(IntStream input, RecognitionException e, BitSet follow)
        throws RecognitionException {
        throw e;
    }

    /**
     *  Test a list of CharjAST nodes to see if any of them has the given token
     *  type.
     */
    public boolean listContainsToken(List<CharjAST> list, int tokenType) {
        if (list == null) return false;
        for (CharjAST node : list) {
            if (node.token.getType() == tokenType) {
                return true;
            }
        }
        return false;
    }
}


// Replace default ANTLR generated catch clauses with this action, allowing early failure.
@rulecatch {
    catch (RecognitionException re) {
        reportError(re);
        throw re;
    }
}


// Starting point for parsing a Charj file.
charjSource[SymbolTable symtab, OutputMode m]
@init {
    this.symtab = symtab;
    this.translator_ = symtab.translator;
    this.mode_ = m;
}
    :   ^(CHARJ_SOURCE (p=packageDeclaration)? 
        ((i+=importDeclaration)
        |(r+=readonlyDeclaration)
        |externDeclaration
        |(t+=typeDeclaration))*)
        -> {emitCC()}? charjSource_cc(basename={basename()}, pd={$p.names}, imports={$i}, types={$t}, ros={$r}, debug={debug()})
        -> {emitCI()}? charjSource_ci(basename={basename()}, pd={$p.names}, imports={$i}, types={$t}, ros={$r}, debug={debug()})
        -> {emitH()}? charjSource_h(basename={basename()}, pd={$p.names}, imports={$i}, types={$t}, ros={$r}, debug={debug()})
        ->
    ;

topLevelDeclaration
    :   importDeclaration -> {$importDeclaration.st;}
    |   typeDeclaration -> {$typeDeclaration.st;}
    ;

packageDeclaration returns [List names]
    :   ^('package' (ids+=IDENT)+)
        {
            $names = $ids;
        }
        ->
    ;

readonlyDeclaration
    :   ^(READONLY lvd=localVariableDeclaration)
        -> {emitCI()}? template(bn={basename()}, v={$lvd.st}) "readonly <v>"
        -> {emitH()}? template(v={$lvd.st}) "extern <v>"
        -> {emitCC()}? {$lvd.st;}
        ->
    ;

externDeclaration
    :   ^(EXTERN qualifiedIdentifier)
        ->
    ;
    
importDeclaration
@init {
    String importID = null;
}
    :   ^('import' qualifiedIdentifier ds='.*'?)
        {
            importID = $qualifiedIdentifier.text;
            if ($ds != null) {
            }
        }
        {$ds == null}? // TODO: add support for importing x.*
        -> {(emitCC() || emitH())}? importDeclaration_cc_h(
            inc_id={importID.replace(".","/")},
            use_id={importID.replace(".","::")})
        ->
    ;
    
typeDeclaration
@init {
    boolean needsMigration = false;
    List<String> inits = new ArrayList<String>();
    List<String> pupInits = new ArrayList<String>();
    List<String> sdagEntries = new ArrayList<String>();
}
    :   ^(TYPE CLASS IDENT (^('extends' su=type))? (^('implements' type+))?
        {
            currentClass = (ClassSymbol)$IDENT.def;

            inits = currentClass.generateInits(currentClass.initializers);
            pupInits = currentClass.generateInits(currentClass.pupInitializers);
        }
        (csds+=classScopeDeclaration)*)
        ->{emitCI()}? chareDeclaration_ci(
                basename={basename()},
                sym={currentClass},
                chareType={"chare"},
                arrayDim={null},
                ident={$IDENT.text}, 
                ext={$su.st}, 
                csds={$csds},
                entries={null})
        -> {emitCC()}? classDeclaration_cc(
                sym={currentClass},
                ident={$IDENT.text}, 
                ext={$su.st}, 
                csds={$csds},
                pupInits={pupInits},
                pupers={currentClass.generatePUPers()},
                inits={inits})
        -> {emitH()}?  classDeclaration_h(
                sym={currentClass},
                ident={$IDENT.text}, 
                ext={$su.st}, 
                csds={$csds},
                needsPupInit={pupInits.size() > 0})
        ->
    |   ^('template' (i0+=IDENT*) ^('class' i1=IDENT (^('extends' su=type))? (^('implements' type+))? (csds+=classScopeDeclaration)*))
        -> {emitH()}? templateDeclaration_h(
            tident={$i0},
            ident={$i1.text},
            ext={$su.st},
            csds={$csds})
        -> 
    |   ^(MESSAGE IDENT (msds+=messageScopeDeclaration)*)
        -> {emitH()}? message_h(basename={basename()}, ident={$IDENT.text}, msds={$msds})
        -> {emitCI()}? message_ci(ident={$IDENT.text}, msds={$msds})
        ->
    |   ^(MULTICAST_MESSAGE IDENT (msds+=messageScopeDeclaration)*)
        -> {emitH()}? multicastMessage_h(basename={basename()}, ident={$IDENT.text}, msds={$msds})
        -> {emitCI()}? multicastMessage_ci(ident={$IDENT.text}, msds={$msds})
        ->
    |   ^(ENUM IDENT (^('implements' type+))? classScopeDeclaration*)
        -> template(t={$text}) "/*ENUM-not implemented*/ <t>"
    |   ^(TYPE chareType IDENT (^('extends' type))? (^('implements' type+))?
        {
            currentClass = (ClassSymbol)$IDENT.def;
            needsMigration = currentClass.isChareArray && !currentClass.hasMigrationCtor;
            sdagEntries = currentClass.generateSDAGEntries();
            inits = currentClass.generateInits(currentClass.initializers);
            pupInits = currentClass.generateInits(currentClass.pupInitializers);
        }
        (csds+=classScopeDeclaration)*)
        -> {emitCC()}? chareDeclaration_cc(
                sym={currentClass},
                ident={$IDENT.text}, 
                ext={$su.st}, 
                csds={$csds},
                pupInits={pupInits},
                pupers={currentClass.generatePUPers()},
                needsMigration={needsMigration},
                inits={inits})
        -> {emitCI()}? chareDeclaration_ci(
                basename={basename()},
                sym={currentClass},
                chareType={$chareType.st},
                arrayDim={null},
                ident={$IDENT.text}, 
                ext={$su.st}, 
                csds={$csds},
                entries={sdagEntries})
        -> {emitH()}? chareDeclaration_h(
                sym={currentClass},
                ident={$IDENT.text}, 
                ext={$su.st}, 
                csds={$csds},
                needsPupInit={pupInits.size() > 0},
                needsMigration={needsMigration})
        ->
    ;

chareType
@init {
$st = %{$start.getText()};
}
    :   CHARE
    |   GROUP
    |   NODEGROUP
    |   MAINCHARE
    |   ^(CHARE_ARRAY ARRAY_DIMENSION)
        -> template(t={$ARRAY_DIMENSION.text}) "array [<t>]"
    ;

enumConstant
    :   ^(IDENT arguments?)
        -> template(t={$text}) "/*enumConstant-not implemented*/ <t>"
    ;

messageScopeDeclaration
    :   ^(PRIMITIVE_VAR_DECLARATION modifierList? simpleType variableDeclaratorList[null])
        -> {!emitCC()}? class_var_decl(
            modl={$modifierList.st},
            type={$simpleType.st},
            declList={$variableDeclaratorList.st})
        ->
    |   ^(OBJECT_VAR_DECLARATION modifierList? objectType variableDeclaratorList[$objectType.st])
        -> {!emitCC()}? class_var_decl(
            modl={$modifierList.st},
            type={$objectType.st},
            declList={$variableDeclaratorList.st})
        ->
    ;


classScopeDeclaration
@init
{
    boolean entry = false;
    boolean migrationCtor = false;
    boolean sdagMethod = false;
}
    :   ^(FUNCTION_METHOD_DECL m=modifierList? g=genericTypeParameterList? 
            ty=type IDENT
            { currentMethod = (MethodSymbol)$IDENT.def; }
            f=formalParameterList
            b=block?)
        -> {emitCC()}? funcMethodDecl_cc(
                classSym={currentClass},
                methodSym={currentMethod},
                modl={$m.st}, 
                gtpl={$g.st}, 
                ty={$ty.st},
                id={$IDENT.text}, 
                fpl={$f.st}, 
                block={$b.st})
        -> {emitH()}? funcMethodDecl_h(
                modl={$m.st}, 
                gtpl={$g.st}, 
                ty={$ty.st},
                id={$IDENT.text}, 
                fpl={$f.st}, 
                block={$b.st})
        -> {emitCI()}? // do nothing, since it's not an entry method
        ->
    |   ^(ENTRY_FUNCTION_DECL m=modifierList? g=genericTypeParameterList? 
            ty=type IDENT
            {
                currentMethod = (MethodSymbol)$IDENT.def;
                sdagMethod = currentMethod.hasSDAG;
            }
            ef=entryFormalParameterList a=domainExpression[null]? 
            b=block?) 
        -> {emitCC()}? entryMethodDecl_cc(
                classSym={currentClass},
                methodSym={currentMethod},
                modl={$m.st}, 
                gtpl={$g.st}, 
                ty={$ty.st},
                id={$IDENT.text}, 
                fpl={$ef.st}, 
                adl={$a.st},
                block={$b.st})
        -> {emitH()}? entryMethodDecl_h(
                modl={$m.st}, 
                gtpl={$g.st}, 
                ty={$ty.st},
                id={$IDENT.text}, 
                fpl={$ef.st}, 
                adl={$a.st},
                block={$b.st})
        -> {emitCI()}? entryMethodDecl_ci(
                modl={$m.st}, 
                gtpl={$g.st}, 
                ty={$ty.st},
                id={$IDENT.text}, 
                fpl={$ef.st}, 
                block={$b.st})
        ->
    |   ^(SDAG_FUNCTION_DECL m=modifierList? g=genericTypeParameterList? 
            ty=type IDENT
            {
            currentMethod = (MethodSymbol)$IDENT.def;
            sdagMethod = currentMethod.hasSDAG;
            }
            ef=entryFormalParameterList a=domainExpression[null]? 
            ^(BLOCK (sdg+=sdagBasicBlock)*))
        -> {emitCI()}? funcMethodDecl_sdag_ci(
                classSym={currentClass},
                methodSym={currentMethod},
                modl={$m.st}, 
                gtpl={$g.st}, 
                ty={$ty.st},
                id={$IDENT.text}, 
                fpl={$ef.st}, 
                adl={$a.st},
                block={$sdg})
        ->
    |   ^(DIVCON_METHOD_DECL modifierList? type IDENT formalParameterList divconBlock)
    |   ^(PRIMITIVE_VAR_DECLARATION modifierList? simpleType variableDeclaratorList[null])
        -> {emitH()}? class_var_decl(
            modl={$modifierList.st},
            type={$simpleType.st},
            declList={$variableDeclaratorList.st})
        ->
    |   ^(OBJECT_VAR_DECLARATION modifierList? objectType variableDeclaratorList[$objectType.st])
        -> {emitH()}? class_var_decl(
            modl={$modifierList.st},
            type={$objectType.st},
            declList={$variableDeclaratorList.st})
        ->
    |   ^(CONSTRUCTOR_DECL m=modifierList? g=genericTypeParameterList? IDENT f=formalParameterList
            {
                currentMethod = (MethodSymbol)$IDENT.def;
            }
            b=block)
        -> {emitCC()}? ctorDecl_cc(
                modl={$m.st},
                gtpl={$g.st}, 
                id={$IDENT.text}, 
                fpl={$f.st}, 
                block={$b.st})
        -> {emitCI()}? // do nothing, it's not an entry constructor
        -> {emitH()}? ctorDecl_h(
                modl={$m.st},
                gtpl={$g.st}, 
                id={$IDENT.text}, 
                fpl={$f.st}, 
                block={$b.st})
        ->
    |   ^(ENTRY_CONSTRUCTOR_DECL m=modifierList? g=genericTypeParameterList? IDENT ef=entryFormalParameterList
            {
                currentMethod = (MethodSymbol)$IDENT.def;
                migrationCtor = currentClass.migrationCtor == $ENTRY_CONSTRUCTOR_DECL;
            }
            b=block)
        -> {emitCC()}? entryCtorDecl_cc(
                modl={$m.st},
                gtpl={$g.st}, 
                id={$IDENT.text}, 
                fpl={$ef.st}, 
                block={$b.st})
        -> {emitCI() && !migrationCtor}? entryCtorDecl_ci(
                modl={$m.st},
                gtpl={$g.st}, 
                id={$IDENT.text}, 
                fpl={$ef.st}, 
                block={$b.st})
        -> {emitH()}? entryCtorDecl_h(
                modl={$m.st},
                gtpl={$g.st}, 
                id={$IDENT.text}, 
                fpl={$ef.st}, 
                block={$b.st})
        ->
    ;
    
variableDeclaratorList[StringTemplate obtype]
    :   ^(VAR_DECLARATOR_LIST (var_decls+=variableDeclarator[obtype])+ )
        -> {emitCI() && currentClass != null && currentMethod != null && currentMethod.hasSDAG}?
                var_decl_list_sdag_ci(var_decls={$var_decls})
        -> {obtype == null ||
            obtype.toString().indexOf("CProxy_") == 0}? var_decl_list(var_decls={$var_decls})
        -> obj_var_decl_list(var_decls={$var_decls})
    ;

variableDeclarator[StringTemplate obtype]
    :   ^(VAR_DECLARATOR id=variableDeclaratorId initializer=variableInitializer[obtype]?)
        -> {emitCC()}? var_decl_cc(id={$id.st}, initializer={$initializer.st})
        -> {emitH()}?  var_decl_h(id={$id.st}, initializer={$initializer.st})
        -> {emitCI() && currentClass != null && currentMethod != null && currentMethod.hasSDAG}?
                var_decl_sdag_ci(id={currentClass.getSDAGLocalName($id.st.toString())}, initializer={$initializer.st})
        -> {emitCI()}? var_decl_ci(id={$id.st}, initializer={$initializer.st})
        ->
    ; 
    
variableDeclaratorId
    :   ^(IDENT de=domainExpression[null]?)
        -> var_decl_id(id={$IDENT.text}, domainExp={$de.st})
    ;

variableInitializer[StringTemplate obtype]
    :   arrayInitializer
        -> {$arrayInitializer.st}
    |   newExpression[obtype]
        -> {$newExpression.st}
    |   expression
        -> {$expression.st}
    ;

rangeItem
    :   e=expression
        -> template(e={$e.st}) "<e>"
    ;

rangeExpression
    :   ^(RANGE_EXPRESSION (ri+=rangeItem)*)
        -> template(t={$ri}) "Range(<t; separator=\",\">)"
    ;

rangeList returns [int len]
    :   (r+=rangeExpression)* { $len = $r.size(); }
        -> template(t={$r}) "<t; separator=\", \">"
    ;

domainExpression[List<StringTemplate> otherParams]
    :   ^(DOMAIN_EXPRESSION rl=rangeList)
        -> range_constructor(range={$rl.st}, others={$otherParams}, len={$rl.len})
    ;

domainExpressionAccess
    :   ^(DOMAIN_EXPRESSION rl=rangeListAccess)
        -> template(t={$rangeListAccess.st}) "<t>"
    ;

rangeListAccess
    :   (r+=rangeExpressionAccess)*
        -> template(t={$r}) "<t; separator=\", \">"
    ;

rangeExpressionAccess
    :   ^(RANGE_EXPRESSION (ri+=rangeItem)*)
        -> {$ri.size() > 1}? template(t={$ri}) "Range(<t; separator=\",\">)"
        -> template(t={$ri}) "<t; separator=\",\">"
    ;

arrayInitializer
    :   ^(ARRAY_INITIALIZER variableInitializer[null]*)
        -> template(t={$text}) "/* arrayInitializer-not implemented */ <t>"
    ;

genericTypeParameterList
    :   ^(GENERIC_TYPE_PARAM_LIST genericTypeParameter+)
        -> template(t={$text}) "/*GENERIC_TYPE_PARAM_LIST-not implemented*/ <t>"
    ;

genericTypeParameter
    :   ^(IDENT bound?)
        -> template(t={$text}) "/*genericTypeParameter-not implemented*/ <t>"
    ;
        
bound
    :   ^(EXTENDS_BOUND_LIST type+)
        -> template(t={$text}) "/*EXTENDS_BOUND_LIST-not implemented*/ <t>"
    ;

throwsClause
    :   ^(THROWS_CLAUSE qualifiedIdentifier+)
        -> template(t={$text}) "/* throwsClause-not implemented */ <t>"
    ;

modifierList
    :   ^(MODIFIER_LIST accessModifierList localModifierList charjModifierList otherModifierList)
        ->  {emitCC()}? mod_list_cc(accmods = {$accessModifierList.names}, localmods = {$localModifierList.names}, charjmods = {$charjModifierList.names}, othermods = {$otherModifierList.names})
        ->  {emitH()}? mod_list_h(accmods = {$accessModifierList.names}, localmods = {$localModifierList.names}, charjmods = {$charjModifierList.names}, othermods = {$otherModifierList.names})
        ->  {emitCI()}? mod_list_ci(accmods = {$accessModifierList.names}, localmods = {$localModifierList.names}, charjmods = {$charjModifierList.names}, othermods = {$otherModifierList.names})
        ->
    ;

modifier
    :   accessModifier
    |   localModifier
    |   charjModifier
    |   otherModifier
    ;

accessModifierList
returns [List names]
    :   ^(ACCESS_MODIFIER_LIST (m+=accessModifier)*)
        {
            $names = $m;
        }
    ;
localModifierList
returns [List names]
    :   ^(LOCAL_MODIFIER_LIST (m+=localModifier)*)
        {
            $names = $m;
        }
        ->  local_mod_list(mods = {$names})
    ;

charjModifierList
returns [List names]
    :   ^(CHARJ_MODIFIER_LIST (m+=charjModifier)*)
        {
            $names = $m;
            // Strip out null entries so we can detect empty modifier lists in the template
            while ($names.remove(null)) {}
        }
    ;

otherModifierList
returns [List names]
    :   ^(OTHER_MODIFIER_LIST (m+=otherModifier)*)
        {
            $names = $m;
        }
    ;
    
localModifier
@init
{
    $st = %{$start.getText()};
}
    :   FINAL
    |   STATIC
    |   VOLATILE
    ;

accessModifier
@init
{
    $st = %{$start.getText()};
}
    :   PUBLIC
    |   PROTECTED
    |   PRIVATE
        {
            $st = %{"private"};
        }
    ;

charjModifier
    :   ENTRY
    |   SDAGENTRY
    |   TRACED
    |   ACCELERATED -> template() "accel"
    |   THREADED -> template() "threaded"
    |   REDUCTIONTARGET -> template() "reductiontarget"
    ;

otherModifier
@init
{
    $st = %{$start.getText()};
}
    :   ABSTRACT
    |   NATIVE
    ;
    
type
    :   simpleType
        -> {$simpleType.st}
    |   objectType 
        -> {$objectType.st}
    |   VOID { $st = %{"void"}; }
    ;

simpleType
    :   ^(SIMPLE_TYPE primitiveType domainExpression[null]?)
        -> simple_type(typeID={$primitiveType.st}, arrDeclList={$domainExpression.st})
    ;

objectType
    : proxyType -> {$proxyType.st;}
    | nonProxyType -> {$nonProxyType.st}
    ;

nonProxyType
    :   ^(OBJECT_TYPE qualifiedTypeIdent domainExpression[null]?)
        -> obj_type(typeID={$qualifiedTypeIdent.st}, arrDeclList={$domainExpression.st})
    |   ^(POINTER_TYPE qualifiedTypeIdent domainExpression[null]?)
        -> pointer_type(typeID={$qualifiedTypeIdent.st}, arrDeclList={$domainExpression.st})
    |   ^(REFERENCE_TYPE qualifiedTypeIdent domainExpression[null]?)
        -> reference_type(typeID={$qualifiedTypeIdent.st}, arrDeclList={$domainExpression.st})
    ;

proxyType
    :   ^(PROXY_TYPE qualifiedTypeIdent domainExpression[null]?)
        -> proxy_type(typeID={$qualifiedTypeIdent.st}, arrDeclList={$domainExpression.st})
	|	^(MESSAGE_TYPE qualifiedTypeIdent)
		-> template(type={$qualifiedTypeIdent.st}) "<type>*"
	|	^(ARRAY_SECTION_TYPE qualifiedTypeIdent domainExpression[null]?)
		-> template(type={$qualifiedTypeIdent.st}) "CProxySection_<type>"
    ;

qualifiedTypeIdent returns [ClassSymbol type]
    :   ^(QUALIFIED_TYPE_IDENT (t+=typeIdent)+)
        {$type = (ClassSymbol)$QUALIFIED_TYPE_IDENT.def;}
        -> template(types={$t}) "<types; separator=\"::\">"
    ;

typeIdent
    :   ^(IDENT templateInstantiation?)
        -> typeIdent(typeID={$IDENT.text}, generics={$templateInstantiation.st})
    ;

primitiveType
@init {
$st = %{$start.getText()};
}
    :   BOOLEAN
        -> template() "bool"
    |   CHAR
    |   BYTE
        -> template() "char"
    |   SHORT
    |   INT
    |   LONG
    |   FLOAT
    |   DOUBLE
    ;

templateArg
    :   genericTypeArgument
        -> {$genericTypeArgument.st}
    |   literal
        -> {$literal.st}
    ;

templateArgList
    :   params+=templateArg+
        -> template(params={$params}) "<params; separator=\", \">"
    ;

templateInstantiation
    :   ^(TEMPLATE_INST templateArgList)
        -> template(args={$templateArgList.st}) "\<<args>\>"
    |   ^(TEMPLATE_INST ts=templateInstantiation)
        -> template(inst={$ts.st}) "\<<inst>\>"
    ;

genericTypeArgumentList
    :   ^(GENERIC_TYPE_ARG_LIST (gta+=genericTypeArgument)+)
        -> template(gtal={$gta}) "\<<gtal; separator=\", \">\>"
    ;

genericTypeArgument
    :   type
        -> {$type.st}
    |   '?'
        -> template(t={$text}) "/* genericTypeArgument: wildcard bound types not implemented */ <t>"
    ;


entryFormalParameterList
    :   ^(FORMAL_PARAM_LIST (efp+=entryFormalParameter)*)
        -> entry_formal_param_list(sdecl={$efp})
    ;


entryFormalParameter
    :   ^(FORMAL_PARAM_STD_DECL ^(SIMPLE_TYPE primitiveType domainExpression[null]?) vdid=variableDeclaratorId)
        -> template(type={$primitiveType.st}, declID={$vdid.st}) "<type> <declID>"
    |   ^(FORMAL_PARAM_STD_DECL ^((OBJECT_TYPE|POINTER_TYPE|REFERENCE_TYPE) qualifiedTypeIdent domainExpression[null]?) vdid=variableDeclaratorId)
        -> template(type={$qualifiedTypeIdent.st}, declID={$vdid.st}) "<type> __<declID>"
    |   ^(FORMAL_PARAM_STD_DECL ^(PROXY_TYPE qualifiedTypeIdent domainExpression[null]?) vdid=variableDeclaratorId)
        -> template(type={$qualifiedTypeIdent.st}, declID={$vdid.st}) "CProxy_<type> <declID>"
    |   ^(FORMAL_PARAM_STD_DECL ^(MESSAGE_TYPE qualifiedTypeIdent) vdid=variableDeclaratorId)
        -> template(type={$qualifiedTypeIdent.st}, declID={$vdid.st}) "<type>* <declID>"
    |   ^(FORMAL_PARAM_STD_DECL ^(ARRAY_SECTION_TYPE qualifiedTypeIdent domainExpression[null]?) vdid=variableDeclaratorId)
        -> template(type={$qualifiedTypeIdent.st}, declID={$vdid.st}) "CProxySection_<type> <declID>"
    ;


formalParameterList
    :   ^(FORMAL_PARAM_LIST (fpsd+=formalParameterStandardDecl)*)
        -> formal_param_list(sdecl={$fpsd})
    ;

    
formalParameterStandardDecl
    :   ^(FORMAL_PARAM_STD_DECL lms=localModifierList? t=type vdid=variableDeclaratorId)
        -> formal_param_decl(modList={$lms.st}, type={$t.st}, declID={$vdid.st})
    ;
    
qualifiedIdentifier
    :   IDENT
        -> {emitCI() && currentClass != null && currentMethod != null && currentMethod.hasSDAG}?
           template(t={currentClass.getSDAGLocalName($text)}) "<t>"
        -> template(t={$text}) "<t>"
    |   ^(DOT qualifiedIdentifier IDENT)
        -> template(t={$text}) "<t>"
    ;
    
block
@init { boolean emptyBlock = true; }
    :   ^(BLOCK (b+=blockStatement)*)
        { emptyBlock = ($b == null || $b.size() == 0); }
        -> {((emitCC() && (currentMethod == null || !currentMethod.hasSDAG)) ||
            (emitCI() && (currentMethod != null && currentMethod.hasSDAG))) && emptyBlock}? template(bsl={$b}) "{ }"
        -> {emitCC() && (currentMethod == null || !currentMethod.hasSDAG)}? block_cc(bsl={$b}, braces={true})
        -> {emitCI() && (currentMethod != null && currentMethod.hasSDAG)}? block_cc(bsl={$b}, braces={true})
        ->
    ;


sdagBlock
    :   ^(BLOCK (sdg+=sdagBasicBlock)*)
        -> block_cc(bsl={$sdg}, braces={true})
    ;

sdagBasicBlock
    :   sdagStatement
        -> {$sdagStatement.st}
    |   (s+=blockStatement)+
        -> block_atomic(s={$s})
    ;
    
blockStatement
    :   localVariableDeclaration
        -> {$localVariableDeclaration.st}
    |   statement
        -> {$statement.st}
    ;


localVariableDeclaration
    :   ^(PRIMITIVE_VAR_DECLARATION localModifierList? simpleType vdl=variableDeclaratorList[null])
        -> {emitCI() && currentClass != null && currentMethod != null && currentMethod.hasSDAG}?
                local_var_decl_sdag_ci(declList={$vdl.st})
        -> local_var_decl(
            modList={$localModifierList.st},
            type={$simpleType.st},
            declList={$vdl.st})
    |   ^(OBJECT_VAR_DECLARATION localModifierList? objectType vdl=variableDeclaratorList[$objectType.st])
        -> {emitCI() && currentClass != null && currentMethod != null && currentMethod.hasSDAG}?
                local_var_decl_sdag_ci(declList={$vdl.st})
        -> local_var_decl(
            modList={$localModifierList.st},
            type={$objectType.st},
            declList={$vdl.st})
    ;


statement
    :   nonBlockStatement
        -> {$nonBlockStatement.st}
    |   block
        -> {$block.st}
    ;

divconBlock
    :   ^(DIVCON_BLOCK divconExpr)
    ;

divconAssignment
    :   ^(LET_ASSIGNMENT IDENT expression)
    ;

divconAssignmentList
    :   divconAssignment+
    ;

divconExpr
    :   ^(IF parenthesizedExpression divconExpr divconExpr?)
    |   ^(LET divconAssignmentList IN divconExpr)
    |   expression
    ;

sdagStatement
    :   ^(OVERLAP sdagBlock)
        -> template(b={$sdagBlock.st}) "overlap <b>"
    |   ^(WHEN (wa+=whenArgument)+ sdagBlock)
        -> template(w={$wa}, b={$sdagBlock.st}) "when <w; separator=\", \"> <b>"
    |   ^(SDAG_IF pe=parenthesizedExpression
            ifblock=sdagBlock elseblock=sdagBlock?)
        -> if(cond={$pe.st}, then={$ifblock.st}, else_={$elseblock.st})
    |   ^(SDAG_FOR forInit? FOR_EXPR cond=expression?
            FOR_UPDATE (update+=expression)* b=sdagBlock)
        -> for(initializer={$forInit.st}, cond={$cond.st},
                update={$update}, body={$b.st})
    |   ^(SDAG_WHILE pe=parenthesizedExpression b=sdagBlock)
        -> while(cond={$pe.st}, body={$b.st})
    |   ^(SDAG_DO b=sdagBlock pe=parenthesizedExpression)
        -> dowhile(cond={$pe.st}, block={$b.st})
    ;

whenArgument
    :   IDENT expression? entryFormalParameterList
		-> whenArgument(ident={$IDENT.text}, expr={$expression.st}, params={$entryFormalParameterList.st})
    ;

nonBlockStatement
    :   ^(ASSERT cond=expression msg=expression?)
        -> assert(cond={$cond.st}, msg={$msg.st})
    |   ^(IF parenthesizedExpression then=block else_=block?)
        -> if(cond={$parenthesizedExpression.st}, then={$then.st}, else_={$else_.st})
    |   ^(FOR forInit? FOR_EXPR cond=expression? FOR_UPDATE (update+=expression)* b=block)
        -> for(initializer={$forInit.st}, cond={$cond.st}, update={$update}, body={$block.st})
    |   ^(FOR_EACH localModifierList? type IDENT expression block) 
        -> template(t={$text}) "/* foreach not implemented */ <t>"
    |   ^(WHILE pe=parenthesizedExpression b=block)
        -> while(cond={$pe.st}, body={$b.st})
    |   ^(DO b=block pe=parenthesizedExpression)
        -> dowhile(cond={$pe.st}, block={$b.st})
    |   ^(SWITCH pe=parenthesizedExpression (scls+=switchCaseLabel)*)
        -> switch(expr={$pe.st}, labels={$scls})
    |   ^(RETURN e=expression?)
        -> return(val={$e.st})
    |   ^(THROW expression)
        -> template(t={$text}) "/* throw not implemented */ <t>"
    |   ^(BREAK IDENT?)
        -> template() "break;" // TODO: support labeling
    |   ^(CONTINUE IDENT?)
        -> template() "continue;" // TODO: support labeling
    |   ^(LABELED_STATEMENT i=IDENT s=statement)
        -> label(text={$i.text}, stmt={$s.st})
    |   expression
        -> template(expr={$expression.st}) "<expr>;"
    |   ^('delete' expression)
        -> template(t={$expression.st}) "delete <t>;"
    |   ^('embed' STRING_LITERAL EMBED_BLOCK)
        ->  embed_cc(str={$STRING_LITERAL.text}, blk={$EMBED_BLOCK.text})
	|	^(CONTRIBUTE_1 e1=expression)
		-> contribute(type={true}, size={null}, data={null}, func={null}, callback={$e1.st} )
	|	^(CONTRIBUTE_2 e1=expression e2=expression q1=qualifiedIdentifier e3=expression)
		-> contribute(type={false}, size={$e1.st}, data={$e2.st}, func={$q1.st}, callback={$e3.st})
    |   ';' // Empty statement.
        -> {%{$start.getText()}}
    ;
        
switchCaseLabel
    :   ^(CASE expression (b+=blockStatement)*)
        -> case(expr={$expression.st}, block={$b})
    |   ^(DEFAULT (b+=blockStatement)*)
        -> template(block={$b}) "default: <block>"
    ;
    
forInit
    :   localVariableDeclaration
        -> template(lvd={$localVariableDeclaration.st}) "<lvd>"
    |   (ex+=expression)+
        -> template(ex={$ex}) "<ex; separator=\", \">;"
    ;

// EXPRESSIONS

parenthesizedExpression
    :   ^(PAREN_EXPR exp=expression)
        -> template(expr={$exp.st}) "(<expr>)"
    ;

expressionArrayAccess
    :   ^(EXPR expr)
        -> {$expr.st}
    |    domainExpressionAccess
        -> {$domainExpressionAccess.st}
    ;
    
expression
    :   ^(EXPR expr)
        -> {$expr.st}
    |   domainExpression[null]
        -> {$domainExpression.st}
    ;

expr
    :   ^(ASSIGNMENT e1=expr e2=expr)
        -> template(e1={$e1.st}, e2={$e2.st}) "<e1> = <e2>"
    |   ^('+=' e1=expr e2=expr)
        -> template(e1={$e1.st}, e2={$e2.st}) "<e1> += <e2>"
    |   ^('-=' e1=expr e2=expr)
        -> template(e1={$e1.st}, e2={$e2.st}) "<e1> -= <e2>"
    |   ^('*=' e1=expr e2=expr)
        -> template(e1={$e1.st}, e2={$e2.st}) "<e1> *= <e2>"
    |   ^('/=' e1=expr e2=expr)
        -> template(e1={$e1.st}, e2={$e2.st}) "<e1> /= <e2>"
    |   ^('&=' e1=expr e2=expr)
        -> template(e1={$e1.st}, e2={$e2.st}) "<e1> &= <e2>"
    |   ^('|=' e1=expr e2=expr)
        -> template(e1={$e1.st}, e2={$e2.st}) "<e1> |= <e2>"
    |   ^('^=' e1=expr e2=expr)
        -> template(e1={$e1.st}, e2={$e2.st}) "<e1> ^= <e2>"
    |   ^('%=' e1=expr e2=expr)
        -> template(e1={$e1.st}, e2={$e2.st}) "<e1> %= <e2>"
    |   ^('>>>=' e1=expr e2=expr)
        -> template(e1={$e1.st}, e2={$e2.st}) "<e1> \>\>\>= <e2>"
    |   ^('>>=' e1=expr e2=expr)
        -> template(e1={$e1.st}, e2={$e2.st}) "<e1> \>\>= <e2>"
    |   ^('<<=' e1=expr e2=expr)
        -> template(e1={$e1.st}, e2={$e2.st}) "<e1> \<\<= <e2>"
    |   ^('?' e1=expr e2=expr e3=expr)
        -> template(e1={$e1.st}, e2={$e2.st}, e3={$e3.st}) "<e1> ? <e2> : <e3>"
    |   ^('||' e1=expr e2=expr)
        -> template(e1={$e1.st}, e2={$e2.st}) "<e1> || <e2>"
    |   ^('&&' e1=expr e2=expr)
        -> template(e1={$e1.st}, e2={$e2.st}) "<e1> && <e2>"
    |   ^(BITWISE_OR e1=expr e2=expr)
        -> template(e1={$e1.st}, e2={$e2.st}) "<e1> | <e2>"
    |   ^('^' e1=expr e2=expr)
        -> template(e1={$e1.st}, e2={$e2.st}) "<e1> ^ <e2>"
    |   ^('&' e1=expr e2=expr)
        -> template(e1={$e1.st}, e2={$e2.st}) "<e1> & <e2>"
    |   ^(EQUALS e1=expr e2=expr)
        -> template(e1={$e1.st}, e2={$e2.st}) "<e1> == <e2>"
    |   ^('!=' e1=expr e2=expr)
        -> template(e1={$e1.st}, e2={$e2.st}) "<e1> != <e2>"
    |   ^('instanceof' expr type)
        -> template(t={$text}) "/* instanceof not implemented */ <t>"
    |   ^('<=' e1=expr e2=expr)
        -> template(e1={$e1.st}, e2={$e2.st}) "<e1> \<= <e2>"
    |   ^('>=' e1=expr e2=expr)
        -> template(e1={$e1.st}, e2={$e2.st}) "<e1> \>= <e2>"
    |   ^('>>>' e1=expr e2=expr)
        -> template(e1={$e1.st}, e2={$e2.st}) "<e1> \>\>\> <e2>"
    |   ^('>>' e1=expr e2=expr)
        -> template(e1={$e1.st}, e2={$e2.st}) "<e1> \>\> <e2>"
    |   ^('>' e1=expr e2=expr)
        -> template(e1={$e1.st}, e2={$e2.st}) "<e1> \> <e2>"
    |   ^('<<' e1=expr e2=expr)
        -> template(e1={$e1.st}, e2={$e2.st}) "<e1> \<\< <e2>"
    |   ^('<' e1=expr e2=expr)
        -> template(e1={$e1.st}, e2={$e2.st}) "<e1> \< <e2>"
    |   ^('+' e1=expr e2=expr)
        -> template(e1={$e1.st}, e2={$e2.st}) "<e1> + <e2>"
    |   ^('-' e1=expr e2=expr)
        -> template(e1={$e1.st}, e2={$e2.st}) "<e1> - <e2>"
    |   ^('*' e1=expr e2=expr)
        -> template(e1={$e1.st}, e2={$e2.st}) "<e1> * <e2>"
    |   ^('/' e1=expr e2=expr)
        -> template(e1={$e1.st}, e2={$e2.st}) "<e1> / <e2>"
    |   ^('%' e1=expr e2=expr)
        -> template(e1={$e1.st}, e2={$e2.st}) "<e1> % <e2>"
    |   ^(UNARY_PLUS e1=expr)
        -> template(e1={$e1.st}) "+<e1>"
    |   ^(UNARY_MINUS e1=expr)
        -> template(e1={$e1.st}) "-<e1>"
    |   ^(PRE_INC e1=expr)
        -> template(e1={$e1.st}) "++<e1>"
    |   ^(PRE_DEC e1=expr)
        -> template(e1={$e1.st}) "--<e1>"
    |   ^(POST_INC e1=expr)
        -> template(e1={$e1.st}) "<e1>++"
    |   ^(POST_DEC e1=expr)
        -> template(e1={$e1.st}) "<e1>--"
    |   ^(TILDE e1=expr)
        -> template(e1={$e1.st}) "~<e1>"
    |   ^(NOT e1=expr)
        -> template(e1={$e1.st}) "!<e1>"
    |   ^(CAST_EXPR ty=type e1=expr)
        -> template(ty={$ty.st}, e1={$e1.st}) "(<ty>)<e1>"
    |   primaryExpression
        -> {$primaryExpression.st}
    |   ^(POINTER_DEREFERENCE e1 = expr)
        ->  template(e = {$e1.st}) "*(<e>)"
    ;

primaryExpression
@init { int dims = 1; boolean isValueType = true; }
    :  ^(ARRAY_ELEMENT_ACCESS pe=primaryExpression ex=expressionArrayAccess) {
            if ($pe.start.symbolType != null && $pe.start.symbolType instanceof PointerType) {
                PointerType p = (PointerType)($pe.start.symbolType);
                if (p.baseType instanceof ClassSymbol) {
                    ClassSymbol cs = (ClassSymbol)(p.baseType);
                    if (cs.templateArgs != null && cs.templateArgs.size() > 1 &&
                        cs.templateArgs.get(1) instanceof LiteralType) {
                        LiteralType l = (LiteralType)(cs.templateArgs.get(1));
                        dims = Integer.valueOf(l.literal);
                    }
                }
            }
            if ($pe.start.symbolType instanceof PointerType) {
              PointerType pt = (PointerType)($pe.start.symbolType);
              ClassSymbol cs = (ClassSymbol)(pt.baseType);
              if (cs != null && cs.templateArgs != null && cs.templateArgs.size() > 0) {
                List<TypeName> list = new ArrayList<TypeName>();
                list.add(new TypeName(cs.templateArgs.get(0).getTypeName()));
                isValueType = symtab.lookupPrimitive(list) != null;
              }
            }
        }
        -> {/*isValueType && */$pe.start.symbolType != null && $pe.start.symbolType instanceof PointerType && dims == 1}?
               template(pe={$pe.st}, ex={$ex.st}) "(*(<pe>))[<ex>]"
        //-> {!isValueType && $pe.start.symbolType != null && $pe.start.symbolType instanceof PointerType && dims == 1}?
               //template(pe={$pe.st}, ex={$ex.st}) "(*((*(<pe>))[<ex>]))"
        -> {$pe.start.symbolType != null && $pe.start.symbolType instanceof PointerType && dims == 2}?
               template(pe={$pe.st}, ex={$ex.st}) "(*(<pe>)).access(<ex>)"
        -> template(pe={$pe.st}, ex={$ex.st}) "(<pe>)[<ex>]"
    |   ^(DOT prim=primaryExpression
            ( IDENT   -> template(id={$IDENT.text}, prim={$prim.st}) "<prim>.<id>"
            | THIS    -> template(prim={$prim.st}) "<prim>.this"
            | SUPER   -> template(prim={$prim.st}) "<prim>.super"
            )
        )
    |   ^(ARROW prim=primaryExpression
            ( IDENT   -> template(id={$IDENT.text}, prim={$prim.st}) "<prim>-><id>"
            | THIS    -> template(prim={$prim.st}) "<prim>->this"
            | SUPER   -> template(prim={$prim.st}) "<prim>->super"
            )
        )
    |   parenthesizedExpression
        -> {$parenthesizedExpression.st}
    |   IDENT
        -> {emitCI() && currentClass != null && currentMethod != null && currentMethod.hasSDAG}?
           template(t={currentClass.getSDAGLocalName($IDENT.text)}) "<t>"
        -> {%{$IDENT.text}}
    |   CHELPER
        -> {%{"constructorHelper"}}
    |   ^(METHOD_CALL pe=primaryExpression gtal=genericTypeArgumentList? args=arguments)
        -> method_call(primary={$pe.st}, generic_types={$gtal.st}, args={$args.st})
    |   ^(ENTRY_METHOD_CALL pe=primaryExpression gtal=genericTypeArgumentList? args=arguments)
        -> method_call(primary={$pe.st}, generic_types={$gtal.st}, args={$args.st})
    |   explicitConstructorCall
        -> {$explicitConstructorCall.st}
    |   literal
        -> {$literal.st}
    |   newExpression[null]
        -> {$newExpression.st}
    |   THIS
        -> {%{$start.getText()}}
    |   arrayTypeDeclarator
        -> {$arrayTypeDeclarator.st}
    |   SUPER
        -> {%{$start.getText()}}
	|	THISINDEX
		->	template() "thisIndex"
	|	THISPROXY
		->	template() "thisProxy"
    |   domainExpression[null]
        ->  {$domainExpression.st}
    |   ^(SIZEOF e=expression)
        -> template(ex={$e.st}) "sizeof(<ex>)"
    |   ^(SIZEOF t=type)
        -> template(ty={$t.st}) "sizeof(<ty>)"
    ;
    
explicitConstructorCall
    :   ^(THIS_CONSTRUCTOR_CALL genericTypeArgumentList? arguments)
        -> template(t={$text}) "<t>"
    |   ^(SUPER_CONSTRUCTOR_CALL primaryExpression? genericTypeArgumentList? arguments)
        -> template(t={$text}) "<t>"
    ;

arrayTypeDeclarator
    :   ^(ARRAY_DECLARATOR (arrayTypeDeclarator | qualifiedIdentifier | primitiveType))
        -> template(t={$text}) "<t>"
    ;

newExpression[StringTemplate obtype]
    :   ^(NEW_EXPRESSION arguments? domainExpression[$arguments.args])
        -> template(domain={$domainExpression.st},type={$obtype}) "new <type>(<domain>)"
    |   ^(NEW proxyType arguments)
        -> template(t={$proxyType.st}, a={$arguments.st}) "<t>::ckNew(<a>)"
    |   ^(NEW nonProxyType arguments)
        -> template(q={$nonProxyType.st}, a={$arguments.st}) "new <q>(<a>)"
    ;

arguments returns [List<StringTemplate> args]
@init {
    $args = new ArrayList<StringTemplate>();
}
    :   ^(ARGUMENT_LIST (e=expression { $args.add($e.st); } )*)        
        ->  arguments(exprs={$args})
    ;

literal
@init {
$st = %{$start.getText()};
}
    :   HEX_LITERAL
    |   OCTAL_LITERAL
    |   DECIMAL_LITERAL
    |   FLOATING_POINT_LITERAL
    |   CHARACTER_LITERAL
    |   STRING_LITERAL
    |   TRUE
    |   FALSE
    |   NULL
    ;


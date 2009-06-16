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
    SymbolTable symtab_ = null;

    PackageScope currentPackage_ = null;
    ClassSymbol currentClass_ = null;
    MethodSymbol currentMethod_ = null;
    LocalScope currentLocalScope_ = null;

    Translator translator_;
    OutputMode mode_;

    private boolean emitCC() { return mode_ == OutputMode.cc; }
    private boolean emitCI() { return mode_ == OutputMode.ci; }
    private boolean emitH() { return mode_ == OutputMode.h; }

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
     *  t// Replace default ANTLR generated catch clauses with this action, allowing early failure.
@rulecatch {
    catch (RecognitionException re) {
        reportError(re);
        throw re;
    }
}ype.
     */
    public boolean listContainsToken(List<CharjAST> list, int tokenType) {
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
    this.symtab_ = symtab;
    this.mode_ = m;
    String closingBraces = "";
}
    :   ^(CHARJ_SOURCE (p=packageDeclaration)? 
        (i+=importDeclaration)* 
        (t=typeDeclaration))
        //(t+=typeDeclaration)*)
        {
            // construct string of }'s to close namespace 
            if ($p.st != null) {
                String temp_p = $p.st.toString();
                for (int idx=0; idx<temp_p.length(); ++idx) {
                    if (temp_p.charAt(idx) == '{') {
                        closingBraces += "} ";
                    }
                }
            }
        }
        -> {emitCC()}? charjSource_cc(
            pd={$p.st}, ids={$i}, tds={$t.st}, cb={closingBraces})
        -> {emitCI()}? charjSource_ci(pd={$p.st}, ids={$i}, tds={$t.st})
        -> {emitH()}? charjSource_h(
            pd={$p.st}, ids={$i}, tds={$t.st}, cb={closingBraces})
        ->
    ;

packageDeclaration
@init { 
    List<String> names = null; 
}
    :   ^(PACKAGE qualifiedIdentifier)  {
            names =  java.util.Arrays.asList(
                    $qualifiedIdentifier.text.split("[.]"));
        }
        -> {(emitCC() || emitH())}? packageDeclaration_cc_h(
            ids={names})
        ->
    ;
    
importDeclaration
@init {
    String importID = null;
}
    :   ^(IMPORT STATIC? qualifiedIdentifier DOTSTAR?)
        {
            importID = $qualifiedIdentifier.text;
            if ($DOTSTAR != null) {
            }
        }
        {$DOTSTAR == null}? // TODO: add support for importing x.*
        -> {(emitCC() || emitH())}? importDeclaration_cc_h(
            inc_id={importID.replace(".","/")},
            use_id={importID.replace(".","::")})
        ->
    ;
    
typeDeclaration
    :   ^(CLASS m=modifierList IDENT g=genericTypeParameterList? 
                e=classExtendsClause? i=implementsClause? c=classTopLevelScope) 
        -> {emitCC()}? classDeclaration_cc(
                mod={$m.st}, 
                ident={$IDENT.text}, 
                gen={$g.st}, 
                ext={$e.st}, 
                impl={$i.st},
                ctls={$c.st})
        -> {emitCI()}? classDeclaration_ci(
                mod={$m.st}, 
                ident={$IDENT.text}, 
                gen={$g.st}, 
                ext={$e.st}, 
                impl={$i.st},
                ctls={$c.st})
        -> {emitH()}? classDeclaration_h(
                mod={$m.st}, 
                ident={$IDENT.text}, 
                gen={$g.st}, 
                ext={$e.st}, 
                impl={$i.st},
                ctls={$c.st})
        ->
    |   ^(INTERFACE modifierList IDENT genericTypeParameterList? 
                interfaceExtendsClause? interfaceTopLevelScope)
        -> template(t={$text}) "/*INTERFACE*/ <t>"
    |   ^(ENUM modifierList IDENT implementsClause? enumTopLevelScope)
        -> template(t={$text}) "/*ENUM*/ <t>"
    ;


classExtendsClause
    :   ^(EXTENDS_CLAUSE t=type) 
        -> {emitCC() || emitH()}? classExtends_cc_h(type={$t.st})
        -> {emitCI()}? classExtends_ci(type={$t.st})
        ->
    ;   

interfaceExtendsClause 
    :   ^(EXTENDS_CLAUSE (typeList+=type)+) 
        -> interfaceExtends(ts={$typeList})
    ;   
    
implementsClause
    :   ^(IMPLEMENTS_CLAUSE type+)
        -> template(t={$text}) "/*IMPLEMENTS_CLAUSE*/ <t>"
    ;
        
genericTypeParameterList
    :   ^(GENERIC_TYPE_PARAM_LIST genericTypeParameter+)
        -> template(t={$text}) "/*GENERIC_TYPE_PARAM_LIST*/ <t>"
    ;

genericTypeParameter
    :   ^(IDENT bound?)
        -> template(t={$text}) "/*genericTypeParameter*/ <t>"
    ;
        
bound
    :   ^(EXTENDS_BOUND_LIST type+)
        -> template(t={$text}) "/*EXTENDS_BOUND_LIST*/ <t>"
    ;

enumTopLevelScope
    :   ^(ENUM_TOP_LEVEL_SCOPE enumConstant+ classTopLevelScope?)
        -> template(t={$text}) "/*enumTopLevelScope*/ <t>"
    ;
    
enumConstant
    :   ^(IDENT arguments? classTopLevelScope?)
        -> template(t={$text}) "/*enumConstant*/ <t>"
    ;
    
    
classTopLevelScope
    :   ^(CLASS_TOP_LEVEL_SCOPE (csd+=classScopeDeclarations)*) 
        -> classTopLevelScope(classScopeDeclarations={$csd})
    ;
    
classScopeDeclarations
@init { boolean entry = false; }
    :   ^(CLASS_INSTANCE_INITIALIZER block)
        -> template(t={$text}) "/*cii*/ <t>"
    |   ^(CLASS_STATIC_INITIALIZER block)
        -> template(t={$text}) "/*csi*/ <t>"
    |   ^(FUNCTION_METHOD_DECL m=modifierList g=genericTypeParameterList? 
            ty=type IDENT f=formalParameterList a=arrayDeclaratorList? 
            tc=throwsClause? b=block?)
        { 
            // determine whether this is an entry method
            entry = listContainsToken($m.start.getChildren(), ENTRY);
        }
        -> {emitCC()}? funcMethodDecl_cc(
                modl={$m.st}, 
                gtpl={$g.st}, 
                ty={$ty.text},
                id={$IDENT.text}, 
                fpl={$f.st}, 
                adl={$a.st},
                tc={$tc.st}, 
                block={$b.st})
        -> {emitH()}? funcMethodDecl_h(
                modl={$m.st}, 
                gtpl={$g.st}, 
                ty={$ty.text},
                id={$IDENT.text}, 
                fpl={$f.st}, 
                adl={$a.st},
                tc={$tc.st}, 
                block={$b.st})
        -> {(emitCI() && entry)}? funcMethodDecl_ci(
                modl={$m.st}, 
                gtpl={$g.st}, 
                ty={$ty.text},
                id={$IDENT.text}, 
                fpl={$f.st}, 
                adl={$a.st},
                tc={$tc.st}, 
                block={$b.st})
        ->
    |   ^(VOID_METHOD_DECL m=modifierList g=genericTypeParameterList? IDENT 
            f=formalParameterList t=throwsClause? b=block?)
        { 
            // determine whether this is an entry method
            entry = listContainsToken($m.start.getChildren(), ENTRY);
        }
        -> {emitCC()}? voidMethodDecl_cc(
                modl={$m.st}, 
                gtpl={$g.st}, 
                id={$IDENT.text}, 
                fpl={$f.st}, 
                tc={$t.st}, 
                block={$b.st})
        -> {emitCI() && entry}? voidMethodDecl_ci(
                modl={$m.st}, 
                gtpl={$g.st}, 
                id={$IDENT.text}, 
                fpl={$f.st}, 
                tc={$t.st}, 
                block={$b.st})
        -> {emitH()}? voidMethodDecl_h(
                modl={$m.st}, 
                gtpl={$g.st}, 
                id={$IDENT.text}, 
                fpl={$f.st}, 
                tc={$t.st}, 
                block={$b.st})
        ->
    |   ^(PRIMITIVE_VAR_DECLARATION modifierList simpleType variableDeclaratorList)
        -> {emitCC() || emitH()}? primitive_var_decl(
            modList={$modifierList.st},
            type={$simpleType.st},
            declList={$variableDeclaratorList.st})
        ->
    |   ^(OBJECT_VAR_DECLARATION modifierList objectType variableDeclaratorList)
        -> {emitCC() || emitH()}? object_var_decl(
            modList={$modifierList.st},
            type={$objectType.st},
            declList={$variableDeclaratorList.st})
        ->
    |   ^(CONSTRUCTOR_DECL m=modifierList g=genericTypeParameterList? IDENT f=formalParameterList 
            t=throwsClause? b=block)
        { 
            // determine whether this is an entry method
            entry = listContainsToken($m.start.getChildren(), ENTRY);
        }
        -> {emitCC()}? ctorDecl_cc(
                modl={$m.st}, 
                gtpl={$g.st}, 
                id={$IDENT.text}, 
                fpl={$f.st}, 
                tc={$t.st}, 
                block={$b.st})
        -> {emitCI() && entry}? ctorDecl_ci(
                modl={$m.st}, 
                gtpl={$g.st}, 
                id={$IDENT.text}, 
                fpl={$f.st}, 
                tc={$t.st}, 
                block={$b.st})
        -> {emitH()}? ctorDecl_h(
                modl={$m.st}, 
                gtpl={$g.st}, 
                id={$IDENT.text}, 
                fpl={$f.st}, 
                tc={$t.st}, 
                block={$b.st})
        ->
    |   d=typeDeclaration
        -> template(t={$d.st}) "/*typeDeclaration*/ <t>"
    ;
    
interfaceTopLevelScope
    :   ^(INTERFACE_TOP_LEVEL_SCOPE interfaceScopeDeclarations*)
        -> template(t={$text}) "<t>"
    ;
    
interfaceScopeDeclarations
    :   ^(FUNCTION_METHOD_DECL modifierList genericTypeParameterList? type IDENT formalParameterList arrayDeclaratorList? throwsClause?)
        -> template(t={$text}) "<t>"
    |   ^(VOID_METHOD_DECL modifierList genericTypeParameterList? IDENT formalParameterList throwsClause?)
        -> template(t={$text}) "<t>"
        // Interface constant declarations have been switched to variable
        // declarations by Charj.g; the parser has already checked that
        // there's an obligatory initializer.
    |   ^(PRIMITIVE_VAR_DECLARATION modifierList simpleType variableDeclaratorList)
        -> template(t={$text}) "<t>"
    |   ^(OBJECT_VAR_DECLARATION modifierList objectType variableDeclaratorList)
        -> template(t={$text}) "<t>"
    |   typeDeclaration
        -> template(t={$text}) "<t>"
    ;

variableDeclaratorList
    :   ^(VAR_DECLARATOR_LIST variableDeclarator+)
        -> template(t={$text}) "/*variableDeclaratorList*/ <t>"
    ;

variableDeclarator
    :   ^(VAR_DECLARATOR variableDeclaratorId variableInitializer?)
        -> template(t={$text}) "/*variableDeclarator*/ <t>"
    ;
    
variableDeclaratorId
    :   ^(IDENT arrayDeclaratorList?)
        -> template(t={$text}) "/*variableDeclaratorId*/ <t>"
    ;

variableInitializer
    :   arrayInitializer
        -> template(t={$text}) "<t>"
    |   expression
        -> template(t={$text}) "<t>"
    ;

arrayDeclarator
    :   LBRACK RBRACK
        -> template(t={$text}) "<t>"
    ;

arrayDeclaratorList
    :   ^(ARRAY_DECLARATOR_LIST ARRAY_DECLARATOR*)  
        -> template(t={$text}) "<t>"
    ;
    
arrayInitializer
    :   ^(ARRAY_INITIALIZER variableInitializer*)
        -> template(t={$text}) "<t>"
    ;

throwsClause
    :   ^(THROWS_CLAUSE qualifiedIdentifier+)
        -> template(t={$text}) "<t>"
    ;

modifierList
    :   ^(MODIFIER_LIST (m+=modifier)*)
        -> template(mod={$m}) "<mod; separator=\" \">"
    ;

modifier
    :   PUBLIC
        -> template(t={$text}) "<t>"
    |   PROTECTED
        -> template(t={$text}) "<t>"
    |   PRIVATE
        -> template(t={$text}) "<t>"
    |   ENTRY
        -> template() "public"
    |   STATIC
        -> template(t={$text}) "<t>"
    |   ABSTRACT
        -> template(t={$text}) "<t>"
    |   NATIVE
        -> template(t={$text}) "<t>"
    |   SYNCHRONIZED
        -> template(t={$text}) "<t>"
    |   TRANSIENT
        -> template(t={$text}) "<t>"
    |   VOLATILE
        -> template(t={$text}) "<t>"
    |   localModifier
        -> template(t={$text}) "<t>"
    ;

localModifierList
    :   ^(LOCAL_MODIFIER_LIST localModifier*)
        -> template(t={$text}) "<t>"
    ;

localModifier
    :   FINAL
        -> template(t={$text}) "<t>"
    ;

    
type
    :   simpleType
        -> template(type={$simpleType.st}) "<type>"
    |   objectType 
        -> template(type={$objectType.st}) "<type>"
    ;

simpleType
    :   ^(TYPE primitiveType arrayDeclaratorList?)
        -> type(typeID={$primitiveType.st}, arrDeclList={$arrayDeclaratorList.st})
    ;

objectType
    :   ^(TYPE qualifiedTypeIdent arrayDeclaratorList?)
        -> type(typeID={$qualifiedTypeIdent.st}, arrDeclList={$arrayDeclaratorList.st})
    ;

qualifiedTypeIdent
    :   ^(QUALIFIED_TYPE_IDENT typeIdent+) 
        -> template(t={$text}) "<t>"
    ;

typeIdent
    :   ^(IDENT genericTypeArgumentList?)
        -> template(t={$text}) "<t>"
    ;

primitiveType
    :   BOOLEAN
        -> template() "bool"
    |   CHAR
        -> template(t={$text}) "<t>"
    |   BYTE
        -> template(t={$text}) "<t>"
    |   SHORT
        -> template(t={$text}) "<t>"
    |   INT
        -> template(t={$text}) "<t>"
    |   LONG
        -> template(t={$text}) "<t>"
    |   FLOAT
        -> template(t={$text}) "<t>"
    |   DOUBLE
        -> template(t={$text}) "<t>"
    ;

genericTypeArgumentList
    :   ^(GENERIC_TYPE_ARG_LIST genericTypeArgument+)
        -> template(t={$text}) "<t>"
    ;
    
genericTypeArgument
    :   type
        -> template(t={$text}) "<t>"
    |   ^(QUESTION genericWildcardBoundType?)
        -> template(t={$text}) "<t>"
    ;

genericWildcardBoundType                                                                                                                      
    :   ^(EXTENDS type)
        -> template(t={$text}) "<t>"
    |   ^(SUPER type)
        -> template(t={$text}) "<t>"
    ;

formalParameterList
    :   ^(FORMAL_PARAM_LIST (fpsd+=formalParameterStandardDecl)* fpvd=formalParameterVarargDecl?)
        -> formal_param_list(sdecl={$fpsd}, vdecl={$fpvd.st})
    ;
    
formalParameterStandardDecl
    :   ^(FORMAL_PARAM_STD_DECL lms=localModifierList t=type vdid=variableDeclaratorId)
        -> formal_param_decl(modList={$lms.st}, type={$t.st}, declID={$vdid.st})
        //-> template(t={$text}) "/*fpsd*/ <t>"
    ;
    
formalParameterVarargDecl
    :   ^(FORMAL_PARAM_VARARG_DECL localModifierList type variableDeclaratorId)
        -> template(t={$text}) "/*fpvd*/ <t>"
    ;
    
qualifiedIdentifier
    :   IDENT
        -> template(t={$text}) "<t>"
    |   ^(DOT qualifiedIdentifier IDENT)
        -> template(t={$text}) "<t>"
    ;
    
block
@init { boolean emptyBlock = true; }
    :   ^(BLOCK_SCOPE (b+=blockStatement)*)
        { emptyBlock = ($b == null || $b.size() == 0); }
        -> {emitCC() && emptyBlock}? template(bsl={$b}) "{ }"
        -> {emitCC()}? block_cc(bsl={$b})
        ->
    ;
    
blockStatement
    :   localVariableDeclaration
        -> {$localVariableDeclaration.st}
    |   typeDeclaration
        -> {$typeDeclaration.st}
    |   statement
        -> {$statement.st}
    ;


localVariableDeclaration
    :   ^(PRIMITIVE_VAR_DECLARATION localModifierList simpleType variableDeclaratorList)
        -> primitive_var_decl(
            modList={null},
            type={$simpleType.st},
            declList={$variableDeclaratorList.st})
    |   ^(OBJECT_VAR_DECLARATION localModifierList objectType variableDeclaratorList)
        -> object_var_decl(
            modList={null},
            type={$objectType.st},
            declList={$variableDeclaratorList.st})
    ;


statement
    :   block
        -> {$block.st}
    |   ^(ASSERT expression expression?)
        -> template(t={$text}) "<t>"
    |   ^(IF parenthesizedExpression statement statement?)
        -> template(t={$text}) "<t>"
    |   ^(FOR forInit forCondition forUpdater statement)
        -> template(t={$text}) "<t>"
    |   ^(FOR_EACH localModifierList type IDENT expression statement) 
        -> template(t={$text}) "<t>"
    |   ^(WHILE parenthesizedExpression statement)
        -> template(t={$text}) "<t>"
    |   ^(DO statement parenthesizedExpression)
        -> template(t={$text}) "<t>"
    |   ^(TRY block catches? block?)  // The second optional block is the optional finally block.
        -> template(t={$text}) "<t>"
    |   ^(SWITCH parenthesizedExpression switchBlockLabels)
        -> template(t={$text}) "<t>"
    |   ^(SYNCHRONIZED parenthesizedExpression block)
        -> template(t={$text}) "<t>"
    |   ^(RETURN expression?)
        -> template(t={$text}) "<t>"
    |   ^(THROW expression)
        -> template(t={$text}) "<t>"
    |   ^(BREAK IDENT?)
        -> template(t={$text}) "<t>"
    |   ^(CONTINUE IDENT?)
        -> template(t={$text}) "<t>"
    |   ^(LABELED_STATEMENT IDENT statement)
        -> template(t={$text}) "<t>"
    |   expression
        -> template(t={$text}) "<t>"
    |   ^(EMBED STRING_LITERAL EMBED_BLOCK)
        ->  embed_cc(str={$STRING_LITERAL.text}, blk={$EMBED_BLOCK.text})
    |   SEMI // Empty statement.
        -> template(t={$text}) "<t>"
    ;
        
catches
    :   ^(CATCH_CLAUSE_LIST catchClause+)
        -> template(t={$text}) "<t>"
    ;
    
catchClause
    :   ^(CATCH formalParameterStandardDecl block)
        -> template(t={$text}) "<t>"
    ;

switchBlockLabels
    :   ^(SWITCH_BLOCK_LABEL_LIST switchCaseLabel* switchDefaultLabel? switchCaseLabel*)
        -> template(t={$text}) "<t>"
    ;
        
switchCaseLabel
    :   ^(CASE expression blockStatement*)
        -> template(t={$text}) "<t>"
    ;
    
switchDefaultLabel
    :   ^(DEFAULT blockStatement*)
        -> template(t={$text}) "<t>"
    ;
    
forInit
    :   ^(FOR_INIT (localVariableDeclaration | expression*)?)
        -> template(t={$text}) "<t>"
    ;
    
forCondition
    :   ^(FOR_CONDITION expression?)
        -> template(t={$text}) "<t>"
    ;
    
forUpdater
    :   ^(FOR_UPDATE expression*)
        -> template(t={$text}) "<t>"
    ;
    
// EXPRESSIONS

parenthesizedExpression
    :   ^(PARENTESIZED_EXPR expression)
        -> template(t={$text}) "<t>"
    ;
    
expression
    :   ^(EXPR expr)
        -> template(t={$text}) "<t>"
    ;

expr
    :   ^(ASSIGN expr expr)
        -> template(t={$text}) "<t>"
    |   ^(PLUS_ASSIGN expr expr)
        -> template(t={$text}) "<t>"
    |   ^(MINUS_ASSIGN expr expr)
        -> template(t={$text}) "<t>"
    |   ^(STAR_ASSIGN expr expr)
        -> template(t={$text}) "<t>"
    |   ^(DIV_ASSIGN expr expr)
        -> template(t={$text}) "<t>"
    |   ^(AND_ASSIGN expr expr)
        -> template(t={$text}) "<t>"
    |   ^(OR_ASSIGN expr expr)
        -> template(t={$text}) "<t>"
    |   ^(XOR_ASSIGN expr expr)
        -> template(t={$text}) "<t>"
    |   ^(MOD_ASSIGN expr expr)
        -> template(t={$text}) "<t>"
    |   ^(BIT_SHIFT_RIGHT_ASSIGN expr expr)
        -> template(t={$text}) "<t>"
    |   ^(SHIFT_RIGHT_ASSIGN expr expr)
        -> template(t={$text}) "<t>"
    |   ^(SHIFT_LEFT_ASSIGN expr expr)
        -> template(t={$text}) "<t>"
    |   ^(QUESTION expr expr expr)
        -> template(t={$text}) "<t>"
    |   ^(LOGICAL_OR expr expr)
        -> template(t={$text}) "<t>"
    |   ^(LOGICAL_AND expr expr)
        -> template(t={$text}) "<t>"
    |   ^(OR expr expr)
        -> template(t={$text}) "<t>"
    |   ^(XOR expr expr)
        -> template(t={$text}) "<t>"
    |   ^(AND expr expr)
        -> template(t={$text}) "<t>"
    |   ^(EQUAL expr expr)
        -> template(t={$text}) "<t>"
    |   ^(NOT_EQUAL expr expr)
        -> template(t={$text}) "<t>"
    |   ^(INSTANCEOF expr type)
        -> template(t={$text}) "<t>"
    |   ^(LESS_OR_EQUAL expr expr)
        -> template(t={$text}) "<t>"
    |   ^(GREATER_OR_EQUAL expr expr)
        -> template(t={$text}) "<t>"
    |   ^(BIT_SHIFT_RIGHT expr expr)
        -> template(t={$text}) "<t>"
    |   ^(SHIFT_RIGHT expr expr)
        -> template(t={$text}) "<t>"
    |   ^(GREATER_THAN expr expr)
        -> template(t={$text}) "<t>"
    |   ^(SHIFT_LEFT expr expr)
        -> template(t={$text}) "<t>"
    |   ^(LESS_THAN expr expr)
        -> template(t={$text}) "<t>"
    |   ^(PLUS expr expr)
        -> template(t={$text}) "<t>"
    |   ^(MINUS expr expr)
        -> template(t={$text}) "<t>"
    |   ^(STAR expr expr)
        -> template(t={$text}) "<t>"
    |   ^(DIV expr expr)
        -> template(t={$text}) "<t>"
    |   ^(MOD expr expr)
        -> template(t={$text}) "<t>"
    |   ^(UNARY_PLUS expr)
        -> template(t={$text}) "<t>"
    |   ^(UNARY_MINUS expr)
        -> template(t={$text}) "<t>"
    |   ^(PRE_INC expr)
        -> template(t={$text}) "<t>"
    |   ^(PRE_DEC expr)
        -> template(t={$text}) "<t>"
    |   ^(POST_INC expr)
        -> template(t={$text}) "<t>"
    |   ^(POST_DEC expr)
        -> template(t={$text}) "<t>"
    |   ^(NOT expr)
        -> template(t={$text}) "<t>"
    |   ^(LOGICAL_NOT expr)
        -> template(t={$text}) "<t>"
    |   ^(CAST_EXPR type expr)
        -> template(t={$text}) "<t>"
    |   primaryExpression
        -> template(t={$text}) "<t>"
    ;
    
primaryExpression
    :   ^(  DOT
            (   primaryExpression
                (   IDENT
                |   THIS
                |   SUPER
                |   innerNewExpression
                |   CLASS
                )
            |   primitiveType CLASS
            |   VOID CLASS
            )
        )
        -> template(t={$text}) "<t>"
    |   parenthesizedExpression
        -> template(t={$text}) "<t>"
    |   IDENT
        -> template(t={$text}) "<t>"
    |   ^(METHOD_CALL primaryExpression genericTypeArgumentList? arguments)
        -> template(t={$text}) "<t>"
    |   explicitConstructorCall
        -> template(t={$text}) "<t>"
    |   ^(ARRAY_ELEMENT_ACCESS primaryExpression expression)
        -> template(t={$text}) "<t>"
    |   literal
        -> template(t={$text}) "<t>"
    |   newExpression
        -> template(t={$text}) "<t>"
    |   THIS
        -> template(t={$text}) "<t>"
    |   arrayTypeDeclarator
        -> template(t={$text}) "<t>"
    |   SUPER
        -> template(t={$text}) "<t>"
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

newExpression
    :   ^(  STATIC_ARRAY_CREATOR
            (   primitiveType newArrayConstruction
            |   genericTypeArgumentList? qualifiedTypeIdent newArrayConstruction
            )
        )
        -> template(t={$text}) "<t>"
    |   ^(CLASS_CONSTRUCTOR_CALL genericTypeArgumentList? qualifiedTypeIdent arguments classTopLevelScope?)
        -> template(t={$text}) "<t>"
    ;

innerNewExpression // something like 'InnerType innerType = outer.new InnerType();'
    :   ^(CLASS_CONSTRUCTOR_CALL genericTypeArgumentList? IDENT arguments classTopLevelScope?)
        -> template(t={$text}) "<t>"
    ;
    
newArrayConstruction
    :   arrayDeclaratorList arrayInitializer
        -> template(t={$text}) "<t>"
    |   expression+ arrayDeclaratorList?
        -> template(t={$text}) "<t>"
    ;

arguments
    :   ^(ARGUMENT_LIST expression*)
        -> template(t={$text}) "<t>"
    ;

literal 
    :   HEX_LITERAL
        -> template(t={$text}) "<t>"
    |   OCTAL_LITERAL
        -> template(t={$text}) "<t>"
    |   DECIMAL_LITERAL
        -> template(t={$text}) "<t>"
    |   FLOATING_POINT_LITERAL
        -> template(t={$text}) "<t>"
    |   CHARACTER_LITERAL
        -> template(t={$text}) "<t>"
    |   STRING_LITERAL
        -> template(t={$text}) "<t>"
    |   TRUE
        -> template(t={$text}) "<t>"
    |   FALSE
        -> template(t={$text}) "<t>"
    |   NULL
        -> template(t={$text}) "<t>"
    ;

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
    private boolean debug() { return translator_.debug(); }

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
    this.symtab_ = symtab;
    this.translator_ = symtab.translator;
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
            pd={$p.st}, ids={$i}, tds={$t.st}, cb={closingBraces}, debug={debug()})
        -> {emitCI()}? charjSource_ci(pd={$p.st}, ids={$i}, tds={$t.st}, debug={debug()})
        -> {emitH()}? charjSource_h(
            pd={$p.st}, ids={$i}, tds={$t.st}, cb={closingBraces}, debug={debug()})
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
        -> template(t={$text}) "/*INTERFACE-not implemented*/ <t>"
    |   ^(ENUM modifierList IDENT implementsClause? enumTopLevelScope)
        -> template(t={$text}) "/*ENUM-not implemented*/ <t>"
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
        -> template(t={$text}) "/*IMPLEMENTS_CLAUSE-not implemented*/ <t>"
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

enumTopLevelScope
    :   ^(ENUM_TOP_LEVEL_SCOPE enumConstant+ classTopLevelScope?)
        -> template(t={$text}) "/*enumTopLevelScope-not implemented*/ <t>"
    ;
    
enumConstant
    :   ^(IDENT arguments? classTopLevelScope?)
        -> template(t={$text}) "/*enumConstant-not implemented*/ <t>"
    ;
    
    
classTopLevelScope
    :   ^(CLASS_TOP_LEVEL_SCOPE (csd+=classScopeDeclarations)*) 
        -> classTopLevelScope(classScopeDeclarations={$csd})
    ;
    
classScopeDeclarations
@init { boolean entry = false; }
    :   ^(CLASS_INSTANCE_INITIALIZER block)
        -> {$block.st}
    |   ^(CLASS_STATIC_INITIALIZER block)
        -> {$block.st}
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
    |   typeDeclaration
        -> {$typeDeclaration.st}
    ;
    
interfaceTopLevelScope
    :   ^(INTERFACE_TOP_LEVEL_SCOPE interfaceScopeDeclarations*)
        -> template(t={$text}) "/*interfaceTopLevelScope-not implemented */ <t>"
    ;
    
interfaceScopeDeclarations
    :   ^(FUNCTION_METHOD_DECL modifierList genericTypeParameterList? type IDENT formalParameterList arrayDeclaratorList? throwsClause?)
        -> template(t={$text}) "/*interfaceScopeDeclarations-not implemented */ <t>"
    |   ^(VOID_METHOD_DECL modifierList genericTypeParameterList? IDENT formalParameterList throwsClause?)
        -> template(t={$text}) "/*interfaceScopeDeclarations-not implemented */ <t>"
        // Interface constant declarations have been switched to variable
        // declarations by Charj.g; the parser has already checked that
        // there's an obligatory initializer.
    |   ^(PRIMITIVE_VAR_DECLARATION modifierList simpleType variableDeclaratorList)
        -> template(t={$text}) "/*interfaceScopeDeclarations-not implemented */ <t>"
    |   ^(OBJECT_VAR_DECLARATION modifierList objectType variableDeclaratorList)
        -> template(t={$text}) "/*interfaceScopeDeclarations-not implemented */ <t>"
    |   typeDeclaration
        -> {$typeDeclaration.st}
    ;

variableDeclaratorList
    :   ^(VAR_DECLARATOR_LIST (var_decls+=variableDeclarator)+)
        -> var_decl_list(var_decls={$var_decls})
    ;

variableDeclarator
    :   ^(VAR_DECLARATOR id=variableDeclaratorId initializer=variableInitializer?)
        -> var_decl(id={$id.st}, initializer={$initializer.st})
    ;
    
variableDeclaratorId
    :   ^(IDENT adl=arrayDeclaratorList?)
        -> var_decl_id(id={$IDENT.text}, arrayDeclList={$adl.st})
    ;

variableInitializer
    :   arrayInitializer
        -> {$arrayInitializer.st}
    |   expression
        -> {$expression.st}
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
        -> template(t={$text}) "/* arrayInitializer-not implemented */ <t>"
    ;

throwsClause
    :   ^(THROWS_CLAUSE qualifiedIdentifier+)
        -> template(t={$text}) "/* throwsClause-not implemented */ <t>"
    ;

modifierList
    :   ^(MODIFIER_LIST (m+=modifier)*)
        -> template(mod={$m}) "<mod; separator=\" \">"
    ;

modifier
@init {
$st = %{$start.getText()};
}
    :   PUBLIC
    |   PROTECTED
    |   PRIVATE
    |   ENTRY
    |   STATIC
    |   ABSTRACT
    |   NATIVE
    |   SYNCHRONIZED
    |   TRANSIENT
    |   VOLATILE
    |   localModifier
        -> {$localModifier.st}
    ;

localModifierList
    :   ^(LOCAL_MODIFIER_LIST (m+=localModifier)*)
        -> template(mod={$m}) "<mod; separator=\" \">"
    ;

localModifier
    :   FINAL
        -> {%{$start.getText()}}
    ;

    
type
    :   simpleType
        -> {$simpleType.st}
    |   objectType 
        -> {$objectType.st}
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
    :   ^(QUALIFIED_TYPE_IDENT (t+=typeIdent)+) 
        -> template(types={$t}) "<types; separator=\".\">"
    ;

typeIdent
    :   ^(IDENT genericTypeArgumentList?)
        -> typeIdent(typeID={$IDENT.text}, generics={$genericTypeArgumentList.st})
    ;

primitiveType
@init {
$st = %{$start.getText()};
}
    :   BOOLEAN
    |   CHAR
    |   BYTE
    |   SHORT
    |   INT
    |   LONG
    |   FLOAT
    |   DOUBLE
    ;

genericTypeArgumentList
    :   ^(GENERIC_TYPE_ARG_LIST (gta+=genericTypeArgument)+)
        -> template(gtal={$gta}) "\<<gtal; separator=\", \">\>"
    ;
    
genericTypeArgument
    :   type
        -> {$type.st}
    |   ^(QUESTION genericWildcardBoundType?)
        -> template(t={$text}) "/* genericTypeArgument: wildcard bound types not implemented */ <t>"
    ;

genericWildcardBoundType                                                                                                                      
    :   ^(EXTENDS type)
        -> template(t={$text}) "/* genericWildcardBoundType not implemented */ <t>"
    |   ^(SUPER type)
        -> template(t={$text}) "/* genericWildcardBoundType not implemented */ <t>"
    ;

formalParameterList
    :   ^(FORMAL_PARAM_LIST (fpsd+=formalParameterStandardDecl)* fpvd=formalParameterVarargDecl?)
        -> formal_param_list(sdecl={$fpsd}, vdecl={$fpvd.st})
    ;
    
formalParameterStandardDecl
    :   ^(FORMAL_PARAM_STD_DECL lms=localModifierList t=type vdid=variableDeclaratorId)
        -> formal_param_decl(modList={$lms.st}, type={$t.st}, declID={$vdid.st})
    ;
    
formalParameterVarargDecl
    :   ^(FORMAL_PARAM_VARARG_DECL localModifierList type variableDeclaratorId)
        -> template(t={$text}) "/*formal parameter varargs not implemented*/ <t>"
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
    |   ^(ASSERT cond=expression msg=expression?)
        -> assert(cond={$cond.st}, msg={$msg.st})
    |   ^(IF parenthesizedExpression then=statement else_=statement?)
        -> if(cond={$parenthesizedExpression.st}, then={$then.st}, else_={$else_.st})
    |   ^(FOR forInit forCondition forUpdater s=statement)
        -> for(initializer={$forInit.st}, cond={$forCondition.st}, update={$forUpdater.st}, body={$s.st})
    |   ^(FOR_EACH localModifierList type IDENT expression statement) 
        -> template(t={$text}) "/* foreach not implemented */ <t>"
    |   ^(WHILE pe=parenthesizedExpression s=statement)
        -> while(cond={$pe.st}, body={$s.st})
    |   ^(DO s=statement pe=parenthesizedExpression)
        -> dowhile(cond={$pe.st}, block={$s.st})
    |   ^(TRY block catches? block?)  // The second optional block is the finally block.
        -> template(t={$text}) "/* try/catch not implemented */ <t>"
    |   ^(SWITCH pe=parenthesizedExpression sbls=switchBlockLabels)
        -> switch(expr={$pe.st}, labels={$sbls.st})
    |   ^(SYNCHRONIZED parenthesizedExpression block)
        -> template(t={$text}) "/* synchronized not implemented */ <t>"
    |   ^(RETURN e=expression?)
        -> return(val={$e.st})
    |   ^(THROW expression)
        -> template(t={$text}) "/* throw not implemented */ <t>"
    |   ^(BREAK IDENT?)
        -> template() "break;" // TODO: support labeling
    |   ^(CONTINUE IDENT?)
        -> template() "continue" // TODO: support labeling
    |   ^(LABELED_STATEMENT i=IDENT s=statement)
        -> label(text={$i.text}, stmt={$s.st})
    |   expression
        -> {$expression.st}
    |   ^(EMBED STRING_LITERAL EMBED_BLOCK)
        ->  embed_cc(str={$STRING_LITERAL.text}, blk={$EMBED_BLOCK.text})
    |   SEMI // Empty statement.
        -> {%{$start.getText()}}
    ;
        
catches
    :   ^(CATCH_CLAUSE_LIST catchClause+)
        -> template(t={$text}) "/* catch not implemented */ <t>"
    ;
    
catchClause
    :   ^(CATCH formalParameterStandardDecl block)
        -> template(t={$text}) "/* catchClause not implemented */ <t>"
    ;

switchBlockLabels
    :   ^(SWITCH_BLOCK_LABEL_LIST (l+=switchCaseLabel)*)
        -> template(labels={$l}) "<labels; separator=\"\n\">"
    ;
        
switchCaseLabel
    :   ^(CASE expression (b+=blockStatement)*)
        -> case(expr={$expression.st}, block={$b})
    |   ^(DEFAULT (b+=blockStatement)*)
        -> template(block={$b}) "default: <block>"
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
        -> {$expr.st}
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
        -> {$primaryExpression.st}
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
        -> {$parenthesizedExpression.st}
    |   IDENT
        -> {%{$start.getText()}}
    |   ^(METHOD_CALL primaryExpression genericTypeArgumentList? arguments)
        -> template(t={$text}) "<t>"
    |   explicitConstructorCall
        -> {$explicitConstructorCall.st}
    |   ^(ARRAY_ELEMENT_ACCESS primaryExpression expression)
        -> template(t={$text}) "<t>"
    |   literal
        -> {$literal.st}
    |   newExpression
        -> {$newExpression.st}
    |   THIS
        -> {%{$start.getText()}}
    |   arrayTypeDeclarator
        -> {$arrayTypeDeclarator.st}
    |   SUPER
        -> {%{$start.getText()}}
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


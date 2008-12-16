/**
 * The semantic phase walks the tree and builds the symbol table, handles
 * all the imports, and does the semantic checks. The resulting tree and
 * symbol table are used by the emitter to generate the output. 
 */

tree grammar CharjSemantics;

options {
    backtrack = true; 
    memoize = true;
    tokenVocab = Charj;
    ASTLabelType = CharjAST;
}

scope ScopeStack {
    Scope current;
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

    Translator translator;

    boolean m_messageCollectionEnabled = false;
    private boolean m_hasErrors = false;
    List<String> m_messages;

    /**
     *  Switches error message collection on or off.
     *
     *  The standard destination for parser error messages is 
     *  <code>System.err</code>.
     *  However, if <code>true</code> gets passed to this method this default
     *  behaviour will be switched off and all error messages will be collected
     *  instead of written to anywhere.
     *
     *  The default value is <code>false</code>.
     *
     *  @param newState  <code>true</code> if error messages should be collected.
     */
    public void enableErrorMessageCollection(boolean newState) {
        m_messageCollectionEnabled = newState;
        if (m_messages == null && m_messageCollectionEnabled) {
            m_messages = new ArrayList<String>();
        }
    }
    
    /**
     *  Collects an error message or passes the error message to <code>
     *  super.emitErrorMessage(...)</code>.
     *
     *  The actual behaviour depends on whether collecting error messages
     *  has been enabled or not.
     *
     *  @param message  The error message.
     */
     @Override
    public void emitErrorMessage(String message) {
        if (m_messageCollectionEnabled) {
            m_messages.add(message);
        } else {
            super.emitErrorMessage(message);
        }
    }
    
    /**
     *  Returns collected error messages.
     *
     *  @return  A list holding collected error messages or <code>null</code> if
     *           collecting error messages hasn't been enabled. Of course, this
     *           list may be empty if no error message has been emited.
     */
    public List<String> getMessages() {
        return m_messages;
    }
    
    /**
     *  Tells if parsing a Charj source has caused any error messages.
     *
     *  @return  <code>true</code> if parsing a Charj source has caused at 
     *           least one error message.
     */
    public boolean hasErrors() {
        return m_hasErrors;
    }

    /**
     *  Test a list of CharjAST nodes to see if any of them has the given token
     *  type.
     */
    public boolean listContainsToken(List<CharjAST> list, int tokenType) {
        for (CharjAST node : list) {
            if (node.token.getType() == tokenType) {
                return true;
            }
        }
        return false;
    }

    public void importPackages(ClassSymbol cs, List<CharjAST> imports) {
        if (imports == null) {
            return;
        }

        for (CharjAST pkg : imports) {
            String pkgName = input.getTokenStream().toString(
                    pkg.getTokenStartIndex(),
                    pkg.getTokenStopIndex());
            // find imported class and add to cs.imports
            PackageScope p = cs.importPackage(pkgName);
            if (p == null) {
                translator.error(
                    this, 
                    "package " + pkgName + " not found.",
                    pkg);
            }
        }
    }

}

// Starting point for parsing a Charj file.
charjSource[SymbolTable _symtab] returns [ClassSymbol cs]
scope ScopeStack; // default scope
@init {
    symtab = _symtab;
    $ScopeStack::current = symtab.getDefaultPkg();
}
    // TODO: go back to allowing multiple type definitions per file, check that
    // there is exactly one public type and return that one.
    :   ^(CHARJ_SOURCE 
        (packageDeclaration)? 
        (importDeclaration)* 
        (typeDeclaration))
        //(typeDeclaration)*)
        { $cs = null; }
    ;

// note: no new scope here--this replaces the default scope
packageDeclaration
@init { 
    List<String> names = null; 
}
    :   ^(PACKAGE qualifiedIdentifier)  {
            String packageName = $qualifiedIdentifier.text;
            PackageScope ps = symtab.resolvePackage(packageName);
            if (ps == null) {
                ps = symtab.definePackage(packageName);
                symtab.addScope(ps);
            }
            currentPackage = ps;
            $ScopeStack::current = ps;
            $qualifiedIdentifier.start.symbol = ps;
        }
    ;
    
importDeclaration
    :   ^(IMPORT STATIC? qualifiedIdentifier DOTSTAR?)
    ;
    
typeDeclaration returns [ClassSymbol sym]
scope ScopeStack; // top-level type scope
    :   ^(CLASS m=modifierList IDENT g=genericTypeParameterList? 
                e=classExtendsClause? i=implementsClause? c=classTopLevelScope) 
        {
            Scope outerScope = $ScopeStack[-1]::current;
            $sym = new ClassSymbol(symtab, $IDENT.text, null, outerScope);
            outerScope.define($sym.name, $sym);
            currentClass = $sym;
            $sym.definition = $typeDeclaration.start;
            $sym.definitionTokenStream = input.getTokenStream();
            $IDENT.symbol = $sym;
            $ScopeStack::current = $sym;
        }
    |   ^(INTERFACE modifierList IDENT genericTypeParameterList? 
                interfaceExtendsClause? interfaceTopLevelScope)
    |   ^(ENUM modifierList IDENT implementsClause? enumTopLevelScope)
    ;


classExtendsClause
    :   ^(EXTENDS_CLAUSE t=type) 
    ;   

interfaceExtendsClause 
    :   ^(EXTENDS_CLAUSE (type)+) 
    ;   
    
implementsClause
    :   ^(IMPLEMENTS_CLAUSE type+)
    ;
        
genericTypeParameterList
    :   ^(GENERIC_TYPE_PARAM_LIST genericTypeParameter+)
    ;

genericTypeParameter
    :   ^(IDENT bound?)
    ;
        
bound
    :   ^(EXTENDS_BOUND_LIST type+)
    ;

enumTopLevelScope
    :   ^(ENUM_TOP_LEVEL_SCOPE enumConstant+ classTopLevelScope?)
    ;
    
enumConstant
    :   ^(IDENT arguments? classTopLevelScope?)
    ;
    
    
classTopLevelScope
    :   ^(CLASS_TOP_LEVEL_SCOPE (classScopeDeclarations)*) 
    ;
    
classScopeDeclarations
    :   ^(CLASS_INSTANCE_INITIALIZER block)
    |   ^(CLASS_STATIC_INITIALIZER block)
    |   ^(FUNCTION_METHOD_DECL m=modifierList g=genericTypeParameterList? 
            ty=type IDENT f=formalParameterList a=arrayDeclaratorList? 
            tc=throwsClause? b=block?)
    |   ^(VOID_METHOD_DECL m=modifierList g=genericTypeParameterList? IDENT 
            f=formalParameterList t=throwsClause? b=block?)
    |   ^(PRIMITIVE_VAR_DECLARATION modifierList simpleType variableDeclaratorList)
    |   ^(OBJECT_VAR_DECLARATION modifierList objectType variableDeclaratorList)
    |   ^(CONSTRUCTOR_DECL m=modifierList g=genericTypeParameterList? IDENT f=formalParameterList 
            t=throwsClause? b=block)
    |   d=typeDeclaration
    ;
    
interfaceTopLevelScope
    :   ^(INTERFACE_TOP_LEVEL_SCOPE interfaceScopeDeclarations*)
    ;
    
interfaceScopeDeclarations
    :   ^(FUNCTION_METHOD_DECL modifierList genericTypeParameterList? type IDENT formalParameterList arrayDeclaratorList? throwsClause?)
    |   ^(VOID_METHOD_DECL modifierList genericTypeParameterList? IDENT formalParameterList throwsClause?)
        // Interface constant declarations have been switched to variable
        // declarations by Charj.g; the parser has already checked that
        // there's an obligatory initializer.
    |   ^(PRIMITIVE_VAR_DECLARATION modifierList simpleType variableDeclaratorList)
    |   ^(OBJECT_VAR_DECLARATION modifierList objectType variableDeclaratorList)
    |   typeDeclaration
    ;

variableDeclaratorList
    :   ^(VAR_DECLARATOR_LIST variableDeclarator+)
    ;

variableDeclarator
    :   ^(VAR_DECLARATOR variableDeclaratorId variableInitializer?)
    ;
    
variableDeclaratorId
    :   ^(IDENT arrayDeclaratorList?)
    ;

variableInitializer
    :   arrayInitializer
    |   expression
    ;

arrayDeclarator
    :   LBRACK RBRACK
    ;

arrayDeclaratorList
    :   ^(ARRAY_DECLARATOR_LIST ARRAY_DECLARATOR*)  
    ;
    
arrayInitializer
    :   ^(ARRAY_INITIALIZER variableInitializer*)
    ;

throwsClause
    :   ^(THROWS_CLAUSE qualifiedIdentifier+)
    ;

modifierList
    :   ^(MODIFIER_LIST (modifier)*)
    ;

modifier
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
    ;

localModifierList
    :   ^(LOCAL_MODIFIER_LIST localModifier*)
    ;

localModifier
    :   FINAL
    ;

type
    :   simpleType
    |   objectType 
    ;

simpleType
    :   ^(TYPE primitiveType arrayDeclaratorList?)
    ;
    
objectType
    :   ^(TYPE qualifiedTypeIdent arrayDeclaratorList?)
    ;

qualifiedTypeIdent
    :   ^(QUALIFIED_TYPE_IDENT typeIdent+) 
    ;

typeIdent
    :   ^(IDENT genericTypeArgumentList?)
    ;

primitiveType
    :   BOOLEAN     { $start.symbol = new Symbol(symtab, "bool_primitive", symtab.resolveBuiltinType("bool")); }
    |   CHAR        { $start.symbol = new Symbol(symtab, "char_primitive", symtab.resolveBuiltinType("char")); }
    |   BYTE        { $start.symbol = new Symbol(symtab, "byte_primitive", symtab.resolveBuiltinType("char")); }
    |   SHORT       { $start.symbol = new Symbol(symtab, "short_primitive", symtab.resolveBuiltinType("short")); }
    |   INT         { $start.symbol = new Symbol(symtab, "int_primitive", symtab.resolveBuiltinType("int")); }
    |   LONG        { $start.symbol = new Symbol(symtab, "long_primitive", symtab.resolveBuiltinType("long")); }
    |   FLOAT       { $start.symbol = new Symbol(symtab, "float_primitive", symtab.resolveBuiltinType("float")); }
    |   DOUBLE      { $start.symbol = new Symbol(symtab, "double_primitive", symtab.resolveBuiltinType("double")); }
    ;

genericTypeArgumentList
    :   ^(GENERIC_TYPE_ARG_LIST genericTypeArgument+)
    ;
    
genericTypeArgument
    :   type
    |   ^(QUESTION genericWildcardBoundType?)
    ;

genericWildcardBoundType                                                                                                                      
    :   ^(EXTENDS type)
    |   ^(SUPER type)
    ;

formalParameterList
    :   ^(FORMAL_PARAM_LIST formalParameterStandardDecl* formalParameterVarargDecl?) 
    ;
    
formalParameterStandardDecl
    :   ^(FORMAL_PARAM_STD_DECL localModifierList type variableDeclaratorId)
    ;
    
formalParameterVarargDecl
    :   ^(FORMAL_PARAM_VARARG_DECL localModifierList type variableDeclaratorId)
    ;
    
qualifiedIdentifier
    :   IDENT
    |   ^(DOT qualifiedIdentifier IDENT)
    ;
    
block
    :   ^(BLOCK_SCOPE (blockStatement)*)
    ;
    
blockStatement
    :   localVariableDeclaration
    |   typeDeclaration
    |   statement
    ;
    
localVariableDeclaration
    :   ^(PRIMITIVE_VAR_DECLARATION localModifierList simpleType variableDeclaratorList)
    |   ^(OBJECT_VAR_DECLARATION localModifierList objectType variableDeclaratorList)
    ;


statement
    :   block
    |   ^(ASSERT expression expression?)
    |   ^(IF parenthesizedExpression statement statement?)
    |   ^(FOR forInit forCondition forUpdater statement)
    |   ^(FOR_EACH localModifierList type IDENT expression statement) 
    |   ^(WHILE parenthesizedExpression statement)
    |   ^(DO statement parenthesizedExpression)
    |   ^(TRY block catches? block?)  // The second optional block is the optional finally block.
    |   ^(SWITCH parenthesizedExpression switchBlockLabels)
    |   ^(SYNCHRONIZED parenthesizedExpression block)
    |   ^(RETURN expression?)
    |   ^(THROW expression)
    |   ^(BREAK IDENT?)
    |   ^(CONTINUE IDENT?)
    |   ^(LABELED_STATEMENT IDENT statement)
    |   expression
    |   ^(EMBED STRING_LITERAL EMBED_BLOCK)
    |   SEMI // Empty statement.
    ;
        
catches
    :   ^(CATCH_CLAUSE_LIST catchClause+)
    ;
    
catchClause
    :   ^(CATCH formalParameterStandardDecl block)
    ;

switchBlockLabels
    :   ^(SWITCH_BLOCK_LABEL_LIST switchCaseLabel* switchDefaultLabel? switchCaseLabel*)
    ;
        
switchCaseLabel
    :   ^(CASE expression blockStatement*)
    ;
    
switchDefaultLabel
    :   ^(DEFAULT blockStatement*)
    ;
    
forInit
    :   ^(FOR_INIT (localVariableDeclaration | expression*)?)
    ;
    
forCondition
    :   ^(FOR_CONDITION expression?)
    ;
    
forUpdater
    :   ^(FOR_UPDATE expression*)
    ;
    
// EXPRESSIONS

parenthesizedExpression
    :   ^(PARENTESIZED_EXPR expression)
    ;
    
expression
    :   ^(EXPR expr)
    ;

expr
    :   ^(ASSIGN expr expr)
    |   ^(PLUS_ASSIGN expr expr)
    |   ^(MINUS_ASSIGN expr expr)
    |   ^(STAR_ASSIGN expr expr)
    |   ^(DIV_ASSIGN expr expr)
    |   ^(AND_ASSIGN expr expr)
    |   ^(OR_ASSIGN expr expr)
    |   ^(XOR_ASSIGN expr expr)
    |   ^(MOD_ASSIGN expr expr)
    |   ^(BIT_SHIFT_RIGHT_ASSIGN expr expr)
    |   ^(SHIFT_RIGHT_ASSIGN expr expr)
    |   ^(SHIFT_LEFT_ASSIGN expr expr)
    |   ^(QUESTION expr expr expr)
    |   ^(LOGICAL_OR expr expr)
    |   ^(LOGICAL_AND expr expr)
    |   ^(OR expr expr)
    |   ^(XOR expr expr)
    |   ^(AND expr expr)
    |   ^(EQUAL expr expr)
    |   ^(NOT_EQUAL expr expr)
    |   ^(INSTANCEOF expr type)
    |   ^(LESS_OR_EQUAL expr expr)
    |   ^(GREATER_OR_EQUAL expr expr)
    |   ^(BIT_SHIFT_RIGHT expr expr)
    |   ^(SHIFT_RIGHT expr expr)
    |   ^(GREATER_THAN expr expr)
    |   ^(SHIFT_LEFT expr expr)
    |   ^(LESS_THAN expr expr)
    |   ^(PLUS expr expr)
    |   ^(MINUS expr expr)
    |   ^(STAR expr expr)
    |   ^(DIV expr expr)
    |   ^(MOD expr expr)
    |   ^(UNARY_PLUS expr)
    |   ^(UNARY_MINUS expr)
    |   ^(PRE_INC expr)
    |   ^(PRE_DEC expr)
    |   ^(POST_INC expr)
    |   ^(POST_DEC expr)
    |   ^(NOT expr)
    |   ^(LOGICAL_NOT expr)
    |   ^(CAST_EXPR type expr)
    |   primaryExpression
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
    |   parenthesizedExpression
    |   IDENT
    |   ^(METHOD_CALL primaryExpression genericTypeArgumentList? arguments)
    |   explicitConstructorCall
    |   ^(ARRAY_ELEMENT_ACCESS primaryExpression expression)
    |   literal
    |   newExpression
    |   THIS
    |   arrayTypeDeclarator
    |   SUPER
    ;
    
explicitConstructorCall
    :   ^(THIS_CONSTRUCTOR_CALL genericTypeArgumentList? arguments)
    |   ^(SUPER_CONSTRUCTOR_CALL primaryExpression? genericTypeArgumentList? arguments)
    ;

arrayTypeDeclarator
    :   ^(ARRAY_DECLARATOR (arrayTypeDeclarator | qualifiedIdentifier | primitiveType))
    ;

newExpression
    :   ^(  STATIC_ARRAY_CREATOR
            (   primitiveType newArrayConstruction
            |   genericTypeArgumentList? qualifiedTypeIdent newArrayConstruction
            )
        )
    |   ^(CLASS_CONSTRUCTOR_CALL genericTypeArgumentList? qualifiedTypeIdent arguments classTopLevelScope?)
    ;

innerNewExpression // something like 'InnerType innerType = outer.new InnerType();'
    :   ^(CLASS_CONSTRUCTOR_CALL genericTypeArgumentList? IDENT arguments classTopLevelScope?)
    ;
    
newArrayConstruction
    :   arrayDeclaratorList arrayInitializer
    |   expression+ arrayDeclaratorList?
    ;

arguments
    :   ^(ARGUMENT_LIST expression*)
    ;

literal 
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

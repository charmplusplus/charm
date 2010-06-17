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
    List<CharjAST> imports = new ArrayList<CharjAST>();
    AstModifier astmod = new AstModifier();

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

    public void addImport(CharjAST importNode) {
        imports.add(importNode);
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
charjSource[SymbolTable _symtab] returns [ClassSymbol cs]
scope ScopeStack; // default scope
@init {
    symtab = _symtab;
    $ScopeStack::current = symtab.getDefaultPkg();
}
    :   ^(CHARJ_SOURCE 
        (packageDeclaration)? 
        (importDeclaration
        | typeDeclaration { $cs = $typeDeclaration.sym; }
        | readonlyDeclaration)*)
    ;

// note: no new scope here--this replaces the default scope
packageDeclaration
@init { 
    List<String> names = null; 
    String packageName = "";
}
    :   ^(PACKAGE ((ids+=IDENT) { packageName += "." + $IDENT.text; })+)
        {
            packageName = packageName.substring(1);
            PackageScope ps = symtab.resolvePackage(packageName);
            if (ps == null) {
                ps = symtab.definePackage(packageName);
                symtab.addScope(ps);
            }
            currentPackage = ps;
            $ScopeStack::current = ps;
        }
    ;
    
importDeclaration
    :   ^(IMPORT qualifiedIdentifier '.*'?)
        { addImport($qualifiedIdentifier.start); }
    ;

readonlyDeclaration
    :   ^(READONLY localVariableDeclaration)
    ;

typeDeclaration returns [ClassSymbol sym]
scope ScopeStack; // top-level type scope
    :   ^(TYPE classType IDENT
            (^('extends' parent=type))? (^('implements' type+))?
            {
            }
            classScopeDeclaration*)
            {
                //System.out.println("Members for type " + $sym.name + ":");
                //for (Map.Entry<String, Symbol> entry : $sym.members.entrySet()) {
                //    System.out.println(entry.getKey());
                //}
            }
    |   ^('template' i1=IDENT* typeDeclaration)
        {
            // JL: Need to fill the templateArgs in ClassSymbol, and push this down
            // to the class subtree
        }
    |   ^('interface' IDENT (^('extends' type+))?  interfaceScopeDeclaration*)
    |   ^('enum' IDENT (^('implements' type+))? enumConstant+ classScopeDeclaration*)
    ;

classType
    :   CLASS
    |   chareType
    ;

chareType
    :   CHARE
    |   GROUP
    |   NODEGROUP
    |   MAINCHARE
    |   ^(CHARE_ARRAY ARRAY_DIMENSION)
    ;

enumConstant
    :   ^(IDENT arguments?)
    ;
    
classScopeDeclaration
scope ScopeStack;
    :   ^(FUNCTION_METHOD_DECL m=modifierList? g=genericTypeParameterList? 
            ty=type IDENT f=formalParameterList a=arrayDeclaratorList? 
            b=block?)
        {
        }
    |   ^(ENTRY_FUNCTION_DECL m=modifierList? g=genericTypeParameterList?
            ty=type IDENT formalParameterList a=arrayDeclaratorList? b=block)
        {
        }
    |   ^(PRIMITIVE_VAR_DECLARATION modifierList? simpleType
            ^(VAR_DECLARATOR_LIST field[$simpleType.type, false]+))
    |   ^(OBJECT_VAR_DECLARATION modifierList? objectType
            ^(VAR_DECLARATOR_LIST field[$objectType.type, false]+))
        {
        }
    |   ^(CONSTRUCTOR_DECL m=modifierList? g=genericTypeParameterList? IDENT f=formalParameterList 
            b=block)
        {
            if (astmod.isMigrationCtor($CONSTRUCTOR_DECL)) currentClass.migrationCtor = $CONSTRUCTOR_DECL;
            if (currentClass != null) {
                currentClass.constructor = $classScopeDeclaration.start;
            }
        }
    |   ^(ENTRY_CONSTRUCTOR_DECL m=modifierList? g=genericTypeParameterList? IDENT f=formalParameterList 
            b=block)
        {
            if (astmod.isMigrationCtor($ENTRY_CONSTRUCTOR_DECL)) currentClass.migrationCtor = $ENTRY_CONSTRUCTOR_DECL;
        }
    ;

field [Type type, boolean localdef]
    :   ^(VAR_DECLARATOR variableDeclaratorId[localdef] variableInitializer?)
    {
    }
    ;
    
interfaceScopeDeclaration
    :   ^(FUNCTION_METHOD_DECL modifierList? genericTypeParameterList? 
            type IDENT formalParameterList arrayDeclaratorList?)
        // Interface constant declarations have been switched to variable
        // declarations by Charj.g; the parser has already checked that
        // there's an obligatory initializer.
    |   ^(PRIMITIVE_VAR_DECLARATION modifierList? simpleType variableDeclaratorList[false])
    |   ^(OBJECT_VAR_DECLARATION modifierList? objectType variableDeclaratorList[false])
    ;

variableDeclaratorList[boolean localdef]
    :   ^(VAR_DECLARATOR_LIST variableDeclarator[localdef]+)
    ;

variableDeclarator[boolean localdef]
    :   ^(VAR_DECLARATOR variableDeclaratorId[localdef] variableInitializer?)
    ;
    
variableDeclaratorId[boolean localdef] returns [String ident]
    :   ^(IDENT domainExpression?
        { 
            if (currentClass != null && !localdef) {
                currentClass.initializers.add($variableDeclaratorId.start);
            }

            $ident = $IDENT.text;
        } )
    ;

rangeItem
    :   DECIMAL_LITERAL
    |   IDENT
    ;

rangeExpression
    :   ^(RANGE_EXPRESSION rangeItem)
    |   ^(RANGE_EXPRESSION rangeItem rangeItem)
    |   ^(RANGE_EXPRESSION rangeItem rangeItem rangeItem)
    ;

rangeList
    :   rangeExpression*
    ;

domainExpression
    :   ^(DOMAIN_EXPRESSION rangeList)
    ;

variableInitializer
    :   arrayInitializer
    |   expression
    ;

arrayDeclaratorList
    :   ^(ARRAY_DECLARATOR_LIST ARRAY_DECLARATOR*)  
    ;
    
arrayInitializer
    :   ^(ARRAY_INITIALIZER variableInitializer*)
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

modifierList
    :   ^(MODIFIER_LIST accessModifierList? localModifierList? charjModifierList? otherModifierList?)
    ;

modifier
    :   accessModifier
    |   localModifier
    |   charjModifier
    |   otherModifier
    ;

localModifierList
    :   ^(LOCAL_MODIFIER_LIST localModifier+)
    ;

accessModifierList
    :   ^(ACCESS_MODIFIER_LIST accessModifier+)
    ;

charjModifierList
    :   ^(CHARJ_MODIFIER_LIST charjModifier+)
    ;

otherModifierList
    :   ^(OTHER_MODIFIER_LIST otherModifier+)
    ;
    
localModifier
    :   FINAL
    |   STATIC
    |   VOLATILE
    ;

accessModifier
    :   PUBLIC
    |   PROTECTED
    |   PRIVATE
    ;

charjModifier
    :   ENTRY
    ;

otherModifier
    :   ABSTRACT
    |   NATIVE
    ;

type
    :   simpleType
    |   objectType
    |   VOID
    ;

simpleType returns [ClassSymbol type]
    :   ^(SIMPLE_TYPE primitiveType arrayDeclaratorList?)
        {
        }
    ;
    
objectType returns [ClassSymbol type]
    :   ^(OBJECT_TYPE qualifiedTypeIdent arrayDeclaratorList?)
    |   ^(REFERENCE_TYPE qualifiedTypeIdent arrayDeclaratorList?)
    |   ^(PROXY_TYPE qualifiedTypeIdent arrayDeclaratorList?)
    |   ^(POINTER_TYPE qualifiedTypeIdent arrayDeclaratorList?)
        {
        }
    ;

qualifiedTypeIdent returns [ClassSymbol type]
@init {
String name = "";
}
    :   ^(QUALIFIED_TYPE_IDENT (typeIdent {name += $typeIdent.name;})+) 
        {
        }
    ;

typeIdent returns [String name]
    :   ^(IDENT templateInstantiation?)
    ;

primitiveType
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
    :   ^(GENERIC_TYPE_ARG_LIST genericTypeArgument+)
    ;

templateArgList
    :   genericTypeArgument+
    ;

templateInstantiation
    :   ^(TEMPLATE_INST templateArgList)
    |   ^(TEMPLATE_INST templateInstantiation)
    ;
    
genericTypeArgument
    :   type
    |   '?'
    ;

formalParameterList
    :   ^(FORMAL_PARAM_LIST formalParameterStandardDecl* formalParameterVarargDecl?) 
    ;
    
formalParameterStandardDecl
    :   ^(FORMAL_PARAM_STD_DECL localModifierList? type variableDeclaratorId[false])
    ;
    
formalParameterVarargDecl
    :   ^(FORMAL_PARAM_VARARG_DECL localModifierList? type variableDeclaratorId[false])
    ;
    
// FIXME: is this rule right? Verify that this is ok, I expected something like:
// IDENT (^(DOT qualifiedIdentifier IDENT))*
qualifiedIdentifier
    :   IDENT
    |   ^(DOT qualifiedIdentifier IDENT)
    ;
    
block
    :   ^(BLOCK (blockStatement)*)
    ;
    
blockStatement
    :   localVariableDeclaration
    |   statement
    ;
    
localVariableDeclaration
    :   ^(PRIMITIVE_VAR_DECLARATION localModifierList? simpleType variableDeclaratorList[true])
    |   ^(OBJECT_VAR_DECLARATION localModifierList? objectType variableDeclaratorList[true])
        {
        }
    ;

statement
    : nonBlockStatement
    | block
    ;

nonBlockStatement
    :   ^(ASSERT expression expression?)
    |   ^(IF parenthesizedExpression block block?)
    |   ^(FOR forInit? FOR_EXPR expression? FOR_UPDATE expression* block)
    |   ^(FOR_EACH localModifierList? type IDENT expression block) 
    |   ^(WHILE parenthesizedExpression block)
    |   ^(DO block parenthesizedExpression)
    |   ^(SWITCH parenthesizedExpression switchCaseLabel*)
    |   ^(RETURN expression?)
    |   ^(THROW expression)
    |   ^(BREAK IDENT?) {
            if ($IDENT != null) {
                translator.error(this, "Labeled break not supported yet, ignoring.", $IDENT);
            }
        }
    |   ^(CONTINUE IDENT?) {
            if ($IDENT != null) {
                translator.error(this, "Labeled continue not supported yet, ignoring.", $IDENT);
            }
        }
    |   ^(LABELED_STATEMENT IDENT statement)
    |   expression
    |   ^('delete' expression)
    |   ^(EMBED STRING_LITERAL EMBED_BLOCK)
    |   ';' // Empty statement.
    |   ^(PRINT expression*)
    |   ^(PRINTLN expression*)
    |   ^(EXIT expression?)
    |   EXITALL
    ;
        
switchCaseLabel
    :   ^(CASE expression blockStatement*)
    |   ^(DEFAULT blockStatement*)
    ;
    
forInit
    :   localVariableDeclaration 
    |   expression+
    ;
    
// EXPRESSIONS

parenthesizedExpression
    :   ^(PAREN_EXPR expression)
    ;

expression
    :   ^(EXPR expr)
    ;

expr
    :   ^(ASSIGNMENT expr expr)
    |   ^(PLUS_EQUALS expr expr)
    |   ^(MINUS_EQUALS expr expr)
    |   ^(TIMES_EQUALS expr expr)
    |   ^(DIVIDE_EQUALS expr expr)
    |   ^(AND_EQUALS expr expr)
    |   ^(OR_EQUALS expr expr)
    |   ^(POWER_EQUALS expr expr)
    |   ^(MOD_EQUALS expr expr)
    |   ^('>>>=' expr expr)
    |   ^('>>=' expr expr)
    |   ^('<<=' expr expr)
    |   ^('?' expr expr expr)
    |   ^(OR expr expr)
    |   ^(AND expr expr)
    |   ^(BITWISE_OR expr expr)
    |   ^(POWER expr expr)
    |   ^(BITWISE_AND expr expr)
    |   ^(EQUALS expr expr)
    |   ^(NOT_EQUALS expr expr)
    |   ^(INSTANCEOF expr type)
    |   ^(LTE expr expr)
    |   ^(GTE expr expr)
    |   ^('>>>' expr expr)
    |   ^('>>' expr expr)
    |   ^(GT expr expr)
    |   ^('<<' expr expr)
    |   ^(LT expr expr)
    |   ^(PLUS expr expr)
    |   ^(MINUS expr expr)
    |   ^(TIMES expr expr)
    |   ^(DIVIDE expr expr)
    |   ^(MOD expr expr)
    |   ^(UNARY_PLUS expr)
    |   ^(UNARY_MINUS expr)
    |   ^(PRE_INC expr)
    |   ^(PRE_DEC expr)
    |   ^(POST_INC expr)
    |   ^(POST_DEC expr)
    |   ^(TILDE expr)
    |   ^(NOT expr)
    |   ^(CAST_EXPR type expr)
    |   primaryExpression
    ;
    
primaryExpression
    :   ^(DOT primaryExpression
                (   IDENT
                |   THIS
                |   SUPER
                )
        )
    |   ^(ARROW primaryExpression
                (   IDENT
                |   THIS
                |   SUPER
                )
        )
    |   parenthesizedExpression
    |   IDENT
    |   ^(METHOD_CALL primaryExpression genericTypeArgumentList? arguments)
    |   ^(ENTRY_METHOD_CALL primaryExpression genericTypeArgumentList? arguments)
    |   explicitConstructorCall
    |   ^(ARRAY_ELEMENT_ACCESS primaryExpression expression)
    |   literal
    |   newExpression
    |   THIS
    |   arrayTypeDeclarator
    |   SUPER
    |   GETNUMPES
    |   GETNUMNODES
    |   GETMYPE
    |   GETMYNODE
    |   GETMYRANK
    ;
    
explicitConstructorCall
    :   ^(THIS_CONSTRUCTOR_CALL genericTypeArgumentList? arguments)
    |   ^(SUPER_CONSTRUCTOR_CALL primaryExpression? genericTypeArgumentList? arguments)
    ;

arrayTypeDeclarator
    :   ^(ARRAY_DECLARATOR (arrayTypeDeclarator | qualifiedIdentifier | primitiveType))
    ;

newExpression
    :   ^(NEW_EXPRESSION arguments? domainExpression)
        {
            if (currentClass != null) {
                currentClass.initializers.add($newExpression.start);
            }
        }
    |   ^(NEW type arguments)
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


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
    // TODO: go back to allowing multiple type definitions per file, check that
    // there is exactly one public type and return that one.
    :   ^(CHARJ_SOURCE 
        (packageDeclaration)? 
        (importDeclarations) 
        (typeDeclaration[$importDeclarations.packageNames]))
        { $cs = $typeDeclaration.sym; }
    ;

// note: no new scope here--this replaces the default scope
packageDeclaration
@init { 
    List<String> names = null; 
}
    :   ^('package' qualifiedIdentifier)  {
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
    
importDeclarations returns [List<CharjAST> packageNames]
@init {
	packageNames = new ArrayList<CharjAST>();
}
    :   (^('import' qualifiedIdentifier './*'?)
		{ packageNames.add($qualifiedIdentifier.start); })*
    ;


typeDeclaration[List<CharjAST> imports] returns [ClassSymbol sym]
scope ScopeStack; // top-level type scope
    :   ^('class' IDENT (^('extends' type))? (^('implements' type+))? classScopeDeclaration*)
        {
            Scope outerScope = $ScopeStack[-1]::current;
            $sym = new ClassSymbol(symtab, $IDENT.text, null, outerScope);
            outerScope.define($sym.name, $sym);
            currentClass = $sym;
            $sym.definition = $typeDeclaration.start;
            $sym.definitionTokenStream = input.getTokenStream();
            $IDENT.symbol = $sym;
            $ScopeStack::current = $sym;
            importPackages($sym, $imports);
        }
    |   ^('interface' IDENT (^('extends' type+))?  interfaceScopeDeclaration*)
    |   ^('enum' IDENT (^('implements' type+))? enumConstant+ classScopeDeclaration*)
    |   ^(chareType IDENT (^('extends' type))? (^('implements' type+))? classScopeDeclaration*)
    |   ^('chare_array' ARRAY_DIMENSION IDENT (^('extends' type))? (^('implements' type+))? classScopeDeclaration*)
    ;

chareType
    :   'chare'
    |   'group'
    |   'nodegroup'
    ;

enumConstant
    :   ^(IDENT arguments?)
    ;
    
classScopeDeclaration
    :   ^(FUNCTION_METHOD_DECL m=modifierList? g=genericTypeParameterList? 
            ty=type IDENT f=formalParameterList a=arrayDeclaratorList? 
            b=block?)
    |   ^(VOID_METHOD_DECL m=modifierList? g=genericTypeParameterList? IDENT 
            f=formalParameterList b=block?)
    |   ^(PRIMITIVE_VAR_DECLARATION modifierList? simpleType variableDeclaratorList)
    |   ^(OBJECT_VAR_DECLARATION modifierList? objectType variableDeclaratorList)
    |   ^(CONSTRUCTOR_DECL m=modifierList? g=genericTypeParameterList? IDENT f=formalParameterList 
            b=block)
    ;
    
interfaceScopeDeclaration
    :   ^(FUNCTION_METHOD_DECL modifierList? genericTypeParameterList? 
            type IDENT formalParameterList arrayDeclaratorList?)
    |   ^(VOID_METHOD_DECL modifierList? genericTypeParameterList? IDENT formalParameterList)
        // Interface constant declarations have been switched to variable
        // declarations by Charj.g; the parser has already checked that
        // there's an obligatory initializer.
    |   ^(PRIMITIVE_VAR_DECLARATION modifierList? simpleType variableDeclaratorList)
    |   ^(OBJECT_VAR_DECLARATION modifierList? objectType variableDeclaratorList)
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
    :   ^(MODIFIER_LIST modifier+)
    ;

modifier
    :   'public'
    |   'protected'
    |   'private'
    |   'entry'
    |   'abstract'
    |   'native'
    |   localModifier
    ;

localModifierList
    :   ^(LOCAL_MODIFIER_LIST localModifier+)
    ;

localModifier
    :   'final'
    |   'static'
    |   'volatile'
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
    :   'boolean'     { $start.symbol = new Symbol(symtab, "bool_primitive", symtab.resolveBuiltinType("bool")); }
    |   'char'        { $start.symbol = new Symbol(symtab, "char_primitive", symtab.resolveBuiltinType("char")); }
    |   'byte'        { $start.symbol = new Symbol(symtab, "byte_primitive", symtab.resolveBuiltinType("char")); }
    |   'short'       { $start.symbol = new Symbol(symtab, "short_primitive", symtab.resolveBuiltinType("short")); }
    |   'int'         { $start.symbol = new Symbol(symtab, "int_primitive", symtab.resolveBuiltinType("int")); }
    |   'long'        { $start.symbol = new Symbol(symtab, "long_primitive", symtab.resolveBuiltinType("long")); }
    |   'float'       { $start.symbol = new Symbol(symtab, "float_primitive", symtab.resolveBuiltinType("float")); }
    |   'double'      { $start.symbol = new Symbol(symtab, "double_primitive", symtab.resolveBuiltinType("double")); }
    ;

genericTypeArgumentList
    :   ^(GENERIC_TYPE_ARG_LIST genericTypeArgument+)
    ;
    
genericTypeArgument
    :   type
    |   '?'
    ;

formalParameterList
    :   ^(FORMAL_PARAM_LIST formalParameterStandardDecl* formalParameterVarargDecl?) 
    ;
    
formalParameterStandardDecl
    :   ^(FORMAL_PARAM_STD_DECL localModifierList? type variableDeclaratorId)
    ;
    
formalParameterVarargDecl
    :   ^(FORMAL_PARAM_VARARG_DECL localModifierList? type variableDeclaratorId)
    ;
    
// FIXME: is this rule right? Verify that this is ok, I expected something like:
// IDENT (^('.' qualifiedIdentifier IDENT))*
qualifiedIdentifier
    :   IDENT
    |   ^('.' qualifiedIdentifier IDENT)
    ;
    
block
    :   ^(BLOCK (blockStatement)*)
    ;
    
blockStatement
    :   localVariableDeclaration
    |   statement
    ;
    
localVariableDeclaration
    :   ^(PRIMITIVE_VAR_DECLARATION localModifierList? simpleType variableDeclaratorList)
    |   ^(OBJECT_VAR_DECLARATION localModifierList? objectType variableDeclaratorList)
    ;

statement
    :   block
    |   ^('assert' expression expression?)
    |   ^('if' parenthesizedExpression statement statement?)
    |   ^('for' forInit? FOR_EXPR expression? FOR_UPDATE expression* statement)
    |   ^(FOR_EACH localModifierList? type IDENT expression statement) 
    |   ^('while' parenthesizedExpression statement)
    |   ^('do' statement parenthesizedExpression)
    |   ^('switch' parenthesizedExpression switchCaseLabel*)
    |   ^('return' expression?)
    |   ^('throw' expression)
    |   ^('break' IDENT?) {
            if ($IDENT != null) {
                translator.error(this, "Labeled break not supported yet, ignoring.", $IDENT);
            }
        }
    |   ^('continue' IDENT?) {
            if ($IDENT != null) {
                translator.error(this, "Labeled continue not supported yet, ignoring.", $IDENT);
            }
        }
    |   ^(LABELED_STATEMENT IDENT statement)
    |   expression
    |   ^('embed' STRING_LITERAL EMBED_BLOCK)
    |   ';' // Empty statement.
    ;
        
switchCaseLabel
    :   ^('case' expression blockStatement*)
    |   ^('default' blockStatement*)
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
    :   ^('=' expr expr)
    |   ^('+=' expr expr)
    |   ^('-=' expr expr)
    |   ^('*=' expr expr)
    |   ^('/=' expr expr)
    |   ^('&=' expr expr)
    |   ^('|=' expr expr)
    |   ^('^=' expr expr)
    |   ^('%=' expr expr)
    |   ^('>>>=' expr expr)
    |   ^('>>=' expr expr)
    |   ^('<<=' expr expr)
    |   ^('?' expr expr expr)
    |   ^('||' expr expr)
    |   ^('&&' expr expr)
    |   ^('|' expr expr)
    |   ^('^' expr expr)
    |   ^('&' expr expr)
    |   ^('==' expr expr)
    |   ^('!=' expr expr)
    |   ^('instanceof' expr type)
    |   ^('<=' expr expr)
    |   ^('>=' expr expr)
    |   ^('>>>' expr expr)
    |   ^('>>' expr expr)
    |   ^('>' expr expr)
    |   ^('<<' expr expr)
    |   ^('<' expr expr)
    |   ^('+' expr expr)
    |   ^('-' expr expr)
    |   ^('*' expr expr)
    |   ^('/' expr expr)
    |   ^('%' expr expr)
    |   ^(UNARY_PLUS expr)
    |   ^(UNARY_MINUS expr)
    |   ^(PRE_INC expr)
    |   ^(PRE_DEC expr)
    |   ^(POST_INC expr)
    |   ^(POST_DEC expr)
    |   ^('~' expr)
    |   ^('!' expr)
    |   ^(CAST_EXPR type expr)
    |   primaryExpression
    ;
    
primaryExpression
    :   ^(  '.' primaryExpression
                (   IDENT
                |   'this'
                |   'super'
                )
        )
    |   parenthesizedExpression
    |   IDENT
    |   ^(METHOD_CALL primaryExpression genericTypeArgumentList? arguments)
    |   explicitConstructorCall
    |   ^(ARRAY_ELEMENT_ACCESS primaryExpression expression)
    |   literal
    |   newExpression
    |   'this'
    |   arrayTypeDeclarator
    |   'super'
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
    |   'true'
    |   'false'
    |   'null'
    ;


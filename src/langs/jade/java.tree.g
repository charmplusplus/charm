header {
//header
package jade;
}

{
//class preamble
import jade.JJ.ASTJ;
import jade.JJ.J;
import java.io.PrintWriter;
import java.io.FileWriter;

}

/** Java 1.3 AST Recognizer Grammar
 *
 * Author: (see java.g preamble)
 * Author: J. DeSouza
 *
 * Assumes that JavaTreeParser1 has been run on the AST.
 */

class JavaTreeParser extends TreeParser;
options {
	importVocab = JavaTreeParser2;
    ASTLabelType = "ASTJ";
}
{
// Class Members

    private void eieio()
    {
    }
}

compilationUnit
	:	(p:packageDefinition {
                String name = J.pE(p.getFirstChild());
                J.tmp.push(name);
		J.pgmName = name;

                J.ci.append((p.status?"main":"") +
                    "module " + name + " {\n");
                J.h.append("\n#include <vector>\nusing std::vector;\n#include \"pup_stl.h\"\n");
                J.h.append("\n#include \"jade.h\"\n");
                // .decl.h is inserted right before main class
                // J.h.append("\n#include \"" + name + ".decl.h\"\n\n");
                J.c.append("\n#include \"" + name + ".h\"\n\n");
                J.indentLevel++;
            }
        )?
		(importDefinition)* // skip
		(c:typeDefinition  //           { ((ClassNode)c).print(); }
            //{ System.out.println(((ASTJ)c).toStringTree()); }
        )*
        {
            String name = J.pE(p.getFirstChild());

            if (p != null) {
                J.indentLevel--;
                J.ci.append("}\n");
                J.c.append("\n\n#include \"" + name + ".def.h\"\n");
            }

//             System.out.println("Done!");
//             System.out.println("================ci================");
//             System.out.println(J.ci);
            try {
                PrintWriter ci = new PrintWriter(new FileWriter(name + ".ci"));
                ci.print(J.ci.toString());
                ci.close();
            } catch (Exception e) {
                System.out.println(e);
                e.printStackTrace();
            }

//             System.out.println("================.h================");
//             System.out.println(J.h);
            try {
                PrintWriter h = new PrintWriter(new FileWriter(name + ".h"));
                h.print(J.h.toString());
                h.close();
            } catch (Exception e) {
                System.out.println(e);
                e.printStackTrace();
            }

//             System.out.println("================.C================");
//             System.out.println(J.c);
            try {
                PrintWriter cFile = new PrintWriter(new FileWriter(name + ".C"));
                cFile.print(J.c.toString());
                cFile.close();
            } catch (Exception e) {
                System.out.println(e);
                e.printStackTrace();
            }

        }
	;

packageDefinition // done
	:	#( PACKAGE_DEF identifier )
	;

importDefinition
	:	#( IMPORT identifierStar )
	;

typeDefinition
	:	#(c:CLASS_DEF
            m:modifiers
            IDENT { J.tmp.push(#IDENT.getText()); }
            e:extendsClause {

//                 if (((ASTJ)c).status)
//                     System.out.println("pass3: " + #IDENT.getText() + "is main");
//                 else
//                     System.out.println("pass3: " + #IDENT.getText() + "is NOT main");

                //{ //((ASTJ)#IDENT).genID(J.h);
                if (J.isX(m, "synchronized")) { // is a chare
//                     J.ciOn();
                    if (e.getFirstChild() == null)
                        J.fatalError( e, "synchronized class " + #IDENT.getText() + " must extend chare, charearray1d, or charearray2d");
                    else if (e.getFirstChild().getText().equalsIgnoreCase(J.strChare))
                        J.ci.append(J.indent() + (c.status?"main":"") + "chare " + #IDENT.getText());
                    else if (e.getFirstChild().getText().equalsIgnoreCase(J.strChareArray1D))
                        J.ci.append(J.indent() + "array [1D] " + #IDENT.getText());
                    else if (e.getFirstChild().getText().equalsIgnoreCase(J.strChareArray2D))
                        J.ci.append(J.indent() + "array [2D] " + #IDENT.getText());
                    else
                        J.fatalError( e, "synchronized class " + #IDENT.getText() + " must extend chare, charearray1d, or charearray2d");
                } else {
//                     J.ciOff();
                }

                // insert .decl.h right before main class
                if(c.status)
                   J.h.append("\n#include \"" + J.pgmName + ".decl.h\"\n\n");
                J.h.append(J.indent() + "class " + #IDENT.getText());
//                 J.c.append(J.indent() + "class " + #IDENT.getText());

                if (null != e) {
                    StringBuffer s = new StringBuffer();
                    s.append(": public ");
                    AST i = e.getFirstChild();
                    while (null != i) {
                        if (J.isStrChareStar(i.getText()) ) {
                            s.append("CBase_" + #IDENT.getText());
                        } else {
                            s.append(J.pE(i));
                        }
                        i = i.getNextSibling();
                        if (null != i) s.append(", ");
                    }

                    J.h.append(s);
//                     J.c.append(s);
                }
            }
            implementsClause // skip
//             {
//                 //System.out.println("Class: " + #IDENT.getText())/*#modifiers_in.getText()*/;
//                 System.out.println(//"class " +
//                     //((#m.getFirstChild()==null)?"":"modi_here ")  +
//                     #IDENT.getText() + " {");
//             }
            o:objBlock[#IDENT, J.isStrChareArray(e.getFirstChild().getText()) ]
        ) { J.tmp.pop(); }
	|	#(INTERFACE_DEF modifiers IDENT extendsClause interfaceBlock )  // skip
	;

typeSpec // done
	:	#(TYPE typeSpecArray)
	;

// arrays are handled in typeSpec, variableDeclarator, and
// newExpression.  e.g. int[] a[] = new int[10][20]; the "int[]"
// is a typeSpec, the "a[]" is a variableDeclarator, and the rest
// is an ASSIGN newExpression.
typeSpecArray // done
	:	#( ARRAY_DECLARATOR typeSpecArray )
    |   #( TEMPLATE typeSpecArray )
	|	type
	;

type // done
    :	identifier
	|	builtInType
	;

builtInType
    :   "void"
    |   "boolean"
    |   "byte"
    |   "char"
    |   "short"
    |   "int"
    |   "float"
    |   "long"
    |   "double"
    ;

modifiers
	:	#( MODIFIERS (m:modifier {
//                     if (!m.getText().equalsIgnoreCase("synchronized")) {
//                         J.c.append(m.getText()+ " ");
//                     }
                }
            )* )
	;

modifier
    :   "private"
    |   "public"
    |   "protected"
    |   "static"
    |   "transient"
    |   "final"
    |   "abstract"
    |   "native"
    |   "threadsafe"
    |   "synchronized"
    |   "const"
    |   "volatile"
	|	"strictfp"
	|	"threaded"
 	|	"blocking"
	|	"readonly"
    ;

extendsClause // done
	:	#(EXTENDS_CLAUSE (identifier)* )
	;

implementsClause
	:	#(IMPLEMENTS_CLAUSE (identifier)* )
	;

interfaceBlock
	:	#(	OBJBLOCK
			(	methodDecl
			|	variableDef[true,true]
			)*
		)
	;

objBlock[AST className, boolean insertMigrateConstructor]
	:	#(	OBJBLOCK {
		if(className!=null) J.curClassName = className.getText();
                J.ci.append(J.indent() + " {\n");
                J.h.append(J.indent() + " {\n");
                J.h.append(J.indent() + "private: void _init();\n");
                J.indentLevel++;
                if (insertMigrateConstructor) {
                    J.h.append(J.indent() + "public: " + J.pE(className) + "(CkMigrateMessage *m) {}\n");
                }
//                 J.c.append(J.indent() + " {\n");

                // initialize end-class strings
                J.endci.push(new StringBuffer(""));
                J.endci2.push(new StringBuffer(""));
                J.endh.push(new StringBuffer(""));
                J.endh2.push(new StringBuffer(""));
                J.classInit.push(new StringBuffer("void "+className.getText()+"::_init() {\n"));
                J.endc.push(new StringBuffer(""));

            }

			(	ctorDef
			|	m:methodDef[className]
			|	variableDef[true,true]
			|	typeDefinition
			|	#(STATIC_INIT slist)
			|	#(INSTANCE_INIT slist)
			)*
            {
                // class migration constructor is generated above
                // Generate the class PUP method here
                J.h.append(J.indent() + "public: virtual void pup(PUP::er &p);\n");
                J.c.append(J.indent() + "void "+J.pE(className)+"::pup(PUP::er &p) {\n");
                J.genPup(J.pE(className));
                J.c.append(J.indent() + "}\n");
                ((StringBuffer)J.classInit.peek()).append("};\n");

                J.indentLevel--;

                // end-class wrapup code
                J.ci.append((StringBuffer)J.endci.pop());
                J.ci.append(J.indent() + "}\n");
                J.ci.append((StringBuffer)J.endci2.pop());

                J.h.append((StringBuffer)J.endh.pop());
                J.h.append(J.indent() + "};\n");
                J.h.append((StringBuffer)J.endh2.pop());

                J.c.append((StringBuffer)J.classInit.pop());
                J.c.append((StringBuffer)J.endc.pop());
            }
		)
	;

ctorDef
	:	#(c:CTOR_DEF modifiers { J.startBlock(); } methodHead
            {
                J.tmp.push("");
                AST modifiers = #c.getFirstChild();
                AST methodName = modifiers.getNextSibling();
                AST parameters = methodName.getNextSibling();
                AST tmp = parameters.getNextSibling();
                AST throwsClause = null;
                AST slist = null;
                if (tmp == null) {
                    throwsClause = null; slist = null;
                } else if (tmp.getText().equalsIgnoreCase("throws")) {
                    throwsClause = tmp;
                    slist = throwsClause.getNextSibling();
                } else {
                    throwsClause = null; slist = tmp;
                }

                String s = methodName.getText() + "("
                    + J.printParamList(parameters)
                    + ")";

                J.ci.append(J.indent() + "entry " + s + ";\n");// J.nl(J.ci);
                J.h.append(J.indent() + "public: " + s + ";\n");// J.nl(J.h);
                J.c.append(J.indent() + methodName.getText() + "::"
                    + s + " {\n");// J.nl(J.h);

                J.indentLevel++;
                J.c.append(J.indent() + "_init();\n");
            }
            ctorSList
            {
                J.endBlock();
                J.tmp.pop();
                J.indentLevel--;
                J.c.append(J.indent() + "}\n");// J.nl(J.h);
            }
        )
	;

methodDecl
	:	#(METHOD_DEF modifiers typeSpec { J.startBlock(); } methodHead { J.endBlock(); })
	;

methodDef[AST className]
	:	#(m:METHOD_DEF modifiers ts:typeSpec
            { J.startBlock(); }
            mh:methodHead
            {
//                 System.out.println("MethodDef " + #mh.getText());

                J.tmp.push(new String(mh.getText()));
                //System.out.println("Method: " + #m.toStringTree());
                AST modifiers = #m.getFirstChild();
//                 AST returnType = ((ASTJ)#m).getChild(1);
                AST returnType = modifiers.getNextSibling();
                AST methodName = returnType.getNextSibling();
                AST parameters = methodName.getNextSibling();
                AST tmp = parameters.getNextSibling();
                AST throwsClause = null;
                AST slist = null;
                if (tmp == null) {
                    throwsClause = null; slist = null;
                } else if (tmp.getText().equalsIgnoreCase("throws")) {
                    throwsClause = tmp;
                    slist = throwsClause.getNextSibling();
                } else {
                    throwsClause = null; slist = tmp;
                }

                // entry method ?
                if (J.isX(modifiers, "public")
                        && returnType.getFirstChild().getType() == LITERAL_void) {
                    //System.out.println("Method: " + #m.toStringTree());
                    String s = methodName.getText() + "("
                        + J.printParamList(parameters) + ")";
                    J.ci.append(J.indent() + "entry " + (J.isX(modifiers, "threaded")?"[threaded] ":"") + "void " + s + ";\n");
                    J.h.append(J.indent() + "public: void " + s + ";\n");
                    J.c.append(J.indent() + "void " + className.getText() + "::" + s + "\n");
                }

                // main method == ClassName(CkArgMsg *) == ClassName()
                else if (methodName.getText().equalsIgnoreCase("main")) {
                    J.ci.append(J.indent() + "entry "
                        + className.getText() +"(CkArgMsg *);\n");
                    J.h.append(J.indent() + "public: "
                        + className.getText() +"(CkArgMsg *);\n");
                    J.c.append(J.indent() //+ "public: "
                        + className.getText() + "::" + className.getText() +"(CkArgMsg *)\n");
                }

                // regular method
                else {
                    String s = methodName.getText() + "("
                        + J.printParamList(parameters) + ")";

                    StringBuffer p = new StringBuffer();
                    if (J.isX(m.getFirstChild(), "public")) {
                        p.append("public");
                    } else if (J.isX(m.getFirstChild(), "protected"))
                        p.append("protected");
                    else {
                        p.append("private");
                    }
                    // @@ other modifiers ?

                    J.h.append(J.indent() + p + ": " + J.printTypeSpec(ts, null) + s + ";\n");
                    J.c.append(J.indent() + J.printTypeSpec(ts, null) + className.getText() + "::" + s + "\n");
                }

//                 System.out.println("entry "
//                     + ((modifiers.getFirstChild()==null)?"":"modi_here ")
//                     + returnType.getFirstChild().getText() + " " //@@
//                     + methodName.getText() + "("
//                     + (parameters.getFirstChild() == null?"":"params_here") + ")"
//                     + ((throwsClause != null)? " throws ": "")
//                     + ((slist == null)? "": " slist_here ")
//                     + ";"
//                 );

//                 J.indentLevel++;
            }
            (slist)?
            {
                J.tmp.pop();
//                 J.indentLevel--;
//                 J.c.append(J.indent() + "}\n");

                J.endBlock();
            }
        )
	;

// We can be defining a class variable or a local variable.
// We have to intercept declarations of Chare's here.
variableDef[boolean classVarq, boolean outputOnNewLine]
	:	#(v:VARIABLE_DEF m:modifiers ts:typeSpec vd:variableDeclarator vi:varInitializer {

                // MODIFIERS
                // "private" "public" protected
                // static
                // transient
                // final
                // abstract
                // native
                // ?? threadsafe
                // synchronized
                // ?? const
                // volatile
                // strictfp
                // threaded
                // sync
                // readonly
                //
                if (J.isX(m, new String[]{"threaded", "blocking", "abstract", "native", "strictfp", "synchronized"}))
                    J.nonfatalError(m, "Variables cannot be threaded, sync, abstract, native, strictfp, synchronized.  Ignoring.");
                if (!classVarq)
                    if (J.isX(m, new String[]{"private", "public", "protected"}))
                        J.nonfatalError(m, "Only class members can be private public protected.  Ignoring.");
                int n = ( (J.isX(m, "private")?1:0) +
                            (J.isX(m, "public")?1:0) +
                            (J.isX(m, "protected")?1:0) ) ;
                if (n>1)
                    J.fatalError(m, "Variable can only be one of private public protected.");

                String varName = J.printVariableDeclarator(vd);
                if (!classVarq){
                    J.localStackPush(varName, v);
                }

                // class variable def
                if (classVarq) {
                    if (J.isX(m, "readonly")) {
                        if (! (J.isX(m, "public") && J.isX(m, "static") ) )
                            J.fatalError(m, "Readonly must be public static");
                    }

                    // class variable def, readonly
                    if (J.isX(m, "public") && J.isX(m, "static") && J.isX(m, "readonly")) {
                        if (ARRAY_DECLARATOR == ts.getType() || ARRAY_DECLARATOR == ts.getFirstChild().getType())
                            J.fatalError(m, "readonly arrays not supported.");
                        else if (J.isStrMSA(ts.getFirstChild().getText()))
                            J.fatalError(m, "readonly msa not supported");
                        // jade: module M { class C { public static readonly int ro; } }
                        // .ci: module M { readonly int ro; }
                        // .h: class C {}; extern int ro;
                        // .C: int ro;
                        String tmp = J.printTypeSpec(ts, J.mangleIfReadonly(varName)) +";\n";
                        ((StringBuffer)J.endci2.peek()).append(J.indent() +"readonly " +tmp);
                        ((StringBuffer)J.endh2.peek()).append(J.indent() +"extern " +tmp);
                        ((StringBuffer)J.endc.peek()).append(
                            J.indent() +"//readonly\n"
                            +J.indent() +tmp);
                        if (vi != null)
                            J.nonfatalError(v, "Cannot initialize readonly in class. readonly can only be initialized in main.  Ignoring init.");

                    // class variable def, not readonly
                    } else {
                        J.h.append(J.indent());

                        if (J.isX(m, "public")) {
                            J.h.append("public: ");
                            // constant = public static final
                            if (J.isX(m, "static") && J.isX(m, "final")){
                                // @@ i use "const static" because with only "const" C++ does not allow an initializer.  Remove the static later, and do the initialzation in the constructor.
                                J.h.append("const ");
                            } else 
                                // @@ if isChare
                                J.nonfatalError(v, "chare variable " + varName + " cannot be public.  Proceeding anyway.");
                        } else if (J.isX(m, "protected"))
                            J.h.append("protected: ");
                        else {
                            J.h.append("private: ");
                        }

                        if (J.isX(m, "static"))
                            J.h.append("static ");

                        J.h.append(J.printTypeSpec(ts, varName));
                        // int[][] a, becomes a.setDimension(2) in classinit
                        if (ARRAY_DECLARATOR == ts.getType() || ARRAY_DECLARATOR == ts.getFirstChild().getType()) {
                            AST t=ts;
                            if (ARRAY_DECLARATOR != t.getType())
                                t = t.getFirstChild();
                            int i = 0;
                            while (ARRAY_DECLARATOR == t.getType()) {
                                ++i; t = t.getFirstChild();
                            }
                            ((StringBuffer)J.classInit.peek()).append(varName+".setDimension("+i+");\n"); // nD array
                        }

                        if (vi != null) {
                            // @@ static variables should be
                            // initialized only once per program.  Let the
                            // class have a boolean staticInitCalled, and let
                            // it guard a _static_init() method which is
                            // called from the class _init().
                            if (J.isX(m, "static")) {
                                J.h.append(" = " + J.printExpression(vi.getFirstChild()));
                            } else { // non-static class var init, put it into classinit
                                String separator = "=";
                                // Are we doing an array, i.e. = new int[]? .resize() should be put into classinit.
                                if ((EXPR == vi.getFirstChild().getType())
                                    && (LITERAL_new == vi.getFirstChild().getFirstChild().getType())
                                    && (ARRAY_DECLARATOR == J.getNode(vi,"fffn").getType())) {
                                    separator = "";
                                }
                                ((StringBuffer)J.classInit.peek()).append(varName + separator
                                        + J.pE(vi.getFirstChild().getFirstChild())+";\n");
                            }
                        };
                        J.h.append(";\n");
                    }
                }

                // local variable, "new" chare, i.e. Chare a = new Hello(params);
                // or, ChareArray a = new Hello[5];
                else if (
                    J.isStrChareStar(ts.getFirstChild().getText())
                    && vi != null
                    && EXPR == vi.getFirstChild().getType()
                    && LITERAL_new == vi.getFirstChild().getFirstChild().getType()) {
                    AST newAST = vi.getFirstChild().getFirstChild();
//                     String cid = "_ckchareid_" + varName;
                    String cproxytype = "CProxy_" + newAST.getFirstChild().getText();
                    J.c.append(J.indent());
//                     J.c.append("CkChareID " + cid + ";\n"
//                         + J.indent() + cproxytype + "::ckNew(" + "params_here"
//                             + ", &" + cid + ");\n"
//                         + J.indent() + cproxytype + " " + varName + "(" + cid + ")");

                    J.c.append(cproxytype + " " + varName + " = "
                        + cproxytype + "::ckNew");

                    if (ts.getFirstChild().getText().equalsIgnoreCase(J.strChare)){
                        // for Chare, we have an ELIST
			J.curClassDim = 0;
                        J.c.append("(" + J.printExpression(newAST.getFirstChild().getNextSibling()) +")");
                    } else if (ts.getFirstChild().getText().equalsIgnoreCase(J.strChareArray1D)) {
                        // For a ChareArray, we need to go down one more level.
			J.curClassDim = 1;
                        J.c.append("(" + J.printExpression(newAST.getFirstChild().getNextSibling().getFirstChild()) +")");
                    } else if (ts.getFirstChild().getText().equalsIgnoreCase(J.strChareArray2D)) {
                        // For a ChareArray2D, we need to create elements using insert
			J.curClassDim = 2;
                        AST e1 = J.getNode(newAST, "fnff"); // expr in first []
                        AST e2 = J.getNode(newAST, "fnfn"); // expr in 2nd []
                        J.c.append("(); {int e1="
                            +J.printExpression(e1)
                            +"; int e2="
                            +J.printExpression(e2)
                            +"; for(int _i=0; _i<e1; _i++) for(int _j=0; _j<e2; _j++) "
                            +varName+"(_i,_j).insert(); "
                            +varName+".doneInserting();}");
                    } else {
                        J.internalError(v, "var def: " + v + " unexpected input");
                    }
                    J.c.append(";\n");
                }

                // local variable, chare with getProxy, i.e.
                // Chare c = id.getProxy(Hello)
                else if (ts.getFirstChild().getText().equalsIgnoreCase(J.strChare)
                    && vi != null
                    && METHOD_CALL == vi.getFirstChild().getFirstChild().getType()
                    && vi.getFirstChild().getFirstChild().getFirstChild().getFirstChild().getNextSibling().getText().equalsIgnoreCase("getProxy")) {
                    String cidName = vi.getFirstChild().getFirstChild().getFirstChild().getFirstChild().getText();
                    String cproxytype = "CProxy_" + vi.getFirstChild().getFirstChild().getFirstChild().getNextSibling().getFirstChild().getFirstChild().getText();
                    J.c.append(J.indent() + cproxytype + " " + varName + "(" + cidName + ");\n");
                }

                // local non-chare/non-ChareArray variable
                else {
                    if (outputOnNewLine) J.c.append(J.indent());
                    J.c.append(J.printTypeSpec(ts, varName)); // convert int[], int[][]. etc. to JArray<int>

                    // if MSA
                    if (ts.isTypeMSA()) {
                        // generate the template instantiations
                        String templateStuff = J.printTemplateList(ts.getNode("ff"), 3);
                        ((StringBuffer)J.endci2.peek()).append(J.indent() + "group MSA_CacheGroup" + templateStuff + ";\n");
                        ((StringBuffer)J.endci2.peek()).append(J.indent() + "array [1D] MSA_PageArray" + templateStuff + ";\n");

                        System.out.println("reached here");
                        System.out.println("ts="+ts.toStringTree());
                    }

                    if (vi != null) {
                        // @@ see handling of vi in class vars above.

                        // Are we doing an array, i.e. = new int[], int[][], etc.
                        if ((EXPR == vi.getFirstChild().getType())
                            && (LITERAL_new == vi.getFirstChild().getFirstChild().getType())
                            && (ARRAY_DECLARATOR == J.getNode(vi, "fffn").getType())) {
                            // nD array
                            int i = 0;
                            AST t = J.getNode(vi, "fffn");
                            while (ARRAY_DECLARATOR == t.getType())
                                { ++i; t = t.getFirstChild(); }
                            J.c.append("("+i+");" + varName);
                        } else
                            J.c.append(" = ");
                        J.c.append(J.printExpression(vi.getFirstChild()));
                    };
                    if (outputOnNewLine) J.c.append(";\n");
                }

            }
        )
	;

// called from methodHead, handled there
// called from handler, TBD
// Adds parameter to localStack.
parameterDef
	:	#(p:PARAMETER_DEF modifiers typeSpec i:IDENT {
                    J.localStackPush(i.getText(), p);
            })
	;

objectinitializer
	:	#(INSTANCE_INIT slist)
	;

variableDeclarator
	:	IDENT
	|	LBRACK variableDeclarator
	;

varInitializer // done in variableDef
	:	#(ASSIGN initializer)
	|
	;

initializer
	:	expression // done
	|	arrayInitializer
	;

arrayInitializer
	:	#(ARRAY_INIT (initializer)*)
	;

methodHead // done in methodDef, ctorDef; TBD methodDecl
	:	IDENT #( PARAMETERS (p:parameterDef)* ) (throwsClause)?
	;

throwsClause
	:	#( "throws" (identifier)* )
	;

templater
    :  #( TEMPLATE (identifier|constant)+ )
    ;

// done as part of printExpression
identifier
	:	#( IDENT (templater)? )
	|	#( DOT identifier #( IDENT (templater)? ) )
	;

identifierStar
	:	IDENT
	|	#( DOT identifier (STAR|IDENT) )
	;

ctorSList
	:	#( SLIST (ctorCall)? (stat)* )
	;

slist // done
	:	#( SLIST
            {J.c.append(J.indent()+"{\n"); J.indentLevel++; }
            (stat)*
            {J.indentLevel--; J.c.append(J.indent()+"}\n");} )
	;

stat
    : typeDefinition  // @@ what to do about nested classes?
	|	variableDef[false,true]  // done
	|	e:expression { J.c.append(J.indent() + J.printExpression(e) + ";\n"); }
	|	#(LABELED_STAT i:IDENT {J.c.append(J.pE(i)+":\n"); } stat)
	|	#("if" ife:expression {
                J.c.append(J.indent() + "if (" + J.printExpression(ife) + ")\n");
            }
            stat ({J.c.append(J.indent() + "else\n");} stat)?
        )

	|	#(	fo:"for"
            {
                J.c.append(J.indent() + "for(");
            }
			#(f:FOR_INIT (variableDef[false,false] | e1:elist)?)
			#(FOR_CONDITION (e2:expression)?)
			#(FOR_ITERATOR (e3:elist)?)
            {
                J.c.append( ((e1!=null)? J.printExpression(e1): "") +";"
                    + J.printExpression(e2) + ";"
                    + J.printExpression(e3) + ")\n");
            }
			stat
		)

	|	#("while" we:expression { J.c.append(J.indent()+"while("+J.printExpression(we)+")");} stat)
	|	#("do" {J.c.append(J.indent()+"do");}
            stat { J.c.append(J.indent()+"while(");}
            dwe:expression {J.c.append(J.pE(dwe) + ");\n");})
	|	#("break" (bi:IDENT)? {J.c.append(J.indent()+"break "+ J.pE(bi) +";\n");} )
	|	#("continue" (ci:IDENT)? {J.c.append(J.indent()+"continue "+ J.pE(ci) +";\n");} )
	|	#("return" (re:expression)? {J.c.append(J.indent()+"return "+ J.printExpression(re) +";\n");} )

	|	#("switch" expression (caseGroup)*)
	|	#("throw" et:expression { J.fatalError(et, "throw not supported yet");})
	|	#("synchronized" expression stat)
	|	t:tryBlock { J.fatalError(t, "try not supported yet");}
	|	{ J.startBlock(); } slist { J.endBlock(); } // nested SLIST // done
	|	EMPTY_STAT { J.c.append(";\n"); } // done
	;

caseGroup
	:	#(CASE_GROUP (#("case" expression) | "default")+ slist)
	;

tryBlock
	:	#( "try" slist (handler)* (#("finally" slist))? )
	;

handler
	:	#( "catch" { J.startBlock(); } parameterDef slist { J.endBlock(); } )
	;

elist // done as part of printExpression
	:	#( ELIST (expression)*)
	;

colonExpression
    :   #(COLON expression expression (expression)?)
    |   expression
    ;

expression // done as part of printExpression
	:	#(EXPR expr)
	;

expr // mostly done
    :	#(QUESTION expr expr expr)	// trinary operator // done
	|	#(ASSIGN expr expr)			// binary operators... // done
	|	#(PLUS_ASSIGN expr expr)// done
	|	#(MINUS_ASSIGN expr expr)// done
	|	#(STAR_ASSIGN expr expr)// done
	|	#(DIV_ASSIGN expr expr)// done
	|	#(MOD_ASSIGN expr expr)// done
	|	#(SR_ASSIGN expr expr)// done
	|	#(BSR_ASSIGN expr expr)// done
	|	#(SL_ASSIGN expr expr)// done
	|	#(BAND_ASSIGN expr expr)// done
	|	#(BXOR_ASSIGN expr expr)// done
	|	#(BOR_ASSIGN expr expr)// done
	|	#(LOR expr expr)// done
	|	#(LAND expr expr)// done
	|	#(BOR expr expr)// done
	|	#(BXOR expr expr)// done
	|	#(BAND expr expr)// done
	|	#(NOT_EQUAL expr expr)// done
	|	#(EQUAL expr expr)// done
	|	#(LT expr expr)// done
	|	#(GT expr expr)// done
	|	#(LE expr expr)// done
	|	#(GE expr expr)// done
	|	#(SL expr expr)// done
	|	#(SR expr expr)// done
	|	#(BSR expr expr)// done
	|	#(PLUS expr expr)// done
	|	#(MINUS expr expr)// done
	|	#(DIV expr expr)// done
	|	#(MOD expr expr)// done
	|	#(STAR expr expr)// done
	|	#(INC expr)// done
	|	#(DEC expr)// done
	|	#(POST_INC expr)// done
	|	#(POST_DEC expr)// done
	|	#(BNOT expr)// done
	|	#(LNOT expr)// done
	|	#("instanceof" expr expr)
	|	#(UNARY_MINUS expr)// done
	|	#(UNARY_PLUS expr)// done
	|	primaryExpression
	;

primaryExpression // done as part of printExpression
    :   IDENT // done
    |   #(	DOT // done
			(	expr // done
				(	IDENT // done
				|	arrayIndex // done
				|	"this"
				|	c:"class"
                    //{ System.out.println("class2")/*#modifiers_in.getText()*/;}
				|	#( "new" IDENT elist )
				|   "super"
				)
			|	#(ARRAY_DECLARATOR typeSpecArray)
			|	builtInType ("class")?
			)
		)
	|	arrayIndex // done
	|	#(METHOD_CALL primaryExpression elist) // done
	|	#(TYPECAST typeSpec expr) // done
	|   newExpression
	|   constant// done
    |   "super"
    |   "true"
    |   "false"
    |   "this"
    |   "null"
	|	typeSpec // type name used with instanceof // done
	;

ctorCall
	:	#( CTOR_CALL elist )
	|	#( SUPER_CTOR_CALL
			(	elist
			|	primaryExpression elist
			)
		 )
	;

arrayIndex // done as part of printExpression
	:	#(INDEX_OP primaryExpression colonExpression)
	;

constant // done
    :   NUM_INT
    |   CHAR_LITERAL
    |   STRING_LITERAL
    |   NUM_FLOAT
    |   NUM_DOUBLE
    |   NUM_LONG
    ;

newExpression
	:	#(	"new" type
			(	newArrayDeclarator (arrayInitializer)?
			|	elist (objBlock[null, false])?
			)
		)

	;

newArrayDeclarator
	:	#( ARRAY_DECLARATOR (newArrayDeclarator)? (expression)? )
	;


package charj.translator;

import java.util.*;

public class ClassSymbol extends SymbolWithScope implements Scope {

    public ClassSymbol superClass;
    public List<String> interfaceImpls;

    Map<String, PackageScope> imports = 
        new LinkedHashMap<String, PackageScope>();
    List<String> includes = new ArrayList<String>();

    /** List of all fields and methods */
    public Map<String, Symbol> members = new LinkedHashMap<String, Symbol>();
    public Map<String, String> aliases = new LinkedHashMap<String, String>();

    /** The set of method names (without signatures) for this class.  Maps
     *  to a list of methods with same name but different args
     protected Map<String, List<MethodSymbol>> methods =
     new HashMap<String, List<MethodSymbol>>();
     */

    /** List of unmangled methods for this class. Used to distinguish
     *  var from method name in expressions.  x = f; // what is f?
     */
    protected Set<String> methodNames = new HashSet<String>();

    public boolean hasCopyCtor = false;

    public ClassSymbol(
            SymbolTable symtab, 
            String name) {
        super(symtab, name);
        type = this;
        for (String pkg : SymbolTable.AUTO_IMPORTS) {
            importPackage(pkg);
        }
    }

    public ClassSymbol(
            SymbolTable symtab, 
            String name,
            ClassSymbol superClass,
            Scope scope) {
        this(symtab, name);
        this.superClass = superClass;
        this.scope = scope;

        // manually add automatic class methods and symbols here
        this.includes.add("charm++.h");
    }

    public Scope getEnclosingScope() {
        // at root?  Then use enclosing scope
        if ( superClass==null ) { 
            return scope;
        }
        return superClass;
    }

    public Map<String, Symbol> getMembers() {
        return members;
    }

    /** Importing a package means adding the package to list of "filters"
     *  used by resolvePackage.  The resolve operation can only see classes
     *  defined in the imported packages.  This method asks the sym tab if
     *  it is known before looking at corresponding dir on disk to see if it 
     *  exists.
     *
     *  Return null if this class is not in sym tab and was not found in path.
     */
    public PackageScope importPackage(String packageName) {
        if (debug()) System.out.println(
                "ClassSymbol.importPackage(" + packageName +
                "): add to " + toString());
        
        PackageScope p = symtab.resolvePackage(packageName);
        if ( p!=null ) {
            imports.put(packageName, p);
            if (debug()) System.out.println(
                    "ClassSymbol.importPackage(" + packageName + "): known");
            return p;
        }

        if ( symtab.translator.findPackage(packageName) != null ) {
            p = symtab.definePackage(packageName);
            imports.put(packageName, p);
        }

        if ( p==null && debug() ) System.out.println(
                "ClassSymbol.importPackage(" + packageName + 
                "): dir not found");
        return p;
    }

    public void alias(
            CharjAST aliasAST, 
            CharjAST methodNameAST) {
        String op = aliasAST.getToken().getText();
        op = op.substring(1,op.length()-1);
        String method = methodNameAST.getToken().getText();
        method = method.substring(1,method.length()-1);
        alias(op, method);
    }

    public void alias(
            String alias, 
            String methodName) {
        aliases.put(alias, methodName);
    }

    public String getMethodNameForOperator(String op) {
        String name = aliases.get(op);
        if ( name==null ) {
            symtab.translator.error(
                    "no such operator for " + this.name + ": " + op);
        }
        return name;
    }

    /** Using the list of imports, resolve a package name like charj.lang.
     *  There may be many packages defined and visible to the symbol table
     *  but this class can only see those packages in the implicit or explicit
     *  import list.
     */
    public PackageScope resolvePackage(String packageName) {
        return imports.get(packageName);
    }

    /** To resolve a type in a class, look it up in each imported package 
     *  including the current package. Classes cannot be defined in the 
     *  superclass so don't look upwards for types.
     *
     *  First check to see if we are resolving enclosing class then
     *  look for type in each imported package.  If not found in existing
     *  packges, walk through imported packages again, trying to load from 
     *  disk.
     */
    public ClassSymbol resolveType(String type) {
        if (debug()) System.out.println(
                "ClassSymbol.resolveType(" + type + "): context is " + name +
                ":" + members.keySet());

        if (type == null) {
            return null;
        }

        if ( name.equals(type) ) {
            if ( debug() ) System.out.println(
                    "ClassSymbol.resolveType(" + type + 
                    "): surrounding class " + name + ":" + members.keySet());
            return this;
        }

        // look for type in classes already defined in imported packages
        for (String packageName : imports.keySet()) {
            PackageScope pkg = resolvePackage(packageName);
            ClassSymbol cs = pkg.resolveType(type);
            if ( cs != null) { // stop looking, found it
                if ( debug() ) System.out.println(
                        "ClassSymbol.resolveType(" + type + 
                        "): found in context " + name + ":" + 
                        members.keySet());
                return cs;
            }
        }

        // not already seen in one of the imported packages, look on disk
        for (String packageName : imports.keySet()) {
            PackageScope pkg = resolvePackage(packageName);
            ClassSymbol cs = symtab.translator.loadType(
                    pkg.getFullyQualifiedName(), type);
            if ( cs!=null ) {
                pkg.define(type, cs); // add to symbol table
                if ( debug() ) System.out.println(
                        "ClassSymbol.resolveType(" + type +
                        "): found after loading in context " + name +
                        ":" + members.keySet());
                return cs;
            }
        }

        if ( debug() ) System.out.println(
                "ClassSymbol.resolveType(" + type + 
                "): not in context " + name + ":" + members.keySet());
        return null;
    }

    public MethodSymbol resolveMethodLocally(
            String name, 
            int numargs) {
        if ( numargs>0 ) {
            name += numargs;
        }
     
        Symbol s = members.get(name);
        if ( s!=null && s.getClass() == MethodSymbol.class ) {
            return (MethodSymbol)s;
        }

        return null;
    }

    public boolean isMethod(String name) {
        if ( methodNames.contains(name) ) {
            return true;
        }
        if ( getEnclosingScope()!=null ) {
            return getEnclosingScope().isMethod(name);
        }
        return false;
    }

    public Symbol define(
            String name, 
            Symbol sym) {
        if ( sym instanceof MethodSymbol ) {
            methodNames.add(sym.name);
        }
        return super.define(name, sym);
    }

    public String toString() {
        return "ClassSymbol[" + name + "]: " + members;
    }

    public String getFullyQualifiedName() {
        String parent = null;
        if ( scope!=null ) { // in a package?
            parent = scope.getFullyQualifiedName();
        }
        if ( parent!=null ) {
            return parent+"."+name;
        }
        return name;
    }

    public String getMangledName() {
        if ( SymbolTable.TYPE_NAMES_TO_MANGLE.contains(name) ) {
            return "m"+name;
        }
        return name;
    }

    public void addInclude(String includeName) {
        includes.add(includeName);
    }

    public String getIncludeString() {
        String includeString = "";
        for (String include : includes) {
            includeString += "#include <" + include + ">\n";
        }
        return includeString;
    }

    public List<String> getPackageNames()
    {
        List<String> list = new ArrayList<String>();
        for(Scope currentScope = scope;
                currentScope.getEnclosingScope() != null;
                currentScope = currentScope.getEnclosingScope()) {
            list.add(currentScope.getScopeName());
        }
        return list;
    }
}

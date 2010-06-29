
package charj.translator;

import java.util.*;

public class ClassSymbol extends SymbolWithScope implements Scope, Type {

    public ClassSymbol superClass;
    public List<String> interfaceImpls;
    public List<String> templateArgs;
    public List<CharjAST> initializers;
    public List<CharjAST> varsToPup;

    public CharjAST constructor;

    Map<String, PackageScope> imports =
        new LinkedHashMap<String, PackageScope>();
    List<String> includes = new ArrayList<String>();
    List<String> usings = new ArrayList<String>();
    Set<String> externs = new TreeSet<String>();

    /** Record of all fields and methods */
    public Map<String, Symbol> members = new LinkedHashMap<String, Symbol>();
    public Map<String, VariableSymbol> fields = new LinkedHashMap<String, VariableSymbol>();
    public Map<String, MethodSymbol> methods = new LinkedHashMap<String, MethodSymbol>();

    public boolean hasCopyCtor = false;
    public boolean isPrimitive = false;
    public boolean isChare = false;
    public boolean isMainChare = false;

    public CharjAST migrationCtor = null;

    public ClassSymbol(
            SymbolTable symtab,
            String name) {
        super(symtab, name);
        type = this;
        for (String pkg : SymbolTable.AUTO_IMPORTS) {
            importPackage(pkg);
        }
	this.initializers = new ArrayList<CharjAST>();
        varsToPup = new ArrayList<CharjAST>();
	constructor = null;
    }

    public ClassSymbol(
            SymbolTable symtab,
            String name,
            ClassSymbol superClass,
            Scope scope) {
        this(symtab, name);
        this.superClass = superClass;
        this.scope = scope;
        this.type = this;
	this.initializers = new ArrayList<CharjAST>();
	constructor = null;

        // manually add automatic class methods and symbols here
        this.includes.add("charm++.h");
        this.includes.add("string");
        this.usings.add("std::string");
        this.includes.add("vector");
        this.usings.add("std::vector");
        this.includes.add("iostream");
        this.usings.add("std::cout");
        this.usings.add("std::endl");
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
    public Type resolveType(String type) {
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

        // Look in our enclosing package
        if (scope != null) {
            Type cs = scope.resolveType(type);
            if (cs != null && cs instanceof ClassSymbol) return (ClassSymbol)cs;
        }

        // look for type in classes already defined in imported packages
        for (String packageName : imports.keySet()) {
            if ( debug() ) System.out.println( "Looking for type " +
                    type + " in package " + packageName);
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

        if ( debug() ) System.out.println(
                "ClassSymbol.resolveType(" + type +
                "): not in context " + name + ":" + members.keySet());
        return null;
    }

    public MethodSymbol resolveMethodLocally(
            String name,
            int numargs) {
        if (numargs > 0) {
            name += numargs;
        }

        return methods.get(name);
    }

    public boolean isMethod(String name) {
        if ( methods.containsKey(name) ) {
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
        if (sym == null) {
            System.out.println("ClassSymbol.define: Uh oh, defining null symbol");
        }
        members.put(name, sym);
        if (sym instanceof MethodSymbol) {
            methods.put(name, (MethodSymbol)sym);
        } else if (sym instanceof VariableSymbol) {
            fields.put(name, (VariableSymbol)sym);
        }
        return super.define(name, sym);
    }

    public String toString() {
        if (isPrimitive) return name;
        else return getFullyQualifiedName() + members;
    }

    public String getFullyQualifiedName() {
        String parent = null;
        if ( scope!=null ) { // in a package?
            parent = scope.getFullyQualifiedName();
        }
        if ( parent!=null ) {
            return parent+"::"+name;
        }
        return name;
    }

    public void addInclude(String includeName) {
        includes.add(includeName);
    }

    public void addExtern(String externName) {
        externs.add(externName);
    }

    public void getUsings(String usingName) {
        usings.add(usingName);
    }

    public List<String> getIncludes()
    {
        return includes;
    }

    public List<String> getUsings()
    {
        return usings;
    }

    public Set<String> getExterns()
    {
        return externs;
    }

    public List<String> getPackageNames()
    {
        List<String> list = new LinkedList<String>();
        for(Scope currentScope = scope;
                currentScope.getEnclosingScope() != null;
                currentScope = currentScope.getEnclosingScope()) {
            list.add(0, currentScope.getScopeName());
        }
        return list;
    }

    private Set<ClassSymbol> getMemberTypes()
    {
        Set<ClassSymbol> types = new HashSet<ClassSymbol>();
        for (Map.Entry<String, VariableSymbol> entry : fields.entrySet()) {
            // note: type info may be null for unknown types, but this might
            // need to be changed at some point.
            Type type = ((VariableSymbol)entry.getValue()).type;
            if (type != null && type instanceof ClassSymbol) types.add((ClassSymbol)type);
        }
        return types;
    }

    public List<String> getMemberTypeNames()
    {
        List<String> names = new ArrayList<String>();
        for (ClassSymbol c : getMemberTypes()) {
            if (c.isPrimitive) continue;
            names.add(c.getName());
        }
        return names;
    }

    public String getName()
    {
        return name;
    }

    public String getTypeName()
    {
        return name;
    }

    private boolean requiresInit() {
        for (CharjAST varAst : varsToPup) {
            if (varAst.def instanceof VariableSymbol &&
                ((VariableSymbol)varAst.def).isPointerType()) {
                return true;
            }
        }
        return false;
    }

    public List<String> generateInits() {
        List<String> inits = new ArrayList<String>();
        for (CharjAST varAst : varsToPup) {
            if (varAst.def instanceof VariableSymbol &&
                ((VariableSymbol)varAst.def).isPointerType()) {
                VariableSymbol vs = (VariableSymbol)varAst.def;
                inits.add(vs.generateInit());
            }
        }
        if (inits.size() == 0)
            return null;
        else
            return inits;
    }

    public List<String> generatePUPers() {
        List<String> PUPers = new ArrayList<String>();
        for (CharjAST varAst : varsToPup) {
            if (varAst.def instanceof VariableSymbol) {
                PUPers.add(((VariableSymbol)varAst.def).generatePUP());
            }
        }
        return PUPers;
    }
}

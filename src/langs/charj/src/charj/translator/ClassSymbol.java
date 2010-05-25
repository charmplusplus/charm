
package charj.translator;

import java.util.*;

public class ClassSymbol extends SymbolWithScope implements Scope {

    public ClassSymbol superClass;
    public List<String> interfaceImpls;

    Map<String, PackageScope> imports = 
        new LinkedHashMap<String, PackageScope>();
    List<String> includes = new ArrayList<String>();

    /** Record of all fields and methods */
    public Map<String, Symbol> members = new LinkedHashMap<String, Symbol>();
    public Map<String, VariableSymbol> fields = new LinkedHashMap<String, VariableSymbol>();
    public Map<String, MethodSymbol> methods = new LinkedHashMap<String, MethodSymbol>();

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
        members.put(name, sym);
        if (sym instanceof MethodSymbol) {
            methods.put(name, (MethodSymbol)sym);
        } else if (sym instanceof VariableSymbol) {
            fields.put(name, (VariableSymbol)sym);
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
            return parent+"::"+name;
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

    private Set<ClassSymbol> getMemberTypes()
    {
        Set<ClassSymbol> types = new HashSet<ClassSymbol>();
        for (Map.Entry<String, VariableSymbol> entry : fields.entrySet()) {
            types.add(((VariableSymbol)entry.getValue()).type);
        }
        return types;
    }

    public List<String> getMemberTypeNames()
    {
        System.out.println("Looking for type names...");
        System.out.println("Found " + fields.size() + " fields...");
        List<String> names = new ArrayList<String>();
        for (ClassSymbol c : getMemberTypes()) {
            names.add(c.getFullyQualifiedName());
            System.out.println("Found type " + c.getFullyQualifiedName());
        }
        return names;
    }

    public String getName()
    {
        return name;
    }
}

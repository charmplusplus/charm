
package charj.translator;

import java.util.HashMap;
import java.util.Map;

public class PackageScope extends SymbolWithScope {

    /** List of packages and classes in this package */
    Map<String, Symbol> members = new HashMap(); // union of types and subpackages
    Scope enclosingScope;

    public PackageScope(
            SymbolTable symtab, 
            String name, 
            Scope enclosingScope) {
        super(symtab, name);
        this.enclosingScope = enclosingScope;
    }

    public Scope getEnclosingScope() {
        return enclosingScope;
    }

    /** See if type is already defined in this package.  If not, look
     *  for type on the disk in same package.  For example, first time
     *  charj.lang.Chare fails to resolve.  Load from disk and put Chare
     *  in package lang which is in package charj.  Next time, Chare will
     *  be found.
     */
    public ClassSymbol resolveType(String type) {
        if (type == null) return null;
        if (debug()) System.out.println(
                " PackageScope.resolveType(" + type + 
                "): examine " + toString());

        ClassSymbol cs = symtab.primitiveTypes.get(type);
        if (cs != null) return cs;

        // break off leading package names and look them up,
        // then look up the base class name within the appropriate package scope.
        String[] nameParts = type.split("[.]", 2);
        if (nameParts.length == 1) return (ClassSymbol)members.get(type);
        PackageScope innerPackage = (PackageScope)members.get(nameParts[0]);
        if (innerPackage == null) {
            if (debug()) System.out.println("Package lookup for " +
                    nameParts[0] + "failed.\n");
            return null;
        }
        return innerPackage.resolveType(nameParts[1]);
    }

    public String getFullyQualifiedName() {
        if ( name.equals(SymbolTable.DEFAULT_PACKAGE_NAME) ) {
            return null;
        }
        return super.getFullyQualifiedName();
    }

    public Map<String, Symbol> getMembers() {
        return members;
    }

    public String toString() {
        return "PackageScope[" + name + "]: " + members.keySet();
    }
}

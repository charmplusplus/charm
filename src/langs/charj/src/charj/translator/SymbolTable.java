
package charj.translator;

import java.lang.reflect.Field;
import java.util.*;

public class SymbolTable {
    public static final String DEFAULT_PACKAGE_NAME = "default";
    public static final List<String> AUTO_IMPORTS = 
        new ArrayList<String>() {
            {
                add("charj.lang");
                add(DEFAULT_PACKAGE_NAME);
            }
        };

    public static final Set<String> TYPE_NAMES_TO_MANGLE = 
        new HashSet<String>() { {} };

    public static final Set<String> METHOD_NAMES_TO_MANGLE = 
        new HashSet<String>() { {} };

    public static Map<String, ClassSymbol> primitiveTypes =
        new HashMap<String, ClassSymbol>() { {} };

    /** Provides runtime variables and loads classes */
    public Translator translator;

    /** The scope for the "default" package */
    PackageScope defaultPkg;

    public Map<String,PackageScope> topLevelPackageScopes = new HashMap();

    /** This is the list of all scopes created during symbol table building. */
    public List scopes = new ArrayList();

    /** Root of the object hierarchy, Charj.lang.Object */
    ClassSymbol objectRoot;

    public SymbolTable(Translator _translator) 
    {
        translator = _translator;
        // define default package
        defaultPkg = new PackageScope(this, DEFAULT_PACKAGE_NAME, null);
        topLevelPackageScopes.put(DEFAULT_PACKAGE_NAME, defaultPkg);
        PackageScope lang = definePackage("charj.lang");
        addScope(defaultPkg);
        initObjectHierarchy();
    }

    public boolean debug() {
        return translator.debug();
    }

    protected void initObjectHierarchy() 
    {
        PackageScope lang = resolvePackage("charj.lang");
        objectRoot = new ClassSymbol(this, "Object", null, lang);
        objectRoot.define("EOF", new VariableSymbol(this,"EOF",null));
        lang.define("Object", objectRoot);

        primitiveTypes.put("int",    new ClassSymbol(this, "int",    null, lang));
        primitiveTypes.put("long",   new ClassSymbol(this, "long",   null, lang));
        primitiveTypes.put("float",  new ClassSymbol(this, "float",  null, lang));
        primitiveTypes.put("double", new ClassSymbol(this, "double", null, lang));
        primitiveTypes.put("char",   new ClassSymbol(this, "char",   null, lang)); 
        primitiveTypes.put("short",  new ClassSymbol(this, "short",  null, lang)); 
        primitiveTypes.put("bool",   new ClassSymbol(this, "bool",   null, lang)); 
    }

    public ClassSymbol resolveBuiltinType(String type) {
        ClassSymbol ptype = primitiveTypes.get(type);
        if (ptype != null) return ptype;
        return objectRoot.resolveType(type);
    }

    /** Given a package like foo or charj.io, define it by breaking it up and
     *  looking up the packages to left of last id.  Add last id to that 
     *  package.
     */
    public PackageScope definePackage(String packageName) {
        String[] packageNames = packageName.split("[.]");
        String outerPackageName = packageNames[0];
        PackageScope outerPackage = (PackageScope)defaultPkg.resolve(
                outerPackageName);
        if (outerPackage == null) {
            if (debug()) {
                System.out.println(
                        " SymbolTable.definePackage(" + packageName + 
                        "): defining outer pkg: " + outerPackageName);
            }
            outerPackage = new PackageScope(this,outerPackageName,defaultPkg);
            defaultPkg.define(outerPackageName, outerPackage);
            topLevelPackageScopes.put(outerPackageName, outerPackage);
        }

        PackageScope enclosingPackage = defaultPkg;
        for (String pname : packageNames) {
            PackageScope p = (PackageScope)enclosingPackage.resolve(pname);
            if (p==null) {
                if (debug()) System.out.println(
                        " SymbolTable.definePackage(" + packageName + 
                        "): defining inner pkg: " + pname +
                        " in "+enclosingPackage.toString());
                
                p = new PackageScope(this,pname,enclosingPackage);
                enclosingPackage.define(pname, p);
            }
            enclosingPackage = p;
        }
        return enclosingPackage;
    }

    public PackageScope getDefaultPkg() {
        return defaultPkg;
    }

    /** Find package starting with its outermost package name.  If
     *  not in sym tab, return null.
     */
    public PackageScope resolvePackage(String packageName) {
        if (debug()) System.out.println(
                " SymbolTable.resolvePackage(" + packageName + 
                "): examine: " + topLevelPackageScopes.keySet());
        String[] packageNames = packageName.split("[.]");
        String outerPackageName = packageNames[0];
        PackageScope enclosingPackage = topLevelPackageScopes.get(
                outerPackageName);

        if (enclosingPackage == null) {
            if (debug()) System.out.println(
                    " SymbolTable.resolvePackage(" + packageName + 
                    "): outer package " +
                    outerPackageName + " not found in top level " +
                    topLevelPackageScopes.keySet());
            return null;
        }

        if (packageNames.length==1) {
            return enclosingPackage; // top-level package
        }

        PackageScope p = null;
        for (int i=1; i<packageNames.length; i++) {
            String pname = packageNames[i];
            p = (PackageScope)enclosingPackage.resolve(pname);
            if (p == null) {
                if (debug()) System.out.println(
                        " SymbolTable.resolvePackage(" + packageName +
                        "): not found in " + topLevelPackageScopes.keySet());
                return null;
            }
            enclosingPackage = p;
        }

        if (p!=null && debug()) System.out.println(
                " SymbolTable.resolvePackage(" + packageName + "): found in " +
                topLevelPackageScopes.keySet());
        return p;
    }

    public ClassSymbol getObjectRoot() {
        return objectRoot;
    }

    public void addScope(Scope s) {
        scopes.add(s);
    }

    // TODO:  shouldn't we include the arguments and do all mangling here?
    public static String mangle(String methodName) {
        if (METHOD_NAMES_TO_MANGLE.contains(methodName)) {
            return "cj" + methodName;
        }
        return methodName;
    }

    public static String unmangle(String methodName) {
        // this is not perfect because perhaps someone makes a method called
        // mtoString() etc.
        String unmangled = methodName.substring(2, methodName.length());
        if (METHOD_NAMES_TO_MANGLE.contains(unmangled)) {
            return unmangled;
        }
        return methodName;
    }

    public static String getCharjTypeName(String className) {
        if (SymbolTable.TYPE_NAMES_TO_MANGLE.contains(className)) {
            return "cj"+className;
        }
        return className;
    }

    public String toString() {
        return scopes.toString();
    }
}


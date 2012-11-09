
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
        PackageScope externals = definePackage("externals");
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
        lang.define("Object", objectRoot);

        ClassSymbol array = new ClassSymbol(this, "Array",  null, lang);
        TypeName typeName = new TypeName("int");
        Type tInt = (Type)(new ClassSymbol(this, "int",    null, lang));
        Type tVoid = (Type)(new ClassSymbol(this, "void",    null, lang));
        array.define("size", new MethodSymbol(this, "size", array, tInt));
        // Not to be used with other Charj code
        array.define("raw", new MethodSymbol(this, "raw", array, tVoid));

        primitiveTypes.put("Array",  array);
        primitiveTypes.put("void",   new ClassSymbol(this, "void",   null, lang));
        primitiveTypes.put("int",    new ClassSymbol(this, "int",    null, lang));
        primitiveTypes.put("long",   new ClassSymbol(this, "long",   null, lang));
        primitiveTypes.put("float",  new ClassSymbol(this, "float",  null, lang));
        primitiveTypes.put("double", new ClassSymbol(this, "double", null, lang));
        primitiveTypes.put("char",   new ClassSymbol(this, "char",   null, lang));
        primitiveTypes.put("short",  new ClassSymbol(this, "short",  null, lang));
        primitiveTypes.put("boolean",new ClassSymbol(this, "bool",   null, lang));
        primitiveTypes.put("string", new ClassSymbol(this, "string", null, lang));
        primitiveTypes.put("byte", primitiveTypes.get("char"));
        for (Map.Entry<String, ClassSymbol> entry : primitiveTypes.entrySet()) {
            ClassSymbol c = entry.getValue();
            lang.define(entry.getKey(), c);
            c.isPrimitive = true;
        }

        defaultPkg.define("CkArgMsg", new ExternalSymbol(this, "CkArgMsg"));
        defaultPkg.define("CkPrintf", new MethodSymbol(this, "CkPrintf"));
        defaultPkg.define("CkNumPes", new MethodSymbol(this, "CkNumPes"));
        defaultPkg.define("CkMyPe", new MethodSymbol(this, "CkMyPe"));
        defaultPkg.define("CkExit", new MethodSymbol(this, "CkExit"));
        defaultPkg.define("CkWallTimer", new MethodSymbol(this, "CkWallTimer"));
        defaultPkg.define("contribute", new MethodSymbol(this, "contribute"));
        defaultPkg.define("CkCallback", new MethodSymbol(this, "CkCallback"));
        defaultPkg.define("CkReductionTarget", new ExternalSymbol(this, "CkReductionTarget"));
    }

    public ClassSymbol lookupPrimitive(List<TypeName> type) {
      if (type.size() == 1) {
        ClassSymbol cs = primitiveTypes.get(type.get(0).name);
        if (cs != null && type.get(0).parameters != null && type.get(0).parameters.size() > 0) {
          if (cs.templateArgs == null)
            cs.templateArgs = new ArrayList<Type>();
          for (Type param : type.get(0).parameters) {
            /*System.out.println("template type param: " + param);*/
            cs.templateArgs.add(param);
          }
        }
        return cs;
      } else
        return null;
    }

    public ClassSymbol resolveBuiltinType(String type) {
        ClassSymbol ptype = primitiveTypes.get(type);
        if (ptype != null) return ptype;
        return (ClassSymbol)objectRoot.resolveType(TypeName.createTypeName(type));
    }

    public ClassSymbol resolveBuiltinLitType(String type, String literal) {
        ClassSymbol ptype = primitiveTypes.get(type);
        if (ptype != null) {
            LiteralType t = new LiteralType(this, ptype);
            t.literal = literal;
            return (ClassSymbol)t;
        }
        return (ClassSymbol)objectRoot.resolveType(TypeName.createTypeName(type));
    }

    public ClassSymbol resolveBuiltinType(String type, String lit) {
        ClassSymbol ptype = primitiveTypes.get(type);
        if (ptype != null) return ptype;
        return (ClassSymbol)objectRoot.resolveType(TypeName.createTypeName(type));
    }

    public ClassSymbol getEnclosingClass(Scope scope) {
        while (scope != null) {
            if (scope instanceof ClassSymbol) return (ClassSymbol)scope;
            scope = scope.getEnclosingScope();
        }
        return null;
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

    public String toString() {
        return scopes.toString();
    }
}


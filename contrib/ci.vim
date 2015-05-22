" Vim syntax file for Charm++ .ci files
" Mostly identical to the C++ syntax file with special keyword additions
"
" To use, add this file to ~/.vim/syntax and add the following
" line to ~/.vim/filetype.vim:
" au! BufRead,BufNewFile *.ci set filetype=ci

if version < 600
  syntax clear
elseif exists("b:current_syntax")
  finish
endif

" Read the C syntax to start with
if version < 600
  so <sfile>:p:h/c.vim
else
  runtime! syntax/c.vim
  unlet b:current_syntax
endif

" C++/Charm extentions
syn keyword cppStatement	new delete this friend using
syn keyword cppStatement	serial atomic overlap when publishes connect
syn keyword cppAccess		public protected private readonly
syn keyword cppType		inline virtual explicit export bool wchar_t
syn keyword cppType             entry
syn keyword cppExceptions	throw try catch
syn keyword cppOperator		operator typeid
syn keyword cppOperator		and bitor or xor compl bitand and_eq or_eq xor_eq not not_eq
syn match cppCast		"\<\(const\|static\|dynamic\|reinterpret\)_cast\s*<"me=e-1
syn match cppCast		"\<\(const\|static\|dynamic\|reinterpret\)_cast\s*$"
syn keyword cppStorageClass	mutable
syn keyword cppStorageClass	aggregate threaded sync exclusive nokeep notrace immediate expedited inline local python accel readwrite writeonly accelblock memcritical packed varsize initproc initnode initcall stacksize createhere createhome reductiontarget
syn keyword cppStructure	class typename template namespace message conditional
syn keyword cppStructure        mainmodule mainchare module chare array group nodegroup
syn keyword cppNumber		NPOS
syn keyword cppBoolean		true false

" The minimum and maximum operators in GNU C++
syn match cppMinMax "[<>]?"

" Default highlighting
if version >= 508 || !exists("did_cpp_syntax_inits")
  if version < 508
    let did_cpp_syntax_inits = 1
    command -nargs=+ HiLink hi link <args>
  else
    command -nargs=+ HiLink hi def link <args>
  endif
  HiLink cppAccess		cppStatement
  HiLink cppCast		cppStatement
  HiLink cppExceptions		Exception
  HiLink cppOperator		Operator
  HiLink cppStatement		Statement
  HiLink cppType		Type
  HiLink cppStorageClass	StorageClass
  HiLink cppStructure		Structure
  HiLink cppNumber		Number
  HiLink cppBoolean		Boolean
  delcommand HiLink
endif

let b:current_syntax = "cpp"

" vim: ts=8

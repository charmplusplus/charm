#set auto-load safe-path /
set auto-load safe-path /
set pagination off
set logging file gdb.txt
set logging on
file simpleZeroCopy
br __ubsan::ScopedReport::ScopedReport
#br __asan::ReportGenericError
#commands
r
#bt
#continue
#end
#info breakpoints
#r
#set logging off
#quit

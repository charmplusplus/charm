// Grab all code snippets and paint it to highlight syntax
$(document).ready( function() {
    $("code").each( function(idx) {
        CodeMirror.runMode($(this).text(), "text/x-c++src", $(this).get(0));
    })
    .addClass("cm-s-lesser-dark");
} )


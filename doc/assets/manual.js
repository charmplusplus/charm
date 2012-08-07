// Grab all code snippets and paint it to highlight syntax
$(document).ready( function() {
    $("pre code").each( function(idx) {
        CodeMirror.runMode($(this).text(), "text/x-charm++", $(this).get(0));
    })
    .addClass("cm-s-lesser-dark")
    .children("span.cm-charmkeyword").css("color", "#dd5ef3");

    $(".navigation")
    .append('<span id="pulldowntab" class="navsymbol">&raquo;</span>')
    .click( function() { $("ul.manual-toc").fadeToggle(); $("#pulldowntab").toggle(); } )
    .mouseleave( function() { $("ul.manual-toc").fadeOut('slow'); $("#pulldowntab").fadeIn('slow'); } )
    .css('cursor','pointer');

} )


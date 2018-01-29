/**
 * This file defines APIs for regular expression calls. Current implementation is c++11 regex library that
 * should be well supported across platform.
 *
 * @author Shaoqin(Bobby) Lu
 */
#ifdef __cplusplus
extern "C" {
#endif

/**
 * find captures in the first matched substring using the provided regular expression pattern on the target string.
 *
 * TODO this feels a little not as flexible. Maybe make a function that takes int n to find first n matches and all captures
 *
 * example: pattern="abcd([0-9]+)"
 *          s="abcd1234abcd5678"
 *          return ["1234", NULL]
 *
 * @param pattern the regular expression pattern
 * @param s the target string
 * @return An array of string representing the captured substrings terminated with a NULL, caller is responsible to free the consumed memory
 */
char ** findFirstCaptures(const char * pattern, const char * s);

#ifdef __cplusplus
}
#endif
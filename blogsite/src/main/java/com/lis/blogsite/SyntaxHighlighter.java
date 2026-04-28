package com.lis.blogsite;

import java.util.Locale;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

final class SyntaxHighlighter {
    private static final Pattern CODE_BLOCK = Pattern.compile(
            "<pre><code class=\"language-([A-Za-z0-9_.+-]+)\">([\\s\\S]*?)</code></pre>");

    private static final Set<String> JVM_KEYWORDS = Set.of(
            "abstract", "actual", "annotation", "as", "assert", "boolean", "break", "byte", "by",
            "case", "catch", "char", "class", "companion", "const", "constructor", "continue",
            "crossinline", "data", "default", "do", "double", "dynamic", "else", "enum",
            "expect", "exports", "extends", "external", "false", "final", "finally", "float",
            "for", "fun", "if", "implements", "import", "in", "infix", "init", "inline",
            "inner", "instanceof", "int", "interface", "internal", "is", "lateinit", "long",
            "module", "native", "new", "noinline", "non", "null", "object", "open", "opens",
            "operator", "out", "override", "package", "permits", "private", "protected",
            "provides", "public", "record", "reified", "requires", "return", "sealed", "short",
            "static", "strictfp", "super", "suspend", "switch", "synchronized", "tailrec",
            "this", "throw", "throws", "to", "transient", "transitive", "true", "try",
            "typealias", "uses", "val", "var", "vararg", "void", "volatile", "when", "where",
            "while", "with", "yield");

    private static final Set<String> SHELL_KEYWORDS = Set.of(
            "cat", "cd", "echo", "export", "for", "gcloud", "gh", "git", "gradle", "if", "in",
            "java", "javap", "then", "while");

    String highlightCodeBlocks(String html) {
        Matcher matcher = CODE_BLOCK.matcher(html);
        StringBuilder highlighted = new StringBuilder(html.length());
        int last = 0;
        while (matcher.find()) {
            highlighted.append(html, last, matcher.start());
            String language = matcher.group(1);
            String code = decodeCodeText(matcher.group(2));
            highlighted.append("<pre><code class=\"language-")
                    .append(language)
                    .append("\">")
                    .append(highlight(language, code))
                    .append("</code></pre>");
            last = matcher.end();
        }
        highlighted.append(html, last, html.length());
        return highlighted.toString();
    }

    private String highlight(String language, String source) {
        String normalized = language.toLowerCase(Locale.ROOT);
        if (Set.of("java", "kt", "kotlin", "groovy", "gradle").contains(normalized)) {
            return highlightSource(source, JVM_KEYWORDS, true);
        }
        if (Set.of("bash", "console", "sh", "shell").contains(normalized)) {
            return highlightSource(source, SHELL_KEYWORDS, false);
        }
        return escapeHtml(source);
    }

    private String highlightSource(String source, Set<String> keywords, boolean markTypes) {
        StringBuilder html = new StringBuilder(source.length() + 256);
        int index = 0;
        while (index < source.length()) {
            if (startsWith(source, index, "//")) {
                int end = source.indexOf('\n', index);
                if (end < 0) {
                    end = source.length();
                }
                appendToken(html, "comment", source.substring(index, end));
                index = end;
            } else if (startsWith(source, index, "/*")) {
                int end = source.indexOf("*/", index + 2);
                end = end < 0 ? source.length() : end + 2;
                appendToken(html, "comment", source.substring(index, end));
                index = end;
            } else if (startsWith(source, index, "\"\"\"")) {
                int end = source.indexOf("\"\"\"", index + 3);
                end = end < 0 ? source.length() : end + 3;
                appendToken(html, "string", source.substring(index, end));
                index = end;
            } else if (source.charAt(index) == '"' || source.charAt(index) == '\'') {
                int end = quotedEnd(source, index, source.charAt(index));
                appendToken(html, "string", source.substring(index, end));
                index = end;
            } else if (source.charAt(index) == '@' && index + 1 < source.length()
                    && isIdentifierStart(source.charAt(index + 1))) {
                int end = identifierEnd(source, index + 1);
                appendToken(html, "annotation", source.substring(index, end));
                index = end;
            } else if (Character.isDigit(source.charAt(index))) {
                int end = numberEnd(source, index);
                appendToken(html, "number", source.substring(index, end));
                index = end;
            } else if (isIdentifierStart(source.charAt(index))) {
                int end = identifierEnd(source, index);
                String word = source.substring(index, end);
                if (keywords.contains(word)) {
                    appendToken(html, "keyword", word);
                } else if (markTypes && Character.isUpperCase(word.charAt(0))) {
                    appendToken(html, "type", word);
                } else if (nextNonWhitespaceIs(source, end, '(')) {
                    appendToken(html, "function", word);
                } else {
                    html.append(escapeHtml(word));
                }
                index = end;
            } else if (source.charAt(index) == '$' && index + 1 < source.length()
                    && isIdentifierStart(source.charAt(index + 1))) {
                int end = identifierEnd(source, index + 1);
                appendToken(html, "variable", source.substring(index, end));
                index = end;
            } else {
                html.append(escapeHtml(source.substring(index, index + 1)));
                index++;
            }
        }
        return html.toString();
    }

    private int quotedEnd(String source, int start, char quote) {
        int index = start + 1;
        while (index < source.length()) {
            char current = source.charAt(index++);
            if (current == '\\' && index < source.length()) {
                index++;
            } else if (current == quote) {
                break;
            }
        }
        return index;
    }

    private int identifierEnd(String source, int start) {
        int index = start;
        while (index < source.length() && isIdentifierPart(source.charAt(index))) {
            index++;
        }
        return index;
    }

    private int numberEnd(String source, int start) {
        int index = start;
        while (index < source.length()) {
            char current = source.charAt(index);
            if (Character.isLetterOrDigit(current) || current == '_' || current == '.' || current == '-') {
                index++;
            } else {
                break;
            }
        }
        return index;
    }

    private boolean nextNonWhitespaceIs(String source, int start, char expected) {
        int index = start;
        while (index < source.length() && Character.isWhitespace(source.charAt(index))) {
            index++;
        }
        return index < source.length() && source.charAt(index) == expected;
    }

    private boolean startsWith(String source, int index, String needle) {
        return source.regionMatches(index, needle, 0, needle.length());
    }

    private boolean isIdentifierStart(char current) {
        return Character.isLetter(current) || current == '_' || current == '$';
    }

    private boolean isIdentifierPart(char current) {
        return Character.isLetterOrDigit(current) || current == '_' || current == '$';
    }

    private void appendToken(StringBuilder html, String token, String text) {
        html.append("<span class=\"tok tok-")
                .append(token)
                .append("\">")
                .append(escapeHtml(text))
                .append("</span>");
    }

    private String decodeCodeText(String text) {
        return text.replace("&lt;", "<")
                .replace("&gt;", ">")
                .replace("&quot;", "\"")
                .replace("&#39;", "'")
                .replace("&amp;", "&");
    }

    private String escapeHtml(String text) {
        return text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;");
    }
}

package com.lis.blogsite;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

class SyntaxHighlighterTest {
    private final SyntaxHighlighter highlighter = new SyntaxHighlighter();

    @Test
    void highlightsJvmTokensAndEscapesCode() {
        String html = highlighter.highlightCodeBlocks("""
                <pre><code class="language-java">@Override
                public String value() {
                    // comment
                    String text = "a<b";
                    int number = 12;
                    return text + number;
                }</code></pre>
                """);

        assertThat(html)
                .contains("tok-annotation")
                .contains("tok-keyword")
                .contains("tok-type")
                .contains("tok-function")
                .contains("tok-comment")
                .contains("tok-string")
                .contains("tok-number")
                .contains("&lt;b");
    }

    @Test
    void highlightsShellAndVariables() {
        String html = highlighter.highlightCodeBlocks("""
                <pre><code class="language-bash">export HOME=/tmp
                echo "$HOME"
                # literal</code></pre>
                """);

        assertThat(html)
                .contains("tok-keyword")
                .contains("tok-variable")
                .contains("tok-string");
    }

    @Test
    void handlesCommentsTripleQuotesEscapesAndUnknownLanguages() {
        String kotlin = highlighter.highlightCodeBlocks("""
                <pre><code class="language-kotlin">/* block */
                val text = """triple"""
                val quote = '\\''
                </code></pre>
                """);
        String unknown = highlighter.highlightCodeBlocks(
                "<pre><code class=\"language-text\">&lt;raw&gt; &amp; value</code></pre>");
        String unchanged = highlighter.highlightCodeBlocks("<p>plain</p>");

        assertThat(kotlin).contains("tok-comment").contains("tok-string");
        assertThat(unknown).contains("&lt;raw&gt; &amp; value");
        assertThat(unchanged).isEqualTo("<p>plain</p>");
    }

    @Test
    void unterminatedTokensDoNotBreakHighlighting() {
        String html = highlighter.highlightCodeBlocks(
                "<pre><code class=\"language-java\">String x = \"unterminated\n/* open</code></pre>");

        assertThat(html).contains("tok-string");
    }
}

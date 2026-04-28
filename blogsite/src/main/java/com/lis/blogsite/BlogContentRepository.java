package com.lis.blogsite;

import org.commonmark.node.Node;
import org.commonmark.parser.Parser;
import org.commonmark.renderer.html.HtmlRenderer;
import org.springframework.core.io.ClassPathResource;
import org.springframework.stereotype.Repository;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

@Repository
public class BlogContentRepository {
    private final Map<String, BlogPost> postsBySlug;

    public BlogContentRepository() {
        Parser parser = Parser.builder().build();
        HtmlRenderer renderer = HtmlRenderer.builder()
                .escapeHtml(true)
                .build();
        SyntaxHighlighter syntaxHighlighter = new SyntaxHighlighter();
        this.postsBySlug = Collections.unmodifiableMap(loadPosts(parser, renderer, syntaxHighlighter));
    }

    public List<BlogPost> findAll() {
        return List.copyOf(postsBySlug.values());
    }

    public Optional<BlogPost> findBySlug(String slug) {
        return Optional.ofNullable(postsBySlug.get(slug));
    }

    private Map<String, BlogPost> loadPosts(Parser parser, HtmlRenderer renderer, SyntaxHighlighter syntaxHighlighter) {
        Map<String, BlogPost> posts = new LinkedHashMap<>();
        for (ManifestEntry entry : loadManifest()) {
            String markdown = readResource(entry.resourcePath());
            Node document = parser.parse(markdown);
            String html = syntaxHighlighter.highlightCodeBlocks(renderer.render(document));
            posts.put(entry.slug(), new BlogPost(
                    entry.slug(),
                    entry.archiveName(),
                    titleFrom(markdown, entry.archiveName()),
                    excerptFrom(markdown),
                    markdown,
                    html
            ));
        }
        return posts;
    }

    private List<ManifestEntry> loadManifest() {
        List<ManifestEntry> entries = new ArrayList<>();
        ClassPathResource manifest = new ClassPathResource("blog-posts/manifest.tsv");
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(
                manifest.getInputStream(), StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                String[] fields = line.split("\t", -1);
                if (fields.length != 3) {
                    throw new IllegalStateException("Invalid blog manifest line: " + line);
                }
                entries.add(new ManifestEntry(fields[0], fields[1], fields[2]));
            }
        } catch (IOException e) {
            throw new UncheckedIOException("Unable to load blog manifest", e);
        }
        return entries;
    }

    private String readResource(String path) {
        ClassPathResource resource = new ClassPathResource(path);
        try {
            return new String(resource.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new UncheckedIOException("Unable to load " + path, e);
        }
    }

    private String titleFrom(String markdown, String fallback) {
        return markdown.lines()
                .map(String::trim)
                .filter(line -> line.startsWith("# "))
                .findFirst()
                .map(line -> line.substring(2).trim())
                .filter(line -> !line.isEmpty())
                .orElse(fallback);
    }

    private String excerptFrom(String markdown) {
        return markdown.lines()
                .map(String::trim)
                .filter(line -> !line.isEmpty())
                .filter(line -> !line.startsWith("#"))
                .filter(line -> !line.startsWith("```"))
                .findFirst()
                .map(this::truncate)
                .orElse("");
    }

    private String truncate(String text) {
        int limit = 220;
        if (text.length() <= limit) {
            return text;
        }
        return text.substring(0, limit - 1).stripTrailing() + "...";
    }

    private record ManifestEntry(String slug, String archiveName, String resourcePath) {
    }
}

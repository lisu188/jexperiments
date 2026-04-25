package com.lis.blogsite;

public record BlogPost(String slug,
                       String archiveName,
                       String title,
                       String excerpt,
                       String markdown,
                       String html) {
}

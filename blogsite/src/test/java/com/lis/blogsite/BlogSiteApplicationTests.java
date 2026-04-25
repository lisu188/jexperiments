package com.lis.blogsite;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.web.server.LocalServerPort;
import org.springframework.http.HttpStatus;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;

import static org.assertj.core.api.Assertions.assertThat;

@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
class BlogSiteApplicationTests {
    private final HttpClient httpClient = HttpClient.newHttpClient();

    @Autowired
    private BlogContentRepository repository;

    @LocalServerPort
    private int port;

    @Test
    void loadsAllConfiguredPosts() {
        assertThat(repository.findAll()).hasSize(13);
        assertThat(repository.findBySlug("bcel"))
                .hasValueSatisfying(post -> assertThat(post.title())
                        .isEqualTo("Generating a Class with BCEL"));
    }

    @Test
    void indexListsRepresentativePosts() throws Exception {
        HttpResponse<String> response = get("/");

        assertThat(response.statusCode()).isEqualTo(HttpStatus.OK.value());
        assertThat(response.body())
                .contains("/posts/bcel")
                .contains("/posts/flowpublisher")
                .contains("Coordinating Subscribers with Java Flow");
    }

    @Test
    void postRouteRendersMarkdownAsHtml() throws Exception {
        HttpResponse<String> response = get("/posts/bcel");

        assertThat(response.statusCode()).isEqualTo(HttpStatus.OK.value());
        assertThat(response.body())
                .contains("Generating a Class with BCEL")
                .contains("<h2>Why this experiment exists</h2>")
                .contains("<code>com.company.Main</code>");
    }

    @Test
    void missingPostReturnsNotFound() throws Exception {
        HttpResponse<String> response = get("/posts/not-a-module");

        assertThat(response.statusCode()).isEqualTo(HttpStatus.NOT_FOUND.value());
    }

    private HttpResponse<String> get(String path) throws Exception {
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create("http://localhost:" + port + path))
                .GET()
                .build();
        return httpClient.send(request, HttpResponse.BodyHandlers.ofString());
    }
}

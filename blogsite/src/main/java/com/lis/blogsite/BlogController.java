package com.lis.blogsite;

import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.server.ResponseStatusException;

@Controller
public class BlogController {
    private final BlogContentRepository repository;

    public BlogController(BlogContentRepository repository) {
        this.repository = repository;
    }

    @GetMapping("/")
    public String index(Model model) {
        model.addAttribute("posts", repository.findAll());
        return "index";
    }

    @GetMapping("/posts/{slug}")
    public String post(@PathVariable String slug, Model model) {
        BlogPost post = repository.findBySlug(slug)
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND));
        model.addAttribute("post", post);
        return "post";
    }
}

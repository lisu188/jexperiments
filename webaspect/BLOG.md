# A Minimal Servlet Session Experiment

## Why this experiment exists

This module is a very small servlet experiment packaged as a WAR. It contains a
single servlet, a mostly empty web descriptor, and a placeholder JSP. The servlet
writes an attribute into the HTTP session when `/servlet` is requested.

The name `webaspect` suggests the module may have been created while exploring
web containers or request/session behavior. The current code is minimal enough
that the most useful reading is as a servlet lifecycle and deployment sketch.

## How it works

`HelloServlet` is annotated with `@WebServlet("/servlet")`, so the servlet
container can discover it without a `web.xml` mapping. Its `doGet` method calls
`req.getSession().setAttribute(...)`, storing a string value under a session
attribute key.

The class also has a static initializer that starts a background thread. That
thread sleeps for 100 seconds and then calls `System.exit(0)`. This is unusual
for servlet code, but it makes the module an interesting lifecycle experiment:
loading the servlet class starts a process-level timer.

The WAR also contains a placeholder `index.jsp` and a minimal Java EE 3.1
`web.xml`. The Gradle build applies the `war` plugin and includes embedded
Tomcat as the servlet API provider, matching the old module intent.

## What to notice

The servlet itself demonstrates how little code is needed to attach state to an
HTTP session. The annotation declares the route, `getSession()` creates or
retrieves the session, and `setAttribute` stores server-side state associated
with that client.

The static initializer is the surprising part. In web applications, code that
exits the JVM from a servlet class is almost never appropriate outside of a
controlled experiment. It can terminate the whole container, not just one web
application. As a learning artifact, though, it makes class loading and servlet
container side effects impossible to miss.

## Sanity check

The module is packaged as a WAR by Gradle and has no tests. The static pass
found that the servlet uses `System.exit(0)` from a background thread started
during class initialization. The JSP is a placeholder and the web descriptor
contains no servlet mappings, relying on annotation scanning.

The repository-level Gradle wrapper build compiles this module successfully.

## Suggested next experiments

A sensible next step would be turning the session write into a visible page or
JSON response so the behavior can be verified from a browser. Another extension
would move lifecycle behavior into a `ServletContextListener`, which is the
container-oriented place to start and stop web-app resources.

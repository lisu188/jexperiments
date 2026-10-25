# A Minimal Servlet Session Experiment

## Why this experiment exists

This experiment shows how a small annotation-mapped servlet mutates HTTP session
state. It deliberately has almost no response logic: one GET request obtains a
session and writes one attribute. That makes servlet mapping, session creation,
and the boundary between request work and container lifecycle easy to inspect.

The module is a WAR project using the classic `javax.servlet` API. Its Tomcat
dependency is on the maintained 9.0 branch, which retains that namespace. Moving
to a Tomcat generation using `jakarta.servlet` would be an API migration rather
than a dependency-only update.

The former version also started a background timer in the servlet's static
initializer and called `System.exit(0)` after 100 seconds. That behavior has
been removed. Lazy servlet initialization could let an unauthenticated request
start the countdown, and the eventual exit terminated the entire container JVM,
including unrelated applications hosted in it. Documenting the timer did not
make that a safe default for a deployable servlet.

## Execution path

The servlet declares its HTTP mapping in an annotation:

```java
@WebServlet("/servlet")
public class HelloServlet extends HttpServlet {
```

The descriptor identifies a Servlet 3.1 web application and adds no explicit
servlet declaration. A compatible container can discover the annotation during
deployment. Depending on its configuration, the container can initialize the
servlet eagerly or when the first matching request arrives. Neither loading
path now schedules work or changes the process lifecycle.

`HttpServlet` dispatches GET requests to the override:

```java
@Override
protected void doGet(HttpServletRequest req, HttpServletResponse resp)
        throws ServletException, IOException {
    req.getSession().setAttribute("halo janusz", "UWAGA");
}
```

Calling `getSession()` obtains the current session or asks the container to
create one. `setAttribute` stores the string value under the same key on every
request. Repeated requests belonging to the same session replace that value;
requests belonging to different sessions operate on distinct session objects.

There is no response body, redirect, explicit status, or view rendering. A
browser can therefore receive an otherwise empty successful response while
the container creates session state and may emit a session cookie. Inspecting
only the body would miss the experiment's observable effect.

## Request work and process lifecycle

The servlet uses only the request and session APIs supplied by the container:

```java
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
```

There is no static initializer, private executor, timer thread, or call to
`System.exit`. That matters beyond the first request. A container can initialize
classes during deployment, annotation scanning, reloads, or test discovery.
Request-facing class initialization must not turn any of those operations into
a delayed shutdown of the host process.

If a future experiment needs scheduled work, it should use an explicitly
managed lifecycle component whose shutdown cancels that work. If an isolated
process-exit demonstration is desired, it belongs in a separately launched
program with a clear entrypoint. It should not run as a side effect of loading
an HTTP servlet.

Session mutation still has its own side effects. It can allocate server-side
state and it is shared by requests carrying the same session identifier:

```java
// The container owns allocation, identifiers, cookies, and session lifetime.
req.getSession().setAttribute("halo janusz", "UWAGA");
```

The stored value is an immutable string, so this example does not require
additional synchronization. Replacing it with a mutable object would require
the normal servlet concurrency analysis. The attribute key is legal, although
a named constant would be clearer if additional request paths used it.

## Regression testing without a container shutdown

`HelloServletTest` runs a probe in a separate JVM. The probe supplies request and
session proxies, invokes the actual GET handler, and checks the stored value:

```java
new HelloServlet().doGet(request, response);
if (!"UWAGA".equals(attributes.get("halo janusz"))) {
    throw new AssertionError("GET no longer writes the expected session attribute");
}
```

The parent test also requires the child process to exit promptly after that
request finishes. The old non-daemon timer kept it alive even though request
handling had returned, so this is a regression check for the unsafe lifecycle
behavior as well as preservation of the session example.

```java
assertTrue(process.waitFor(10, TimeUnit.SECONDS),
        "Servlet initialization/request must not leave a process-exit timer running");
```

The test forcibly destroys a child that exceeds the deadline. It does not wait
for the former 100-second shutdown and does not let a servlet experiment end
the test worker JVM. These proxies exercise the handler and initialization;
they do not establish annotation scanning, real HTTP dispatch, browser cookies,
or session expiration behavior in a running Tomcat instance.

## Runtime behavior and follow-up experiments

The module is still a minimal session demonstration. It provides no application
authentication, authorization policy, useful response content, or configurable
session behavior. Deployment must supply the container configuration and the
access controls appropriate to the surrounding application.

A useful next step is an integration test that deploys the WAR into Tomcat,
requests `/servlet` twice using the same cookie jar, and checks that both
requests address the same session. A second client should receive an independent
session. That would complement the focused handler test with real mapping and
cookie evidence.

Other extensions could return whether a session was created, set an explicit
inactivity timeout, and observe invalidation through a managed session listener.
Comparing eager and lazy servlet loading would also be instructive now that
both paths are free of process-wide shutdown side effects.

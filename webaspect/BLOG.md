# A Minimal Servlet Session Experiment

## Why this experiment exists

This experiment is a tiny servlet module that demonstrates two servlet
lifecycle side effects. First, loading the servlet class starts a background
thread that eventually exits the JVM. Second, handling a GET request creates or
retrieves an HTTP session and writes an attribute into it. There is almost no
application logic, which makes the lifecycle behavior impossible to miss.

The module name hints at web aspects, but the current source is best understood
as a servlet container experiment. It asks what happens when work is triggered
by class initialization rather than by an explicit application lifecycle hook.
It also shows how little code is required to mutate session state from a
request handler.

For experienced Java developers, this is a useful cautionary example. Static
initializers in servlet classes are global process behavior, not request-scoped
behavior. A servlet container may load a servlet because of annotation scanning,
first request handling, startup configuration, redeployment, or tooling. If
class loading starts a timer that calls `System.exit(0)`, the entire container
process becomes coupled to that loading moment.

## Execution path

The servlet is annotation-mapped:

```java
@WebServlet("/servlet")
public class HelloServlet extends HttpServlet {
```

The `web.xml` file is effectively empty, so the annotation is the mapping that
matters. In a Servlet 3.1-compatible container, annotation scanning can discover
the servlet without explicit descriptor entries.

The static initializer starts immediately when the class is initialized:

```java
static {
    new Thread() {
        public void run() {
            try {
                Thread.sleep(100000);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            System.exit(0);
        }
    }.start();
}
```

That code creates a new non-daemon thread, sleeps for 100 seconds, and then
terminates the JVM. It is independent of any request. If the servlet class is
loaded during deployment, the countdown begins before a user ever visits
`/servlet`.

```java
// Process-wide side effect: this does not stop only the web application.
Thread.sleep(100000);
System.exit(0);
```

Those two lines are the reason the post treats the module as unsafe outside an
isolated JVM.

The static block also has no guard around repeated class-loader creation. In a
normal single deployment, it runs once. During development reloads, container
rescans, or repeated tests with fresh class loaders, each initialized copy can
start its own countdown. That is a classic reason servlet code avoids unmanaged
threads in static initializers.

## Core code walkthrough

The request handler is intentionally small:

```java
@Override
protected void doGet(HttpServletRequest req, HttpServletResponse resp) throws ServletException, IOException {
    req.getSession().setAttribute("halo janusz", "UWAGA");
}
```

`req.getSession()` creates a session if one does not already exist. The
attribute key contains a space and the value is a string. There is no response
body, redirect, status change, or view rendering. The observable HTTP response
is therefore mostly the container default, plus any session cookie behavior the
container applies.

The imports show that this is classic servlet API code:

```java
import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
```

The module is configured as a WAR project in the root Gradle build, so it is
meant to compile against the `javax.servlet` API supplied by the embedded
Tomcat dependency used in the project.

The descriptor contains no servlet declaration:

```java
public class HelloServlet extends HttpServlet {
```

That absence is part of the experiment. The behavior is driven by annotation
metadata and class initialization side effects rather than XML wiring.

The empty descriptor still matters because it declares a Servlet 3.1 web
application in `web.xml`. Annotation scanning is available in that generation
of the servlet spec, so the annotation and descriptor are not contradictory.
The XML simply does not add behavior beyond identifying the web application
schema version.

## Important implementation details

Static initialization is the most important detail. It runs once per class
loader, not once per servlet instance and not once per request. In a servlet
container, redeploying the application can create a new class loader and run
the initializer again. Multiple deployments or reloads can therefore create
multiple countdown threads over time if old loaders are not cleaned up
promptly.

The thread is anonymous and unmanaged. The servlet does not keep a reference to
it, mark it daemon, interrupt it during undeploy, or coordinate it with
`ServletContextListener`. If the application is stopped before the sleep ends,
the thread may still hold the old webapp class loader until it exits.

The session mutation is also minimal. `getSession()` has side effects: it may
allocate server-side session state and cause a `Set-Cookie` header. Because the
handler does not write a body, a caller has to inspect cookies or server-side
session state to see the effect. The attribute name is legal but awkward for
conventional code because it is not a constant and contains a space.

```java
// getSession() may allocate server-side state and emit a session cookie.
req.getSession().setAttribute("halo janusz", "UWAGA");
```

The handler's entire visible purpose is this session mutation.

No synchronization is needed for the handler itself because each request only
touches the `HttpServletRequest` session object supplied by the container.
However, session attributes are shared across requests belonging to the same
session. If later code stored mutable objects under that key, normal servlet
concurrency rules would apply.

## Runtime behavior and caveats

This module is intentionally unsafe to run in a shared servlet container. Once
`HelloServlet` is initialized, the background thread will call `System.exit(0)`
after roughly 100 seconds. That exits the whole JVM, not just the web
application. In embedded Tomcat it stops the process. In an application server
hosting multiple apps, it can terminate unrelated workloads.

The timing is also container-dependent. Some containers initialize annotated
servlets lazily on first request. Others may load classes during deployment or
scanning. That means the 100-second timer is not reliably tied to a user action.

There are no tests, no response assertions, and no cleanup hooks. The module
compiles as a demonstration of servlet mechanics, but it should be treated as a
behavioral warning rather than a pattern. If you need delayed shutdown or
session instrumentation, those concerns belong in managed lifecycle components
with explicit configuration.

The safest way to explore the module is in an isolated JVM that can be allowed
to exit. Running it inside a shared IDE application server, a local container
used for other work, or a CI worker that expects the JVM to stay alive can
produce surprising failures because `System.exit(0)` is process-wide.

## Suggested next experiments

Move the delayed action into a `ServletContextListener` and cancel it in
`contextDestroyed`. Replace `System.exit(0)` with a logged event so the session
behavior can be tested safely. Add a response body that reports whether a
session was created. Store the session attribute key as a constant. Finally,
add an integration test that verifies `/servlet` creates a session without
allowing class initialization to terminate the test JVM.

Another useful variant would compare lazy and eager servlet loading. Adding
`loadOnStartup` through a descriptor or registration API would make the static
initializer timing deterministic, which would also make the danger easier to
demonstrate in documentation.

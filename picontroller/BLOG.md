# A Raspberry Pi Monitoring Dashboard

## Why this experiment exists

This experiment combines embedded Tomcat, WebSocket broadcasting, Pi4J GPIO
access, and simple system monitors into a Raspberry Pi dashboard. It starts
monitor threads for light, memory, and CPU temperature, exposes static web
resources through a servlet, and pushes value changes to connected WebSocket
sessions at `/info`.

The code is a hardware-bound sketch rather than a portable web application.
That is what makes it worth documenting carefully. It touches several runtime
surfaces that behave differently on a development laptop and on a Raspberry Pi:
GPIO pin provisioning, Pi4J system calls, embedded servlet lifecycle, Tomcat
WebSocket setup, static singleton state, and a `/kill` endpoint that stops the
server.

For experienced Java developers, the module is a compact example of application
lifecycle side effects. Starting the process does not merely bind an HTTP port;
it provisions pins, starts threads, accumulates monitor history, and begins
broadcasting state changes to every connected session.

## Execution path

The entrypoint is `com.lis.pi.MainClass`:

```java
public static void main(String[] args) {
    try {
        Monitor.startMonitors();
        TomcatManager.getInstance().getTomcat().start();
        TomcatManager.getInstance().getTomcat().getServer().await();
        Monitor.stopMonitors();
```

The ordering matters. Monitors start before Tomcat, so a WebSocket client that
connects immediately can receive current values. `await()` blocks the main
thread until Tomcat stops. `stopMonitors()` is after `await()`, so a forced JVM
exit or startup failure may skip clean monitor termination.

There is no dependency injection container. The startup sequence is plain Java
static calls and singleton access. That keeps the experiment easy to follow,
but it also means tests or alternate runtime configurations cannot replace the
Tomcat manager, monitor set, or Pi4J access points without editing source.

Tomcat is configured programmatically. The manager creates `new Tomcat()`, sets
port `8080`, creates a root `StandardContext`, and uses the system temp
directory as the document base. Servlets are added directly rather than through
a packaged WAR. The mappings are intentionally broad: `/kill` routes to
`KillServlet`, while `/` routes to `DefaultServlet`.

```java
// Embedded Tomcat is configured directly, not through a WAR descriptor.
this.tomcat = new Tomcat();
tomcat.setPort(8080);
final Context rootCtx = new StandardContext();
rootCtx.setPath("");
```

This setup makes startup self-contained, but it also hard-codes the port and
context shape into application code.

Requests to `/kill` stop the server, while all other root requests are handled
by the default resource servlet. That makes the dashboard self-contained, but
it also means routing, static resources, and administrative shutdown all live in
one embedded container setup.

## Core code walkthrough

The root context registers a WebSocket listener during the Tomcat lifecycle:

```java
if (rootCtx.getState().equals(LifecycleState.STARTING_PREP)) {
    if (!added) {
        rootCtx.getServletContext().addListener(
                new PiContextListener(rootCtx
                        .getServletContext()));
        added = true;
    }
}
```

`PiContextListener` extends Tomcat's `WsContextListener` and then reads the
`javax.websocket.server.ServerContainer` servlet-context attribute so it can
call `serverContainer.addEndpoint(new InfoEndpointConfig())`. The endpoint path
comes from `InfoEndpointConfig.getPath()`, which returns `"/info"`. That keeps
WebSocket registration code separate from the endpoint implementation.

Monitor startup creates three concrete monitors:

```java
public static void startMonitors() {
    monitors.add(new LightMonitor());
    monitors.add(new MemMonitor());
    monitors.add(new TempMonitor());
    for (Monitor monitor : monitors) {
        monitor.start();
    }
}
```

`LightMonitor` provisions two GPIO inputs, while memory and temperature monitors
poll Pi4J `SystemInfo` once per second.

## Important implementation details

The base `Monitor` class serializes updates with a lock and broadcasts only
when the value changes:

```java
protected void setValue(float value) {
    lock.lock();
    try {
        if (value != this.value) {
            this.value = value;
            valueChanged();
        }
    } finally {
        lock.unlock();
    }
}
```

`valueChanged()` updates the in-memory history and sends JSON text built by
`getValueString()`, which concatenates the monitor type and numeric value into a
small object payload. That is acceptable for enum-and-number payloads in the
sketch, but it would not scale to strings or richer data without escaping. New
WebSocket sessions receive all current monitor values in `onOpen`, then future
changes are pushed through `InfoEndpoint.broadcast`.

```java
// Manual JSON is safe only because the payload is enum-plus-number.
String json = "{" + "\"type\":\"" + type.toString()
        + "\",\"value\":" + String.valueOf(value) + "}";
return json;
```

The string construction is intentionally visible, which makes the serialization
shortcut easy to spot.

The static session set is synchronized for collection mutation, and the
`broadcast` method itself is synchronized. That avoids simple concurrent
modification problems, but it does not handle failed asynchronous sends, slow
clients, or session cleanup beyond `onClose`. A robust dashboard would remove
sessions on send failure and expose connection metrics.

```java
// Every current WebSocket session receives the same monitor update.
public static synchronized void broadcast(String message) {
    for (Session session : sessions) {
        session.getAsyncRemote().sendText(message);
    }
}
```

This is the fan-out point between monitor threads and connected browsers.

## Runtime behavior and caveats

This module is environment-bound. `LightMonitor` calls Pi4J GPIO APIs in its
constructor:

```java
lightPin1 = GpioFactory.getInstance().provisionDigitalInputPin(
        RaspiPin.GPIO_08, PinPullResistance.PULL_DOWN);
lightPin2 = GpioFactory.getInstance().provisionDigitalInputPin(
        RaspiPin.GPIO_09, PinPullResistance.PULL_DOWN);
```

On non-Raspberry Pi hardware, or without the required permissions and native
libraries, startup can fail before Tomcat is available. Even on a Pi, the code
assumes specific pins and pull-down wiring.

The polling monitors have their own environment assumptions. `MemMonitor`
divides `SystemInfo.getMemoryUsed()` by `SystemInfo.getMemoryTotal()`, rounds
the ratio to two decimal places, calls `setValue`, and sleeps for one second.
`TempMonitor` follows the same polling shape for CPU temperature. Those values
are good enough for a dashboard demo, but they are not a time-series monitoring
strategy with sampling metadata, error states, or retention policy.

There are also operational caveats. The application binds port `8080` with no
configuration option. `/kill` starts a thread that stops Tomcat after a short
sleep and has no authentication. Static singletons hold sessions and monitor
history. `MonitorData` clears all history after 10000 samples. The default
servlet returns silently when a resource is missing, so missing assets may look
like empty responses rather than clear 404s.

The code is therefore best read as a hardware and lifecycle experiment, not as
a secure dashboard template.

One more subtlety is resource serving. `DefaultServlet` resolves paths with
`ResourcesManager.getWebResourceAsStream(path)`, which delegates to
`getClass().getResourceAsStream(path)`. That means request paths map to
classpath resources relative to `com.lis.pi.web.ResourcesManager`, not to a
normal servlet document root. The behavior is compact, but it is different from
standard static-file serving.

## Suggested next experiments

Make the port, GPIO pins, and monitor set configurable. Replace manual JSON
construction with a JSON library or a small encoder that handles escaping. Add
authentication or remove the `/kill` servlet. Make monitor shutdown join the
threads and release GPIO resources. Finally, add a hardware abstraction layer so
memory and temperature monitors can be tested on non-Pi systems while GPIO
behavior is mocked.

Another practical extension would persist monitor samples outside the JVM,
perhaps to a small time-series file or database. That would turn the dashboard
from a live-only view into something that can answer questions about recent
history after a browser reconnects or the server restarts. A retention limit
could then be configured deliberately instead of being the current hard-coded
clear-after-10000-samples behavior.
Adding timestamps to broadcast messages would also help clients distinguish a
fresh sensor update from a value replayed during WebSocket connection setup or
reconnection after a temporary network drop.

A second useful extension would split the embedded server from monitor startup.
That would let the HTTP layer be tested with fake monitor data and would make
hardware initialization failures reportable through a controlled status page
instead of preventing the server from starting at all.
It would also let development machines run the web UI without pretending to
have GPIO hardware during local testing.

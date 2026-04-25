# A Raspberry Pi Monitoring Dashboard

## Why this experiment exists

This module combines embedded Tomcat, Pi4J, servlets, WebSockets, and a small
Bootstrap page into a Raspberry Pi monitoring dashboard. It watches memory,
temperature, and light sensor state, then broadcasts changes to connected web
clients.

The experiment sits at the boundary between Java server code and physical
hardware. It is not just a web app mockup: `LightMonitor` provisions GPIO pins
through Pi4J, while the temperature and memory monitors read system information
through Pi4J's `SystemInfo` API.

## How it works

`MainClass` starts the monitors, starts embedded Tomcat, waits for the server,
and stops monitors when the server exits. `TomcatManager` configures Tomcat on
port `8080`, installs a root context, maps a default servlet to `/`, and maps a
kill servlet to `/kill`.

The monitor model is centered on the abstract `Monitor` class. Each monitor is a
thread with a `ValueType` and a current float value. When a value changes, the
monitor updates `MonitorData` and broadcasts a JSON string through
`InfoEndpoint`.

`TempMonitor` polls CPU temperature once per second. `MemMonitor` polls memory
usage and sends a ratio. `LightMonitor` provisions GPIO pins 8 and 9 with
pull-down resistors, listens for digital state changes, and maps the two input
states to a light value from 0 to 2.

The web UI is served from resources under `com/lis/pi/web`. The Gradle build
keeps non-Java files under `src/main/java` on the runtime classpath so
`ResourcesManager` can load the HTML, JavaScript, CSS, and font files.

## What to notice

The compactness of the module comes from embedding the web server directly in
the Java process. There is no external servlet container to configure. The same
application owns hardware initialization, monitoring threads, HTTP resource
serving, and WebSocket broadcasting.

The monitor abstraction is also simple: each sensor decides how to produce
values, while the base class handles change notification. That keeps
WebSocket-specific code out of the sensor implementations.

## Sanity check

The Gradle build keeps the original embedded Tomcat, Log4j, and Pi4J
dependencies. There are no tests.

The code assumes Raspberry Pi hardware and GPIO access. It is unlikely to run
unchanged on a normal development machine. The default servlet silently returns
when a resource is missing, and the web page still contains visible Bootstrap
template content. The `/kill` endpoint stops Tomcat after a short delay and has
no access control, which is acceptable for an experiment but not for a deployed
service.

The repository-level Gradle wrapper build compiles this module successfully.

## Suggested next experiments

A strong next step would be adding a development mode that replaces Pi4J sensor
reads with simulated values, making the dashboard runnable without hardware.
Another useful extension would be serving correct content types and reducing the
Bootstrap template page to just the live monitoring UI.

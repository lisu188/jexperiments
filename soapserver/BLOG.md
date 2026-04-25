# A SOAP Weather Proxy around a JSON API

## Why this experiment exists

This module experiments with SOAP messages, servlets, and translating a JSON
weather API into a SOAP-shaped response. It contains both server-side servlet
code and client/proxy code. The server receives a SOAP request asking for
weather by location and unit format, calls OpenWeatherMap's daily forecast API,
parses the JSON, and returns weather data as SOAP elements.

The result is a useful snapshot of integration code: one protocol on the
outside, another protocol behind it, and a Java object model in between.

## How it works

`WeatherServlet` is the main server-side piece. Its `doPost` method delegates to
`sendWeatherInfo`, which reads MIME headers and creates a `SOAPMessage` from the
request body. The servlet looks for a `Weather` body element, extracts
`methodName`, `location`, and `unitFormat`, and handles the
`getWeatherForLocation` method by calling `getWeatherInformation`.

`getWeatherInformation` constructs an OpenWeatherMap URL, performs an HTTP GET,
reads the JSON response into a string, and passes it to `getWeatherDataFromJson`.
That method parses city metadata and forecast entries, then builds a SOAP
response containing city coordinates and a list of days with humidity, pressure,
wind, temperatures, description, and weather id.

`SoapWeatherClient` builds the matching SOAP request and posts it to a hardcoded
server URL. `WeatherClientProxy` is a separate servlet that reads a serialized
`WeatherQuery`, calls the SOAP client, and writes back a serialized
`WeatherMessage`.

## What to notice

The experiment shows translation at several layers. The server maps SOAP request
structure to an HTTP JSON API call, maps JSON fields to SOAP response elements,
and the proxy maps Java serialized objects to SOAP and back.

It also includes a checked-in JSON package, so the module is mostly
self-contained apart from servlet, Tomcat, and SOAP dependencies. That makes the
protocol work more visible than it would be behind a framework.

## Sanity check

The module is packaged as a WAR by Gradle. The current Gradle build includes
embedded Tomcat plus SOAP API/runtime dependencies needed on modern JDKs. There
are no tests.

The static pass found several runtime assumptions. The OpenWeatherMap endpoint
is the older `forecast/daily` URL and no API key is included. The SOAP client
posts to a hardcoded private-network address. The proxy uses Java object
serialization over HTTP, and exception handling mostly prints stack traces. This
is best treated as an integration experiment rather than a deployable service.

The repository-level Gradle wrapper build compiles this module successfully.

## Suggested next experiments

A useful next step would be replacing hardcoded URLs with configuration and
adding a small recorded JSON fixture for parser tests. Another extension would
define the SOAP request and response contract explicitly, then validate messages
before translating them.

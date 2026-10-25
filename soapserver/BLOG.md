# A SOAP Weather Proxy around a JSON API

The Gradle build uses Tomcat 9.0.123 for this module's existing `javax.servlet`
API. The former Tomcat 8.0.23 dependency belonged to an unsupported release
line; the current Java EE branch supplies maintained container fixes without
changing these servlet imports. See [dependency security notes](../SECURITY_DEPENDENCIES.md)
for the upstream sources and focused regression boundary.

## Why this experiment exists

This experiment wraps a JSON weather API with a SOAP-facing servlet and a
data-only HTTP proxy. It is a useful snapshot of integration code where several
wire formats meet: HTTP servlet requests, SAAJ `SOAPMessage` objects, JSON
responses from OpenWeatherMap, domain objects such as `WeatherMessage`, and
bounded form requests and JSON responses at the HTTP proxy servlet.

The module is not a modern weather service template. That is why the mechanics
are worth making explicit. The server parses a SOAP request, extracts a method
name and location properties, calls an external JSON endpoint, translates the
JSON payload back into SOAP elements, and writes the SOAP response to the
servlet output stream. A separate proxy accepts URL-encoded `location` and
`units` fields, builds an internal `WeatherQuery`, calls the SOAP client, and
returns the `WeatherMessage` as JSON.

For experienced Java developers, this code is a compact study in translation
boundaries. Every boundary has a schema, even if the schema is implicit in
string constants and element names. The experiment shows how quickly those
implicit contracts accumulate when SOAP, JSON, servlet APIs, and domain objects
are combined.

## Execution path

The main request handler is `hrab.WeatherServlet`. A POST request enters
`doPost` and delegates immediately:

```java
protected void doPost(HttpServletRequest req, HttpServletResponse res)
        throws ServletException, IOException {
    System.out.println("dostalem posta!");
    sendWeatherInfo(req, res);
}
```

`sendWeatherInfo` reconstructs MIME headers, builds a `SOAPMessage` from the
request input stream, finds the `Weather` body element, and extracts the method
and location fields. Only `getWeatherForLocation` is recognized. The handler
then reads `location` and `unitFormat`, calls `getWeatherInformation(location,
units)`, sets `Content-Type` to `text/xml`, and writes the reply with
`realReply.writeTo(os)`.

```java
// SOAP is written manually to the servlet response stream.
res.setHeader("Content-Type", "text/xml");
OutputStream os = res.getOutputStream();
realReply.writeTo(os);
os.flush();
```

That snippet is the whole response layer; there is no generated SOAP endpoint
or framework dispatcher.

There is no framework-level SOAP endpoint here. The servlet manually parses and
writes SAAJ messages.

Manual parsing makes control flow visible, but it also means the servlet has to
act as dispatcher, validator, translator, and response writer at the same time.
The recognized operation is encoded as a string comparison instead of a method
binding generated from WSDL. That is workable for one operation and brittle as
soon as more operations or versions are added.

## Core code walkthrough

The outbound JSON request is built with string concatenation:

```java
URL url = new URL(
        "http://api.openweathermap.org/data/2.5/forecast/daily?q="
                + locationQuery + "&mode=json&units=" + units
                + "&cnt=7");
```

The code opens an `HttpURLConnection`, reads the response into a string, and
passes that string to `getWeatherDataFromJson`. The JSON parser is the bundled
`json` package in the module rather than Jackson, Gson, or JSON-P.

The SOAP response root is created manually:

```java
Name listingElementName = soapFactory.createName("hahr",
        "WeatherResponse",
        "http://schemas.realhouses.com/listingSubmission");
SOAPBodyElement listingElement = body
        .addBodyElement(listingElementName);

SOAPElement humidityNode = day
        .addChildElement(WeatherEntry.COLUMN_HUMIDITY);
humidityNode.addTextNode("" + humidity);
SOAPElement pressureNode = day
        .addChildElement(WeatherEntry.COLUMN_PRESSURE);
pressureNode.addTextNode("" + pressure);
```

City fields are added as child elements, and forecast entries are added under
`Days/day`, using column-name constants from `WeatherEntry`.

Those constants act as the schema bridge between JSON names and SOAP element
names. The JSON field `speed`, for example, becomes the SOAP element named by
`WeatherEntry.COLUMN_WIND_SPEED`.

The domain object layer mirrors the same fields. `DayForecast` stores pressure,
humidity, wind speed, wind direction, high and low temperatures, description,
and weather id. `WeatherMessage` wraps the city name, coordinates, and list of
days. These legacy domain classes retain their serializable interfaces, but
the HTTP proxy never reads or writes Java object streams.

## Important implementation details

The client side constructs the SOAP request with the same element names the
servlet expects:

```java
SOAPElement methodNAme = listingElement.addChildElement("methodName");
methodNAme.addTextNode("getWeatherForLocation");
SOAPElement locationProperties = listingElement
        .addChildElement("locationProperties");
SOAPElement location = locationProperties.addChildElement("location");
location.addTextNode(locationTag);
```

The SOAP endpoint URL is hard-coded in `SoapWeatherClient` as
`http://192.168.43.95:8888/SOAP_sever/hrab.WeatherServlet`, and the call is
made with `con.call(message, url)`. That makes the client environment-specific.
It is fine for a local network experiment, but it prevents the code from being
portable without editing source.

```java
// Local-network endpoint baked into the client experiment.
String url = "http://192.168.43.95:8888/SOAP_sever/hrab.WeatherServlet";
SOAPMessage soapResponse = con.call(message, url);
```

The hard-coded address is worth documenting because it is the first runtime
blocker outside the original environment.

`WeatherClientProxy` adds another translation layer:

```java
byte[] body = source.readNBytes(MAX_QUERY_BYTES + 1);
if (body.length > MAX_QUERY_BYTES)
    throw new InvalidObjectException("Query is too large");
String key = URLDecoder.decode(pair.substring(0, equals), StandardCharsets.UTF_8);
String value = URLDecoder.decode(pair.substring(equals + 1), StandardCharsets.UTF_8);
```

The proxy accepts only flat form data: `location=Krak%C3%B3w&units=metric`.
POST requires `application/x-www-form-urlencoded`; GET uses the same fields
in its query string. The encoded body is bounded to 16 KiB before decoding,
unknown or duplicate fields are rejected, and the two nonblank strings are
limited to 256 and 32 Java characters. No request data selects classes or an
object graph. The legacy serialized-object request format is deliberately
unsupported; clients must migrate to these data fields. Responses use
`application/json` and UTF-8, generated from the known weather domain object.
Malformed forms return HTTP 400, incompatible media types return 415, and
upstream failures return 502 before a response body is opened. Regressions
exercise valid GET/POST translations with a synthetic weather provider,
invalid encodings, duplicate/unknown fields, size limits and legacy binary
input rejection without calling the original network endpoint.

`SoapWeatherClient.saveData` performs the reverse mapping from SOAP elements
back to Java objects. It finds the `WeatherResponse` body element, extracts
city fields, iterates `Days/day`, parses each child value, and constructs
`DayForecast` instances. The server and client therefore both carry mapping
logic, which is a maintenance risk if element names change.

## Runtime behavior and caveats

This module is highly environment-bound. The OpenWeatherMap URL uses the old
`forecast/daily` endpoint and does not include an API key. Modern deployments of
that API generally require authentication and may not support the same path.
The SOAP client points to a private `192.168.43.95` address and an application
path containing `SOAP_sever`, so it will not work unchanged outside the
original network.

Security caveats remain significant. Removing Java object serialization from
the proxy closes that input mechanism, but the servlets still lack
authentication and the SOAP servlet retains optimistic element lookups.
Missing SOAP elements can
become `NoSuchElementException` or `NullPointerException` paths. External HTTP
failures are mostly logged or printed rather than returned as SOAP faults.

There are also protocol caveats. The code manually mirrors element names in the
client and server instead of generating them from a WSDL or shared schema. It
sets `Content-Type` with `setHeader` instead of `setContentType`. The response
message is printed to standard output during construction, which is useful for
debugging but noisy in a servlet container.

The JSON request also omits URL encoding for the location query. A city name
with spaces or non-ASCII characters can produce an invalid or unintended URL.
The code catches `IOException` and logs a generic message, so requesters may see
an empty or broken SOAP response rather than a clear explanation. This is
another place where separating translation code from servlet code would make
testing and error reporting easier.

## Suggested next experiments

Parameterize the SOAP endpoint URL and external weather API key, then replace
the remaining unauthenticated HTTP connections with authenticated TLS. Add
SOAP fault responses for invalid requests and upstream failures. Extract the
JSON-to-domain and domain-to-SOAP mapping into testable methods. Finally, add a
small fixture JSON response so the translation can be validated without calling
the live weather API.

It would also be worth introducing a contract test that feeds a SOAP request
fixture into `WeatherServlet` and asserts the generated SOAP response structure.
That would document the implicit schema even before a WSDL or generated binding
is introduced, and it would protect the client/server element-name agreement.

The same fixture could drive the Java proxy, proving the object boundary and
SOAP boundary stay aligned together.

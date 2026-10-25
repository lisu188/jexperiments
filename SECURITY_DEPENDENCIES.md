# Embedded Java dependency security

The Pi controller, SOAP servlet experiment and WebAspect compile against Tomcat
9.0.123. Tomcat 8.0 reached end of support in 2018; the previous 8.0.23 dependency
could not receive current security fixes. Tomcat 9 is the supported Java EE branch
that preserves these experiments' `javax.servlet` and `javax.websocket` APIs.
The Pi controller's core and WebSocket artifacts use the same version.

The Pi controller uses Log4j Core 2.26.1, replacing 2.5. The old version falls in
the affected Log4Shell range. `LoggingSecurityTest` checks that a log message
containing a harmless system-property lookup remains literal. It exercises the
message-substitution mechanism without using JNDI, opening sockets or disclosing
environment values. The test fails with the previous 2.5 API/Core jars and passes
with 2.26.1.

These dependency updates do not establish safe deployment of the historical
applications. The Pi entrypoint requires physical hardware, and servlet examples
still need an appropriate access policy before deployment. Run the repository's
module tests and coverage gates in addition to focused compatibility checks.

Upstream sources checked on 2026-10-25:

- [Tomcat versions and support](https://tomcat.apache.org/whichversion.html)
- [Tomcat 9 security advisories](https://tomcat.apache.org/security-9.html)
- [Log4j release](https://logging.apache.org/log4j/2.x/download.html)
- [Apache Logging security advisories](https://logging.apache.org/security.html)

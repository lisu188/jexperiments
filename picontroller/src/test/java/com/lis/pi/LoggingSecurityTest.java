package com.lis.pi;

import org.apache.logging.log4j.Level;
import org.apache.logging.log4j.core.config.DefaultConfiguration;
import org.apache.logging.log4j.core.impl.Log4jLogEvent;
import org.apache.logging.log4j.core.layout.PatternLayout;
import org.apache.logging.log4j.message.SimpleMessage;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class LoggingSecurityTest {
    @Test
    void loggedMessagesDoNotEvaluateLookupExpressions() {
        String property = "jexperiments.logging.lookup.probe";
        String previous = System.getProperty(property);
        System.setProperty(property, "lookup-was-evaluated");
        try {
            String message = "untrusted ${sys:" + property + "}";
            PatternLayout layout = PatternLayout.newBuilder()
                    .withConfiguration(new DefaultConfiguration())
                    .withPattern("%m")
                    .build();
            Log4jLogEvent event = Log4jLogEvent.newBuilder()
                    .setLoggerName("security-regression")
                    .setLevel(Level.INFO)
                    .setMessage(new SimpleMessage(message))
                    .build();

            assertEquals(message, layout.toSerializable(event));
        } finally {
            if (previous == null) {
                System.clearProperty(property);
            } else {
                System.setProperty(property, previous);
            }
        }
    }
}

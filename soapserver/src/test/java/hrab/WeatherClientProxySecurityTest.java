package hrab;

import json.JSONObject;
import org.junit.jupiter.api.Test;
import javax.servlet.ReadListener;
import javax.servlet.ServletInputStream;
import javax.servlet.ServletOutputStream;
import javax.servlet.WriteListener;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.*;
import java.lang.reflect.Proxy;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import static org.junit.jupiter.api.Assertions.*;

class WeatherClientProxySecurityTest {
    private static ByteArrayInputStream form(String text) {
        return new ByteArrayInputStream(text.getBytes(StandardCharsets.UTF_8));
    }
    @Test void acceptsOnlyBoundedDataFields() throws Exception {
        WeatherQuery query = WeatherClientProxy.readQuery(form("location=Krak%C3%B3w&units=metric"));
        assertEquals("Kraków", query.getLocation());
        assertEquals("metric", query.getUnits());
        assertEquals("New York", WeatherClientProxy.readQuery(form("location=New+York&units=imperial")).getLocation());
    }
    @Test void rejectsLegacyObjectBytesAndMalformedOrAmbiguousForms() {
        assertThrows(IOException.class, () -> WeatherClientProxy.readQuery(new ByteArrayInputStream(new byte[]{(byte) 0xac, (byte) 0xed, 0, 5})));
        for (String value : List.of("", "location=Krakow", "location=&units=metric", "location=Krakow&units=", "location=a&location=b&units=metric", "class=Anything&location=a&units=b", "location=%xx&units=metric")) {
            assertThrows(IOException.class, () -> WeatherClientProxy.readQuery(form(value)), value);
        }
    }
    @Test void rejectsOversizedBodiesAndFields() {
        assertThrows(IOException.class, () -> WeatherClientProxy.readQuery(new ByteArrayInputStream(new byte[WeatherClientProxy.MAX_QUERY_BYTES + 1])));
        assertThrows(IOException.class, () -> WeatherClientProxy.readQuery(form("location=" + "x".repeat(257) + "&units=metric")));
        assertThrows(IOException.class, () -> WeatherClientProxy.readQuery(form("location=Krakow&units=" + "x".repeat(33))));
    }
    static class Exchange {
        final ByteArrayOutputStream output = new ByteArrayOutputStream();
        final AtomicInteger error = new AtomicInteger();
        final AtomicReference<String> type = new AtomicReference<>();
        HttpServletRequest request(String contentType, String body, String query) {
            ByteArrayInputStream bytes = form(body);
            ServletInputStream input = new ServletInputStream() {
                public int read() { return bytes.read(); }
                public boolean isFinished() { return bytes.available() == 0; }
                public boolean isReady() { return true; }
                public void setReadListener(ReadListener ignored) { throw new UnsupportedOperationException(); }
            };
            return (HttpServletRequest) Proxy.newProxyInstance(getClass().getClassLoader(), new Class<?>[]{HttpServletRequest.class}, (proxy, method, args) -> {
                switch (method.getName()) {
                    case "getInputStream": return input;
                    case "getContentType": return contentType;
                    case "getQueryString": return query;
                    default: throw new AssertionError(method.getName());
                }
            });
        }
        HttpServletResponse response() {
            ServletOutputStream stream = new ServletOutputStream() {
                public void write(int value) { output.write(value); }
                public boolean isReady() { return true; }
                public void setWriteListener(WriteListener ignored) { throw new UnsupportedOperationException(); }
            };
            return (HttpServletResponse) Proxy.newProxyInstance(getClass().getClassLoader(), new Class<?>[]{HttpServletResponse.class}, (proxy, method, args) -> {
                switch (method.getName()) {
                    case "sendError": error.set((Integer) args[0]); return null;
                    case "setContentType": type.set((String) args[0]); return null;
                    case "setCharacterEncoding": assertEquals("UTF-8", args[0]); return null;
                    case "setContentLength": return null;
                    case "getOutputStream": return stream;
                    default: throw new AssertionError(method.getName());
                }
            });
        }
    }
    private static WeatherClientProxy fixture(AtomicInteger calls) {
        return new WeatherClientProxy() {
            protected WeatherMessage fetchWeather(WeatherQuery query) {
                calls.incrementAndGet();
                return new WeatherMessage(query.getLocation(), 19.94, 50.06, List.of());
            }
        };
    }
    @Test void validPostAndGetReturnJsonWithoutCallingALiveService() throws Exception {
        for (boolean get : new boolean[]{false, true}) {
            Exchange exchange = new Exchange();
            AtomicInteger calls = new AtomicInteger();
            String form = "location=Krak%C3%B3w&units=metric";
            HttpServletRequest request = exchange.request("application/x-www-form-urlencoded; charset=UTF-8", form, form);
            if (get) fixture(calls).doGet(request, exchange.response());
            else fixture(calls).doPost(request, exchange.response());
            assertEquals(1, calls.get());
            assertEquals(0, exchange.error.get());
            assertEquals("application/json", exchange.type.get());
            assertEquals("Kraków", new JSONObject(exchange.output.toString(StandardCharsets.UTF_8)).getString("cityName"));
        }
    }
    @Test void invalidPostDoesNotReachTheWeatherService() throws Exception {
        for (String type : new String[]{"application/x-java-serialized-object", "application/x-www-form-urlencoded"}) {
            Exchange exchange = new Exchange();
            AtomicInteger calls = new AtomicInteger();
            fixture(calls).doPost(exchange.request(type, "invalid", null), exchange.response());
            assertEquals(type.equals("application/x-www-form-urlencoded") ? 400 : 415, exchange.error.get());
            assertEquals(0, calls.get());
            assertEquals(0, exchange.output.size());
        }
    }
}

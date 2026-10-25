package hrab;

import json.JSONObject;
import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import javax.xml.soap.SOAPException;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InvalidObjectException;
import java.net.URLDecoder;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;

/** A bounded form-in, JSON-out facade for the legacy SOAP weather client. */
@WebServlet("/hrab.WeatherClientProxy")
public class WeatherClientProxy extends HttpServlet {
    private static final long serialVersionUID = 1L;
    static final int MAX_QUERY_BYTES = 16 * 1024;

    static WeatherQuery readQuery(InputStream source) throws IOException {
        byte[] body = source.readNBytes(MAX_QUERY_BYTES + 1);
        if (body.length > MAX_QUERY_BYTES) throw new InvalidObjectException("Query is too large");
        // A flat form has no class names, object hooks or recursive object graph.
        String form = StandardCharsets.UTF_8.newDecoder().decode(ByteBuffer.wrap(body)).toString();
        Map<String, String> fields = new HashMap<>();
        try {
            for (String pair : form.split("&", -1)) {
                if (pair.isEmpty()) continue;
                int equals = pair.indexOf('=');
                if (equals < 1) throw new InvalidObjectException("Expected form fields");
                String key = URLDecoder.decode(pair.substring(0, equals), StandardCharsets.UTF_8);
                String value = URLDecoder.decode(pair.substring(equals + 1), StandardCharsets.UTF_8);
                if ((!key.equals("location") && !key.equals("units")) || fields.putIfAbsent(key, value) != null)
                    throw new InvalidObjectException("Unexpected or duplicate form field");
            }
        } catch (IllegalArgumentException invalidEncoding) {
            throw new InvalidObjectException("Invalid form encoding");
        }
        String location = fields.get("location");
        String units = fields.get("units");
        if (location == null || location.isBlank() || location.length() > 256 ||
                units == null || units.isBlank() || units.length() > 32)
            throw new InvalidObjectException("Invalid weather query fields");
        return new WeatherQuery(location, units);
    }

    protected WeatherMessage fetchWeather(WeatherQuery query) throws SOAPException, IOException {
        return SoapWeatherClient.getWeatherData(query.getLocation(), query.getUnits());
    }

    private void writeWeather(WeatherQuery query, HttpServletResponse response) throws IOException {
        final byte[] body;
        try {
            String json = new JSONObject(fetchWeather(query)).toString();
            if (json == null) throw new IllegalStateException("Invalid weather response");
            body = json.getBytes(StandardCharsets.UTF_8);
        } catch (SOAPException | IOException | RuntimeException upstreamFailure) {
            response.sendError(HttpServletResponse.SC_BAD_GATEWAY, "Weather service unavailable");
            return;
        }
        response.setContentType("application/json");
        response.setCharacterEncoding("UTF-8");
        response.setContentLength(body.length);
        response.getOutputStream().write(body);
    }

    @Override
    protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        WeatherQuery query;
        try {
            String form = request.getQueryString();
            if (form == null || form.length() > MAX_QUERY_BYTES)
                throw new InvalidObjectException("Missing or oversized query");
            query = readQuery(new ByteArrayInputStream(form.getBytes(StandardCharsets.UTF_8)));
        } catch (IOException invalidQuery) {
            response.sendError(HttpServletResponse.SC_BAD_REQUEST, "Invalid weather query");
            return;
        }
        writeWeather(query, response);
    }

    @Override
    protected void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        String contentType = request.getContentType();
        if (contentType == null || !contentType.split(";", 2)[0].trim().equalsIgnoreCase("application/x-www-form-urlencoded")) {
            response.sendError(HttpServletResponse.SC_UNSUPPORTED_MEDIA_TYPE, "Use application/x-www-form-urlencoded");
            return;
        }
        WeatherQuery query;
        try {
            query = readQuery(request.getInputStream());
        } catch (IOException invalidQuery) {
            response.sendError(HttpServletResponse.SC_BAD_REQUEST, "Invalid weather query");
            return;
        }
        writeWeather(query, response);
    }
}

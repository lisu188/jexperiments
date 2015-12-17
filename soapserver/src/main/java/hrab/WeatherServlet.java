package hrab;

import json.JSONArray;
import json.JSONException;
import json.JSONObject;

import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import javax.xml.soap.*;
import java.io.*;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.Enumeration;
import java.util.Iterator;
import java.util.StringTokenizer;

/**
 * Servlet implementation class hrab.WeatherServlet
 */
@WebServlet("/hrab.WeatherServlet")
public class WeatherServlet extends HttpServlet {
    private static final long serialVersionUID = 1L;
    private static final String LOG_TAG = WeatherServlet.class.getSimpleName();

    /**
     * @see HttpServlet#HttpServlet()
     */
    public WeatherServlet() {
        super();
    }

    private static void log(String className, String msg) {
        System.out.println("[" + className + "]" + " " + msg);
    }

    /**
     * @see HttpServlet#doGet(HttpServletRequest request, HttpServletResponse
     * response)
     */
    protected void doGet(HttpServletRequest request,
                         HttpServletResponse response) throws ServletException, IOException {
        System.out.println("ktos sie podlaczyl getem");
    }

    /**
     * @see HttpServlet#doPost(HttpServletRequest request, HttpServletResponse
     * response)
     */
    protected void doPost(HttpServletRequest req, HttpServletResponse res)
            throws ServletException, IOException {
        System.out.println("dostalem posta!");
        sendWeatherInfo(req, res);
    }

    private SOAPMessage getWeatherInformation(String... params) {

        // If there's no zip code, there's nothing to look up. Verify size of
        // params.
        if (params.length == 0) {
            return null;
        }
        String locationQuery = params[0];
        SOAPMessage message = null;
        // These two need to be declared outside the try/catch
        // so that they can be closed in the finally block.
        HttpURLConnection urlConnection = null;
        BufferedReader reader = null;

        // Will contain the raw JSON response as a string.
        String forecastJsonStr = null;
        String units = params[1];

        try {
            URL url = new URL(
                    "http://api.openweathermap.org/data/2.5/forecast/daily?q="
                            + locationQuery + "&mode=json&units=" + units
                            + "&cnt=7");
            System.out.println(url);
            // Create the request to OpenWeatherMap, and open the connection
            urlConnection = (HttpURLConnection) url.openConnection();
            urlConnection.setRequestMethod("GET");
            urlConnection.connect();

            // Read the input stream into a String
            InputStream inputStream = urlConnection.getInputStream();
            StringBuilder buffer = new StringBuilder();
            if (inputStream == null) {
                // Nothing to do.
                return null;
            }
            reader = new BufferedReader(new InputStreamReader(inputStream));

            String line;
            while ((line = reader.readLine()) != null) {
                // Since it's JSON, adding a newline isn't necessary (it won't
                // affect parsing)
                // But it does make debugging a *lot* easier if you print out
                // the completed
                // buffer for debugging.
                buffer.append(line + "\n");
            }

            if (buffer.length() == 0) {
                // Stream was empty. No point in parsing.
                return null;
            }
            forecastJsonStr = buffer.toString();
            message = getWeatherDataFromJson(forecastJsonStr, locationQuery);
        } catch (IOException e) {
            log(LOG_TAG, "Error ");
            // If the code didn't successfully get the weather data, there's no
            // point in attempting
            // to parse it.
        } catch (JSONException e) {
            log(LOG_TAG, e.getMessage());
            e.printStackTrace();
        } catch (SOAPException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        } finally {
            if (urlConnection != null) {
                urlConnection.disconnect();
            }
            if (reader != null) {
                try {
                    reader.close();
                } catch (final IOException e) {
                    log(LOG_TAG, "Error closing stream");
                }
            }
        }
        return message;
    }

    private SOAPMessage getWeatherDataFromJson(String forecastJsonStr,
                                               String locationQuery) throws SOAPException {
        MessageFactory factory = MessageFactory.newInstance();
        SOAPFactory soapFactory = SOAPFactory.newInstance();
        SOAPMessage message = factory.createMessage();
        SOAPBody body = message.getSOAPBody();

        // These are the names of the JSON objects that need to be extracted.

        // Location information
        final String OWM_CITY = "city";
        final String OWM_CITY_NAME = "name";
        final String OWM_COORD = "coord";

        // Location coordinate
        final String OWM_LATITUDE = "lat";
        final String OWM_LONGITUDE = "lon";

        // Weather information. Each day's forecast info is an element of the
        // "list" array.
        final String OWM_LIST = "list";

        final String OWM_PRESSURE = "pressure";
        final String OWM_HUMIDITY = "humidity";
        final String OWM_WINDSPEED = "speed";
        final String OWM_WIND_DIRECTION = "deg";

        // All temperatures are children of the "temp" object.
        final String OWM_TEMPERATURE = "temp";
        final String OWM_MAX = "max";
        final String OWM_MIN = "min";

        final String OWM_WEATHER = "weather";
        final String OWM_DESCRIPTION = "main";
        final String OWM_WEATHER_ID = "id";

        try {
            JSONObject forecastJson = new JSONObject(forecastJsonStr);
            JSONArray weatherArray = forecastJson.getJSONArray(OWM_LIST);

            JSONObject cityJson = forecastJson.getJSONObject(OWM_CITY);
            String cityName = cityJson.getString(OWM_CITY_NAME);

            JSONObject cityCoord = cityJson.getJSONObject(OWM_COORD);
            double cityLatitude = cityCoord.getDouble(OWM_LATITUDE);
            double cityLongitude = cityCoord.getDouble(OWM_LONGITUDE);

            // long locationId = addLocation(locationQuery, cityName,
            // cityLatitude, cityLongitude);

            Name listingElementName = soapFactory.createName("hahr",
                    "WeatherResponse",
                    "http://schemas.realhouses.com/listingSubmission");
            SOAPBodyElement listingElement = body
                    .addBodyElement(listingElementName);
            SOAPElement cityNameNode = listingElement
                    .addChildElement("cityName");
            cityNameNode.addTextNode(cityName);
            SOAPElement cityLatitudeNode = listingElement
                    .addChildElement("cityLatitude");
            cityLatitudeNode.addTextNode("" + cityLatitude);
            SOAPElement cityLongitudeNode = listingElement
                    .addChildElement("cityLongitude");
            cityLongitudeNode.addTextNode("" + cityLongitude);

            SOAPElement daysNodes = listingElement.addChildElement("Days");
            for (int i = 0; i < weatherArray.length(); i++) {
                // These are the values that will be collected.
                SOAPElement day = daysNodes.addChildElement("day");
                double pressure;
                int humidity;
                double windSpeed;
                double windDirection;

                double high;
                double low;

                String description;
                int weatherId;

                // Get the JSON object representing the day
                JSONObject dayForecast = weatherArray.getJSONObject(i);

                // Cheating to convert this to UTC time, which is what we want
                // anyhow

                pressure = dayForecast.getDouble(OWM_PRESSURE);
                humidity = dayForecast.getInt(OWM_HUMIDITY);
                windSpeed = dayForecast.getDouble(OWM_WINDSPEED);
                windDirection = dayForecast.getDouble(OWM_WIND_DIRECTION);

                // Description is in a child array called "weather", which is 1
                // element long.
                // That element also contains a weather code.
                JSONObject weatherObject = dayForecast
                        .getJSONArray(OWM_WEATHER).getJSONObject(0);
                description = weatherObject.getString(OWM_DESCRIPTION);
                weatherId = weatherObject.getInt(OWM_WEATHER_ID);

                // Temperatures are in a child object called "temp". Try not to
                // name variables
                // "temp" when working with temperature. It confuses everybody.
                JSONObject temperatureObject = dayForecast
                        .getJSONObject(OWM_TEMPERATURE);
                high = temperatureObject.getDouble(OWM_MAX);
                low = temperatureObject.getDouble(OWM_MIN);

                // day.addChildElement(hrab.WeatherEntry.COLUMN_LOC_KEY, locationId);

                SOAPElement humidityNode = day
                        .addChildElement(WeatherEntry.COLUMN_HUMIDITY);
                humidityNode.addTextNode("" + humidity);
                SOAPElement pressureNode = day
                        .addChildElement(WeatherEntry.COLUMN_PRESSURE);
                pressureNode.addTextNode("" + pressure);
                SOAPElement windSpeedNode = day
                        .addChildElement(WeatherEntry.COLUMN_WIND_SPEED);
                windSpeedNode.addTextNode("" + windSpeed);
                SOAPElement windDirectionNode = day
                        .addChildElement(WeatherEntry.COLUMN_DEGREES);
                windDirectionNode.addTextNode("" + windDirection);
                SOAPElement highTempNode = day
                        .addChildElement(WeatherEntry.COLUMN_MAX_TEMP);
                highTempNode.addTextNode("" + high);
                SOAPElement lowTempNode = day
                        .addChildElement(WeatherEntry.COLUMN_MIN_TEMP);
                lowTempNode.addTextNode("" + low);
                SOAPElement descriptionNode = day
                        .addChildElement(WeatherEntry.COLUMN_SHORT_DESC);
                descriptionNode.addTextNode(description);
                SOAPElement weatherIdNode = day
                        .addChildElement(WeatherEntry.COLUMN_WEATHER_ID);
                weatherIdNode.addTextNode("" + weatherId);

            }

        } catch (JSONException e) {
            e.printStackTrace();
        }
        try {
            message.writeTo(System.out);
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        return message;
    }

    private void sendWeatherInfo(HttpServletRequest req, HttpServletResponse res)
            throws IOException {
        MimeHeaders mimeHeaders = new MimeHeaders();
        Enumeration en = req.getHeaderNames();
        while (en.hasMoreElements()) {
            String headerName = (String) en.nextElement();
            String headerVal = req.getHeader(headerName);
            StringTokenizer tk = new StringTokenizer(headerVal, ",");
            while (tk.hasMoreTokens()) {
                mimeHeaders.addHeader(headerName, tk.nextToken().trim());
            }
        }
        SOAPMessage realReply = null;
        try {
            MessageFactory messageFactory = MessageFactory.newInstance();
            SOAPMessage message = messageFactory.createMessage(mimeHeaders,
                    req.getInputStream());
            SOAPFactory soapFactory = SOAPFactory.newInstance();
            SOAPBody body = message.getSOAPBody();
            Name listingElName = soapFactory.createName("hahr", "Weather",
                    "http://schemas.realhouses.com/listingSubmission");
            Iterator listings = body.getChildElements(listingElName);
            while (listings.hasNext()) {
                // The listing and its ID
                SOAPElement listing = (SOAPElement) listings.next();
                // The listing agency name
                Iterator ageIt = listing.getChildElements(soapFactory
                        .createName("methodName"));
                SOAPElement methodName = (SOAPElement) ageIt.next();
                String methodNameString = methodName.getValue();
                if (methodNameString.equals("getWeatherForLocation")) {
                    Iterator propertiesIt = listing
                            .getChildElements(soapFactory
                                    .createName("locationProperties"));
                    SOAPElement prop = (SOAPElement) propertiesIt.next();
                    System.out.println(prop.getValue());
                    Iterator locationIt = prop.getChildElements(soapFactory
                            .createName("location"));
                    Iterator unitsIt = prop.getChildElements(soapFactory
                            .createName("unitFormat"));
                    SOAPElement locationElem = (SOAPElement) locationIt.next();
                    SOAPElement unitsElem = (SOAPElement) unitsIt.next();
                    System.out.println(locationElem.getValue()
                            + unitsElem.getValue());
                    realReply = getWeatherInformation(locationElem.getValue(),
                            unitsElem.getValue());
                }
            }
            res.setHeader("Content-Type", "text/xml");
            OutputStream os = res.getOutputStream();
            realReply.writeTo(os);
            os.flush();

        } catch (SOAPException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }
}

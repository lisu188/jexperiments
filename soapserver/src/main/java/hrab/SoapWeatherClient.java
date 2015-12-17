package hrab;

import json.JSONException;

import javax.xml.soap.*;
import java.io.IOException;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;

public class SoapWeatherClient {

    public static void main(String[] args) throws UnsupportedOperationException, SOAPException, IOException {

    }

    public static WeatherMessage getWeatherData(String locationTag, String units)
            throws UnsupportedOperationException, SOAPException, IOException {
        SOAPConnectionFactory spConFactory = SOAPConnectionFactory
                .newInstance();
        SOAPConnection con = spConFactory.createConnection();
        SOAPFactory soapFactory = SOAPFactory.newInstance();

        MessageFactory factory = MessageFactory.newInstance();

        SOAPMessage message = factory.createMessage();

        SOAPBody body = message.getSOAPBody();
        Name listingElementName = soapFactory.createName("hahr", "Weather",
                "http://schemas.realhouses.com/listingSubmission");
        SOAPBodyElement listingElement = body
                .addBodyElement(listingElementName);
        SOAPElement methodNAme = listingElement.addChildElement("methodName");
        methodNAme.addTextNode("getWeatherForLocation");
        SOAPElement locationProperties = listingElement
                .addChildElement("locationProperties");
        SOAPElement location = locationProperties.addChildElement("location");
        location.addTextNode(locationTag);
        SOAPElement unitFormat = locationProperties
                .addChildElement("unitFormat");
        unitFormat.addTextNode(units);
        message.writeTo(System.out);
        String url = "http://192.168.43.95:8888/SOAP_sever/hrab.WeatherServlet";
        SOAPMessage soapResponse = con.call(message, url);

        WeatherMessage decodedResponse = saveData(soapResponse, null);
        System.out.println(decodedResponse);
        return decodedResponse;
    }

    private static WeatherMessage saveData(SOAPMessage message,
                                           String locationSetting) throws JSONException, SOAPException {
        MessageFactory messageFactory = MessageFactory.newInstance();
        SOAPFactory soapFactory = SOAPFactory.newInstance();
        SOAPBody body = message.getSOAPBody();
        Name listingElName = soapFactory.createName("hahr", "WeatherResponse",
                "http://schemas.realhouses.com/listingSubmission");
        Iterator listings = body.getChildElements(listingElName);

        String cityName = "default name";
        double cityLatitude = 0;
        double cityLongitude = 0;
        List<DayForecast> forecastList = new LinkedList<>();
        try {

            while (listings.hasNext()) {
                SOAPElement listing = (SOAPElement) listings.next();
                Iterator cityNameIt = listing.getChildElements(soapFactory
                        .createName("cityName"));
                SOAPElement cityNameElement = (SOAPElement) cityNameIt.next();
                cityName = cityNameElement.getValue();

                Iterator latitudeIt = listing.getChildElements(soapFactory
                        .createName("cityLatitude"));
                SOAPElement cityLatitudeElement = (SOAPElement) latitudeIt.next();
                cityLatitude = Double.parseDouble(cityLatitudeElement.getValue());

                Iterator longitudeIt = listing.getChildElements(soapFactory
                        .createName("cityLongitude"));
                SOAPElement cityLongitudeElement = (SOAPElement) longitudeIt.next();
                cityLongitude = Double.parseDouble(cityLongitudeElement.getValue());

                Iterator weatherIterator = listing.getChildElements(soapFactory
                        .createName("Days"));
                SOAPElement dayElements = (SOAPElement) weatherIterator.next();
                Iterator daysIterator = dayElements.getChildElements(soapFactory
                        .createName("day"));

                while (daysIterator.hasNext()) {
                    // These are the values that will be collected.
                    SOAPElement dayForecast = (SOAPElement) daysIterator.next();
                    long dateTime;
                    double pressure;
                    int humidity;
                    double windSpeed;
                    double windDirection;

                    double high;
                    double low;

                    String description;
                    int weatherId;

                    Iterator pressureIterator = dayForecast.getChildElements(soapFactory.createName(WeatherEntry.COLUMN_PRESSURE));
                    SOAPElement pressureElement = (SOAPElement) pressureIterator.next();
                    pressure = Double.parseDouble(pressureElement.getValue());

                    Iterator humidityIterator = dayForecast.getChildElements(soapFactory.createName(WeatherEntry.COLUMN_HUMIDITY));
                    SOAPElement humidityElement = (SOAPElement) humidityIterator.next();
                    humidity = Integer.parseInt(humidityElement.getValue());

                    Iterator windSpeedIterator = dayForecast.getChildElements(soapFactory.createName(WeatherEntry.COLUMN_WIND_SPEED));
                    SOAPElement windSpeedElement = (SOAPElement) windSpeedIterator.next();
                    windSpeed = Double.parseDouble(windSpeedElement.getValue());

                    Iterator windDirectionIterator = dayForecast.getChildElements(soapFactory.createName(WeatherEntry.COLUMN_DEGREES));
                    SOAPElement windDirectionElement = (SOAPElement) windDirectionIterator.next();
                    windDirection = Double.parseDouble(windDirectionElement.getValue());

                    Iterator descriptionIterator = dayForecast.getChildElements(soapFactory.createName(WeatherEntry.COLUMN_SHORT_DESC));
                    SOAPElement descriptionElement = (SOAPElement) descriptionIterator.next();
                    description = descriptionElement.getValue();

                    Iterator weatherIdIterator = dayForecast.getChildElements(soapFactory.createName(WeatherEntry.COLUMN_WEATHER_ID));
                    SOAPElement weatherIdElement = (SOAPElement) weatherIdIterator.next();
                    weatherId = Integer.parseInt(weatherIdElement.getValue());

                    Iterator highIterator = dayForecast.getChildElements(soapFactory.createName(WeatherEntry.COLUMN_MAX_TEMP));
                    SOAPElement highElement = (SOAPElement) highIterator.next();
                    high = Double.parseDouble(highElement.getValue());

                    Iterator lowIterator = dayForecast.getChildElements(soapFactory.createName(WeatherEntry.COLUMN_MIN_TEMP));
                    SOAPElement lowElement = (SOAPElement) lowIterator.next();
                    low = Double.parseDouble(lowElement.getValue());
                    DayForecast dayForecastObject = new DayForecast(pressure, humidity, windSpeed, windDirection, high, low, description, weatherId);
                    forecastList.add(dayForecastObject);
                }
            }

            int inserted = 0;
            // add to database
            /**if (cVVector.size() > 0) {
             ContentValues[] cvArray = new ContentValues[cVVector.size()];
             cVVector.toArray(cvArray);
             inserted = mContext.getContentResolver().bulkInsert(
             hrab.WeatherEntry.CONTENT_URI, cvArray);
             }

             Log.d(LOG_TAG, "FetchWeatherTask Complete. " + inserted
             + " Inserted");*/

        } catch (JSONException e) {
            e.printStackTrace();
        }

        WeatherMessage weatherMessage = new WeatherMessage(cityName, cityLongitude, cityLatitude, forecastList);
        return weatherMessage;
    }
}



import hrab.WeatherQuery;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import javax.xml.soap.SOAPException;

/**
 * Servlet implementation class WeatherClientProxy
 */
@WebServlet("/WeatherClientProxy")
public class WeatherClientProxy extends HttpServlet {
	private static final long serialVersionUID = 1L;
       
    /**
     * @see HttpServlet#HttpServlet()
     */
    public WeatherClientProxy() {
        super();
        // TODO Auto-generated constructor stub
    }

	/**
	 * @see HttpServlet#doGet(HttpServletRequest request, HttpServletResponse response)
	 */
	protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
		doPost(request, response);
	}

	/**
	 * @see HttpServlet#doPost(HttpServletRequest request, HttpServletResponse response)
	 */
	protected void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
		ObjectInputStream in = new ObjectInputStream(request.getInputStream());
		WeatherQuery query = null;
		try {
		    query = (WeatherQuery) in.readObject();
		} catch (ClassNotFoundException e) {
		    e.printStackTrace();
		}
		in.close();

		ObjectOutputStream oos = new ObjectOutputStream(response.getOutputStream());
		try {
			oos.writeObject(SoapWeatherClient.getWeatherData(query.getLocation(), query.getUnits()));
		} catch (UnsupportedOperationException | SOAPException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		oos.close();
	}

}

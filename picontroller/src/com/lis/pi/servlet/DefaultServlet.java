package com.lis.pi.servlet;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import com.lis.pi.web.ResourcesManager;

public class DefaultServlet extends HttpServlet {
	public long copy(InputStream from, OutputStream to) throws IOException {
		byte[] buf = new byte[1024];
		long total = 0;
		while (true) {
			int r = from.read(buf);
			if (r == -1) {
				break;
			}
			to.write(buf, 0, r);
			total += r;
		}
		return total;
	}

	@Override
	protected void doGet(HttpServletRequest req, HttpServletResponse resp)
			throws ServletException, IOException {
		String path = req.getRequestURI().substring(
				req.getContextPath().length() + 1);

		InputStream in = ResourcesManager.getInstance().getWebResourceAsStream(
				path);
		if (in == null) {
			return;
		}
		OutputStream out = resp.getOutputStream();
		copy(in, out);
	}
}

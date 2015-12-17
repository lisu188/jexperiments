package com.lis.distributed.thread.pool.client;

import java.io.IOException;
import java.net.Socket;

import com.lis.distributed.thread.pool.SocketAccesor;
import com.lis.distributed.thread.pool.server.ThreadPoolServer;

public class ClientConnectionProcessor extends SocketAccesor<ThreadPoolClient,ThreadPoolServer> {

	public ClientConnectionProcessor(ThreadPoolClient context, Socket accept)
			throws IOException {
		super(context, accept);
	}
}

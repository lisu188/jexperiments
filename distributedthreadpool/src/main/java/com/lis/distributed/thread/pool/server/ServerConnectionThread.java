package com.lis.distributed.thread.pool.server;

import java.net.Socket;
import java.util.concurrent.atomic.AtomicInteger;

import com.lis.distributed.thread.pool.SocketAccesor;
import com.lis.distributed.thread.pool.client.ThreadPoolClient;
import com.lis.distributed.thread.pool.func.FuncUtils;

public class ServerConnectionThread extends
		SocketAccesor<ThreadPoolServer, ThreadPoolClient> {

	private static AtomicInteger ID_GEN = new AtomicInteger();
	private int id;

	public ServerConnectionThread(ThreadPoolServer context, Socket accept)
			throws Exception {
		super(context, accept);
		this.id = ID_GEN.incrementAndGet();
		context.registerClient(id, this);
		context.callOnClient(id,
				FuncUtils.bind((Integer id, ThreadPoolClient ctx) -> {
					ctx.setId(id);
					return true;
				}, id));
	}

}

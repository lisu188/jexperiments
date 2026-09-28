package com.lis.distributed.thread.pool.client;

import com.lis.distributed.thread.pool.SocketAccessor;

import java.io.IOException;
import java.net.Socket;
import java.util.concurrent.ExecutorService;

final class ClientConnectionProcessor extends SocketAccessor<ThreadPoolClient> {
    ClientConnectionProcessor(
            ThreadPoolClient context,
            Socket socket,
            ExecutorService invocationExecutor,
            Options options) throws IOException {
        super(context, socket, invocationExecutor, options);
        startTransport();
    }

    @Override
    protected void onRegistration(int clientId) {
        context().setId(clientId);
    }
}

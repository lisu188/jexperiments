package com.lis.distributed.thread.pool.server;

import com.lis.distributed.thread.pool.SocketAccessor;
import com.lis.distributed.thread.pool.client.ThreadPoolClient;

import java.io.IOException;
import java.net.Socket;
import java.util.concurrent.ExecutorService;

final class ServerConnectionThread extends SocketAccessor<ThreadPoolServer> {
    private final int id;

    ServerConnectionThread(
            ThreadPoolServer context,
            Socket socket,
            int id,
            ExecutorService invocationExecutor,
            Options options) throws IOException {
        super(context, socket, invocationExecutor, options);
        this.id = id;
        context.registerClient(id, this);
        startTransport();
        sendRegistration(id).whenComplete((ignored, failure) -> {
            if (failure != null) {
                close();
            }
        });
    }

    int id() {
        return id;
    }

    @Override
    protected void onClosed(Throwable failure) {
        context().unregisterClient(id, this);
    }
}

package com.lis.distributed.thread.pool.client;

import java.io.IOException;
import java.net.Socket;
import java.net.UnknownHostException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import com.lis.distributed.thread.pool.DataRepository;
import com.lis.distributed.thread.pool.Numbers;
import com.lis.distributed.thread.pool.func.FuncUtils;
import com.lis.distributed.thread.pool.func.SerializableConsumer;
import com.lis.distributed.thread.pool.func.SerializableSupplier;
import com.lis.distributed.thread.pool.server.ThreadPoolServer;

public class ThreadPoolClient {

	private ExecutorService executor = Executors.newSingleThreadExecutor();

	private ClientConnectionProcessor clientConnectionProcessor;
	private int id;

	private DataRepository repository = new DataRepository();

	public ThreadPoolClient(String string, int i) throws UnknownHostException,
			IOException {
		clientConnectionProcessor = new ClientConnectionProcessor(this,
				new Socket(string, i));
		executor.submit(clientConnectionProcessor);
	}

	public static void main(String[] args) throws Exception {
		ThreadPoolClient threadPoolClient = new ThreadPoolClient("127.0.0.1",
				55555);
		for (int i = 0; i < 100; i++) {
			Thread.sleep(1000);
			threadPoolClient.callOnServer(() -> {
				return Numbers.getId();
			}, System.out::println);
		}
	}

	public void setId(int id) {
		this.id = id;
	}

	public <T, U> void callOnServer(SerializableSupplier<T> target,
			SerializableConsumer<T> callback) throws Exception {
		callOnServer(FuncUtils.bind((Integer id, ThreadPoolServer context) -> {
			T result = target.get();
			if (callback != null) {
				context.callOnClient(id, (client) -> {
					callback.accept(result);
				});
			}
		}, this.id));
	}

	public <T, U> T callOnServer(SerializableSupplier<T> target)
			throws Exception {
		int msgId = repository.lock();
		callOnServer(FuncUtils.bind((Integer id, ThreadPoolServer context) -> {
			T t = target.get();
			context.callOnClient(id, (ThreadPoolClient ctx) -> {
				ctx.repository.setValue(msgId, t);
			});
		}, this.id));
		return (T) repository.getValue(msgId);
	}

	public void callOnServer(SerializableConsumer<ThreadPoolServer> target)
			throws Exception {
		clientConnectionProcessor.postMessage(target);
	}
}

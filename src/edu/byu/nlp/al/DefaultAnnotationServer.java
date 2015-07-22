/**
 * Copyright 2012 Brigham Young University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package edu.byu.nlp.al;

import java.io.PrintWriter;
import java.rmi.RemoteException;
import java.util.Map;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import java.util.logging.Logger;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.Maps;

import edu.byu.nlp.util.Nullable;

/**
 * No timeout mechanism.
 * 
 * @author rah67
 *
 */
// FIXME(rhaertel): query the InstanceManager from time-to-time to allow for clean shutdown.
public class DefaultAnnotationServer<D, L> implements AnnotationServer<D, L> {
	
	private static final Logger logger = Logger.getLogger(DefaultAnnotationServer.class.getName());
	
	private class AnnotationHandler implements Runnable {

		private final BlockingQueue<AnnotationInfo<L>> q;
		private final PrintWriter out;
		private volatile boolean shutdown = false;
		
		public AnnotationHandler(BlockingQueue<AnnotationInfo<L>> q, PrintWriter out) {
			this.q = q;
			this.out = out;
		}

		/** {@inheritDoc} */
		@Override
		public void run() {
		    if (out != null) {
		        out.write("source, annotator_id, annotation, annotation_time_nanos, wait_time_nanos\n");
		    }
			try {
				while (!shutdown && !Thread.currentThread().isInterrupted()) {
				    // TODO(rhaertel): this should be configurable.
					AnnotationInfo<L> annotationInfo = q.poll(1, TimeUnit.SECONDS);
					if (annotationInfo == null) {
					    // This allows us to periodically check if a shutdown has been issued.
					    continue;
					}
					
					AnnotationRequest<D, L> ar = outstandingRequests.remove(annotationInfo.getRequestId());
					// TODO: do we need a response rather than an exception?
					if (ar == null) {
						logger.severe("Could not find request " + annotationInfo.getRequestId());
					}
					ar.storeAnnotation(annotationInfo);
					logger.fine(String.format("Rcvd annotation for annotator %d: %s",
							ar.getAnnotatorId(),
							annotationInfo));
					writeAnnotation(annotationInfo, ar);
				}
			} catch (InterruptedException e) {
			    Thread.currentThread().interrupt();
			}
			if (out != null) {
			    out.flush();
			}
		}

		private void writeAnnotation(AnnotationInfo<L> ai, AnnotationRequest<D, L> ar) {
			if (out != null) {
				// CSV: source, annotator_id, annotation, duration
				out.printf("%s, %d, %s, %d, %d\n",
						ar.getInstance().getSource(),
						ar.getAnnotatorId(),
						ai.getAnnotation(),
						ai.getAnnotationEvent().getDurationNanos(),
						ai.getWaitEvent().getDurationNanos());
			}
		}
		
		public void shutdown() {
		    shutdown = true;
		}
	}
	
	private final InstanceManager<D, L> instanceManager;
	private final long timeout;
	private final TimeUnit timeoutTimeUnit;
	private final Map<Long, AnnotationRequest<D, L>> outstandingRequests;
	private final AtomicLong idCounter;
	private final BlockingQueue<AnnotationInfo<L>> eventQueue;
	private final AnnotationHandler annotationHandler;
	
	public DefaultAnnotationServer(@Nullable InstanceManager<D, L> instanceManager, long timeout,
			TimeUnit timeoutTimeUnit, @Nullable PrintWriter out) {
		this(instanceManager, timeout, timeoutTimeUnit, Maps.<Long, AnnotationRequest<D, L>>newConcurrentMap(),
				new AtomicLong(0L), new LinkedBlockingQueue<AnnotationInfo<L>>(), out);
	}
	
	@VisibleForTesting
	DefaultAnnotationServer(InstanceManager<D, L> instanceManager,
			long timeout,
			TimeUnit TimeoutTimeUnit,
			Map<Long, AnnotationRequest<D, L>> outstandingRequests,
			AtomicLong idCounter,
			BlockingQueue<AnnotationInfo<L>> eventQueue,
			PrintWriter out) {
		this.instanceManager = instanceManager;
		this.timeout = timeout;
		this.timeoutTimeUnit = TimeoutTimeUnit;
		this.outstandingRequests = outstandingRequests;
		this.idCounter = idCounter;
		this.eventQueue = eventQueue;
		this.annotationHandler = new AnnotationHandler(eventQueue, out);
		// FIXME(rhaertel): should not happen in the constructor
		new Thread(annotationHandler, "AnnotationServerEventThread").start();
	}

	/** {@inheritDoc} */
	@Override
	public void cancelInstanceRequest(long requestId) throws RemoteException {
		AnnotationRequest<D, L> ar = outstandingRequests.remove(requestId);
		ar.cancelRequest();
	}

	/** {@inheritDoc} */
	@Override
	public RequestInstanceResponse<D> requestInstanceFor(int annotatorId) throws RemoteException {
		// FIXME(rhaertel): prevent an annotator from having more than one outstanding request
		try {
			AnnotationRequest<D, L> ar = instanceManager.requestInstanceFor(annotatorId, timeout, timeoutTimeUnit);
			if (ar == null) {
			    if (instanceManager.isDone()) {
			        shutdown();
			        return RequestInstanceResponse.finished();
			    }
				return RequestInstanceResponse.noInstances();
			}
			InstanceForAnnotation<D> ifa =
					new InstanceForAnnotation<D>(idCounter.incrementAndGet(), ar.getInstance().getData(), ar.getInstance().getInstanceId());
			outstandingRequests.put(ifa.getRequestId(), ar);
			return RequestInstanceResponse.success(ifa);
		} catch (InterruptedException e) {
			return RequestInstanceResponse.timeout();
		}
	}

	/** {@inheritDoc} */
	@Override
	public void storeAnnotation(AnnotationInfo<L> annotationInfo) throws RemoteException {
		// FIXME(rhaertel). I don't think I need this method to be synchronized; I just need a read lock on
		// the map. Above, I can use read/write locks effectively. Then the map doesn't have to be concurrent, either.
		// AND stores and request can proceed concurrently, under the correct circumstances.
		Preconditions.checkNotNull(annotationInfo);
		if (!outstandingRequests.containsKey(annotationInfo.getRequestId())) {
			throw new RemoteException("Do not recognize request id = " + annotationInfo.getRequestId() + 
					"; " + annotationInfo);
		}
		try {
			eventQueue.put(annotationInfo);
		} catch (InterruptedException e) {
			throw new RemoteException("InterruptedException. Could not add request id = " + 
			        annotationInfo.getRequestId() + "to the event queue; " + annotationInfo);
		}
	}
	
	public void shutdown() {
	    annotationHandler.shutdown();
	}

}

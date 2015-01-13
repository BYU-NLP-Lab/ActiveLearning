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
package edu.byu.nlp.al.simulation;

import java.util.concurrent.Callable;

import edu.byu.nlp.al.AnnotationInfo;
import edu.byu.nlp.al.AnnotationServer;
import edu.byu.nlp.al.InstanceForAnnotation;
import edu.byu.nlp.al.RequestInstanceResponse;
import edu.byu.nlp.al.RequestInstanceResponse.Status;
import edu.byu.nlp.util.TimedEvent;
import edu.byu.nlp.util.Timers.Stoppable;
import edu.byu.nlp.util.Timers.Timer;

/**
 * An annotator who produces annotation non-stop on the clock.
 * @author rah67
 *
 */
public class TirelessAnnotator<D, L> implements Callable<Void> {

	private final long annotatorId;
	private final AnnotationServer<D, L> server;
	private final AnnotationInfoProvider<D, L> annotationInfoProvider;
	private final boolean realTime;
	private final Timer timer;

	/**
	 * If realTime is true, then this Annotator's thread sleeps for the amount of time returned by the
	 * annotationInfoProvider for each instance. 
	 */
	public TirelessAnnotator(long annotatorId, AnnotationServer<D, L> server,
			AnnotationInfoProvider<D, L> annotationInfoProvider, boolean realTime, Timer timer) {
		this.annotatorId = annotatorId;
		this.server = server;
		this.annotationInfoProvider = annotationInfoProvider;
		this.realTime = realTime;
        this.timer = timer;
	}

	@Override
	public Void call() throws Exception {
		// TODO(rah67): consider using mocks to test this behavior
		Stoppable waitTimer = timer.start();
		while(true) {
			RequestInstanceResponse<D> response = server.requestInstanceFor(annotatorId);
			InstanceForAnnotation<D> ifa = response.getInstanceForAnnotation();
            if (ifa == null) {
                if (response.getStatus() == Status.FINISHED) {
                    return null;
                }
                if (response.getStatus() == Status.NO_INSTANCES) {
                    // Give the instance manager some time to find us an instance. This also allows us to recheck
                    // the finished status in case this annotator is done but not everybody else is.
                    // TODO(rhaertel): configure this wait time.
                    Thread.sleep(100);
                    continue;
                }
                throw new IllegalStateException(String.format("InstanceForAnnotation was null and Status was %s",
                        response.getStatus()));
            }
			TimedEvent waitEvent = waitTimer.stop();
			
			// <---- Exclude this block from timing the user wait time as this part is annotation time.
			AnnotationInfo<L> ai = annotationInfoProvider.annotationInfoFor(ifa, waitEvent);
			if (realTime) {
				long time = ai.getAnnotationEvent().getDurationNanos();
				Thread.sleep(time / 1000000L, (int) (time % 1000000L));
			}
			// <---- End
			
			waitTimer = timer.start();
			server.storeAnnotation(ai);
		}
	}
}

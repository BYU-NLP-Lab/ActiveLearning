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

import edu.byu.nlp.util.TimedEvent;

/**
 * @author rah67
 *
 */
public class AnnotationInfo<L> {
	
	private final long requestId;
	private final L annotation;
	// Start and end times for annotation (No provisions for pauses right now)
	private final TimedEvent annotationEvent;
	private final TimedEvent waitEvent;

	public AnnotationInfo(long requestId, L annotation, TimedEvent annotationEvent, TimedEvent waitEvent) {
		this.requestId = requestId;
		this.annotation = annotation;
		this.annotationEvent = annotationEvent;
		this.waitEvent = waitEvent;
	}

	public long getRequestId() {
		return requestId;
	}

	public L getAnnotation() {
		return annotation;
	}
	
	public TimedEvent getAnnotationEvent() {
		return annotationEvent;
	}
	
	/** The time spent waiting for the server to respond **/
	public TimedEvent getWaitEvent() {
		return waitEvent;
	}

	/** {@inheritDoc} */
	@Override
	public String toString() {
		return "AnnotationInfo [requestId=" + requestId + ", annotation=" + annotation + ", annotationEvent="
				+ annotationEvent + ", waitEvent=" + waitEvent + "]";
	}
}

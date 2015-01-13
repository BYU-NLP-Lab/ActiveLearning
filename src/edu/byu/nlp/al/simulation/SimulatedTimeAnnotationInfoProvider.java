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

import edu.byu.nlp.al.AnnotationInfo;
import edu.byu.nlp.al.InstanceForAnnotation;
import edu.byu.nlp.crowdsourcing.LabelProvider;
import edu.byu.nlp.util.TimedEvent;

/**
 * @author rah67
 *
 */
public class SimulatedTimeAnnotationInfoProvider<D, L> implements AnnotationInfoProvider<D, L> {

	private final LabelProvider<D, L> labelProvider;
	private final TimeSimulator<D> timeSimulator;
	
	public SimulatedTimeAnnotationInfoProvider(LabelProvider<D, L> labelProvider,
			TimeSimulator<D> timeSimulator) {
		this.labelProvider = labelProvider;
		this.timeSimulator = timeSimulator;
	}

	/** {@inheritDoc} */
	@Override
	public AnnotationInfo<L> annotationInfoFor(InstanceForAnnotation<D> ifa, TimedEvent waitTime) {
		long startTime = System.nanoTime();
		long endTime = startTime + timeSimulator.annotationTimeInNanoSecsFor(ifa.getInstance());
		TimedEvent annotationEvent = new TimedEvent(startTime, endTime);
		L goldLabel = labelProvider.labelFor(ifa.getSource(), ifa.getInstance());
		return new AnnotationInfo<L>(ifa.getRequestId(), goldLabel, annotationEvent, waitTime);
	}

}

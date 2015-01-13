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

import edu.byu.nlp.data.FlatInstance;


/**
 * Note: This is NOT a request message (that would otherwise correspond to AnnotationResponse). 
 * @author rah67
 */
public interface AnnotationRequest<D, L> {
	long getAnnotatorId();
	FlatInstance<D, L> getInstance();
	// TODO: ideally, the API enforces that cancel/store can only be invoked once.
	void cancelRequest();
	// TODO: add a "refuse()" method
	boolean storeAnnotation(AnnotationInfo<L> annotationInfo);
}
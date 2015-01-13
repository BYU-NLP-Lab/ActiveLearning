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

import java.util.List;

import com.google.common.collect.Lists;

import edu.byu.nlp.annotationinterface.java.BasicInstance;

/**
 * @author rah67
 *
 */
public class InstanceProviderTestUtil {

	private InstanceProviderTestUtil() { }
	

  // FIXME (pfelt): repair this test after resolving the issues described in the 
  // note in LabelUncertaintyScorer
//	public static Instance<Integer, Integer> dummyInstance(Integer label, String source, Integer datum) {
//		return BasicInstance.of(label, source, datum);
//	}
//	
//	public static List<Instance<Integer, Integer>> testData() {
//		List<Instance<Integer, Integer>> data = Lists.newArrayList();
//		Instance<Integer, Integer> instance;
//		
//		instance = dummyInstance(0, "0", 1);
//		instance.getAnnotations().put(0L, BasicTimedAnnotation.of(0));
//		instance.getAnnotations().put(1L, BasicTimedAnnotation.of(0));
//		instance.getAnnotations().put(1L, BasicTimedAnnotation.of(0));
//		instance.getAnnotations().put(1L, BasicTimedAnnotation.of(1));
//		instance.getAnnotations().put(2L, BasicTimedAnnotation.of(1));
//		instance.getAnnotations().put(2L, BasicTimedAnnotation.of(2));
//		data.add(instance);
//		
//		instance = dummyInstance(1, "1", 2);
//		instance.getAnnotations().put(0L, BasicTimedAnnotation.of(1));
//		instance.getAnnotations().put(2L, BasicTimedAnnotation.of(1));
//		instance.getAnnotations().put(2L, BasicTimedAnnotation.of(0));
//		data.add(instance);
//
//		instance = dummyInstance(1, "2", 1);
//		data.add(instance);
//
//		instance = dummyInstance(1, "3", 3);
//		instance.getAnnotations().put(0L, BasicTimedAnnotation.of(0));
//		instance.getAnnotations().put(1L, BasicTimedAnnotation.of(1));
//		instance.getAnnotations().put(2L, BasicTimedAnnotation.of(2));
//		instance.getAnnotations().put(3L, BasicTimedAnnotation.of(3));
//		data.add(instance);
//		
//		return data;
//	}
}

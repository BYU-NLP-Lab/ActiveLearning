/**
 * Copyright 2014 Brigham Young University
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

import edu.byu.nlp.al.AbstractInstanceManager.AnnotationRecorder;
import edu.byu.nlp.data.BasicFlatInstance;
import edu.byu.nlp.data.FlatInstance;
import edu.byu.nlp.data.measurements.ClassificationMeasurements;
import edu.byu.nlp.data.types.Dataset;
import edu.byu.nlp.data.types.SparseFeatureVector;
import edu.byu.nlp.dataset.Datasets;

/**
 * @author pfelt
 *
 */
public class DatasetAnnotationRecorder implements AnnotationRecorder<SparseFeatureVector, Integer>{

  private Dataset dataset;
  private boolean recordMeasurements;

  public DatasetAnnotationRecorder(Dataset dataset, boolean recordMeasurements){
    this.dataset=dataset;
    this.recordMeasurements=recordMeasurements;
  }
  
  /** {@inheritDoc} */
  @SuppressWarnings("unchecked")
  @Override
  public void recordAnnotation(FlatInstance<SparseFeatureVector, Integer> baseAnnotation) {
    
    // include the base annotation
    List<FlatInstance<SparseFeatureVector, Integer>> annotations = Lists.newArrayList(baseAnnotation);
    
//    // optionally include add a bunch of measurements encoding the same annotation info (one per class) 
//    if (recordMeasurements){
//      // transform the base annotation into K measurments; 
//      // 1 positive (+1) and the rest negative (-1)
//      for (int label=0; label<dataset.getInfo().getNumClasses(); label++){
//        // for simulation purposes, assume perfect (dis)agreement values and perfect confidence
//        double measurementValue = baseAnnotation.getAnnotation().equals(label) ? 1: 0; 
////        if (!baseAnnotation.getAnnotation().equals(label)){
////          continue;
////        }
//        double confidence = 1; 
//        annotations.add(
//            new BasicFlatInstance<SparseFeatureVector, Integer>(
//              baseAnnotation.getInstanceId(), 
//              baseAnnotation.getSource(), 
//              baseAnnotation.getAnnotator(), 
//              null, // no annotation; we'll use measurements instead
//              new ClassificationMeasurements.BasicClassificationAnnotationMeasurement(
//                  baseAnnotation.getAnnotator(), measurementValue, confidence, baseAnnotation.getSource(),
//                  label), 
//              baseAnnotation.getStartTimestamp(), 
//              baseAnnotation.getEndTimestamp()
//              ));
//      }
//    }
    
    // now add all the annotations
    for (FlatInstance<SparseFeatureVector, Integer> ann: annotations){
      Datasets.addAnnotationToDataset(dataset, ann);
    }
    
  }

}

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

import edu.byu.nlp.al.AbstractInstanceManager.AnnotationRecorder;
import edu.byu.nlp.data.FlatInstance;
import edu.byu.nlp.data.types.Dataset;
import edu.byu.nlp.data.types.SparseFeatureVector;
import edu.byu.nlp.dataset.Datasets;

/**
 * @author pfelt
 *
 */
public class DatasetAnnotationRecorder implements AnnotationRecorder<SparseFeatureVector, Integer>{

  private Dataset dataset;

  public DatasetAnnotationRecorder(Dataset dataset){
    this.dataset=dataset;
  }
  
  /** {@inheritDoc} */
  @Override
  public void recordAnnotation(FlatInstance<SparseFeatureVector, Integer> baseAnnotation) {
      Datasets.addAnnotationToDataset(dataset, baseAnnotation);
  }

}

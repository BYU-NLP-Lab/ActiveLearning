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

import static org.fest.assertions.Assertions.assertThat;

import java.util.List;
import java.util.concurrent.TimeUnit;

import org.apache.commons.math3.random.RandomGenerator;
import org.junit.Test;

import edu.byu.nlp.data.FlatInstance;
import edu.byu.nlp.data.types.Dataset;
import edu.byu.nlp.data.types.SparseFeatureVector;
import edu.byu.nlp.dataset.Datasets;

/**
 * @author rah67
 *
 */
public class GeneralizedRoundRobinInstanceProviderTest {

  // Note(rhaertel): we could consider narrowing the RandomGenerator interface
  // with a specialized nextInt() interface
  // Only implements nextInt and returns the same value passed to it.
  private static class FakeRandomGenerator implements RandomGenerator {

    // The only method that should be called....
    @Override
    public int nextInt(int i) {
      return i;
    }

    @Override
    public boolean nextBoolean() {
      throw new UnsupportedOperationException();
    }

    @Override
    public void nextBytes(byte[] arg0) {
      throw new UnsupportedOperationException();
    }

    @Override
    public double nextDouble() {
      throw new UnsupportedOperationException();
    }

    @Override
    public float nextFloat() {
      throw new UnsupportedOperationException();
    }

    @Override
    public double nextGaussian() {
      throw new UnsupportedOperationException();
    }

    @Override
    public int nextInt() {
      throw new UnsupportedOperationException();
    }

    @Override
    public long nextLong() {
      throw new UnsupportedOperationException();
    }

    @Override
    public void setSeed(int arg0) {
      throw new UnsupportedOperationException();
    }

    @Override
    public void setSeed(int[] arg0) {
      throw new UnsupportedOperationException();
    }

    @Override
    public void setSeed(long arg0) {
      throw new UnsupportedOperationException();
    }
  }

  // FIXME (pfelt): repair this test after resolving the issues described in the 
  // note in LabelUncertaintyScorer
//  @Test
//  public void testRequestInstanceFor() throws InterruptedException {
//    // This fake should have the same effect as keeping everything in order.
//    FakeRandomGenerator rnd = new FakeRandomGenerator();
//
//    Dataset dataset = InstanceProviderTestUtil.testData();
//    AbstractInstanceManager<SparseFeatureVector, Integer> provider = GeneralizedRoundRobinInstanceManager
//        .newManager(1, dataset, null, rnd);
//    List<FlatInstance<SparseFeatureVector, Integer>> data = Datasets
//        .instancesIn(dataset);
//
//    // Queue: [0, 1, 2, 3] (the numbers are the indices in the original data).
//    // annotatorId field is ignored, so we use -1
//    AnnotationRequest<SparseFeatureVector, Integer> requestOriginalIndex0 = provider
//        .requestInstanceFor(-1, 1, TimeUnit.SECONDS);
//    assertThat(requestOriginalIndex0.getInstance()).isSameAs(data.get(0));
//
//    // Queue: [1, 2, 3] (the numbers are the indices in the original data).
//    AnnotationRequest<SparseFeatureVector, Integer> requestOriginalIndex1 = provider
//        .requestInstanceFor(-1, 1, TimeUnit.SECONDS);
//    assertThat(requestOriginalIndex1.getInstance()).isSameAs(data.get(1));
//
//    // Queue: [2, 3] (the numbers are the indices in the original data).
//    // This will force it to be at the end of the list
//    AnnotationInfo<Integer> ai = new AnnotationInfo<Integer>(-1, 1, null, null);
//    requestOriginalIndex1.getInstance().getAnnotations()
//        .put(-1L, BasicTimedAnnotation.of(-1));
//    requestOriginalIndex1.getInstance().getAnnotations()
//        .put(-1L, BasicTimedAnnotation.of(-1));
//    requestOriginalIndex1.getInstance().getAnnotations()
//        .put(-1L, BasicTimedAnnotation.of(-1));
//    requestOriginalIndex1.getInstance().getAnnotations()
//        .put(-1L, BasicTimedAnnotation.of(-1));
//    requestOriginalIndex1.getInstance().getAnnotations()
//        .put(-1L, BasicTimedAnnotation.of(-1));
//    requestOriginalIndex1.getInstance().getAnnotations()
//        .put(-1L, BasicTimedAnnotation.of(-1));
//    requestOriginalIndex1.getInstance().getAnnotations()
//        .put(-1L, BasicTimedAnnotation.of(-1));
//    requestOriginalIndex1.getInstance().getAnnotations()
//        .put(-1L, BasicTimedAnnotation.of(-1));
//    requestOriginalIndex1.storeAnnotation(ai);
//
//    // Queue: [2, 3, 1] (the numbers are the indices in the original data).
//    AnnotationRequest<Integer, Integer> request = provider.requestInstanceFor(
//        -1, 1, TimeUnit.SECONDS);
//    assertThat(request.getInstance()).isSameAs(data.get(2));
//    request = provider.requestInstanceFor(-1, 1, TimeUnit.SECONDS);
//    assertThat(request.getInstance()).isSameAs(data.get(3));
//    request = provider.requestInstanceFor(-1, 1, TimeUnit.SECONDS);
//    assertThat(request.getInstance()).isSameAs(data.get(1));
//
//    // All of the instances are out for annotation.
//    assertThat(provider.requestInstanceFor(-1, 1, TimeUnit.SECONDS)).isNull();
//  }
}

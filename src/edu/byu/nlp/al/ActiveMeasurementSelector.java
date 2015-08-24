/**
 * Copyright 2015 Brigham Young University
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

import java.util.Collection;
import java.util.List;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.random.RandomGenerator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.collect.Lists;
import com.google.common.collect.Multimap;

import edu.byu.nlp.classify.util.ModelTraining;
import edu.byu.nlp.crowdsourcing.measurements.MeasurementExpectation;
import edu.byu.nlp.crowdsourcing.measurements.MeasurementModelBuilder;
import edu.byu.nlp.crowdsourcing.measurements.classification.ClassificationMeasurementExpectations;
import edu.byu.nlp.crowdsourcing.measurements.classification.ClassificationMeasurementModel;
import edu.byu.nlp.crowdsourcing.measurements.classification.ClassificationMeasurementModel.State;
import edu.byu.nlp.data.BasicFlatInstance;
import edu.byu.nlp.data.FlatInstance;
import edu.byu.nlp.data.measurements.ClassificationMeasurementParser;
import edu.byu.nlp.data.streams.JSONFileToAnnotatedDocumentList.MeasurementPojo;
import edu.byu.nlp.data.types.Dataset;
import edu.byu.nlp.data.types.Measurement;
import edu.byu.nlp.data.types.SparseFeatureVector;
import edu.byu.nlp.data.util.EmpiricalAnnotations;
import edu.byu.nlp.dataset.Datasets;
import edu.byu.nlp.util.ArgMinMaxTracker;
import edu.byu.nlp.util.DoubleArrays;
import edu.byu.nlp.util.Matrices;

/**
 * @author plf1
 *
 */
public class ActiveMeasurementSelector implements MeasurementSelector{
  private static Logger logger = LoggerFactory.getLogger(ActiveMeasurementSelector.class);
  
  private static final String CANDIDATE_TRAINING_OPS = "maximize-all-1"; 

  private List<FlatInstance<SparseFeatureVector, Integer>> candidates = Lists.newArrayList();
  private MeasurementModelBuilder modelBuilder;
  private int numSamples;
  private Dataset dataset;
  private String trainingOperations;
  private RandomGenerator rnd;
  private double thinningRate;
  private int minCandidates;

  /**
   * @param modelBuilder
   * @param annotations 
   */
  public ActiveMeasurementSelector(
      MeasurementModelBuilder modelBuilder, Dataset dataset, EmpiricalAnnotations<SparseFeatureVector, Integer> annotations, 
      int numSamples, String trainingOperations, double thinningRate, int minCandidates, RandomGenerator rnd) {
    this.modelBuilder=modelBuilder;
    this.numSamples=numSamples;
    this.dataset=dataset;
    this.rnd=rnd;
    this.trainingOperations=trainingOperations;
    this.thinningRate=thinningRate;
    this.minCandidates=minCandidates;
    // we want to add all measurements that are not already taken (used as seed set contained in dataset)
    // FIXME: this is horrifically inefficient! Fix it! 
    for (FlatInstance<SparseFeatureVector, Integer> meas: annotations.getMeasurements()){
      if (!dataset.getMeasurements().contains(meas)){
        candidates.add(meas);
      }
    }
    for (Multimap<Integer, FlatInstance<SparseFeatureVector, Integer>> perAnnotatorAnnotations: annotations.getPerInstancePerAnnotatorAnnotations().values()){
      for (FlatInstance<SparseFeatureVector, Integer> meas: perAnnotatorAnnotations.values()){
        candidates.add(meas);
      }
    }
  }
  
  public Collection<FlatInstance<SparseFeatureVector, Integer>> selectNext(int batchSize){
    State currentModel = modelTrainedOn(dataset, trainingOperations, null);
//    ClassificationMeasurementModelExpectations expectations = ClassificationMeasurementModelExpectations.from(currentModel);
    ArgMinMaxTracker<Double, FlatInstance<SparseFeatureVector, Integer>> candidateTracker = new ArgMinMaxTracker<>(rnd,batchSize);
    
    int candidatesConsidered = 0;
    while(candidatesConsidered<minCandidates){ 
      for (FlatInstance<SparseFeatureVector, Integer> candidate: candidates){
        // don't repeat an answer we already have
        if (candidateTracker.argmax().contains(candidate)){
          continue;
        }
        // skip a random subset of the available candidates (ensuring we evaluate SOMEONE) 
        if (rnd.nextDouble()>thinningRate){
          continue;
        }
        candidatesConsidered += 1;
        
        int annotatorIndex = candidate.getMeasurement().getAnnotator();
        String rawAnnotator = dataset.getInfo().getAnnotatorIdIndexer().get(annotatorIndex);
        MeasurementExpectation<Integer> candExpectation = ClassificationMeasurementExpectations.fromMeasurement(candidate.getMeasurement(), dataset, currentModel.getInstanceIndices(), currentModel.getLogNuY());
  
        // calculate parameters to p(tau|x,y,w)
        double mean_jk = candExpectation.sumOfExpectedValuesOfSigma();
        double alpha = currentModel.getNuSigma2()[annotatorIndex][0], beta = currentModel.getNuSigma2()[annotatorIndex][1];
        double var_jk = beta / (alpha - 1); // point estimate (ignoring uncertainty in w)
  
        double mean_utility_jk = 0;
        for (int t=0; t<numSamples; t++){
  //        double var_jk = 1.0/new GammaDistribution(alpha, beta).sample(); // sample variance (integrating over w). note: probably incorrect
          double tau_jkt = new NormalDistribution(mean_jk, Math.sqrt(var_jk)).sample();
          MeasurementPojo speculativeMeasurementPojo = candidate.getMeasurement().getPojo().copy();
          speculativeMeasurementPojo.value = tau_jkt; 
          Measurement speculativeMeasurement = ClassificationMeasurementParser.pojoToMeasurement(speculativeMeasurementPojo, rawAnnotator, candidate.getSource(), 
              candidate.getStartTimestamp(), candidate.getEndTimestamp(), dataset.getInfo().getIndexers());
          FlatInstance<SparseFeatureVector, Integer> speculativeMeasurementInst = new BasicFlatInstance<SparseFeatureVector, Integer>(
              candidate.getInstanceId(), candidate.getSource(), annotatorIndex, candidate.getAnnotation(), speculativeMeasurement, candidate.getStartTimestamp(), candidate.getEndTimestamp());
  
          // add the speculative measurement and train
          Datasets.addAnnotationToDataset(dataset, speculativeMeasurementInst);
          State model = modelTrainedOn(dataset, CANDIDATE_TRAINING_OPS, currentModel);
          // remove the speculative measurement
          Datasets.removeAnnotationFromDataset(dataset, speculativeMeasurementInst);
          
          // calculate utility U=R-C of this model (where reward = accuracy or equivalently, negative hamming loss)
          // and cost is constant
          double[][] logNuY = model.getLogNuY();
          for (int i=0; i<logNuY.length; i++){
            mean_utility_jk += Math.exp(DoubleArrays.max(logNuY[i]));
          }
        }
        mean_utility_jk /= numSamples;
        candidateTracker.offer(mean_utility_jk, candidate);
        
      }
    }
    
    // return top k (and remove from future candidates)
    logger.info("\n**********************************************************\n"
            + "******* Selected batch of size "+candidateTracker.argmax().size()+" *******\n"
            + "**********************************************************\n");
    candidates.removeAll(candidateTracker.argmax());
    return candidateTracker.argmax();
  }
  
  
  
  private State modelTrainedOn(Dataset data, String trainingOperations, State previousState){
    ClassificationMeasurementModel model = modelBuilder.setData(data).build();
    if (previousState!=null){
      // copy previous state (for faster convergence
      System.arraycopy(previousState.getNuTheta(), 0, model.getCurrentState().getNuTheta(), 0, model.getCurrentState().getNuTheta().length); 
      Matrices.copyInto(previousState.getNuSigma2(), model.getCurrentState().getNuSigma2());
      Matrices.copyInto(previousState.getLogNuY(), model.getCurrentState().getLogNuY());
    }
    // train a little more
    ModelTraining.doOperations(trainingOperations, model, null);
    return model.getCurrentState();
  }

  
  
}

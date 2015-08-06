package edu.byu.nlp.al.simulation;

import java.util.Map;
import java.util.Random;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.berkeley.Counter;
import org.deeplearning4j.berkeley.PriorityQueue;

import com.google.common.base.Preconditions;
import com.google.common.collect.Maps;

import edu.byu.nlp.crowdsourcing.LabelProvider;
import edu.byu.nlp.data.measurements.ClassificationMeasurements;
import edu.byu.nlp.data.types.Dataset;
import edu.byu.nlp.data.types.DatasetInstance;
import edu.byu.nlp.data.types.Measurement;
import edu.byu.nlp.util.DoubleArrays;
import edu.byu.nlp.util.Matrices;
import edu.byu.nlp.util.Pair;

public class FallibleMeasurementProvider<D> implements LabelProvider<D,Measurement> {

  private final double labelProportionSD = 0.1;
  private final double labelPredicateSD = 0.1;
  private final double labelPredicateSmoothing = 0.1;
  
  private FallibleAnnotationProvider<D, Integer> labelProvider;
  private Dataset dataset;
//  private RandomGenerator rnd;
  private Random rnd;
  private int annotator;
  private Counter<Integer> labelCounter;
  private double accuracy;
  private double[][] perWordClassCounts;
  private double[] perWordCounts;
  private Counter<Pair<Integer,Integer>> wordLabelPairs;

  public FallibleMeasurementProvider(FallibleAnnotationProvider<D,Integer> labelProvider, Dataset dataset, int annotator, double accuracy, RandomGenerator rnd){
    this.labelProvider=labelProvider;
    this.dataset=dataset;
    this.annotator=annotator;
    this.accuracy=accuracy;
    this.rnd=new Random(rnd.nextLong());
  }
  
  @Override
  public Measurement labelFor(String source, D datum) {
    double choice = rnd.nextDouble();
    
    // how much of which kinds of error?
    Map<String,Double> proportions = Maps.newHashMap();
    proportions.put("annotation", .5);
    proportions.put("labeled_predicate", 0.5);
    proportions.put("label_proportion", 0.);
    Preconditions.checkState(
        Math.abs(1-DoubleArrays.sum(DoubleArrays.fromDoubleCollection(proportions.values())))<1e-20,
        "Illegal Measurement proportions! Must sum to approx 1");
    
    // regular annotation
    choice -= proportions.get("annotation");
    if (choice<=0){
      Integer label = labelProvider.labelFor(source, datum);
      return new ClassificationMeasurements.BasicClassificationAnnotationMeasurement(annotator, 1.0, 1.0, source, label) ;
    }
    
    // labeled predicate
    choice -= proportions.get("labeled_predicate");
    if (choice<=0){
      // choose word^label pair
      Pair<Integer, Integer> wordLabel = sampleWordClassPair(labelPredicateSmoothing);
      Integer wordIndex = wordLabel.getFirst();
      String word = dataset.getInfo().getFeatureIndexer().get(wordIndex);
      Integer label = wordLabel.getSecond();
      String labelRaw = dataset.getInfo().getLabelIndexer().get(label);
      // noisy word^label count
      double effectiveSD = (1/accuracy)*labelPredicateSD*perWordCounts[wordIndex];
      double trueCount = perWordClassCounts[wordIndex][label];
      double corruptCount = Math.max(0, new NormalDistribution(trueCount, effectiveSD).sample()); 
      return new ClassificationMeasurements.BasicClassificationLabeledPredicateMeasurement(annotator, corruptCount, 1.0, label, word);
    }
    
    // label proportion
    else {
      // use noisy truth to determine which label we measure
      Integer label = labelProvider.labelFor(source, datum); 
      // corrupt the label count (according to a gaussian)
      double truth = labelCount(label);
      double effectiveSD = (1/accuracy)*labelProportionSD*dataset.getInfo().getNumDocuments();
      double corruptCount = Math.max(0, new NormalDistribution(truth, effectiveSD).sample());
      return new ClassificationMeasurements.BasicClassificationLabelProportionMeasurement(annotator, corruptCount, 1, label);
    }

  }
  
  private Pair<Integer,Integer> sampleWordClassPair(double smoothing){
    if (perWordClassCounts==null){
      // get counts and normalize
      double[][] perClassWordCounts = new double[dataset.getInfo().getNumClasses()][dataset.getInfo().getNumFeatures()];
      for (DatasetInstance instance: dataset){
        instance.asFeatureVector().addTo(perClassWordCounts[instance.getLabel()]);
      }
      Matrices.addToSelf(perClassWordCounts, smoothing);
      perWordCounts = Matrices.sumOverFirst(perClassWordCounts);
      perWordClassCounts = Matrices.transpose(perClassWordCounts);
      double[][] perWordClassDistribution = Matrices.clone(perWordClassCounts);
      Matrices.normalizeRowsToSelf(perWordClassDistribution);

      // rank word/label pairs by p(label|word)
      wordLabelPairs = new Counter<>();
      for (int w=0; w<perWordClassDistribution.length; w++){
        for (int l=0; l<perWordClassDistribution[w].length; l++){
          wordLabelPairs.incrementCount(Pair.of(w, l), perWordClassDistribution[w][l]); 
        }
      }
//      wordLabelPairs.normalize();
    }
    
    return wordLabelPairs.sample(rnd);
  }
  
  private double labelCount(Integer label){
    // make sure we've got totaled label counts
    precomputeLabelProportions();
    return (double)labelCounter.getCount(label);
  }

  private void precomputeLabelProportions(){
    if (labelCounter==null){
      labelCounter = new Counter<Integer>();
      for (DatasetInstance item: dataset){
        if (item.hasLabel()){
          labelCounter.incrementCount(item.getLabel(), 1);
        }
      }
    }
  }
  
}

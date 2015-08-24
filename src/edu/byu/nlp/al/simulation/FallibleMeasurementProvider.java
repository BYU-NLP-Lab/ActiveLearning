package edu.byu.nlp.al.simulation;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.random.RandomAdaptor;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.berkeley.Counter;

import com.google.common.base.Preconditions;

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
  private RandomGenerator rnd;
  private int annotator;
  private Counter<Integer> labelCounter;
  private double accuracy;
  private double[][] perWordClassCounts;
  private double[] perWordCounts;
  private Counter<Pair<Integer,Integer>> wordLabelPairs;
  private double annotationRate;
  private double labeledPredicateRate;
  private double labelProportionRate;

  public FallibleMeasurementProvider(FallibleAnnotationProvider<D,Integer> labelProvider, Dataset dataset, int annotator, double accuracy,
      double annotationRate, double labeledPredicateRate, double labelProportionRate, RandomGenerator rnd){
    Preconditions.checkArgument(annotationRate>=0);
    Preconditions.checkArgument(labeledPredicateRate>=0);
    Preconditions.checkArgument(labelProportionRate>=0);
    Preconditions.checkArgument(annotationRate+labeledPredicateRate+labelProportionRate>0);
    this.labelProvider=labelProvider;
    this.dataset=dataset;
    this.annotator=annotator;
    this.accuracy=accuracy;
    this.annotationRate=annotationRate;
    this.labeledPredicateRate=labeledPredicateRate;
    this.labelProportionRate=labelProportionRate;
    this.rnd=rnd;
  }
  
  @Override
  public Measurement labelFor(String source, D datum) {
    double choice = rnd.nextDouble();
    double value = 1, defaultConfidence = 1;
    long starttime = rnd.nextLong();
    long endtime = starttime + 1000;
    
    // how much of which kinds of error?
    // [annotations, labeled predicates, label proportions]
    double[] measurementProportions = new double[]{annotationRate, labeledPredicateRate, labelProportionRate};
    DoubleArrays.normalizeToSelf(measurementProportions);
    
    // regular annotation
    choice -= measurementProportions[0];
    if (choice<=0){
      Integer label = labelProvider.labelFor(source, datum);
      return new ClassificationMeasurements.BasicClassificationAnnotationMeasurement(null, annotator, value, defaultConfidence, source, label, starttime, endtime);
    }
    
    // labeled predicate
    choice -= measurementProportions[1];
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
      double corruptCount = Math.max(0, new NormalDistribution(rnd, trueCount, effectiveSD).sample()); 
      return new ClassificationMeasurements.BasicClassificationLabeledPredicateMeasurement(null, annotator, corruptCount, defaultConfidence, label, word, starttime, endtime);
    }
    
    // label proportion
    else {
      // use noisy truth to determine which label we measure
      Integer label = labelProvider.labelFor(source, datum);  
      // corrupt the label count (according to a gaussian)
      double truth = labelCount(label);
      double effectiveSD = (1/accuracy)*labelProportionSD*dataset.getInfo().getNumDocuments();
      double corruptCount = Math.max(0, new NormalDistribution(rnd, truth, effectiveSD).sample());
      return new ClassificationMeasurements.BasicClassificationLabelProportionMeasurement(null, annotator, corruptCount, defaultConfidence, label, starttime, endtime);
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
    }
    
    return wordLabelPairs.sample(new RandomAdaptor(rnd));
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

/**
 * Copyright 2013 Brigham Young University
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
package edu.byu.nlp.al.app;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.TimeUnit;

import org.apache.commons.io.FileUtils;
import org.apache.commons.math3.analysis.MultivariateFunction;
import org.apache.commons.math3.optim.PointValuePair;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.slf4j.MDC;

import com.google.common.base.Function;
import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.base.Predicates;
import com.google.common.base.Stopwatch;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import com.google.common.io.ByteStreams;

import edu.byu.nlp.al.ABArbiterInstanceManager;
import edu.byu.nlp.al.AnnotationInfo;
import edu.byu.nlp.al.AnnotationRequest;
import edu.byu.nlp.al.DatasetAnnotationRecorder;
import edu.byu.nlp.al.EmpiricalAnnotationInstanceManager;
import edu.byu.nlp.al.EmpiricalAnnotationLayersInstanceManager;
import edu.byu.nlp.al.GeneralizedRoundRobinInstanceManager;
import edu.byu.nlp.al.InstanceManager;
import edu.byu.nlp.al.NDeepInstanceManager;
import edu.byu.nlp.al.RateLimitedAnnotatorInstanceManager;
import edu.byu.nlp.al.simulation.FallibleAnnotationProvider;
import edu.byu.nlp.al.simulation.FallibleMeasurementProvider;
import edu.byu.nlp.al.simulation.GoldLabelProvider;
import edu.byu.nlp.al.util.MetricComputers.AnnotatorAccuracyComputer;
import edu.byu.nlp.al.util.MetricComputers.DatasetMetricComputer;
import edu.byu.nlp.al.util.MetricComputers.LogJointComputer;
import edu.byu.nlp.al.util.MetricComputers.MachineAccuracyComputer;
import edu.byu.nlp.al.util.MetricComputers.PredictionTabulator;
import edu.byu.nlp.al.util.MetricComputers.RmseAnnotatorAccuracyComputer;
import edu.byu.nlp.al.util.MetricComputers.RmseMachineAccuracyVsTestComputer;
import edu.byu.nlp.classify.NaiveBayesLearner;
import edu.byu.nlp.classify.UncertaintyPreservingNaiveBayesLearner;
import edu.byu.nlp.classify.data.DatasetBuilder;
import edu.byu.nlp.classify.data.DatasetLabeler;
import edu.byu.nlp.classify.data.GoldLabelLabeler;
import edu.byu.nlp.classify.data.LabelChooser;
import edu.byu.nlp.classify.data.RandomLabelLabeler;
import edu.byu.nlp.classify.data.SingleLabelLabeler;
import edu.byu.nlp.classify.eval.AccuracyComputer;
import edu.byu.nlp.classify.eval.ConfusionMatrixComputer;
import edu.byu.nlp.classify.eval.ConfusionMatrixDistribution;
import edu.byu.nlp.classify.eval.OverallAccuracy;
import edu.byu.nlp.classify.eval.Predictions;
import edu.byu.nlp.classify.eval.ProbabilisticLabelErrorFunction;
import edu.byu.nlp.classify.util.ModelTraining;
import edu.byu.nlp.classify.util.ModelTraining.IntermediatePredictionLogger;
import edu.byu.nlp.classify.util.ModelTraining.OperationType;
import edu.byu.nlp.classify.util.ModelTraining.SupportsTrainingOperations;
import edu.byu.nlp.crowdsourcing.AnnotatorAccuracySetting;
import edu.byu.nlp.crowdsourcing.ArbiterVote;
import edu.byu.nlp.crowdsourcing.EmpiricalMeasurementProvider;
import edu.byu.nlp.crowdsourcing.LabelProvider;
import edu.byu.nlp.crowdsourcing.MajorityVote;
import edu.byu.nlp.crowdsourcing.ModelInitialization;
import edu.byu.nlp.crowdsourcing.ModelInitialization.AssignmentInitializer;
import edu.byu.nlp.crowdsourcing.ModelInitialization.MatrixAssignmentInitializer;
import edu.byu.nlp.crowdsourcing.MultiAnnDatasetLabeler;
import edu.byu.nlp.crowdsourcing.MultiAnnModelBuilders;
import edu.byu.nlp.crowdsourcing.MultiAnnModelBuilders.MultiAnnModelBuilder;
import edu.byu.nlp.crowdsourcing.PriorSpecification;
import edu.byu.nlp.crowdsourcing.SerializableCrowdsourcingState;
import edu.byu.nlp.crowdsourcing.SerializedLabelLabeler;
import edu.byu.nlp.crowdsourcing.measurements.classification.BasicClassificationMeasurementModel;
import edu.byu.nlp.crowdsourcing.measurements.classification.ClassificationMeasurementModelLabeler;
import edu.byu.nlp.crowdsourcing.models.em.CSLDADiscreteModelLabeler;
import edu.byu.nlp.crowdsourcing.models.em.CSLDADiscretePipelinedModelLabeler;
import edu.byu.nlp.crowdsourcing.models.em.FullyDiscriminativeCrowdsourcingModelLabeler;
import edu.byu.nlp.crowdsourcing.models.em.LogRespModelLabeler;
import edu.byu.nlp.crowdsourcing.models.gibbs.BlockCollapsedMultiAnnModel;
import edu.byu.nlp.crowdsourcing.models.gibbs.BlockCollapsedMultiAnnModelMath.DiagonalizationMethod;
import edu.byu.nlp.crowdsourcing.models.gibbs.BlockCollapsedMultiAnnModelNeutered;
import edu.byu.nlp.crowdsourcing.models.gibbs.CollapsedItemResponseModel;
import edu.byu.nlp.crowdsourcing.models.meanfield.MeanFieldItemRespModel;
import edu.byu.nlp.crowdsourcing.models.meanfield.MeanFieldLogRespModel;
import edu.byu.nlp.crowdsourcing.models.meanfield.MeanFieldMomRespModel;
import edu.byu.nlp.crowdsourcing.models.meanfield.MeanFieldMultiAnnLabeler;
import edu.byu.nlp.crowdsourcing.models.meanfield.MeanFieldMultiRespModel;
import edu.byu.nlp.data.docs.CountCutoffFeatureSelectorFactory;
import edu.byu.nlp.data.docs.DocPipes;
import edu.byu.nlp.data.docs.DocumentDatasetBuilder;
import edu.byu.nlp.data.docs.FeatureSelectorFactories;
import edu.byu.nlp.data.docs.JSONDocumentDatasetBuilder;
import edu.byu.nlp.data.docs.JSONVectorDocumentDatasetBuilder;
import edu.byu.nlp.data.docs.TopNPerDocumentFeatureSelectorFactory;
import edu.byu.nlp.data.docs.VectorDocumentDatasetBuilder;
import edu.byu.nlp.data.measurements.ClassificationMeasurements.ClassificationAnnotationMeasurement;
import edu.byu.nlp.data.streams.EmailHeaderStripper;
import edu.byu.nlp.data.streams.EmoticonTransformer;
import edu.byu.nlp.data.streams.PorterStemmer;
import edu.byu.nlp.data.streams.ShortWordFilter;
import edu.byu.nlp.data.streams.StopWordRemover;
import edu.byu.nlp.data.types.Dataset;
import edu.byu.nlp.data.types.Measurement;
import edu.byu.nlp.data.types.SparseFeatureVector;
import edu.byu.nlp.data.util.EmpiricalAnnotations;
import edu.byu.nlp.dataset.Datasets;
import edu.byu.nlp.dataset.Datasets.AnnotatorClusterMethod;
import edu.byu.nlp.io.Files2;
import edu.byu.nlp.io.Paths;
import edu.byu.nlp.math.optimize.MultivariateOptimizers;
import edu.byu.nlp.math.optimize.MultivariateOptimizers.OptimizationMethod;
import edu.byu.nlp.util.Arrays;
import edu.byu.nlp.util.DoubleArrays;
import edu.byu.nlp.util.Functions2;
import edu.byu.nlp.util.Indexer;
import edu.byu.nlp.util.Matrices;
import edu.byu.nlp.util.Pair;
import edu.byu.nlp.util.Strings;
import edu.byu.nlp.util.TimedEvent;
import edu.byu.nlp.util.jargparser.ArgumentParser;
import edu.byu.nlp.util.jargparser.ArgumentValues;
import edu.byu.nlp.util.jargparser.annotations.Option;

/**
 * @author rah67
 * @author plf1
 * 
 */
public class CrowdsourcingLearningCurve {

  // using slf4j logging + MDC for context-sensitive prefixes (to indicate when we are doing hyper
  // parameter optimization vs other)
  private static Logger logger = LoggerFactory.getLogger(CrowdsourcingLearningCurve.class);

  @Option(help = "base directory of the documents") 
  private static String basedir = "./20_newsgroups";

  @Option(help="what kind of hyperparameter optimization to do. In the form maximize-[varname]-[type]-[maxiterations]. type in HyperparamOpt. Default is NONE.")
  private static String hyperparamTraining = "none";

  @Option(help="Should we perform inline hyperparameter tuning?")
  private static boolean inlineHyperparamTuning = false;

  @Option(help="Should we simulate annotators with varying rates?")
  private static boolean varyAnnotatorRates = false;
  
  private enum DatasetType{NEWSGROUPS, REUTERS, ENRON, NB2, NB20, CFGROUPS1000, R8, R52, NG, CADE12, WEBKB, WEATHER, TWITTER, COMPANIES, INDEXED_VEC, JSON_VEC}
  
  @Option(help = "base directory of the documents")
  private static DatasetType datasetType = DatasetType.NEWSGROUPS;
  
  @Option
  private static String dataset = "tiny_set";
  
  @Option(help = "any features that don't appear more than this are discarded")
  private static int featureCountCutoff = 1;
  
  @Option (help = "-1 ignores this filter. Otherwise, all but the N most frequent words in this document are discarded.")
  private static int topNFeaturesPerDocument = -1; 
  
  @Option
  private static String split = "all";
  
  @Option
  private static int evalPoint = -1; 

  @Option (help = "model parameters are saved to this file at the end of the experiment")
  public static String serializeToFile = null;

  @Option
  public static String resultsFile = null;
  
  @Option
  public static String annotationsFile = null;
  
  @Option
  public static String tabularFile = null;
  
  @Option
  public static String debugFile = null;

  @Option
  private static int numObservedLabelsPerClass = 0;
  
  @Option
  private static long dataSeed = System.nanoTime();

  @Option
  private static long algorithmSeed = System.nanoTime();

  @Option(help = "If true, eliminates unannotated data from the dataset before running"
      + "inference of any kind.")
  private static boolean truncateUnannotatedData = false;
  
//  @Option (help = "Has no no effect on baselines. " 
//      + "If labeledDocumentWeight is 1 (default), no unannotated document weighting happens. "
//      + "If >=0, annotated document vectors are normalized and scaled by "
//      + "this factor. "
//      + "If 'binary_classifier', a binary classifier is trained "
//      + "to distinguish annotated from unannotated data, and the probability of being annotated "
//      + "is used as a document weight.")
//  private static String labeledDocumentWeight = "1";

  @Option (help = "Has no effect on baselines. "
      + "If lambda is 1 (default), no unannotated document weighting happens. "
      + "If >=0, unannotated document vectors are normalized and scaled by "
      + "this factor. "
      + "If 'binary_classifier', a binary classifier is trained "
      + "to distinguish annotated from unannotated data, and the probability of being annotated "
      + "is used as a document weight.")
  private static String lambda = "1";

  @Option (help = "Has no effect on baselines. "
      + "If validationPercent is -1, "
      + "no hyperparameter learning happens. "
      + "If 0<validation<100, hyperparams are set to values "
      + "that maximize labeled accuracy on a validation set "
      + "of the size (in percent) specified by validationPercent "
      + "(with annotations produced according to the current settings)."
      + "validationPercent + trainingPercent < 0, and "
      + "100-(validationPercent + trainingPercent)==testPercent.")
  private static double validationPercent = 0;

  @Option (help = "Has no effect on baselines. "
      + "If validationPercent is -1, "
      + "no unannotated document weighting happens. "
      + "If >=0, unannotated document vectors are normalized and scaled by "
      + "the factor that maximizes labeled accuracy on a validation set "
      + "averaged over this many folds. If ==1, then all validation data is used.")
  private static int lambdaValidationFolds = 1;
  
  @Option (help = "If -1, no effect. Otherwise, before any algorithm is run, all "
      + "document feature vectors are scaled to sum to this constant value. "
      + "(e.g., 1 is equivalent to document feature normalization).")
  private static int featureNormalizationConstant = -1;

  private enum LabelingStrategy {MULTIRESP, UBASELINE, BASELINE, MOMRESP, ITEMRESP, LOGRESP_ST, LOGRESP, DISCRIM, VARLOGRESP, VARMULTIRESP, VARMOMRESP, VARITEMRESP, CSLDA, CSLDALEX, CSLDAP, RANDOM, GOLD, PASS, MEAS}; 
  
  /* -------------  Initialization Methods  ------------------- */

  @Option
  private static LabelingStrategy initializationStrategy = LabelingStrategy.RANDOM;
  
  @Option(help = "A sequence of colon-delimited training operations with valid values "
      + "sample,maximize,none where "
      + "sample operations take hyphen-delimited arguments [variablename]-[samples]-[annealingTemp]. "
      + "For example, --training=sample-m-1-1:maximize-all:maximize-y will "
      + "take one sample of all the variables at temp=1, then will do "
      + "joint maximization followed by marginal maximization of y.")
  private static String initializationTraining = "maximize-all";

  /* -------------  Dataset Labeler Methods  ------------------- */

  @Option
  private static LabelingStrategy labelingStrategy = LabelingStrategy.UBASELINE;

  @Option(help = "A sequence of colon-delimited training operations with valid values "
      + "sample,maximize,none where "
      + "sample operations take hyphen-delimited arguments [variablename]-[samples]-[annealingTemp]. "
      + "For example, --training=sample-m-1-1:maximize-all:maximize-y will "
      + "take one sample of all the variables at temp=1, then will do "
      + "joint maximization followed by marginal maximization of y.")
  private static String training = "maximize-all";

  @Option(help = "base the prediction on the single final state of the model. "
      + "Otherwise, the model tracks all samples during the final round of "
      + "annealing and makes a prediction based on marginal distribution.")
  private static boolean predictSingleLastSample = false;
  
  
  /* -------------  Instance Selection Methods  ------------------- */

  @Option(help = "Specifies a number of options that a single anntotator gets to annotate at once (no repeats)")
  private static int annotateTopKChoices = 1; 
  
  @Option(optStrings={"-k","--num-anns-per-instance"})
  private static int k = 1; // Used by grr/ab

  private enum AnnotationStrategy {grr, kdeep, ab, real, reallayers}; 
  @Option
  private static AnnotationStrategy annotationStrategy = AnnotationStrategy.kdeep;
  
  /* -------------  Model Params  ------------------- */
  
  @Option
  private static int maxAnnotations = Integer.MAX_VALUE;
  
  @Option(help = "Accuracy levels of annotators. The first one assumed arbiter for ab1 and ab2")
  private static AnnotatorAccuracySetting annotatorAccuracy = AnnotatorAccuracySetting.HIGH;

  @Option(help = "If present, reads in a json file containing an array of confusion matrices and uses"
      + "them to parameterize the annotators. Warning! They must have the same number of dimensions as "
      + "classes are in the dataset being worked with.")
  private static String annotatorFile = null;
  
  private static final ImmutableSet<Long> arbiters = ImmutableSet.of(0L); // the arbitrator is the first annotator by convention

  @Option(help = "The number of topics used by confused sLDA")
  private static int numTopics = 50;
  
  @Option
  private static double bTheta = 1.0;
  
  @Option
  private static double bMu = 0.6;
  
  @Option
  private static double cMu = 10;
  
  @Option
  private static double bGamma = 0.80;
  
  @Option
  private static double cGamma = 10;
  
  @Option
  private static double bPhi = 0.1;

  @Option
  private static double trainingPercent = 85;

  @Option(help = "The prior Gaussian variance on eta--the logistic regression weights of cslda")
  private static double etaVariance = 1;

  @Option(help = "the label switching cheat should use the entire confusion matrix (including unannotated data). "
      + "This can have a bad effect on learning curves and should only be used for generating correlation plots.")
  private static boolean diagonalizationWithFullConfusionMatrix = false; 


  @Option(help = "How should the class correspondence (label switching) problem be "
  		+ "solved post-hoc? GOLD assumes some amount of annotated data has been given "
  		+ "a gold-standard labeling for calibration. AVG_GAMMA assumes that on "
  		+ "average, inferred annotator error matrices (gamma) should be diagonal. "
  		+ "MAX_GAMMA assumes that the most self-consistent annotator (according to "
  		+ "gamma) should be diagonal.")
  public static DiagonalizationMethod diagonalizationMethod = DiagonalizationMethod.NONE;
  
  @Option(help = "If using --diagonalization-method=GOLD, how many instances should be assumed "
  		+ "to have a gold labeling to be used for diagonalization.")
  public static int goldInstancesForDiagonalization = -1;
  
  @Option
  public static AnnotatorClusterMethod clusterMethod = AnnotatorClusterMethod.KM_MV;
  
  @Option(help = "Group annotators using kmeans clustering on their empirical confusion matrices wrt majority vote."
      + "If -1, don't do any annotator clustering.")
  public static int numAnnotatorClusters = -1;
  
  private static PrintWriter prepareOutput(String path, PrintWriter defaultWriter) throws IOException{
    if (path==null){
      return defaultWriter;
    }
    else{
      // ensure enclosing dir exists
      FileUtils.forceMkdir(new File(path).getParentFile());
      return new PrintWriter(path);
    }
  }
  
  public static void main(String[] args) throws InterruptedException, IOException{
    // parse CLI arguments
    ArgumentValues opts = new ArgumentParser(CrowdsourcingLearningCurve.class).parseArgs(args);

    if (labelingStrategy.toString().contains("LDA")){
      Preconditions.checkArgument(numTopics>0,"LDA-based strategies require numTopics>0");
    }
    Preconditions.checkArgument(evalPoint>0,"evalPoint must be greater than 0");
    Preconditions.checkArgument(new File(basedir).exists(),"basedir must exist "+basedir);
    Preconditions.checkArgument(new File(basedir).isDirectory(),"basedir must be a directory "+basedir);
    Preconditions.checkArgument(new File(basedir).exists(),"basedir must exist "+basedir);
    Preconditions.checkArgument(annotateTopKChoices>0, "--annotate-top-k-choices must be greater than 0");
    
    // this generator deals with data creation (so that all runs with the same annotation strategy
    // settings get the same datasets, regardless of the algorithm run on them)
    RandomGenerator dataRnd = new MersenneTwister(dataSeed);
    RandomGenerator algRnd = new MersenneTwister(algorithmSeed);
    
    // Read in chains of parameter settings (if any).
    // Each file is assumed to represent a single chain.
    List<SerializableCrowdsourcingState> initializationChains = Lists.newArrayList();
    for (String chainFile: opts.getPositionalArgs()){
    	initializationChains.add(SerializableCrowdsourcingState.deserializeFrom(chainFile));
    }
    // aggregate chains (this is calculating max marginal val of variables according to chains)
    initializationChains = (initializationChains.size()==0)? null: initializationChains;  
    SerializableCrowdsourcingState initializationState = SerializableCrowdsourcingState.majorityVote(initializationChains, algRnd);
    
    // ensure encosing 
    
    // open IO streams
    PrintWriter debugOut = prepareOutput(debugFile, nullWriter());
    PrintWriter annotationsOut = prepareOutput(annotationsFile, nullWriter()); 
    PrintWriter tabularPredictionsOut = prepareOutput(tabularFile, nullWriter()); 
    PrintWriter resultsOut = prepareOutput(resultsFile, new PrintWriter(System.out)); 
    PrintWriter serializeOut = prepareOutput(serializeToFile, nullWriter()); 
    
    // pass on to the main program
    CrowdsourcingLearningCurve.run(args, debugOut, annotationsOut, tabularPredictionsOut, resultsOut, serializeOut, initializationState, dataRnd, algRnd);
  }
  
  private static PrintWriter nullWriter(){
	  return new PrintWriter(ByteStreams.nullOutputStream());
  }
  
  
  
  public static void run(String[] args, PrintWriter debugOut, PrintWriter annotationsOut, PrintWriter tabularPredictionsOut, PrintWriter resultsOut, PrintWriter serializeOut, 
		  SerializableCrowdsourcingState initializationState, RandomGenerator dataRnd, RandomGenerator algRnd) throws InterruptedException, IOException {
    ArgumentValues opts = new ArgumentParser(CrowdsourcingLearningCurve.class).parseArgs(args);
    
    // record options
    debugOut.print(opts.optionsMap());
    
    /////////////////////////////////////////////////////////////////////
    // Read and prepare the data
    /////////////////////////////////////////////////////////////////////
    final Stopwatch stopwatchData = Stopwatch.createStarted();
    // currently nothing lda-related can handle fractional word counts
    featureNormalizationConstant = labelingStrategy.name().contains("CSLDA")? -1: featureNormalizationConstant;
    Dataset fullData = readData(dataRnd,featureNormalizationConstant);

    Preconditions.checkArgument(annotateTopKChoices<=fullData.getInfo().getNumClasses(), "--annotate-top-k-choices must not be greater than the number of classes");
    // transform the annotations (if requested) via annotation clustering
    if (numAnnotatorClusters>0){
      double parameterSmoothing = 0.01;
      fullData = Datasets.withClusteredAnnotators(fullData, numAnnotatorClusters, clusterMethod, parameterSmoothing, dataRnd);
    }
    
    logger.info("\nDataset after annotator clustering: \n"+Datasets.summaryOf(fullData,1));

    // Save annotations for future use (if we're using an empirical annotation strategy)
    final EmpiricalAnnotations<SparseFeatureVector, Integer> annotations = EmpiricalAnnotations.fromDataset(fullData);

    // ensure the dataset knows about all the annotators it will need to deal with.
    // if we are dealing with real data, we read in annotators with the data. Otherwise, 
    // we'll have to change it. 
    annotatorAccuracy.generateConfusionMatrices(dataRnd, varyAnnotatorRates, fullData.getInfo().getNumClasses(), annotatorFile);
    if (!annotationStrategy.toString().contains("real")){
      fullData = Datasets.withNewAnnotators(fullData, annotatorAccuracy.getAnnotatorIdIndexer());
    }
    if (annotationStrategy==AnnotationStrategy.kdeep){
      Preconditions.checkState(annotatorAccuracy.getNumAnnotators()>=k, "Not enough simulated annotators ("+annotatorAccuracy.getNumAnnotators()+") to implement kdeep="+k+" (remember kdeep samples annotators WITHOUT replacement)");
    }
    
    List<Dataset> dataSplits = splitData(fullData, trainingPercent, validationPercent, dataRnd);
    final Dataset trainingData = dataSplits.get(0);
    final Dataset validationData = dataSplits.get(1);
    final Dataset testData = dataSplits.get(2); 

    stopwatchData.stop();
    
    boolean returnLabeledAccuracy = true;
    // initialize model variables
    SerializableCrowdsourcingState initialization = trainEval(nullWriter(), nullWriter(), nullWriter(), nullWriter(),
            null, dataRnd, algRnd, stopwatchData, 
            trainingData, false, testData, annotations, // train on training data (also use unannotated, unlabeled validation data) 
            bTheta, bMu, bPhi, bGamma, cGamma, 
            lambda, evalPoint, initializationStrategy, initializationTraining, returnLabeledAccuracy);
    
    // cross-validation sweep unannotated-document-weight (optional)
    if (validationPercent>0){
    	MDC.put("context", "hyperopt");
    	int validationEvalPoint = (int)Math.round(validationData.getInfo().getNumDocuments()/((double)trainingData.getInfo().getNumDocuments()) * evalPoint);
    	ModelTraining.doOperations(hyperparamTraining, 
    	    new CrowdsourcingHyperparameterOptimizer(initialization, validationData, annotations, validationEvalPoint));
    	MDC.remove("context");
    }

    // final go
    SerializableCrowdsourcingState finalState = trainEval(debugOut, annotationsOut, tabularPredictionsOut, resultsOut,
        initialization, dataRnd, algRnd, stopwatchData, 
        trainingData, false, testData, annotations, // train on training data (also use unannotated, unlabeled validation data) 
        bTheta, bMu, bPhi, bGamma, cGamma, 
        lambda, evalPoint, labelingStrategy, training, returnLabeledAccuracy);
    
    // serialize state out
    finalState.serializeTo(serializeOut);
    
    debugOut.close();
    annotationsOut.close();
    tabularPredictionsOut.close();
    resultsOut.close();
    serializeOut.close();
  }

  
  private static class CrowdsourcingHyperparameterOptimizer implements SupportsTrainingOperations{
	private Dataset validationData;
	private EmpiricalAnnotations<SparseFeatureVector, Integer> annotations;
	private int validationEvalPoint;
	private Set<Double> bThetaGrid = Sets.newHashSet(0.01, 0.1, 0.5, 1.0);
	private Set<Double> bPhiGrid = Sets.newHashSet(0.01, 0.1, 0.5, 1.0);
	private Set<Double> bGammaGrid = Sets.newHashSet(0.1, 0.5, 0.9);
	private Set<Double> cGammaGrid = Sets.newHashSet(.1, 1.0, 10.0, 50.0);
	private double bTheta,bPhi,bGamma,cGamma; // determined at run time
	private List<Set<Double>> grid = null; // determined at run time
	private double[] startPoint = null; // determined at run time
	private double[][] boundaries = null; // determined at run time
	private SerializableCrowdsourcingState initialization;
	public CrowdsourcingHyperparameterOptimizer(
		  SerializableCrowdsourcingState initialization, 
	      Dataset validationData,
	      EmpiricalAnnotations<SparseFeatureVector, Integer> annotations,
	      int evalPoint){
		  this.initialization=initialization;
		  this.validationData=validationData;
		  this.annotations=annotations;
		  this.validationEvalPoint=evalPoint;
	  }
	@Override
	public Double sample(String variableName, int iteration, String[] args) {
		throw new UnsupportedOperationException("not implemented");
	}
  @Override
  public DatasetLabeler getIntermediateLabeler() {
    throw new UnsupportedOperationException("not implemented");
  }
	/**
	 * args are in the form maximize-1-[params]-[maxiterations]-[training]
	 * where [params] has a comma-separated list of parameter names to be updated
	 * and [training] has the same form as what is given to the --training param. 
	 * (the global --training args are used as default values).
	 */
	@Override
	public Double maximize(final String parameterNames, int iteration, final String[] args) {
		// parameterNames: comma-separated list of param names to be updated
		importParams(parameterNames);
		preparePointsAndBoundaries(parameterNames);
		// args[0]: hyperparam optimization strategy
		OptimizationMethod optMethod = (args.length==0)? OptimizationMethod.BOBYQA: OptimizationMethod.valueOf(args[0]);
		// args[1]: how many hyperparam iterations?
		int maxEvaluations = (args.length>=2)? Integer.parseInt(args[1]): 50;
		// args[2]: labeling strategy (guiding hyperparam search) 
		final LabelingStrategy hyperLabelingStrategy = (args.length>=3)? LabelingStrategy.valueOf(args[2]): labelingStrategy;
		// args[3]: evaluation criterion (either 'acc' for supervised labeled accuracy or else 'joint' for unsupervised log joint)
		final boolean returnLabeledAccuracy = (args.length>=4)? args[3].toLowerCase().equals("acc"): true;
		// args[4:]: training strategy (guiding hyperparam search) 
		final String hyperTraining = (args.length>=5)? Strings.join(Arrays.subsequence(args,4),"-"): training;
		
		//////////////////////////////////////
	    // Optimize ItemResp hyperparams (theta, gamma)
		//////////////////////////////////////
		PointValuePair optimum = MultivariateOptimizers.optimize(optMethod, maxEvaluations, startPoint, boundaries, 
	    		// grid
				grid,
				// objective function
				new MultivariateFunction(){
					private int iterations = 0;
			        @Override
			        public double value(double[] point) {
			          iterations++;
			    	  adoptParams(parameterNames, point); // writes values onto bTheta, bPhi, etc
			          
			          // ensure results are consistent by using same seed
			          RandomGenerator algRnd = new MersenneTwister(algorithmSeed);
			          RandomGenerator dataRnd = new MersenneTwister(dataSeed);
			          PrintWriter nullPipe = nullWriter();
			          boolean onlyAnnotateLabeledData = true;
			          double val = trainEval(nullPipe, nullPipe, nullPipe, nullPipe, 
			              initialization, dataRnd, algRnd, null, // null stopwatch
			              validationData, onlyAnnotateLabeledData,  
			              Datasets.emptyDataset(validationData.getInfo()), // no test data 
			              annotations, 
			              bTheta, CrowdsourcingLearningCurve.bMu, bPhi, bGamma, cGamma, CrowdsourcingLearningCurve.lambda, validationEvalPoint,
			              hyperLabelingStrategy, hyperTraining, returnLabeledAccuracy).getGoodness(); // use indicated training regime
			          logger.info("ItemResp hyperparam search iteration "+iterations+" {bTheta="+bTheta+" bPhi="+bPhi+" bGamma="+bGamma+" cGamma="+cGamma+"}="+val);
			          return val;
		        }
		       });

		// adopt the best values
		adoptParams(parameterNames, optimum.getPointRef());
	    logger.info("final hyperparameter values: bTheta="+bTheta+" bPhi="+bPhi+" bGamma="+bGamma+" cGamma="+cGamma);
	    // export the best values to static variables
	    exportParams(parameterNames); 
	    return null; // return value is ignored 
	}
	private void preparePointsAndBoundaries(String parameterNames){
		List<Double> startPointList = Lists.newArrayList();
		List<Double> boundariesStartList = Lists.newArrayList();
		List<Double> boundariesEndList = Lists.newArrayList();
		grid = Lists.newArrayList();
		if (parameterNames.contains("btheta")){
			startPointList.add(CrowdsourcingLearningCurve.bTheta);
			boundariesStartList.add(edu.byu.nlp.util.Sets.min(bThetaGrid));
			boundariesEndList.add(edu.byu.nlp.util.Sets.max(bThetaGrid));
			grid.add(bThetaGrid);
		}
		if (parameterNames.contains("bphi")){
			startPointList.add(CrowdsourcingLearningCurve.bPhi);
			boundariesStartList.add(edu.byu.nlp.util.Sets.min(bPhiGrid));
			boundariesEndList.add(edu.byu.nlp.util.Sets.max(bPhiGrid));
			grid.add(bPhiGrid);
		} 
		if (parameterNames.contains("bgamma")){
			startPointList.add(CrowdsourcingLearningCurve.bGamma);
			boundariesStartList.add(edu.byu.nlp.util.Sets.min(bGammaGrid));
			boundariesEndList.add(edu.byu.nlp.util.Sets.max(bGammaGrid));
			grid.add(bGammaGrid);
		} 
		if (parameterNames.contains("cgamma")){
			startPointList.add(CrowdsourcingLearningCurve.cGamma);
			boundariesStartList.add(edu.byu.nlp.util.Sets.min(cGammaGrid));
			boundariesEndList.add(edu.byu.nlp.util.Sets.max(cGammaGrid));
			grid.add(cGammaGrid);
		} 
		startPoint = DoubleArrays.fromList(startPointList);
		boundaries = new double[][]{DoubleArrays.fromList(boundariesStartList), DoubleArrays.fromList(boundariesEndList)};
		Preconditions.checkState(startPoint.length>0,"no valid hyperparameters were recognized in the string \""+parameterNames);
	}
	private void exportParams(String parameterNames){
		CrowdsourcingLearningCurve.bTheta = this.bTheta;
		CrowdsourcingLearningCurve.bPhi = this.bPhi;
		CrowdsourcingLearningCurve.bGamma = this.bGamma;
		CrowdsourcingLearningCurve.cGamma = this.cGamma;
	}
	private void importParams(String parameterNames){
		this.bTheta = CrowdsourcingLearningCurve.bTheta;
		this.bPhi = CrowdsourcingLearningCurve.bPhi;
		this.bGamma = CrowdsourcingLearningCurve.bGamma;
		this.cGamma = CrowdsourcingLearningCurve.cGamma;
	}
	private void adoptParams(String parameterNames, double[] values){
		int index = 0;
		if (parameterNames.contains("btheta")){
			this.bTheta = values[index++];
		}
		if (parameterNames.contains("bphi")){
			this.bPhi = values[index++];
		} 
		if (parameterNames.contains("bgamma")){
			this.bGamma = values[index++];
		} 
		if (parameterNames.contains("cgamma")){
			this.cGamma = values[index++];
		} 
	}
  }

  private static int nextAnnotatorId = 0;
  /**
   * @param returnLabeledAccuracy if false, returns log joint
   * @return
   */
  private static SerializableCrowdsourcingState trainEval(PrintWriter debugOut,
      PrintWriter annotationsOut, PrintWriter tabularPredictionsOut,
      PrintWriter resultsOut, SerializableCrowdsourcingState initialState, 
      RandomGenerator dataRnd, RandomGenerator algRnd,
      Stopwatch stopwatchData, Dataset trainingData, boolean onlyAnnotateLabeledData,  
      final Dataset testData, final EmpiricalAnnotations<SparseFeatureVector, Integer> annotations,
      double bTheta, double bMu, double bPhi, double bGamma, double cGamma, 
      String lambda, int evalPoint, LabelingStrategy labelingStrategy, String training, 
      boolean returnLabeledAccuracy) {
    
    /////////////////////////////////////////////////////////////////////
    // Prepare data. 
    /////////////////////////////////////////////////////////////////////
    // remove any existing annotations from the training data; this is only relevant if doing multiple evaluations in a single run
    Datasets.clearAnnotations(trainingData);
    // all ground-truth labels are hidden for crowdsourcing inference (unless we decide to have a few observed)
//    if (!annotationStrategy.toString().contains("real")){ // if labels were observed in real data, observe them here
    trainingData = Datasets.hideAllLabelsButNPerClass(trainingData, numObservedLabelsPerClass, dataRnd);
//    }
    logger.info("Trusted labels available for " + trainingData.getInfo().getNumDocumentsWithObservedLabels() + " instances");
    logger.info("No labels available for " + trainingData.getInfo().getNumDocumentsWithoutObservedLabels() + " instances");

    
    Preconditions.checkArgument(trainingData.getInfo().getNumDocuments()>0,"Training dataset contained 0 documents. Cannot train a model with no training data.");
    logger.info("======================================================================================");
    logger.info("============= Train + eval ("+labelingStrategy+" bTheta="+bTheta+" bPhi="+bPhi+" bGamma="+bGamma+" cGamma="+cGamma+", evalpoint="+evalPoint+") ==============");
    logger.info("======================================================================================");
    logger.info("data seed "+dataSeed);
    logger.info("algorithm seed "+algorithmSeed);
    logger.info("hyperparameters: bTheta="+bTheta+" bPhi="+bPhi+" bGamma="+bGamma+" cGamma="+cGamma);

    
//    // TODO; data with known and observed labels is suitable for adding as extra supervision to models (assuming they know how to deal with it)
//    Dataset observedLabelsTrainingData = Datasets.divideInstancesWithObservedLabels(trainingData).getFirst();
    // data with known but concealed labels is suitable for simulating annotators and doing evaluation 
    Dataset concealedLabelsTrainingData = Datasets.divideInstancesWithLabels(trainingData).getFirst();

    Stopwatch stopwatchInference = Stopwatch.createStarted();

    /////////////////////////////////////////////////////////////////////
    // Annotators
    /////////////////////////////////////////////////////////////////////
    List<? extends LabelProvider<SparseFeatureVector,Measurement>> annotators;
    if (annotationStrategy.toString().contains("real")){
      annotators = createEmpiricalAnnotators(annotations);
      logger.info("Number of Human Annotators = " + annotators.size());
      annotatorAccuracy = null; // avoid reporting misleading stats based on unused simulation parameters
    }
    else{
      annotators = createAnnotators(concealedLabelsTrainingData, annotatorAccuracy, concealedLabelsTrainingData.getInfo().getNumClasses(), dataRnd);
      logger.info("Number of Simulated Annotators = " + annotators.size());
      for (int i=0; i<annotatorAccuracy.getAccuracies().length; i++){
    	  logger.info("annotator #"+i+" accuracy="+annotatorAccuracy.getAccuracies()[i]);
      }
    }

    
    
    /////////////////////////////////////////////////////////////////////
    // Choose an annotation_strategy+label_chooser combination
    // valid options: grr+majority, single+majority, ab-arbitrator+arbitrator
    //
    // we do annotation on the main training set. Annotations are added in-place, 
    // mutating the dataset. Note that since we didn't do a deep copy, all splits 
    // of the dataset and all references to the datasetinstance objects should be 
    // receiving annotations as well. This doesn't currently affect any logic, 
    // since only annotations mutate and all code except for crowdsourcing training 
    // code ignores annotations, but it's worth noting.
    /////////////////////////////////////////////////////////////////////
    InstanceManager<SparseFeatureVector, Integer> instanceManager;
    LabelChooser baselineChooser;
    switch(annotationStrategy){
    case ab:
      trainingData = Datasets.divideInstancesWithLabels(trainingData).getFirst(); // can't simulate annotations for unlabeled instances
      baselineChooser = new ArbiterVote(arbiters, algRnd);
      instanceManager = ABArbiterInstanceManager.newManager(trainingData, k==1, arbiters);
      break;
    case grr:
      trainingData = Datasets.divideInstancesWithLabels(trainingData).getFirst(); // can't simulate annotations for unlabeled instances
      baselineChooser = new MajorityVote(algRnd);
      instanceManager = GeneralizedRoundRobinInstanceManager.newManager(k, trainingData, new DatasetAnnotationRecorder(trainingData), dataRnd);
      break;
    case kdeep:
      trainingData = Datasets.divideInstancesWithLabels(trainingData).getFirst(); // can't simulate annotations for unlabeled instances
      baselineChooser = new MajorityVote(algRnd);
      instanceManager = NDeepInstanceManager.newManager(k, 1, trainingData, new DatasetAnnotationRecorder(trainingData), dataRnd);
      break;
    case real:
      baselineChooser = new MajorityVote(algRnd);
      instanceManager = EmpiricalAnnotationInstanceManager.newManager(onlyAnnotateLabeledData? concealedLabelsTrainingData: trainingData, annotations);
      break;
    case reallayers:
      baselineChooser = new MajorityVote(algRnd);
      instanceManager = EmpiricalAnnotationLayersInstanceManager.newManager(onlyAnnotateLabeledData? concealedLabelsTrainingData: trainingData, annotations, dataRnd);
      break;
    default:
        throw new IllegalArgumentException("Unknown annotation strategy: " + annotationStrategy.name());
    }
    // rate limited annotators
    if (annotatorAccuracy!=null){
    	instanceManager = new RateLimitedAnnotatorInstanceManager<>(instanceManager, identityAnnotatorRatesMap(annotatorAccuracy.getAnnotatorRates()), dataRnd);
    }


    /////////////////////////////////////////////////////////////////////
    // Annotate until the eval point
    /////////////////////////////////////////////////////////////////////
    annotationsOut.println("source, annotator_id, annotation, annotation_time_nanos, wait_time_nanos"); // annotation file header
    for (int numAnnotations = 0; numAnnotations<maxAnnotations && numAnnotations<evalPoint; numAnnotations++) {

      // Get an instance and annotator assignment
      AnnotationRequest<SparseFeatureVector, Integer> request = null;
      // keep trying until we get one right. The instance  
      // manager should only return isdone==true when 
      // 1) doing ab-arbitrator when we we ask 
      // for an arbitrator instance and there are no conflicts.
      // 2) the realistic annotator enforces historical order.
      while (request==null && !instanceManager.isDone()){
        // Pick an annotator at random
        int annotatorId = dataRnd.nextInt(annotators.size()); // TODO: restore this
//        int annotatorId = ((int)Math.floor(numAnnotations/trainingData.getInfo().getNumClasses()))%(annotators.size()); 
        try {
          request = instanceManager.requestInstanceFor(annotatorId, 1, TimeUnit.MILLISECONDS);
        } catch (InterruptedException e) {
          e.printStackTrace();
          throw new IllegalStateException();
        }
      }
      
      // instance manager has finished (only ab-arbitrator and real will do this. Grr goes forever!)
      if (request==null){
        logger.info("Out of annotations. Finishing with "+numAnnotations+" annotations.");
        break;
      }
      // Annotate (ignore timing information)
      else {

        LabelProvider<SparseFeatureVector, Measurement> annotator = annotators.get((int)request.getAnnotatorId());
        Measurement measurement = annotator.labelFor(request.getInstance().getSource(), request.getInstance().getData());
        Integer label = measurement instanceof ClassificationAnnotationMeasurement? ((ClassificationAnnotationMeasurement)measurement).getLabel(): null;
        AnnotationInfo<Integer> ai = new AnnotationInfo<>(new Long(numAnnotations), label, measurement, TimedEvent.Zeros(), TimedEvent.Zeros());
        request.storeAnnotation(ai);
        writeAnnotation(annotationsOut, ai, request);
        
      }
    } // end annotate

    // truncate unannotated training data (if requested). Note that some 
    // algorithms do this below manually no matter what option is specified
    if (truncateUnannotatedData){
    	trainingData = truncateUnannotatedUnlabeledData(trainingData);
    }
    
    // report final state of the dataset before training
    logger.info("\nFinal training data (actually used for training): \n"+Datasets.summaryOf(trainingData,1));
    logger.info("\nFinal test data (actually used for testing): \n"+Datasets.summaryOf(testData,1));
    
    /////////////////////////////////////////////////////////////////////
    // Build a dataset labeler
    // valid options: 1) multiannotator model(s)
    //                2) use chooser strategy (ab,majority) on labeled portion, and naive bayes on unlabeled/test 
    /////////////////////////////////////////////////////////////////////
    DatasetLabeler labeler;
    
    // state initialization
    AssignmentInitializer yInitializer = ModelInitialization.backoffStateInitializerForY(initialState, 
    		new ModelInitialization.BaselineInitializer(algRnd)); 
    AssignmentInitializer mInitializer = ModelInitialization.backoffStateInitializerForM(initialState, 
			new ModelInitialization.BaselineInitializer(algRnd, true)); // slightly noisier default initialization for m 
    MatrixAssignmentInitializer zInitializer = ModelInitialization.backoffStateInitializerForZ(initialState, 
    		ModelInitialization.uniformRowMatrixInitializer(new ModelInitialization.UniformAssignmentInitializer(numTopics, algRnd))); 
    
    final Dataset evalData = trainingData;
    IntermediatePredictionLogger predictionLogger = new IntermediatePredictionLogger() {
      @Override
      public void logPredictions(int iteration, OperationType opType,
          String variableName, String[] args, DatasetLabeler intermediateLabeler) {
        if (opType==OperationType.MAXIMIZE || iteration%25==0){
          // print intermediate confusion matrices
          Predictions predictions = intermediateLabeler.label(evalData, testData);
          logger.info("confusion matrix after "+opType+"-"+variableName+" (args="+Joiner.on('-').join(args)+" iteration="+iteration+")\n"
              +new ConfusionMatrixComputer(evalData.getInfo().getLabelIndexer()).compute(predictions.labeledPredictions()).toString());
          logAccuracy("accuracy after "+opType+"-"+variableName+" (args="+Joiner.on('-').join(args)+" iteration="+iteration+")\n", 
              new AccuracyComputer().compute(predictions, annotations.getDataInfo().getNullLabel()));
        }
      }
    };
    
    PriorSpecification priors = new PriorSpecification(bTheta, bMu, cMu, bGamma, cGamma, bPhi, etaVariance, inlineHyperparamTuning, annotators.size());
    switch(labelingStrategy){

    // uniform random predictions
    case RANDOM:
    	labeler = new RandomLabelLabeler(algRnd);
    	break;
    	
    // use initial values unchanged
    case PASS:
    	labeler = new SerializedLabelLabeler(initialState);
    	break;
    	
    case GOLD:
    	labeler = new GoldLabelLabeler();
    	break;
    
    case UBASELINE:
      // this labeler reports labeled data without alteration, and trains an uncertainty-
      // preserving naive bayes variant on it to label the unlabeled and test portions
      labeler = new SingleLabelLabeler(new UncertaintyPreservingNaiveBayesLearner(), new DatasetBuilder(baselineChooser), annotators.size());
      break;
      
    case BASELINE:
      // this labeler reports labeled data without alteration, and trains a naive bayes 
      // on it to label the unlabeled and test portions
      labeler = new SingleLabelLabeler(new NaiveBayesLearner(), new DatasetBuilder(baselineChooser), annotators.size());
      break;

    case LOGRESP_ST:
      labeler = new LogRespModelLabeler(trainingData,  priors, true);
      break;

    case LOGRESP:
      trainingData = truncateUnannotatedUnlabeledData(trainingData);
      labeler = new LogRespModelLabeler(trainingData,  priors, false);
      break;

    case DISCRIM:
      trainingData = truncateUnannotatedUnlabeledData(trainingData);
      labeler = new FullyDiscriminativeCrowdsourcingModelLabeler(trainingData,  priors, false);
      break;

    case CSLDA:
      Preconditions.checkState(featureNormalizationConstant == -1, "cslda can't handle fractional doc counts: "+featureNormalizationConstant); // cslda code currently can't handle fractional word counts
      labeler = new CSLDADiscreteModelLabeler(trainingData, numTopics, training, zInitializer, yInitializer, priors, predictionLogger, predictSingleLastSample, false, algRnd);
      break;

    case CSLDALEX:
      Preconditions.checkState(featureNormalizationConstant == -1, "cslda can't handle fractional doc counts: "+featureNormalizationConstant); // cslda code currently can't handle fractional word counts
      labeler = new CSLDADiscreteModelLabeler(trainingData, numTopics, training, zInitializer, yInitializer, priors, predictionLogger, predictSingleLastSample, true, algRnd);
      break;
      
    case CSLDAP:
      Preconditions.checkState(featureNormalizationConstant == -1, "LOGRESP_LDA can't handle fractional doc counts: "+featureNormalizationConstant); // cslda code currently can't handle fractional word counts
      labeler = new CSLDADiscretePipelinedModelLabeler(trainingData, numTopics, training, zInitializer, yInitializer, priors, predictionLogger, predictSingleLastSample, algRnd);
      break;
      
    case VARLOGRESP:
  	  trainingData = truncateUnannotatedUnlabeledData(trainingData);
      labeler = new MeanFieldMultiAnnLabeler(MultiAnnModelBuilders.initModelBuilder(new MeanFieldLogRespModel.ModelBuilder(), 
    				  priors, trainingData, yInitializer, mInitializer, algRnd),
    		  training, predictionLogger);
      break;
      
    case VARMULTIRESP:
        labeler = new MeanFieldMultiAnnLabeler(MultiAnnModelBuilders.initModelBuilder(new MeanFieldMultiRespModel.ModelBuilder(), 
				  priors, trainingData, yInitializer, mInitializer, algRnd),
		  training, predictionLogger);
      break;
      
    case VARMOMRESP:
        labeler = new MeanFieldMultiAnnLabeler(MultiAnnModelBuilders.initModelBuilder(new MeanFieldMomRespModel.ModelBuilder(), 
				  priors, trainingData, yInitializer, mInitializer, algRnd),
	    		  training, predictionLogger);
      break;
      
    case VARITEMRESP:
  	  trainingData = truncateUnannotatedUnlabeledData(trainingData);
      labeler = new MeanFieldMultiAnnLabeler(MultiAnnModelBuilders.initModelBuilder(new MeanFieldItemRespModel.ModelBuilder(), 
				  priors, trainingData, yInitializer, mInitializer, algRnd),
	    		  training, predictionLogger);
      break;
      
    case MEAS:
      labeler = new ClassificationMeasurementModelLabeler(
          new BasicClassificationMeasurementModel.Builder().setPriors(priors).setYInitializer(yInitializer).setRnd(algRnd).setData(trainingData),
          training, predictionLogger);
      break;
      
    // some variant of multiannotator model sampling
    default:
      MultiAnnModelBuilder multiannModelBuilder;
      
      switch(labelingStrategy){
      case MULTIRESP:
        multiannModelBuilder = new BlockCollapsedMultiAnnModel.ModelBuilder();
        break;
        
      case MOMRESP:
        multiannModelBuilder = new BlockCollapsedMultiAnnModelNeutered.ModelBuilder();
        break;
        
      case ITEMRESP:
	    trainingData = truncateUnannotatedUnlabeledData(trainingData);
        multiannModelBuilder = new CollapsedItemResponseModel.ModelBuilder();
        break;
        
      default:
        throw new IllegalArgumentException("Unknown labeling strategy: " + labelingStrategy.name());
      }
      
      labeler = new MultiAnnDatasetLabeler(
    		  MultiAnnModelBuilders.initModelBuilder(multiannModelBuilder, 
    				  priors, trainingData, yInitializer, mInitializer, algRnd),
    		  debugOut, predictSingleLastSample, training, 
	          diagonalizationMethod, diagonalizationWithFullConfusionMatrix, goldInstancesForDiagonalization, trainingData, 
	          lambda, predictionLogger, algRnd);
      
    }
    
    
    /////////////////////////////////////////////////////////////////////
    // Eval the labeler
    /////////////////////////////////////////////////////////////////////
    DatasetMetricComputer annotationsCounter = new DatasetMetricComputer();
    LogJointComputer jointComputer = new LogJointComputer();
    AccuracyComputer accuracyComputer = new AccuracyComputer();
    AccuracyComputer top3AccuracyComputer = new AccuracyComputer(3);
    AnnotatorAccuracyComputer annAccComputer = new AnnotatorAccuracyComputer(annotators.size());
    RmseAnnotatorAccuracyComputer rmseComputer = new RmseAnnotatorAccuracyComputer(annotatorAccuracy==null?null:annotatorAccuracy.getAccuracies());
    RmseAnnotatorConfusionMatrixComputer rmseMatrixComputer = new RmseAnnotatorConfusionMatrixComputer(annotatorAccuracy==null?null:annotatorAccuracy.getConfusionMatrices());
    MachineAccuracyComputer machineAccComputer = new MachineAccuracyComputer();
    RmseMachineAccuracyVsTestComputer machineRmseComputer = new RmseMachineAccuracyVsTestComputer();
    RmseMachineConfusionMatrixVsTestComputer machineMatRmseComputer = new RmseMachineConfusionMatrixVsTestComputer(trainingData.getInfo().getLabelIndexer());
    ExperimentSettingsComputer settingsComputer = new ExperimentSettingsComputer();

    // file headers
    resultsOut.println(Joiner.on(',').join(
        annotationsCounter.csvHeader(), jointComputer.csvHeader(), accuracyComputer.csvHeader(), top3AccuracyComputer.csvHeader(),
        annAccComputer.csvHeader(), rmseComputer.csvHeader(), rmseMatrixComputer.csvHeader(),
        machineAccComputer.csvHeader(), machineRmseComputer.csvHeader(), machineMatRmseComputer.csvHeader(),
        settingsComputer.csvHeader()
        ));

    // predict/eval on all instances 
    Predictions predictions = labeler.label(trainingData, testData);
    
    // record results
    logger.info("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
    logger.info("!! "+CrowdsourcingLearningCurve.class.getSimpleName()+" complete! Writing results.");
    logger.info("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
    int nullLabel = annotations.getDataInfo().getNullLabel();
    stopwatchInference.stop();
    OverallAccuracy accResults = accuracyComputer.compute(predictions, nullLabel);
    OverallAccuracy acc3Results = top3AccuracyComputer.compute(predictions, nullLabel);
    double jointResults = jointComputer.compute(predictions);
    resultsOut.println(Joiner.on(',').join(
        annotationsCounter.compute(trainingData),
        jointResults,
        accResults.toCsv(),
        acc3Results.toCsv(),
        annAccComputer.compute(predictions).toCsv(),
        rmseComputer.compute(predictions),
        rmseMatrixComputer.compute(predictions),
        machineAccComputer.compute(predictions),
        machineRmseComputer.compute(predictions, trainingData.getInfo().getNullLabel()),
        machineMatRmseComputer.compute(predictions),
        settingsComputer.compute(
            (stopwatchData==null)? 0: (int)stopwatchData.elapsed(TimeUnit.SECONDS),
            (int)stopwatchInference.elapsed(TimeUnit.SECONDS),
            priors)
        ));
    resultsOut.flush();
    PredictionTabulator.writeTo(predictions, tabularPredictionsOut);
    
    // for debugging only
    try {
      Files2.write(Datasets.toAnnotationCsv(trainingData), "/tmp/"+Paths.baseName(trainingData.getInfo().getSource())+".csv");
    } catch (IOException e) {
      e.printStackTrace();
    }

    logger.info("confusion matrix\n"+new ConfusionMatrixComputer(trainingData.getInfo().getLabelIndexer()).compute(predictions.labeledPredictions()).toString());
    logger.info("annacc = " + DoubleArrays.toString(predictions.annotatorAccuracies()));
    logger.info("machacc = " + predictions.machineAccuracy());
//    logger.info("machacc_mat = " + Matrices.toString(predictions.machineConfusionMatrix()));
    logger.info("log joint = " + jointResults);
    logAccuracy("", accResults);
    logAccuracy("top3", acc3Results);

    SerializableCrowdsourcingState modelState = SerializableCrowdsourcingState.of(predictions);
    // hyperparam optimization is based on this result
    if (returnLabeledAccuracy){
    	// labeled accuracy (of interest but very noisy/jumpy)
    	modelState.setGoodness(accResults.getLabeledAccuracy().getAccuracy());
    }
    else{
    	// smooth (but not necessarily what we are most interested in)
    	modelState.setGoodness(jointResults);
    }
    return modelState;
  }

  


  private static void logAccuracy(String prefix, OverallAccuracy acc){
    logger.info(prefix+"test_acc = " + acc.getTestAccuracy().getAccuracy()+" ("+acc.getTestAccuracy().getCorrect()+"/"+acc.getTestAccuracy().getTotal()+")");
    logger.info(prefix+"labeled_acc = " + acc.getLabeledAccuracy().getAccuracy()+" ("+acc.getLabeledAccuracy().getCorrect()+"/"+acc.getLabeledAccuracy().getTotal()+")");
    logger.info(prefix+"unlabeled_acc = " + acc.getUnlabeledAccuracy().getAccuracy()+" ("+acc.getUnlabeledAccuracy().getCorrect()+"/"+acc.getUnlabeledAccuracy().getTotal()+")");
    logger.info(prefix+"overall_acc = " + acc.getOverallAccuracy().getAccuracy()+" ("+acc.getOverallAccuracy().getCorrect()+"/"+acc.getOverallAccuracy().getTotal()+")");
  }

  private static Dataset truncateUnannotatedUnlabeledData(Dataset data){
      logger.info("truncating unlabeled/unannotated instances");
      Dataset truncated = Datasets.removeDataWithoutAnnotationsOrObservedLabels(data); 
      logger.info("data size: before truncation="+data.getInfo().getNumDocuments()+" after truncation="+truncated.getInfo().getNumDocuments());
      return truncated;
  }


  private static List<? extends LabelProvider<SparseFeatureVector,Measurement>> createEmpiricalAnnotators(
                                                         EmpiricalAnnotations<SparseFeatureVector, Integer> annotations) {
    List<EmpiricalMeasurementProvider<SparseFeatureVector>> annotators = Lists.newArrayList();
    for (String annotator: annotations.getDataInfo().getAnnotatorIdIndexer()){
      int annotatorIndex = annotations.getDataInfo().getAnnotatorIdIndexer().indexOf(annotator);
      annotators.add(new EmpiricalMeasurementProvider<SparseFeatureVector>(annotatorIndex, annotations));
    }
    return annotators;
  }

  public static List<? extends LabelProvider<SparseFeatureVector,Measurement>> createAnnotators(Dataset concealedLabeledTrainingData,
                                                                                    AnnotatorAccuracySetting accuracySetting, int numLabels,
                                                                                    RandomGenerator rnd) {
    GoldLabelProvider<SparseFeatureVector,Integer> goldLabelProvider = GoldLabelProvider.from(concealedLabeledTrainingData);
    List<FallibleMeasurementProvider<SparseFeatureVector>> annotators = Lists.newArrayList();
    double[][][] annotatorConfusions = accuracySetting.getConfusionMatrices();
    double[] annotatorRates = accuracySetting.getAnnotatorRates();
    // scale annotator rates up until the first one hits 1 (won't change proportions but will decrease failure rate)
    DoubleArrays.multiplyToSelf(annotatorRates, 1.0/DoubleArrays.max(annotatorRates));
    
    for (int j=0; j<annotatorConfusions.length; j++) {
      ProbabilisticLabelErrorFunction<Integer> labelErrorFunction = 
          new ProbabilisticLabelErrorFunction<Integer>(new ConfusionMatrixDistribution(annotatorConfusions[j]),rnd);
      FallibleAnnotationProvider<SparseFeatureVector,Integer> fallibleLabelProvider = 
          new FallibleAnnotationProvider<SparseFeatureVector, Integer>(goldLabelProvider, labelErrorFunction);
      boolean generateNonTrivialMeasurements = labelingStrategy==LabelingStrategy.MEAS; 
      FallibleMeasurementProvider<SparseFeatureVector> annotator = new FallibleMeasurementProvider<>(
          fallibleLabelProvider, concealedLabeledTrainingData, j, annotatorAccuracy.getAccuracies()[j], 
          generateNonTrivialMeasurements, generateNonTrivialMeasurements, rnd);
      annotators.add(annotator);
    }
    return annotators;
  }


  
  private static Dataset readData(RandomGenerator rnd, int featureNormalizationConstant) throws IOException {
    // transforms per dataset
    Function<String, String> docTransform = null;
    Function<String, String> tokenTransform = null;
    switch(datasetType){
    // simulated datasets need no transform
    case NB2:
    case NB20:
    // pre-processed datasets need no transform
    case R8:
    case R52:
    case CADE12:
    case WEBKB:
    case NG:
    case JSON_VEC:
    case INDEXED_VEC:
      break;
    // Web Pages
    case COMPANIES:
      tokenTransform = Functions2.compose(
          new ShortWordFilter(2),
          new PorterStemmer(),
          StopWordRemover.malletStopWordRemover()
          );
      break;
    // tweets
    case TWITTER:
      // preserved tweeted emoticons as text
      docTransform = new EmoticonTransformer();
      // order of ops is from bottom up
      tokenTransform = Functions2.compose( 
          new ShortWordFilter(1),
          new PorterStemmer(),
          StopWordRemover.twitterStopWordRemover()
          );
      break;
    case WEATHER:
      // preserved tweeted emoticons as text
      docTransform = new EmoticonTransformer();
      // order of ops is from bottom up
      tokenTransform = Functions2.compose( 
          new ShortWordFilter(1),
          new PorterStemmer(),
          StopWordRemover.twitterStopWordRemover(),
          StopWordRemover.fromWords(Sets.newHashSet("weather"))
          );
      break;
    // email 
    case ENRON:
    case NEWSGROUPS:
    case CFGROUPS1000:
    case REUTERS:
      docTransform = new EmailHeaderStripper();
      // order of ops is from bottom up
      tokenTransform = Functions2.compose(
          new ShortWordFilter(2),
          new PorterStemmer(),
          StopWordRemover.malletStopWordRemover()
          );
      break;
    default:
      throw new IllegalStateException("unknown dataset type: " + datasetType);
    }
    // -1 => null
    Integer featureNormalizer = featureNormalizationConstant<0? null: featureNormalizationConstant;
    
    // data reader pipeline per dataset
    // build a dataset, doing all the tokenizing, stopword removal, and feature normalization
    Dataset data;
    switch(datasetType){
    // json annotation stream
    case CFGROUPS1000:
    case WEATHER:
    case TWITTER:
    case COMPANIES:
      data = new JSONDocumentDatasetBuilder(basedir, dataset, 
          docTransform, DocPipes.opennlpSentenceSplitter(), DocPipes.McCallumAndNigamTokenizer(), tokenTransform,
          FeatureSelectorFactories.conjoin(
              new CountCutoffFeatureSelectorFactory<String>(featureCountCutoff), 
              (topNFeaturesPerDocument<0)? null: new TopNPerDocumentFeatureSelectorFactory(topNFeaturesPerDocument)),
          featureNormalizer)
          .dataset();
      break;
    case ENRON:
    case NB2:
    case NB20:
    case R8:
    case R52:
    case NG:
    case WEBKB:
    case CADE12:
    case NEWSGROUPS:
    case REUTERS:
      data = new DocumentDatasetBuilder(basedir, dataset, split, 
          docTransform, DocPipes.opennlpSentenceSplitter(), DocPipes.McCallumAndNigamTokenizer(), tokenTransform,
          FeatureSelectorFactories.conjoin(
              new CountCutoffFeatureSelectorFactory<String>(featureCountCutoff), 
              (topNFeaturesPerDocument<0)? null: new TopNPerDocumentFeatureSelectorFactory(topNFeaturesPerDocument)),
          featureNormalizer)
          .dataset();
      break;
    case INDEXED_VEC:
      data = new VectorDocumentDatasetBuilder(basedir, dataset, split).dataset();
      break;
    case JSON_VEC:
      data = new JSONVectorDocumentDatasetBuilder(basedir, dataset).dataset();
      break;
    default:
      throw new IllegalStateException("unknown dataset type: " + datasetType);
    }
    
//    // randomize order 
//    data.shuffle(rnd);
    
    // Postprocessing: remove all documents with duplicate sources or empty features
    data = Datasets.filteredDataset(data, Predicates.and(Datasets.filterDuplicateSources(), Datasets.filterNonEmpty()));

    logger.info("\nDataset on import: \n"+Datasets.summaryOf(data,1));

//    for (DatasetInstance inst: data){
//      Preconditions.checkState(inst.asFeatureVector().sum()>0,"document "+inst.getInfo().getSource()+" was empty");
//      
//      // print document data to make sure import didn't mess things up
//      System.out.println(inst.getInfo().getSource()+": "+Datasets.wordsIn(inst, data.getInfo().getFeatureIndexer()));
//    }
    
    return data;
  }

  // note (pfelt): borrowed from DefaultAnnotationServer.AnnotationHandler
  private static void writeAnnotation(PrintWriter out, AnnotationInfo<Integer> ai, AnnotationRequest<SparseFeatureVector, Integer> ar) {
    if (out != null) {
      // CSV: source, annotator_id, annotation, duration
      out.printf("%s, %d, %s, %d, %d\n",
          ar.getInstance().getSource(),
          ar.getAnnotatorId(),
          ai.getAnnotation(),
          ai.getAnnotationEvent().getDurationNanos(),
          ai.getWaitEvent().getDurationNanos());
      out.flush();
    }
  }
  
  public static class ExperimentSettingsComputer {
    public String csvHeader() {
      return Joiner.on(',').join(new String[]{
          "k",
          "labeling_strategy",
          "annotation_strategy",
          "training",
          "data_seed",
          "algorithm_seed",
          "basedir",
          "dataset",
          "dataset_type",
          "annotator_accuracy",
          "unannotated_document_weight",
          "pre_normalize_documents",
          "data_secs",
          "inference_secs",
          "initialization_strategy",
          "initialization_training",
          "diagonalization_method",
          "diagonalization_gold_instances",
          "btheta",
          "bgamma",
          "cgamma",
          "bmu",
          "cmu",
          "bphi",
          "eta_variance",
          "truncate_unannotated_data",
          "hyperparam_training",
          "num_topics",
          "annotator_cluster_method",
          "inline_hyperparam_tuning",
          "annotate_top_k_choices",
          "vary_annotator_rates",
          });
    }
    public String compute(int dataSecs, int inferenceSecs, PriorSpecification priors) {
      return Joiner.on(',').join(new String[]{
          ""+k,
          ""+labelingStrategy,
          ""+annotationStrategy,
          ""+training,
          ""+dataSeed,
          ""+algorithmSeed,
          basedir,
          dataset,
          ""+datasetType,
          ""+annotatorAccuracy,
          ""+lambda,
          ""+featureNormalizationConstant,
          ""+dataSecs,
          ""+inferenceSecs,
          ""+initializationStrategy,
          ""+initializationTraining,
          diagonalizationMethod.toString(),
          ""+goldInstancesForDiagonalization,
          priors==null? "": ""+priors.getBTheta(),
          priors==null? "": ""+priors.getBGamma(),
          priors==null? "": ""+priors.getCGamma(),
          priors==null? "": ""+priors.getBMu(),
          priors==null? "": ""+priors.getCMu(),
          priors==null? "":  ""+priors.getBPhi(),
          priors==null? "":  ""+priors.getEtaVariance(),
          ""+truncateUnannotatedData,
          hyperparamTraining,
          ""+numTopics,
          ""+clusterMethod,
          ""+inlineHyperparamTuning,
          ""+annotateTopKChoices,
          ""+varyAnnotatorRates,
        });
    }
  }
  

  public static class RmseAnnotatorConfusionMatrixComputer {
    private final double[][][] actualConfusionMatrices;

    public RmseAnnotatorConfusionMatrixComputer(double[][][] actualConfusionMatrices) {
      this.actualConfusionMatrices = actualConfusionMatrices;
    }

    public double compute(Predictions predictions) {
      if (actualConfusionMatrices==null){
        return -1;
      }
      // flatten matrices into a single array and calculate rmse over elements.
      double[][] actualFlattenedMatrices = new double[actualConfusionMatrices.length][];
      double[][] predictedFlattenedMatrices = new double[actualConfusionMatrices.length][];
      for (int a=0; a<actualConfusionMatrices.length; a++){
        actualFlattenedMatrices[a] = Matrices.flatten(actualConfusionMatrices[a]);
        predictedFlattenedMatrices[a] = Matrices.flatten(predictions.annotatorConfusionMatrices()[a]);
      }
      double[] actualArray = Matrices.flatten(actualFlattenedMatrices);
      double[] predictedArray = Matrices.flatten(predictedFlattenedMatrices);
      return DoubleArrays.rmse(actualArray, predictedArray);
    }

    public String csvHeader() {
      return "annacc_mat_rmse";
    }
  }

  public static class RmseMachineConfusionMatrixVsTestComputer {
    public ConfusionMatrixComputer computer;
    public RmseMachineConfusionMatrixVsTestComputer(Indexer<String> labels){
      this.computer = new ConfusionMatrixComputer(labels);
    }
    public double compute(Predictions predictions) {
      double[] actualArray = Matrices.flatten(predictions.machineConfusionMatrix());
      if (actualArray==null){
        return -1;
      }
      double[][] testConfusionMatrix = computer.compute(predictions.testPredictions()).getData();
      double[] testArray = Matrices.flatten(testConfusionMatrix);
      return DoubleArrays.rmse(actualArray, testArray);
    }
    public String csvHeader() {
      return "machacc_mat_rmse";
    }
  }

  private static Map<Integer,Double> identityAnnotatorRatesMap(double[] annotatorRates) {
    Map<Integer,Double> rates = Maps.newHashMap();
    for (int j=0; j<annotatorRates.length; j++){
      rates.put(j, annotatorRates[j]);
    }
    return rates;
  }
  
  /**
   * How to do data splits in a reasonable way less obvious than it seems like it should be for multiannotator data. 
   * Each item may either have a gold label or not, and each item may also either have one or more annotations or not.
   * We want a training set, a validation set, and a test set. The test set only needs items with gold labels. 
   * The train and validation sets should split the remaining “supervised” data (has label and/or annotation). 
   * The train and validation sets should both share all “unsupervised” data (no labels or annotations). 
   * Furthermore, we’d like the data to be randomized such that adding unsupervised data does not change 
   * which supervised items get allocated to train/validation/test. Otherwise, you get weird situations 
   * where adding unannotated and unlabeled data causes majority vote to apparently perform differently. 
   * All of these properties are respected by the following split scheme.
   * 
   *                labeled     -->  randomize -->  split("train1", "validation1", "test1")
   *               / 
   *      annotated 
   *     /         \
   *    /           not labeled --> "extra"
   * all                
   *    \             
   *     \             /labeled -->  randomize -->  split("train1", "validation1", "test1")
   *      not annotated
   *                   \
   *                    not labeled --> "extra"
   *                    
   * train = train1 + train2 + extra
   * validation = validation1 + validation2 + extra
   * test = test1 + test2
   */
  private static List<Dataset> splitData(Dataset fullData, double trainingPercent, double validationPercent, RandomGenerator rnd){

    // check ranges
    Preconditions.checkArgument(0<=trainingPercent && trainingPercent<=100,"trainingPercent must be between 0 and 100 (inclusive) "+trainingPercent);
    Preconditions.checkArgument(0<=validationPercent && validationPercent<=100,"validationPercent must be between 0 and 100 (inclusive) "+validationPercent);
    Preconditions.checkArgument(validationPercent+trainingPercent<=100,"trainingPercent+validationPercent must be between 0 and 100 (inclusive) "+trainingPercent+"+"+validationPercent);
    
    // create split tree as shown in function javadoc
    Pair<? extends Dataset, ? extends Dataset> allSplit = Datasets.divideInstancesWithAnnotations(fullData);
    
    Dataset annDataset = allSplit.getFirst();
    Pair<? extends Dataset, ? extends Dataset> annSplit = Datasets.divideInstancesWithLabels(annDataset);
    Dataset annLabDataset = annSplit.getFirst();
    Dataset annNolabDataset = annSplit.getSecond();
    List<Dataset> annLabSplits = Datasets.split(Datasets.shuffled(annLabDataset, rnd),
        new double[]{trainingPercent,validationPercent,(100-(trainingPercent+validationPercent))});
    
    Dataset noannDataset = allSplit.getSecond();
    Pair<? extends Dataset, ? extends Dataset> noannSplit = Datasets.divideInstancesWithLabels(noannDataset);
    Dataset noannLabDataset = noannSplit.getFirst();
    Dataset noannNolabDataset = noannSplit.getSecond();
    List<Dataset> noannLabSplits = Datasets.split(Datasets.shuffled(noannLabDataset, rnd),
        new double[]{trainingPercent,validationPercent,(100-(trainingPercent+validationPercent))});

    final Dataset trainingData = Datasets.join(annLabSplits.get(0), noannLabSplits.get(0), annNolabDataset, noannNolabDataset);
    final Dataset validationData = Datasets.join(annLabSplits.get(1), noannLabSplits.get(1), annNolabDataset, noannNolabDataset);
    final Dataset testData = Datasets.join(annLabSplits.get(2), noannLabSplits.get(2)); // note: we could ensure we don't waste any observed labels here, but it's not critical
    
    logger.info("Data after original split");
    logger.info("\ntraining data split: \n"+Datasets.summaryOf(trainingData,1));
    logger.info("\nvalidation data split: \n"+Datasets.summaryOf(validationData,1));
    logger.info("\ntest data split: \n"+Datasets.summaryOf(testData,1));
    Preconditions.checkState(testData.getInfo().getNumDocumentsWithoutLabels()==0,"test data must all have labels");
    
    return Lists.newArrayList(trainingData, validationData, testData);
  }
  
}

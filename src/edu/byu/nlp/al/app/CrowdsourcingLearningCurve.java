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

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.logging.Logger;

import org.apache.commons.math3.analysis.MultivariateFunction;
import org.apache.commons.math3.optim.InitialGuess;
import org.apache.commons.math3.optim.MaxEval;
import org.apache.commons.math3.optim.PointValuePair;
import org.apache.commons.math3.optim.SimpleBounds;
import org.apache.commons.math3.optim.nonlinear.scalar.GoalType;
import org.apache.commons.math3.optim.nonlinear.scalar.MultivariateOptimizer;
import org.apache.commons.math3.optim.nonlinear.scalar.ObjectiveFunction;
import org.apache.commons.math3.optim.nonlinear.scalar.noderiv.BOBYQAOptimizer;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.vfs2.FileSystemException;

import com.google.common.base.Function;
import com.google.common.base.Functions;
import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.base.Predicates;
import com.google.common.base.Stopwatch;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Lists;
import com.google.common.io.ByteStreams;

import edu.byu.nlp.al.ABArbiterInstanceManager;
import edu.byu.nlp.al.AnnotationInfo;
import edu.byu.nlp.al.AnnotationRequest;
import edu.byu.nlp.al.DatasetAnnotationRecorder;
import edu.byu.nlp.al.EmpiricalAnnotationInstanceManager;
import edu.byu.nlp.al.GeneralizedRoundRobinInstanceManager;
import edu.byu.nlp.al.InstanceManager;
import edu.byu.nlp.al.simulation.FallibleAnnotationProvider;
import edu.byu.nlp.al.simulation.GoldLabelProvider;
import edu.byu.nlp.al.util.MetricComputers.AnnotationsCounter;
import edu.byu.nlp.al.util.MetricComputers.AnnotatorAccuracyComputer;
import edu.byu.nlp.al.util.MetricComputers.LogJointComputer;
import edu.byu.nlp.al.util.MetricComputers.MachineAccuracyComputer;
import edu.byu.nlp.al.util.MetricComputers.PredictionTabulator;
import edu.byu.nlp.al.util.MetricComputers.RmseAnnotatorAccuracyComputer;
import edu.byu.nlp.al.util.MetricComputers.RmseMachineAccuracyVsTestComputer;
import edu.byu.nlp.classify.NaiveBayesLearner;
import edu.byu.nlp.classify.UncertaintyPreservingNaiveBayesLearner;
import edu.byu.nlp.classify.data.DatasetBuilder;
import edu.byu.nlp.classify.data.DatasetLabeler;
import edu.byu.nlp.classify.data.LabelChooser;
import edu.byu.nlp.classify.data.SingleLabelLabeler;
import edu.byu.nlp.classify.eval.AccuracyComputer;
import edu.byu.nlp.classify.eval.ConfusionMatrixComputer;
import edu.byu.nlp.classify.eval.ConfusionMatrixDistribution;
import edu.byu.nlp.classify.eval.OverallAccuracy;
import edu.byu.nlp.classify.eval.Predictions;
import edu.byu.nlp.classify.eval.ProbabilisticLabelErrorFunction;
import edu.byu.nlp.crowdsourcing.AnnotatorAccuracySetting;
import edu.byu.nlp.crowdsourcing.ArbiterVote;
import edu.byu.nlp.crowdsourcing.EmpiricalAnnotationProvider;
import edu.byu.nlp.crowdsourcing.LabelProvider;
import edu.byu.nlp.crowdsourcing.MajorityVote;
import edu.byu.nlp.crowdsourcing.ModelInitialization;
import edu.byu.nlp.crowdsourcing.ModelInitialization.AssignmentInitializer;
import edu.byu.nlp.crowdsourcing.MultiAnnDatasetLabeler;
import edu.byu.nlp.crowdsourcing.MultiAnnModelBuilders;
import edu.byu.nlp.crowdsourcing.MultiAnnModelBuilders.MultiAnnModelBuilder;
import edu.byu.nlp.crowdsourcing.PriorSpecification;
import edu.byu.nlp.crowdsourcing.em.ConfusedSLDADiscreteModelLabeler;
import edu.byu.nlp.crowdsourcing.em.RaykarModelLabeler;
import edu.byu.nlp.crowdsourcing.gibbs.BlockCollapsedMultiAnnModel;
import edu.byu.nlp.crowdsourcing.gibbs.BlockCollapsedMultiAnnModelMath.DiagonalizationMethod;
import edu.byu.nlp.crowdsourcing.gibbs.BlockCollapsedMultiAnnModelNeutered;
import edu.byu.nlp.crowdsourcing.gibbs.CollapsedItemResponseModel;
import edu.byu.nlp.crowdsourcing.meanfield.MeanFieldItemRespModel;
import edu.byu.nlp.crowdsourcing.meanfield.MeanFieldLabeler;
import edu.byu.nlp.crowdsourcing.meanfield.MeanFieldMomRespModel;
import edu.byu.nlp.crowdsourcing.meanfield.MeanFieldMultiRespModel;
import edu.byu.nlp.crowdsourcing.meanfield.MeanFieldRaykarModel;
import edu.byu.nlp.data.docs.CountCutoffFeatureSelectorFactory;
import edu.byu.nlp.data.docs.DocumentDatasetBuilder;
import edu.byu.nlp.data.docs.FeatureSelectorFactories;
import edu.byu.nlp.data.docs.JSONDocumentDatasetBuilder;
import edu.byu.nlp.data.docs.TokenizerPipes;
import edu.byu.nlp.data.docs.TopNPerDocumentFeatureSelectorFactory;
import edu.byu.nlp.data.pipes.EmailHeaderStripper;
import edu.byu.nlp.data.pipes.PorterStemmer;
import edu.byu.nlp.data.pipes.ShortWordFilter;
import edu.byu.nlp.data.types.Dataset;
import edu.byu.nlp.data.types.SparseFeatureVector;
import edu.byu.nlp.data.util.EmpiricalAnnotations;
import edu.byu.nlp.dataset.Datasets;
import edu.byu.nlp.io.Files2;
import edu.byu.nlp.util.DoubleArrays;
import edu.byu.nlp.util.Indexer;
import edu.byu.nlp.util.IntArrays;
import edu.byu.nlp.util.Matrices;
import edu.byu.nlp.util.Pair;
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
  private static final Logger logger = Logger.getLogger(CrowdsourcingLearningCurve.class.getName());

  @Option(help = "base directory of the documents") 
  private static String basedir = "./20_newsgroups";

  private enum DatasetType{NEWSGROUPS, REUTERS, ENRON, NB2, NB20, DREDZE, CFGROUPS1000, R8, R52, NG, CADE12, WEBKB}
  
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

  @Option (help = "m and y values are saved to this file at the end of the experiment")
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
      + "inference of any kind. (This is untested if you are trying to run for more "
      + "than a single evalpoint, although it will probably work).")
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

  @Option(help = "A sequence of colon-delimited training operations with valid values "
      + "sample,samplem,sampley,maximize,maximizem,maximizey,none where "
      + "sample operations take hyphen-delimited arguments samples:annealingTemp. "
      + "For example, --training=samplem-1-1:maximize:maximizey will "
      + "take one sample of all the m variables at temp=1, then will do "
      + "joint maximization followed by marginal maximization of y.")
  private static String training = "samplem-1-1:maximize:maximizey";

  @Option(help = "base the prediction on the single final state of the model. "
      + "Otherwise, the model tracks all samples during the final round of "
      + "annealing and makes a prediction based on marginal distribution.")
  private static boolean predictSingleLastSample = false;


  /* -------------  Dataset Labeler Methods  ------------------- */

  private enum LabelingStrategy {multiresp, ubaseline, baseline, momresp, itemresp, raykar, rayktrunc, varrayk, varmultiresp, varmomresp, varitemresp, cslda};
  @Option
  private static LabelingStrategy labelingStrategy = LabelingStrategy.multiresp;
  
  /* -------------  Instance Selection Methods  ------------------- */
  
  @Option(optStrings={"-k","--num-anns-per-instance"})
  private static int k = 1; // Used by grr/ab

  private enum AnnotationStrategy {grr, ab, real}; 
  @Option
  private static AnnotationStrategy annotationStrategy = AnnotationStrategy.grr;
  
  /* -------------  Model Params  ------------------- */
  
  @Option
  private static int maxAnnotations = Integer.MAX_VALUE;
  
  @Option(help = "Accuracy levels of annotators. The first one assumed arbiter for ab1 and ab2")
  private static AnnotatorAccuracySetting accuracyLevel = AnnotatorAccuracySetting.HIGH;
  
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
//  private static double[] bGamma = new double[] { 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75 };
  private static double bGamma = 0.80;
  
  @Option
  private static double cGamma = 10;
  
  @Option
  private static double bPhi = 0.1;

  @Option
  private static double trainingPercent = 85;


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
  
  
  
  public static void main(String[] args) throws FileNotFoundException, InterruptedException, FileSystemException{
    // parse CLI arguments
    ArgumentValues opts = new ArgumentParser(CrowdsourcingLearningCurve.class).parseArgs(args);
    
    // read in marginal chains (if any)
    String[] initializeWithChains = opts.getPositionalArgs();
    int[][] yChains = new int[initializeWithChains.length][];
    int[][] mChains = new int[initializeWithChains.length][];
    for (int i = 0; i < initializeWithChains.length; i++) {
      String file = initializeWithChains[i];
      Iterator<String> lineItr = Files2.open(file).iterator();
      yChains[i] = IntArrays.parseIntArray(lineItr.next());
      if (lineItr.hasNext()){
        mChains[i] = IntArrays.parseIntArray(lineItr.next());
      }
      // if only one line was in the file, use it for m as well
      else{
        mChains[i] = yChains[i];
      }
    }
    
    // open IO streams
    PrintWriter debugOut = debugFile == null ? new PrintWriter(ByteStreams.nullOutputStream()) : new PrintWriter(debugFile);
    PrintWriter annotationsOut = annotationsFile == null ? new PrintWriter(ByteStreams.nullOutputStream()) : new PrintWriter(annotationsFile);
    PrintWriter tabularPredictionsOut = tabularFile == null ? new PrintWriter(ByteStreams.nullOutputStream()) : new PrintWriter(tabularFile);
    PrintWriter resultsOut = resultsFile == null ? new PrintWriter(System.out) : new PrintWriter(resultsFile);
    PrintWriter serializeOut = serializeToFile==null ? new PrintWriter(ByteStreams.nullOutputStream()) : new PrintWriter(serializeToFile);
    
    // pass on to the main program
    CrowdsourcingLearningCurve.run(args, debugOut, annotationsOut, tabularPredictionsOut, resultsOut, serializeOut, yChains, mChains);
  }
  
  
  
  
  
  public static void run(String[] args, PrintWriter debugOut, PrintWriter annotationsOut, PrintWriter tabularPredictionsOut, PrintWriter resultsOut, PrintWriter serializeOut, final int[][] yChains, final int[][] mChains) throws InterruptedException, FileNotFoundException, FileSystemException {
    Preconditions.checkArgument(yChains==mChains || yChains.length==mChains.length); // must both be null or the same length
    
    ArgumentValues opts = new ArgumentParser(CrowdsourcingLearningCurve.class).parseArgs(args);

    // this generator deals with data creation (so that all runs with the same annotation strategy
    // settings get the same datasets, regardless of the algorithm run on them)
    RandomGenerator dataRnd = new MersenneTwister(dataSeed);
    RandomGenerator algRnd = new MersenneTwister(algorithmSeed);
    
    // record options
    debugOut.print(opts.optionsMap());
    
    /////////////////////////////////////////////////////////////////////
    // Read and prepare the data
    /////////////////////////////////////////////////////////////////////
    final Stopwatch stopwatchData = Stopwatch.createStarted();
    // currently cslda can't handle fractional word counts
    featureNormalizationConstant = labelingStrategy==LabelingStrategy.cslda? -1: featureNormalizationConstant;
    Dataset fullData = readData(dataRnd,featureNormalizationConstant);

    // Save annotations for future use (if we're using an empirical annotation strategy)
    final EmpiricalAnnotations<SparseFeatureVector, Integer> annotations = EmpiricalAnnotations.fromDataset(fullData);

    // ensure the dataset knows about all the annotators it will need to deal with.
    // if we are dealing with real data, we read in annotators with the data. Otherwise, 
    // we'll have to change it. 
    if (annotationStrategy!=AnnotationStrategy.real){
      fullData = Datasets.withNewAnnotators(fullData, accuracyLevel.getAnnotatorIdIndexer());
    }
    
//    // FIXME: ensures all annotated instances appear before all unannotated. Not necessary, but helps ensure 
//    // results match previous results
//    Pair<? extends Dataset, ? extends Dataset> fullDataDivided = Datasets.divideInstancesWithAnnotations(fullData);
//    fullData = Datasets.join(fullDataDivided.getFirst(), fullDataDivided.getSecond());
    
    // split into training, validation, test
    // strategy: split the labeled portion of the dataset into train/validate/test
    // and then append unlabeled instances back onto validation and train. 
    Pair<? extends Dataset, ? extends Dataset> fullDataDivided = Datasets.divideInstancesWithLabels(fullData);
    Dataset labeledDataset = fullDataDivided.getFirst();
    Dataset unlabeledDataset = fullDataDivided.getSecond();
    Preconditions.checkArgument(0<=trainingPercent && trainingPercent<=100,"trainingPercent must be between 0 and 100 (inclusive)");
    Preconditions.checkArgument(0<=validationPercent && validationPercent<=100,"validationPercent must be between 0 and 100 (inclusive)");
    Preconditions.checkArgument(validationPercent+trainingPercent<=100,"trainingPercent+validationPercent must be between 0 and 100 (inclusive)");
    List<Dataset> annotatedDatasetSplits = Datasets.split(labeledDataset,new double[]{trainingPercent,validationPercent,(100-(trainingPercent+validationPercent))});
    final Dataset trainingData = Datasets.join(annotatedDatasetSplits.get(0), unlabeledDataset);
    final Dataset validationData = Datasets.join(annotatedDatasetSplits.get(1),unlabeledDataset);
    final Dataset testData = annotatedDatasetSplits.get(2); // note: we could ensure we don't waste any observed labels here, but it's not critical
    
    logger.info("training data labeled "+trainingData.getInfo().getNumDocumentsWithLabels());
    logger.info("training data unlabeled "+trainingData.getInfo().getNumDocumentsWithoutLabels());
    logger.info("test data labeled "+testData.getInfo().getNumDocumentsWithLabels());
    Preconditions.checkState(testData.getInfo().getNumDocumentsWithoutLabels()==0,"test data must all have labels");
    logger.info("validation data labeled "+validationData.getInfo().getNumDocumentsWithLabels());
    logger.info("validation data unlabeled "+unlabeledDataset.getInfo().getNumDocumentsWithoutLabels());

    stopwatchData.stop();
    
    // cross-validation sweep unannotated-document-weight (optional)
    if (validationPercent>0){
      // see advice here http://commons.apache.org/proper/commons-math/apidocs/org/apache/commons/math3/optim/nonlinear/scalar/noderiv/BOBYQAOptimizer.html
      PointValuePair hyperparams = getHyperparametersViaCrossValidation(yChains, mChains, validationData, trainingData, annotations, stopwatchData, evalPoint);
      
      // set hyperparameters
      CrowdsourcingLearningCurve.bTheta = hyperparams.getPointRef()[0];
      CrowdsourcingLearningCurve.bPhi = hyperparams.getPointRef()[1];
      CrowdsourcingLearningCurve.bGamma = hyperparams.getPointRef()[2];
      CrowdsourcingLearningCurve.cGamma = hyperparams.getPointRef()[3];
    }

//    if (validationPercent>0){
//    UnivariatePointValuePair bestLambda = null;
//      UnivariateOptimizer lambdaOptimizer = new BrentOptimizer(rel, abs);
//      bestLambda = lambdaOptimizer.optimize(
//        new MaxEval(100), 
//        new UnivariateObjectiveFunction(new UnivariateFunction() {
//          @Override
//          public double value(double lambda) {
//    
//            double total = 0;
//            int numtrials = 1;
//            for (int trials=0; trials<numtrials; trials++){
//              // ensure results are consistent for the purposes of optimization
//              RandomGenerator algRnd = new MersenneTwister(algorithmSeed);
//              RandomGenerator dataRnd = new MersenneTwister(dataSeed);
//              
//              PrintWriter nullPipe = new PrintWriter(ByteStreams.nullOutputStream());
//              double result = trainEval(
//                  nullPipe, nullPipe, nullPipe, nullPipe, nullPipe, 
//                  yChains, mChains, dataRnd, algRnd, 
//                  stopwatchData, validationData, true, testData, annotations, ""+lambda, (int)validationPercent*3);
//              total += result;
////                total += result.getLabeledAccuracy().getAccuracy();
////                total += result.getTestAccuracy().getAccuracy(); 
//            }
//            logger.info("tested lambda="+lambda+" ==> avg_acc="+(total/numtrials));
//            
//            return total/numtrials;
//          }
//        }), 
//        GoalType.MAXIMIZE, 
//        new SearchInterval(0, 1)
//        );
//      
//      logger.info("best lambda (unannotated doc weight) "+bestLambda.getPoint()+" had avg_joint of "+bestLambda.getValue());
//      lambda = ""+bestLambda.getPoint();
//    }
    
    // this generator deals with algorithm estimation (to separate these from the random number 
    // stream generating data)
//    algRnd = new MersenneTwister(algorithmSeed);
//    dataRnd = new MersenneTwister(dataSeed);
    
    // final go
    trainEval(debugOut, annotationsOut, tabularPredictionsOut, resultsOut,
        serializeOut, yChains, mChains, dataRnd, algRnd, stopwatchData, 
        trainingData, false, validationData, testData, annotations, // train on training data (also use unannotated, unlabeled validation data) 
        bTheta, bMu, bPhi, bGamma, cGamma, 
        lambda, evalPoint);
    
    debugOut.close();
    annotationsOut.close();
    tabularPredictionsOut.close();
    resultsOut.close();
    serializeOut.close();
  }

  
  private static PointValuePair getHyperparametersViaCrossValidation(final int[][] yChains, final int[][] mChains, 
      final Dataset validationData,
      final Dataset trainingData,
      final EmpiricalAnnotations<SparseFeatureVector, Integer> annotations,
      final Stopwatch stopwatchData,
      final int evalPoint){
    int additionalInterpolationPoints = 1;
    int maxEvaluations = 100;
    double zero = 0.01;
    double one = 1.-zero;
    double[] startPoint = {0.1, 0.1, 0.1, 5}; // b_theta, b_phi, b_gamma, c_gamma
    double[][] boundaries = {{zero,zero,zero,zero},{2,2,one,20}};
    int dims = startPoint.length; 
    int numberOfInterpolationPoints = 2 * dims + 1 + additionalInterpolationPoints; 
    MultivariateOptimizer optim = new BOBYQAOptimizer(numberOfInterpolationPoints); 
    final int validationEvalPoint  = (int)Math.round(validationData.getInfo().getNumDocuments()/((double)trainingData.getInfo().getNumDocuments()) * evalPoint);
    return
        optim.optimize(new MaxEval(maxEvaluations),
                       new ObjectiveFunction(new MultivariateFunction(){
                        @Override
                        public double value(double[] point) {
                          double btheta = point[0];
                          double bPhi = point[1];
                          double bGamma = point[2];
                          double cGamma = point[3];
                          
                          // ensure results are consistent by using same seed
                          RandomGenerator algRnd = new MersenneTwister(algorithmSeed);
                          RandomGenerator dataRnd = new MersenneTwister(dataSeed);
                          PrintWriter nullPipe = new PrintWriter(ByteStreams.nullOutputStream());
                          boolean onlyAnnotateLabeledData = true;
                          double val = trainEval(nullPipe, nullPipe, nullPipe, nullPipe, nullPipe, 
                              yChains, mChains, dataRnd, algRnd, stopwatchData, 
                              validationData, onlyAnnotateLabeledData, trainingData, // train on validation (also include unannotated/unlabeled training data) 
                              Datasets.emptyDataset(trainingData.getInfo()), // no test data 
                              annotations, 
                              btheta, bMu, bPhi, bGamma, cGamma, lambda, validationEvalPoint);
                          logger.info("hyperparam search iteration {bTheta="+bTheta+" bPhi="+bPhi+" bGamma="+bGamma+" cGamma="+cGamma+"}="+val);
                          return val;
                        }
                       }),
                       GoalType.MAXIMIZE,
                       new InitialGuess(startPoint),
                       new SimpleBounds(boundaries[0],
                                        boundaries[1]));
  }
  

  private static double trainEval(PrintWriter debugOut,
      PrintWriter annotationsOut, PrintWriter tabularPredictionsOut,
      PrintWriter resultsOut, PrintWriter serializeOut, int[][] yChains,
      int[][] mChains, RandomGenerator dataRnd, RandomGenerator algRnd,
      Stopwatch stopwatchData, Dataset trainingData, boolean onlyAnnotateLabeledData, Dataset extraUnlabeledData,  
      Dataset testData, EmpiricalAnnotations<SparseFeatureVector, Integer> annotations,
      double bTheta, double bMu, double bPhi, double bGamma, double cGamma, 
      String lambda, int evalPoint) {
    
    /////////////////////////////////////////////////////////////////////
    // Prepare data. 
    /////////////////////////////////////////////////////////////////////
    // remove any existing annotations; this is only relevant if doing multiple evaluations in a single run
    Datasets.clearAnnotations(trainingData);
    // most or all ground-truth labels are hidden for crowdsourcing inference 
    if (annotationStrategy!=AnnotationStrategy.real){
      trainingData = Datasets.hideAllLabelsButNPerClass(trainingData, numObservedLabelsPerClass, dataRnd);
    }
    logger.info("Trusted labels available for " + trainingData.getInfo().getNumDocumentsWithObservedLabels() + " instances");
    logger.info("No labels available for " + trainingData.getInfo().getNumDocumentsWithoutObservedLabels() + " instances");
    // truncate unannotated data (if requested) by moving it from training into test
    // for historical reasons this parameter is used by some of the labelers below, who 
    // implement it internally, but those should all be rendered obsolete by this global removal.
    // (the one downside is that we don't get "unannotated accuracy" in these cases, but it was 
    // redundant with test case ("test") accuracy anyway.
    if (truncateUnannotatedData){
      trainingData = Datasets.removeDataWithoutAnnotationsOrObservedLabels(trainingData);
      extraUnlabeledData = Datasets.emptyDataset(extraUnlabeledData.getInfo());
    }

    logger.info("\n======================================================================================");
    logger.info("\n============= Train + eval (lambda="+lambda+", evalpoint="+evalPoint+") ==============");
    logger.info("\n======================================================================================");
    logger.info("data seed "+dataSeed);
    logger.info("algorithm seed "+algorithmSeed);
    logger.info("training data labeled "+trainingData.getInfo().getNumDocumentsWithLabels());
    logger.info("training data unlabeled "+trainingData.getInfo().getNumDocumentsWithoutLabels());
    logger.info("extra unlabeled data "+extraUnlabeledData.getInfo().getNumDocuments());
    logger.info("test data labeled "+testData.getInfo().getNumDocumentsWithLabels());
    logger.info("test data unlabeled "+testData.getInfo().getNumDocumentsWithoutLabels());
    logger.info("hyperparameters: bTheta="+bTheta+" bPhi="+bPhi+" bGamma="+bGamma+" cGamma="+cGamma);
    
    
    // data with known and observed labels is suitable for adding as extra supervision to models (TODO)
    Dataset observedLabelsTrainingData = Datasets.divideInstancesWithObservedLabels(trainingData).getFirst();
    // data with known but concealed labels is suitable for simulating annotators and doing evaluation 
    Dataset concealedLabelsTrainingData = Datasets.divideInstancesWithLabels(trainingData).getFirst();

    Stopwatch stopwatchInference = Stopwatch.createStarted();
    
    /////////////////////////////////////////////////////////////////////
    // Annotators
    /////////////////////////////////////////////////////////////////////
    List<? extends LabelProvider<SparseFeatureVector, Integer>> annotators;
    if (annotationStrategy==AnnotationStrategy.real){
      annotators = createEmpiricalAnnotators(annotations);
      accuracyLevel = null; // avoid reporting misleading stats based on unused simulation parameters
    }
    else{
      annotators = createAnnotators(concealedLabelsTrainingData, accuracyLevel, concealedLabelsTrainingData.getInfo().getNumClasses(), dataRnd);
    }

    logger.info("Number of Annotators = " + annotators.size());
    
    
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
    LabelChooser chooser;
    switch(annotationStrategy){
    case ab:
      chooser = new ArbiterVote(arbiters, algRnd);
      instanceManager = ABArbiterInstanceManager.newManager(trainingData, k==1, arbiters);
      break;
    case grr:
      chooser = new MajorityVote(algRnd);
      instanceManager = GeneralizedRoundRobinInstanceManager.newManager(k, trainingData, new DatasetAnnotationRecorder(trainingData), dataRnd);
      break;
    case real:
      chooser = new MajorityVote(algRnd);
      Dataset instances = onlyAnnotateLabeledData? concealedLabelsTrainingData: trainingData;
      instanceManager = EmpiricalAnnotationInstanceManager.newManager(k, instances, annotations, dataRnd);
      break;
    default:
        throw new IllegalArgumentException("Unknown annotation strategy: " + annotationStrategy.name());
    }


    /////////////////////////////////////////////////////////////////////
    // Annotate until the eval point
    /////////////////////////////////////////////////////////////////////
    annotationsOut.println("source, annotator_id, annotation, annotation_time_nanos, wait_time_nanos"); // annotation file header
    for (int numAnnotations = 0; numAnnotations<maxAnnotations && numAnnotations<=evalPoint; numAnnotations++) {

      // Get an instance and annotator assignment
      AnnotationRequest<SparseFeatureVector, Integer> request = null;
      // keep trying until we get one right. The instance  
      // manager should only fail when 
      // 1) doing ab-arbitrator when we we ask 
      // for an arbitrator instance and there are no conflicts.
      // 2) the realistic annotator enforces historical order.
      while (request==null && !instanceManager.isDone()){
        // Pick an annotator at random
        long annotatorId = dataRnd.nextInt(annotators.size());
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
        LabelProvider<SparseFeatureVector, Integer> annotator = annotators.get((int)request.getAnnotatorId());
        Integer label = annotator.labelFor(request.getInstance().getSource(), request.getInstance().getData());
        AnnotationInfo<Integer> ai = new AnnotationInfo<Integer>(new Long(numAnnotations), label, TimedEvent.Zeros(), TimedEvent.Zeros());
        request.storeAnnotation(ai);
        writeAnnotation(annotationsOut, ai, request);
      }
    } // end annotate

    // add extra unlabeled data to the training set
    trainingData = Datasets.join(trainingData,extraUnlabeledData);

    
    /////////////////////////////////////////////////////////////////////
    // Build a dataset labeler
    // valid options: 1) multiannotator model(s)
    //                2) use chooser strategy (ab,majority) on labeled portion, and naive bayes on unlabeled/test 
    /////////////////////////////////////////////////////////////////////
    DatasetLabeler labeler;
    
    // this builder uses the chooser to arbitrate the labeled portion of the dataset
    // for the baseline approaches
    DatasetBuilder datasetBuilder = new DatasetBuilder(chooser);

    PriorSpecification priors = new PriorSpecification(bTheta, bMu, cMu, DoubleArrays.of(bGamma, annotators.size()), cGamma, bPhi);
    switch(labelingStrategy){

    case ubaseline:
      // this labeler reports labeled data without alteration, and trains an uncertainty-
      // preserving naive bayes variant on it to label the unlabeled and test portions
      labeler = new SingleLabelLabeler(new UncertaintyPreservingNaiveBayesLearner(), datasetBuilder, annotators.size(), serializeOut);
      break;
      
    case baseline:
      // this labeler reports labeled data without alteration, and trains a naive bayes 
      // on it to label the unlabeled and test portions
      labeler = new SingleLabelLabeler(new NaiveBayesLearner(), datasetBuilder, annotators.size(), serializeOut);
      break;

    case raykar:
      labeler = new RaykarModelLabeler(trainingData,  priors, true);
      break;

    case rayktrunc:
      truncateUnannotatedData = true;
      labeler = new RaykarModelLabeler(trainingData,  priors, false);
      break;

    case cslda:
      Preconditions.checkState(featureNormalizationConstant == -1); // cslda code currently can't handle fractional word counts
      AssignmentInitializer zInitializer = new ModelInitialization.UniformAssignmentInitializer(numTopics, algRnd);
      AssignmentInitializer yInitializer = new ModelInitialization.BaselineInitializer(algRnd);
      labeler = new ConfusedSLDADiscreteModelLabeler(trainingData, numTopics, training, 
          zInitializer, yInitializer, priors, algRnd);
      break;
      
    case varrayk:
      truncateUnannotatedData = true;
      labeler = new MeanFieldLabeler(initMultiannModelBuilder(
          new MeanFieldRaykarModel.ModelBuilder(), algRnd, trainingData, priors, yChains, mChains), training);
      break;
      
    case varmultiresp:
      labeler = new MeanFieldLabeler(initMultiannModelBuilder(
          new MeanFieldMultiRespModel.ModelBuilder(), algRnd, trainingData, priors, yChains, mChains), training);
      break;
      
    case varmomresp:
      labeler = new MeanFieldLabeler(initMultiannModelBuilder(
          new MeanFieldMomRespModel.ModelBuilder(), algRnd, trainingData, priors, yChains, mChains), training);
      break;
      
    case varitemresp:
      truncateUnannotatedData = true;
      labeler = new MeanFieldLabeler(initMultiannModelBuilder(
          new MeanFieldItemRespModel.ModelBuilder(), algRnd, trainingData, priors, yChains, mChains), training);
      break;
      
    // some variant of multiannotator model sampling
    default:
      MultiAnnModelBuilder multiannModelBuilder;
      
      switch(labelingStrategy){
      case multiresp:
        multiannModelBuilder = new BlockCollapsedMultiAnnModel.ModelBuilder();
        break;
        
      case momresp:
        multiannModelBuilder = new BlockCollapsedMultiAnnModelNeutered.ModelBuilder();
        break;
        
      case itemresp:
        truncateUnannotatedData = true;
        multiannModelBuilder = new CollapsedItemResponseModel.ModelBuilder();
        break;
        
      default:
        throw new IllegalArgumentException("Unknown labeling strategy: " + labelingStrategy.name());
      }
      initMultiannModelBuilder(multiannModelBuilder, algRnd, trainingData, priors, yChains, mChains);
      
      labeler = new MultiAnnDatasetLabeler(multiannModelBuilder, debugOut, 
          serializeOut, predictSingleLastSample, training, 
          diagonalizationMethod, diagonalizationWithFullConfusionMatrix, goldInstancesForDiagonalization, trainingData, 
          lambda, truncateUnannotatedData, algRnd);
      
    }
    
    
    /////////////////////////////////////////////////////////////////////
    // Eval the labeler
    /////////////////////////////////////////////////////////////////////
    AnnotationsCounter annotationsCounter = new AnnotationsCounter();
    LogJointComputer jointComputer = new LogJointComputer();
    AccuracyComputer accuracyComputer = new AccuracyComputer();
    AccuracyComputer top3AccuracyComputer = new AccuracyComputer(3);
    AnnotatorAccuracyComputer annAccComputer = new AnnotatorAccuracyComputer(annotators.size());
    RmseAnnotatorAccuracyComputer rmseComputer = new RmseAnnotatorAccuracyComputer(accuracyLevel==null?null:accuracyLevel.getAccuracies());
    RmseAnnotatorConfusionMatrixComputer rmseMatrixComputer = new RmseAnnotatorConfusionMatrixComputer(accuracyLevel==null?null:accuracyLevel.getConfusionMatrices());
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
            (int)stopwatchData.elapsed(TimeUnit.SECONDS),
            (int)stopwatchInference.elapsed(TimeUnit.SECONDS),
            yChains==null? 0: yChains.length,
            priors)
        ));
    resultsOut.flush();
    PredictionTabulator.writeTo(predictions, tabularPredictionsOut);

    logger.info("annacc = " + DoubleArrays.toString(predictions.annotatorAccuracies()));
    logger.info("machacc = " + predictions.machineAccuracy());
    logger.info("machacc_mat = " + Matrices.toString(predictions.machineConfusionMatrix()));
    logAccuracy("", accResults);
    logAccuracy("top3", acc3Results);

    return accResults.getLabeledAccuracy().getAccuracy();
  }

  private static void logAccuracy(String prefix, OverallAccuracy acc){
    logger.info(prefix+"test_acc = " + acc.getTestAccuracy().getAccuracy()+" ("+acc.getTestAccuracy().getCorrect()+"/"+acc.getTestAccuracy().getTotal()+")");
    logger.info(prefix+"labeled_acc = " + acc.getLabeledAccuracy().getAccuracy()+" ("+acc.getLabeledAccuracy().getCorrect()+"/"+acc.getLabeledAccuracy().getTotal()+")");
    logger.info(prefix+"unlabeled_acc = " + acc.getUnlabeledAccuracy().getAccuracy()+" ("+acc.getUnlabeledAccuracy().getCorrect()+"/"+acc.getUnlabeledAccuracy().getTotal()+")");
    logger.info(prefix+"overall_acc = " + acc.getOverallAccuracy().getAccuracy()+" ("+acc.getOverallAccuracy().getCorrect()+"/"+acc.getOverallAccuracy().getTotal()+")");
  }

  private static MultiAnnModelBuilder initMultiannModelBuilder(MultiAnnModelBuilder builder, 
                              RandomGenerator algRnd, Dataset trainingData, PriorSpecification priors, 
                              int[][] yChains, int[][] mChains) {
//  ModelBuilder builder = BlockCollapsedMultiAnnModel.newModelBuilderWithUniform(priors, trainingData, rnd);
    if (yChains!=null && yChains.length>0){
      return MultiAnnModelBuilders.initModelBuilderWithSerializedChains(builder, priors, trainingData, yChains, mChains, algRnd);
    }
    else{
      return MultiAnnModelBuilders.initModelBuilderWithBaselineInit(builder, priors, trainingData, truncateUnannotatedData, algRnd);
    }
  }


  private static List<? extends LabelProvider<SparseFeatureVector, Integer>> createEmpiricalAnnotators(
                                                         EmpiricalAnnotations<SparseFeatureVector, Integer> annotations) {
    List<EmpiricalAnnotationProvider<SparseFeatureVector, Integer>> annotators = Lists.newArrayList();
    for (Long annotator: annotations.getDataInfo().getAnnotatorIdIndexer()){
      annotators.add(new EmpiricalAnnotationProvider<SparseFeatureVector, Integer>(annotator, annotations));
    }
    return annotators;
  }

  public static List<? extends LabelProvider<SparseFeatureVector, Integer>> createAnnotators(Dataset concealedLabeledTrainingData,
                                                                                    AnnotatorAccuracySetting accuracySetting, int numLabels,
                                                                                    RandomGenerator rnd) {
    GoldLabelProvider<SparseFeatureVector,Integer> goldLabelProvider = GoldLabelProvider.from(concealedLabeledTrainingData);
    List<FallibleAnnotationProvider<SparseFeatureVector,Integer>> annotators = Lists.newArrayList();
    for (double[][] confusionMatrix : accuracySetting.generateConfusionMatrices(rnd, numLabels)) {
      ProbabilisticLabelErrorFunction<Integer> labelErrorFunction =
          new ProbabilisticLabelErrorFunction<Integer>(new ConfusionMatrixDistribution(confusionMatrix),rnd);
      FallibleAnnotationProvider<SparseFeatureVector,Integer> annotator = 
          FallibleAnnotationProvider.from(goldLabelProvider, labelErrorFunction);
      annotators.add(annotator);
    }
    return annotators;
  }


  
  private static Dataset readData(RandomGenerator rnd, int featureNormalizationConstant) throws FileSystemException, FileNotFoundException {
    // transforms per dataset
    Function<String, String> docTransform = null;
    Function<List<String>, List<String>> tokenTransform = null;
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
      break;
    // tweets
    case DREDZE:
      // order of ops is from bottom up
      tokenTransform = Functions.compose( 
          new ShortWordFilter(2),
          new PorterStemmer()
          );
      break;
    // email 
    case ENRON:
    case NEWSGROUPS:
    case CFGROUPS1000:
    case REUTERS:
      docTransform = new EmailHeaderStripper();
      // order of ops is from bottom up
      tokenTransform = Functions.compose(
          new ShortWordFilter(2),
          new PorterStemmer()
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
    case DREDZE:
      data = new JSONDocumentDatasetBuilder(basedir, dataset, 
          docTransform, TokenizerPipes.McCallumAndNigam(), tokenTransform, 
          FeatureSelectorFactories.conjoin(
              new CountCutoffFeatureSelectorFactory<String>(featureCountCutoff), 
              (topNFeaturesPerDocument<0)? null: new TopNPerDocumentFeatureSelectorFactory<String>(topNFeaturesPerDocument)),
          featureNormalizer, rnd)
          .dataset();
      
//      System.out.println(data.getInfo().getFeatureIndexer());
//      for (DatasetInstance inst: data){
//        System.out.println(inst.getInfo().getSource());
//        System.out.println(inst.getInfo().getNumAnnotations());
//        System.out.println(inst.getAnnotations());
//      }
      
      // don't randomize order (unlabeled data are at the rear)
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
          docTransform, TokenizerPipes.McCallumAndNigam(), tokenTransform, 
          FeatureSelectorFactories.conjoin(
              new CountCutoffFeatureSelectorFactory<String>(featureCountCutoff), 
              (topNFeaturesPerDocument<0)? null: new TopNPerDocumentFeatureSelectorFactory<String>(topNFeaturesPerDocument)),
          featureNormalizer)
          .dataset();
      
      // randomize order 
      data.shuffle(rnd);
      break;
    default:
      throw new IllegalStateException("unknown dataset type: " + datasetType);
    }
    // Postprocessing: remove all documents with duplicate sources or empty features
    data = Datasets.filteredDataset(data, Predicates.and(Datasets.filterDuplicateSources(), Datasets.filterNonEmpty()));
    
    logger.info("Number of labeled instances = " + data.getInfo().getNumDocumentsWithObservedLabels());
    logger.info("Number of unlabeled instances = " + data.getInfo().getNumDocumentsWithoutObservedLabels());
    logger.info("Number of tokens = " + data.getInfo().getNumTokens());
    logger.info("Number of features = " + data.getInfo().getNumFeatures());
    logger.info("Number of classes = " + data.getInfo().getNumClasses());
    logger.info("Average Document Size = " + (data.getInfo().getNumTokens()/data.getInfo().getNumDocuments()));

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
          "dataset",
          "corpus",
          "accuracy_level",
          "unannotated_document_weight",
          "pre_normalize_documents",
          "data_secs",
          "inference_secs",
          "initialization_chains",
          "diagonalization_method",
          "diagonalization_gold_instances",
          "btheta",
          "bgamma",
          "cgamma",
          "bmu",
          "cmu",
          "bphi",
          "truncate_unannotated_data",
          });
    }
    public String compute(int dataSecs, int inferenceSecs, int initializationChains, PriorSpecification priors) {
      return Joiner.on(',').join(new String[]{
          ""+k,
          ""+labelingStrategy,
          ""+annotationStrategy,
          ""+training,
          ""+dataSeed,
          ""+algorithmSeed,
          dataset,
          ""+datasetType,
          ""+accuracyLevel,
          ""+lambda,
          ""+featureNormalizationConstant,
          ""+dataSecs,
          ""+inferenceSecs,
          ""+initializationChains,
          diagonalizationMethod.toString(),
          ""+goldInstancesForDiagonalization,
          priors==null? "": ""+priors.getBTheta(),
          priors==null? "": ""+priors.getBGamma(0),
          priors==null? "": ""+priors.getCGamma(),
          priors==null? "": ""+priors.getBMu(),
          priors==null? "": ""+priors.getCMu(),
          priors==null? "":  ""+priors.getBPhi(),
          ""+truncateUnannotatedData,
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
  
  
}

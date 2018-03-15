package cn.rocket;

import java.io.File;
import java.io.IOException;
import java.util.Random;

import org.apache.commons.io.FilenameUtils;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class CNN_MNIST {
	private static Logger log = LoggerFactory.getLogger(CNN_MNIST.class);

	public static void main(String[] args) throws IOException {
		int nChannels = 1;
		int outputNum = 10; // The number of possible outcomes
		int batchSize = 64; // Test batch size
		int nEpochs = 10; // Number of training epochs
		int iterations = 1; // Number of training iterations
		int seed = 123; //
		final String DATA_PATH = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "dl4j_Mnist/");

		File trainData = new File(DATA_PATH + "/mnist_png/training");
		File testData = new File(DATA_PATH + "/mnist_png/testing");

		// FileSplit train = new FileSplit(trainData,
		// NativeImageLoader.ALLOWED_FORMATS, new Random(123));

		FileSplit train = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, new Random(123));
		FileSplit test = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, new Random(123));

		ImageRecordReader recordReader = new ImageRecordReader(28, 28, 1, new ParentPathLabelGenerator());
		recordReader.initialize(train);

		DataSetIterator mnistTrain = new RecordReaderDataSetIterator(recordReader, batchSize, 1, outputNum);
		DataNormalization scaler = new ImagePreProcessingScaler(0, 1);

		scaler.fit(mnistTrain);
		mnistTrain.setPreProcessor(scaler);

		MultiLayerConfiguration conf = new NeuralNetConfiguration
				.Builder()
				.seed(seed)
				.iterations(iterations)
				.regularization(true)
				.l2(0.0005)
				.learningRate(0.01)
				.weightInit(WeightInit.XAVIER)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.updater(Updater.NESTEROVS)
				.momentum(0.9)
				.list()
				.layer(0,new ConvolutionLayer.Builder(5, 5)
						.nIn(nChannels)
						.stride(1, 1)
						.nOut(20)
						.activation(Activation.IDENTITY).
						build())
				.layer(1,new SubsamplingLayer
						.Builder(SubsamplingLayer.PoolingType.MAX)
						.kernelSize(2, 2)
						.stride(2, 2)
						.build())
				.layer(2,new ConvolutionLayer.Builder(5, 5)
						.stride(1, 1)
						.nOut(50)
						.activation(Activation.IDENTITY)
						.build())
				.layer(3,new SubsamplingLayer
						.Builder(SubsamplingLayer.PoolingType.MAX)
						.kernelSize(2, 2)
						.stride(2, 2)
						.build())
				.layer(4, new DenseLayer
						.Builder()
						.activation(Activation.RELU)
						.nOut(500)
						.build())
				.layer(5,new OutputLayer
						.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).nOut(outputNum)
						.activation(Activation.SOFTMAX)
						.build())
				.setInputType(InputType.convolutionalFlat(28, 28, 1))
				.backprop(true).pretrain(false)
				.build();

		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();

		model.setListeners(new ScoreIterationListener(1));

		log.info("******EVALUATE MODEL******");

		for (int i = 0; i < nEpochs; i++) {
			model.fit(mnistTrain);
			log.info("*** Completed epoch {} ***", i);
		}

		recordReader.reset();
		recordReader.initialize(test);

		DataSetIterator testIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, outputNum);
		scaler.fit(testIter);
		testIter.setPreProcessor(scaler);

		log.info(recordReader.getLabels().toString());

		// Create Eval object with 10 possible classes
		Evaluation eval = new Evaluation(outputNum);

		// Evaluate the network
		while (testIter.hasNext()) {
			DataSet next = testIter.next();
			@SuppressWarnings("deprecation")
			INDArray output = model.output(next.getFeatureMatrix());
			// Compare the Feature Matrix from the model
			// with the labels from the RecordReader
			eval.eval(next.getLabels(), output);

		}

		log.info(eval.stats());

		log.info("******SAVE TRAINED MODEL******");
		// Details

		// Where to save model
		File locationToSave = new File("trained_mnist_model.zip");

		// boolean save Updater
		boolean saveUpdater = false;

		// ModelSerializer needs modelname, saveUpdater, Location

		ModelSerializer.writeModel(model, locationToSave, saveUpdater);

	}
}

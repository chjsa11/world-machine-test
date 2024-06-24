package com.glass.machine.test1;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class SimpleMnistClassifier {

    public static void main(String[] args) throws Exception {
        // MNIST 데이터셋을 불러옵니다.
        DataSetIterator mnistTrain = new MnistDataSetIterator(64, true, 12345);
        DataSetIterator mnistTest = new MnistDataSetIterator(64, false, 12345);

        // 신경망 구성 설정
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(28 * 28) // 입력 크기
                        .nOut(100) // 첫 번째 은닉층의 뉴런 수
                        .activation(Activation.RELU)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(100) // 첫 번째 은닉층의 출력 크기
                        .nOut(10) // 출력층의 뉴런 수 (10개의 클래스)
                        .activation(Activation.SOFTMAX)
                        .build())
                .build();

        // 다층 퍼셉트론(MLP) 생성
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(10));

        // 모델 훈련
        for (int i = 0; i < 5; i++) { // 5 에포크 동안 훈련
            model.fit(mnistTrain);
            System.out.println("Epoch " + i + " complete");
        }

        // 모델 평가
        Evaluation eval = model.evaluate(mnistTest);
        System.out.println(eval.stats());
    }
}


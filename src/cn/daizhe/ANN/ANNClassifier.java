package cn.daizhe.ANN;

import cn.daizhe.oper.MatrixOperate;
import cn.daizhe.oper.exception.MatrixNotMatchException;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

import static cn.daizhe.oper.MatrixOperate.*;

/**
 * 神经网络分类器
 * Created by venice on 2017/5/29.
 */
public class ANNClassifier {
    // 神经网络模型
    private ANNModel ann;
    // 训练数据
    public static final List<DataModel> trainData = new ArrayList<>();
    // 测试数据
    public static final List<DataModel> testData = new ArrayList<>();

    // 训练参数
    public static int m = 0;
    public static double alpha = 0.95;
    public static double lambda = 0.45;
    public static int limitEpoch = 1000;
    public static double limitLoss = 0.01;
    public static double limitDiffLoss = 0.00001;


    public ANNClassifier() {
    }

    public ANNClassifier(ANNModel ann) {
        this.ann = ann;
    }

    public ANNClassifier(ANNModel ann, double alpha, double lambda, int limitEpoch, double limitLoss, double limitDiffLoss) {
        this.ann = ann;
        this.alpha = alpha;
        this.lambda = lambda;
        this.limitEpoch = limitEpoch;
        this.limitLoss = limitLoss;
        this.limitDiffLoss = limitDiffLoss;
    }

    public ANNClassifier(ANNModel ann, List<DataModel> trainData) {
        this.ann = ann;
        for (int i = 0; i < trainData.size(); i++) {
            this.trainData.add(new DataModel(trainData.get(i).x, trainData.get(i).y));
        }
        this.m = trainData.size();
        this.ann.setTrainDataNum(trainData.size());
    }

    // 初始化训练数据，也可以使用构造方法创建数据
    public final void initData() throws IOException {
        //trainData.add(new DataModel(new int[]{2, 2}, new int[]{1}));
        //trainData.add(new DataModel(new int[]{3, 2}, new int[]{0}));
        //trainData.add(new DataModel(new int[]{3, 4}, new int[]{1}));
        //trainData.add(new DataModel(new int[]{4, 8}, new int[]{1}));
        //trainData.add(new DataModel(new int[]{4, 3}, new int[]{0}));
        //trainData.add(new DataModel(new int[]{5, 6}, new int[]{0}));
        //trainData.add(new DataModel(new int[]{6, 7}, new int[]{1}));
        //trainData.add(new DataModel(new int[]{8, 6}, new int[]{0}));
        //m = 8;
        //testData.add(new DataModel(new int[]{4, 2}, new int[]{0}));
        //testData.add(new DataModel(new int[]{4, 9}, new int[]{1}));
        //testData.add(new DataModel(new int[]{5, 5}, new int[]{0}));
        //testData.add(new DataModel(new int[]{6, 9}, new int[]{1}));
        try {
            BufferedReader br = new BufferedReader(new FileReader(new File("iris.data.txt")));
            String line = "";
            int trainNum = 0, testNum = 0, i = 0;
            while ((line = br.readLine()) != null) {
                String attr[] = line.split(",");
                double[] x = new double[4];
                int[] y = new int[3];
                try {
                    x[0] = Double.parseDouble(attr[0]);
                    x[1] = Double.parseDouble(attr[1]);
                    x[2] = Double.parseDouble(attr[2]);
                    x[3] = Double.parseDouble(attr[3]);

                    if ("Iris-setosa".equals(attr[4])) {
                        y[0] = 1;
                    } else if ("Iris-versicolor".equals(attr[4])) {
                        y[1] = 1;
                    } else if ("Iris-virginica".equals(attr[4])) {
                        y[2] = 1;
                    }
                    DataModel dm = new DataModel(x, y);
                    if (i % 10 == 0) {
                        testData.add(dm);
                        testNum++;
                    } else {
                        trainData.add(dm);
                        trainNum++;
                    }
                    i++;
                } catch (Exception e) {
                    continue;
                }
            }
            m = trainNum;
            br.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
            throw e;
        } catch (IOException e) {
            e.printStackTrace();
            throw e;
        }
        //printData();
    }

    public final void printData() {
        for (int i = 0; i < trainData.size(); i++) {
            DataModel dm = trainData.get(i);
            System.out.println("data " + i + " int trainDataList :");
            printMatrix(dm.x, "x = ");
            printMatrix(dm.y, "y = ");
        }
        for (int i = 0; i < testData.size(); i++) {
            DataModel dm = testData.get(i);
            System.out.println("data " + i + " int testDataList :");
            printMatrix(dm.x, "x = ");
            printMatrix(dm.y, "y = ");
        }
    }

    /**
     * 前向传播
     *
     * @throws MatrixNotMatchException
     */
    public final void forwardAll() throws MatrixNotMatchException {
        for (int i = 0; i < m; i++) {
            forward(i);
        }
        //printMatrix(ann.a3All, "a3All");
    }

    public final void forward(int i) throws MatrixNotMatchException {
        DataModel dm = trainData.get(i);
        ann.z2All[i] = addMat(multi(dm.x, ann.w1), ann.b1);
        for (int j = 0; j < ann.hNum; j++) {
            ann.a2All[i][j] = activeFunction(ann.z2All[i][j]);
        }
        ann.z3All[i] = addMat(multi(ann.a2All[i], ann.w2), ann.b2);
        for (int j = 0; j < ann.oNum; j++) {
            ann.a3All[i][j] = activeFunction(ann.z3All[i][j]);
        }
        //printMatrix(ann.a3All, "a3All");
    }

    /**
     * 对第i个训练数据，计算各层delta=(a3-y)*a3*(1-a3)=-(y-a3)*a3*(1-a3)
     *
     * @param i
     */
    public final void computeDelta(int i) {
        DataModel dm = trainData.get(i);
        try {
            ann.d3All[i] = dot(
                    minus(
                            ann.a3All[i],
                            dm.y
                    ),
                    derivativeFunction(ann.a3All[i])
            );
            ann.d2All[i] = dot(
                    multi(
                            ann.w2,
                            ann.d3All[i]
                    ),
                    derivativeFunction(ann.a2All[i]));

        } catch (MatrixNotMatchException e) {
            e.printStackTrace();
        }
    }

    /**
     * 计算损失
     *
     * @return
     */
    public final double computeLossWithRegular() {
        return computeLoss(2);
    }

    public final double computeLoss() {
        return computeLoss(1);
    }

    public final double computeLoss(int type) {
        double Jwb = 0.0;
        double sd = 0.0;
        for (int i = 0; i < m; i++) {
            DataModel dm = trainData.get(i);
            // 2范数
            double norm = 0.0;
            for (int j = 0; j < dm.y.length; j++) {
                norm += (dm.y[j] - ann.a3All[i][j]) * (dm.y[j] - ann.a3All[i][j]);
            }
            norm = Math.sqrt(norm);
            sd += norm * norm / 2;
        }
        sd /= m;
        double regu = 0.0;
        if (type == 2) {
            for (int i = 0; i < ann.iNum; i++) {
                for (int j = 0; j < ann.hNum; j++) {
                    regu += ann.w1[i][j] * ann.w1[i][j];
                }
            }
            for (int i = 0; i < ann.hNum; i++) {
                for (int j = 0; j < ann.oNum; j++) {
                    regu += ann.w2[i][j] * ann.w2[i][j];
                }
            }
            regu = regu * lambda / 2;
        }

        Jwb = sd + regu;
        return Jwb;
    }

    /**
     * 更新权重和偏置量
     *
     * @throws MatrixNotMatchException
     */
    public final void refreshWeight() throws MatrixNotMatchException {
        for (int i = 0; i < m; i++) {
            computeDelta(i);
            ann.dw2 = addMat(ann.dw2, multi(ann.a2All[i], ann.d3All[i], true));
            ann.db2 = addMat(ann.db2, ann.d3All[i]);

            ann.dw1 = addMat(ann.dw1, multi(trainData.get(i).x, ann.d2All[i], true));
            ann.db1 = addMat(ann.db1, ann.d2All[i]);
        }

        MatrixOperate.copyMatric(ann.w1, ann.pre_w1);
        MatrixOperate.copyMatric(ann.w2, ann.pre_w2);
        MatrixOperate.copyMatric(ann.b1, ann.pre_b1);
        MatrixOperate.copyMatric(ann.b2, ann.pre_b2);

        //printMatrix(ann.w2, "w2");
        //printMatrix(ann.dw2, "dw2");
        //ann.w2 = addMat(minus(multi(ann.w2, (1 - alpha * lambda)), multi(ann.dw2, alpha / m)), ann.pre_dw2);
        //ann.w1 = addMat(minus(multi(ann.w1, (1 - alpha * lambda)), multi(ann.dw1, alpha / m)), ann.pre_dw1);
        ann.w2 = minus(multi(ann.w2, (1 - alpha * lambda)), multi(ann.dw2, alpha / m));
        ann.w1 = minus(multi(ann.w1, (1 - alpha * lambda)), multi(ann.dw1, alpha / m));

        ann.pre_dw2 = multi(ann.dw2, -1 * alpha / m);
        ann.pre_dw1 = multi(ann.dw1, -1 * alpha / m);

        //printMatrix(ann.w2, "w2");

        ann.b2 = minus(ann.b2, multi(ann.db2, alpha / m));
        ann.b1 = minus(ann.b1, multi(ann.db1, alpha / m));

    }

    /**
     * 更新权重，只更新权重w，不更新偏置量b<br/>
     * w(l+1)=w(l)+alpha*a*deltaT
     * @param i
     * @throws MatrixNotMatchException
     */
    public final void refreshWeight(int i) throws MatrixNotMatchException {
        computeDelta(i);
        ann.dw2 = multi(ann.a2All[i], ann.d3All[i], true);
        //ann.db2 = ann.d3All[i];

        ann.dw1 = multi(trainData.get(i).x, ann.d2All[i], true);
        //ann.db1 = ann.d2All[i];

        ann.w2 = minus(ann.w2, multi(ann.dw2, alpha));
        ann.w1 = minus(ann.w1, multi(ann.dw1, alpha));

        //ann.b2 = minus(ann.b2, multi(ann.db2, alpha / m));
        //ann.b1 = minus(ann.b1, multi(ann.db1, alpha / m));

    }

    public final void rollBackWeight() throws MatrixNotMatchException {
        MatrixOperate.copyMatric(ann.pre_w1, ann.w1);
        MatrixOperate.copyMatric(ann.pre_w2, ann.w2);
        MatrixOperate.copyMatric(ann.pre_b1, ann.b1);
        MatrixOperate.copyMatric(ann.pre_b2, ann.b2);
    }

    /**
     * 激活函数
     *
     * @param x
     * @return
     */
    public final double activeFunction(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    /**
     * 求导数，实际上是f'(z)=f(z)*(1-f(z))，又f(z)=a，所以f'(z)=a*(1-a)，因此是a
     *
     * @param a
     * @return
     */
    public final double[] derivativeFunction(double[] a) {
        double[] b = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            b[i] = a[i] * (1 - a[i]);
        }
        return b;
    }


    /**
     * 分析权重参数
     */
    public final void analysWeight() {
        // 试验权重随机值的系数，2.67左右，a3比较有区分度
        for (int i = 1; i < 1000; i++) {
            System.out.println(i * 0.01);
            this.ann.setAlpha(i * 0.01);
            try {
                this.forwardAll();
                printMatrix(ann.a3All, "a3All");
            } catch (MatrixNotMatchException e) {
                e.printStackTrace();
            }
        }
    }

    public final void analysParams() {
        double step = 0.05;
        double maxJwbDiff = 0.0;
        double maxLambda = step, maxAlpha = step;
        for (int i = 1; (this.alpha = i * step) < 1; i++) {
            for (int j = 1; (this.lambda = j * step) < 1; j++) {
                System.out.println("alpha=" + alpha + "\tlambda=" + lambda);
                double Jwb = 0.0;
                double preJwb = 0.0;
                try {
                    this.forwardAll();
                    preJwb = computeLoss();
                    refreshWeight();
                    this.forwardAll();
                    Jwb = computeLoss();
                    double jwbDiff = preJwb / Jwb;
                    if (jwbDiff > 0) {
                        if (jwbDiff > maxJwbDiff) {
                            maxJwbDiff = jwbDiff;
                            maxLambda = this.lambda;
                            maxAlpha = this.alpha;
                        }
                        System.out.println("Jwb diff=" + jwbDiff);
                    } else {
                        System.out.println("====== Jwb increased ======");
                    }
                    rollBackWeight();
                } catch (MatrixNotMatchException e) {
                    e.printStackTrace();
                }

            }
        }
        System.out.println("========== Max Diff find with (alpha=" + maxAlpha + ",lambda=" + maxLambda + ")==========");
        this.alpha = maxAlpha;
        this.lambda = maxLambda;
    }

    /**
     * 训练过程
     *
     * @throws MatrixNotMatchException
     */
    public final void train() throws MatrixNotMatchException {
        System.out.println("========== start training ==========");
        int epoch = 0;
        double Jwb = 0.0;
        double preJwb = 0.0;
        while (epoch < limitEpoch) {
            forwardAll();
            Jwb = computeLossWithRegular();
            System.out.println("======== epoch " + epoch + " ========");
            System.out.println("======== Jwb = " + Jwb + " ========");
            if (Jwb < limitLoss) {
                break;
            }
            if (epoch > 0) {
                //if (preJwb < Jwb) {
                //    break;
                //}
                if ((Math.abs(preJwb - Jwb)) < limitDiffLoss) {
                    break;
                }
            }
            System.out.println(preJwb - Jwb);
            preJwb = Jwb;
            refreshWeight();
            epoch++;
        }
        ann.printANNModel();
        System.out.println("========== end of training ==========");
    }

    public final void train2() throws MatrixNotMatchException {
        System.out.println("========== start training ==========");
        int epoch = 0;
        double Jwb = 0.0;
        while (epoch < limitEpoch) {
            System.out.println("======== epoch " + epoch + " ========");
            for (int i = 0; i < trainData.size(); i++) {
                forward(i);
                Jwb = computeLoss();
                System.out.println("======== Jwb = " + Jwb + " for data i=" + i + " ========");
                if (Jwb < limitLoss) {
                    break;
                }
                refreshWeight(i);
            }
            epoch++;
        }
        ann.printANNModel();
        System.out.println("========== end of training ==========");
    }

    /**
     * 预测，指定测试数据
     *
     * @param dataModel
     * @return
     * @throws MatrixNotMatchException
     */
    public final double[] predict(DataModel dataModel) throws MatrixNotMatchException {
        double[] z2 = addMat(multi(dataModel.x, ann.w1), ann.b1);
        double[] a2 = new double[ann.hNum];
        for (int i = 0; i < ann.hNum; i++) {
            a2[i] = activeFunction(z2[i]);
        }
        double[] z3 = addMat(multi(a2, ann.w2), ann.b2);
        double[] a3 = new double[ann.oNum];
        for (int i = 0; i < ann.oNum; i++) {
            a3[i] = activeFunction(z3[i]);
        }
        printMatrix(dataModel.x, "predict value of ");
        printMatrix(a3, "is : ");
        return a3;
    }

    /**
     * 预测，对数据初始化时已经指定测试数据的情况
     *
     * @throws MatrixNotMatchException
     */
    public final void predict() throws MatrixNotMatchException {
        int rightPredictNum = 0;
        for (int i = 0; i < testData.size(); i++) {
            System.out.println("predict result for testData " + i);
            int rightNum = 0;
            DataModel dm = testData.get(i);
            double y[] = predict(dm);
            printMatrix(dm.y, "expected value : ");
            for (int j = 0; j < y.length; j++) {
                int predictValue = 0;
                if (y[j] > 0.5) {
                    predictValue = 1;
                }
                System.out.println("predictValue" + predictValue + "\t dm.y" + dm.y[j]);
                if (dm.y[j] == predictValue) {
                    rightNum++;
                } else {
                    break;
                }
            }
            if (rightNum == y.length) {
                rightPredictNum++;
            }
        }
        System.out.println(" testData size = " + testData.size()
                + "\t rightNum = " + rightPredictNum
                + "\tPrecision = " + (((double) rightPredictNum / testData.size()) * 100) + "%");
    }

    /**
     * 主函数
     *
     * @param args
     */
    public static void main(String args[]) {
        ANNClassifier annClassifier = new ANNClassifier(new ANNModel(4, 30, 3));
        try {
            annClassifier.initData();
            annClassifier.ann.setTrainDataNum(annClassifier.trainData.size());
            annClassifier.ann.printANNModel();
            // 试验权重随机值的系数，2.67左右，a3比较有区分度
            //annClassifier.analysWeight();
            // 没有这个比例似乎也可以，只是对训练数据范围内的比较准确
            //annClassifier.ann.setAlpha(0.1);
            // 分析参数取值，找更新一次权重后误差变化最大的参数，作为模型参数（alpha和lambda）
            //annClassifier.analysParams();
            // 开始训练模型
            //annClassifier.train();
            annClassifier.train2();
            // 预测
            annClassifier.predict();
            //} catch (MatrixNotMatchException e) {
            //    e.printStackTrace();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}

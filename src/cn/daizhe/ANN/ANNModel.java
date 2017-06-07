package cn.daizhe.ANN;

import cn.daizhe.oper.exception.MatrixNotMatchException;

import static cn.daizhe.oper.MatrixOperate.multi;
import static cn.daizhe.oper.MatrixOperate.printMatrix;

/**
 * 神经网络模型，三层：输入层、1个隐层、输出层
 * Created by venice on 2017/5/29.
 */
public class ANNModel {

    // 各层节点个数
    public int iNum = 2;
    public int hNum = 5;
    public int oNum = 1;

    // 权重矩阵
    public double[][] w1;
    public double[][] w2;

    public double[][] pre_w1;
    public double[][] pre_w2;

    // 偏置向量
    public double[] b1;
    public double[] b2;

    public double[] pre_b1;
    public double[] pre_b2;

    // 保留所有m个输入数据对应的各层的z和a
    public double[][] z2All;
    public double[][] a2All;

    public double[][] z3All;
    public double[][] a3All;

    // m个输入数据对应的delta
    public double[][] d2All;
    public double[][] d3All;

    // delta_w
    public double[][] dw1;
    public double[][] dw2;

    // 上一次循环时的delta_w
    public double[][] pre_dw1;
    public double[][] pre_dw2;

    // delta_b
    public double[] db1;
    public double[] db2;

    // 上一次循环时的delta_b
    public double[] pre_db1;
    public double[] pre_db2;

    public ANNModel(int iNum, int hNum, int oNum) {
        // 权重和偏置量随机初始化的放大比例
        double alpha = 1;
        // 各层节点数
        this.iNum = iNum;
        this.hNum = hNum;
        this.oNum = oNum;
        // 随机初始化权重矩阵
        this.w1 = new double[iNum][hNum];
        this.pre_w1= new double[iNum][hNum];
        for (int i = 0; i < iNum; i++) {
            for (int j = 0; j < hNum; j++) {
                this.w1[i][j] = Math.random() * alpha;
            }
        }
        this.w2 = new double[hNum][oNum];
        this.pre_w2 = new double[hNum][oNum];
        for (int i = 0; i < hNum; i++) {
            for (int j = 0; j < oNum; j++) {
                this.w2[i][j] = Math.random() * alpha;
            }
        }
        //随机初始化偏置向量
        this.b1 = new double[hNum];
        this.pre_b1 = new double[hNum];
        for (int i = 0; i < hNum; i++) {
            this.b1[i] = Math.random() * alpha;
        }
        this.b2 = new double[oNum];
        this.pre_b2 = new double[oNum];
        for (int i = 0; i < oNum; i++) {
            this.b2[i] = Math.random() * alpha;
        }

        //初始化delta_w
        this.dw1 = new double[iNum][hNum];
        this.dw2 = new double[hNum][oNum];
        //初始化前一次的delta_w
        this.pre_dw1 = new double[iNum][hNum];
        this.pre_dw2 = new double[hNum][oNum];

        // 初始化delta_b
        this.db1 = new double[hNum];
        this.db2 = new double[oNum];
        // 初始化前一次的delta_b
        this.pre_db1 = new double[hNum];
        this.pre_db2 = new double[oNum];

    }

    public ANNModel(int iNum, int hNum, int oNum, int trainDataNum) {
        // 权重和偏置量随机初始化的放大比例
        double alpha = 1;
        // 各层节点数
        this.iNum = iNum;
        this.hNum = hNum;
        this.oNum = oNum;
        // 随机初始化权重矩阵
        this.w1 = new double[iNum][hNum];
        for (int i = 0; i < iNum; i++) {
            for (int j = 0; j < hNum; j++) {
                this.w1[i][j] = Math.random() * alpha;
            }
        }
        this.w2 = new double[hNum][oNum];
        for (int i = 0; i < hNum; i++) {
            for (int j = 0; j < oNum; j++) {
                this.w2[i][j] = Math.random() * alpha;
            }
        }
        //随机初始化偏置向量
        this.b1 = new double[hNum];
        for (int i = 0; i < hNum; i++) {
            this.b1[i] = Math.random() * alpha;
        }
        this.b2 = new double[oNum];
        for (int i = 0; i < oNum; i++) {
            this.b2[i] = Math.random() * alpha;
        }

        //初始化delta_w
        this.dw1 = new double[iNum][hNum];
        this.dw2 = new double[hNum][oNum];
        //初始化前一次的delta_w
        this.pre_dw1 = new double[iNum][hNum];
        this.pre_dw2 = new double[hNum][oNum];

        // 初始化delta_b
        this.db1 = new double[hNum];
        this.db2 = new double[oNum];
        // 初始化前一次的delta_b
        this.pre_db1 = new double[hNum];
        this.pre_db2 = new double[oNum];

        // 初始化保留所有m个输入数据对应的各层的z和a
        this.z2All = new double[trainDataNum][hNum];
        this.a2All = new double[trainDataNum][hNum];

        this.z3All = new double[trainDataNum][oNum];
        this.a3All = new double[trainDataNum][oNum];

        // 初始化delta
        this.d2All = new double[trainDataNum][hNum];
        this.d3All = new double[trainDataNum][oNum];

    }

    /**
     * 输出神经网络模型
     */
    public final void printANNModel() {
        System.out.println("======== the ANN Model: ========");
        System.out.println("iNum=" + iNum + ",hNum=" + hNum + ",oNum=" + oNum);
        printMatrix(w1, "w1");
        printMatrix(w2, "w2");
        printMatrix(b1, "b1");
        printMatrix(b2, "b2");
    }

    /**
     * 设置权重和偏置量的放大比例，根据输入和输出不同，先试验其比例，使得初始的a3在0.5和1之间，且各个a3有所区别，便于模型训练
     *
     * @param alpha
     */
    public final void setAlpha(double alpha) {
        try {
            this.w1 = multi(this.w1, alpha);
            this.w2 = multi(this.w2, alpha);
            this.b1 = multi(this.b1, alpha);
            this.b2 = multi(this.b2, alpha);
        } catch (MatrixNotMatchException e) {
            e.printStackTrace();
        }
    }

    public final void setTrainDataNum(int trainDataNum){
        // 初始化保留所有m个输入数据对应的各层的z和a
        this.z2All = new double[trainDataNum][hNum];
        this.a2All = new double[trainDataNum][hNum];

        this.z3All = new double[trainDataNum][oNum];
        this.a3All = new double[trainDataNum][oNum];

        // 初始化delta
        this.d2All = new double[trainDataNum][hNum];
        this.d3All = new double[trainDataNum][oNum];
    }
}

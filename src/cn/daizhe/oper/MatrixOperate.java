package cn.daizhe.oper;

import cn.daizhe.oper.exception.IllegalMatrixMNException;
import cn.daizhe.oper.exception.MatrixNotMatchException;

/**
 * Created by venice on 2017/5/29.
 */
public class MatrixOperate {

    /**
     * 矩阵相加
     *
     * @param a
     * @param b
     * @return
     * @throws MatrixNotMatchException
     */
    public static final double[][] addMat(double[][] a, double[][] b) throws MatrixNotMatchException {
        if (a == null || b == null) {
            throw new NullPointerException();
        }
        int am = getM(a), an = getN(a), bm = getM(b), bn = getN(b);

        if (am != bm || an != bn) {
            throw new MatrixNotMatchException("a.shape=[" + am + "," + an + "],b.shape=[" + bm + "," + bn + "]");
        }
        double[][] c = new double[a.length][];
        for (int i = 0; i < a.length; i++) {
            if (a[i] == null || b[i] == null) {
                throw new NullPointerException();
            }
            c[i] = new double[a[i].length];
            for (int j = 0; j < a[i].length; j++) {
                c[i][j] = a[i][j] + b[i][j];
            }
        }
        return c;
    }

    /**
     * 矩阵相加
     *
     * @param a
     * @param b
     * @return
     * @throws MatrixNotMatchException
     */
    public static final double[] addMat(double[] a, double[] b) throws MatrixNotMatchException {
        if (a == null || b == null) {
            throw new NullPointerException();
        }
        if (a.length != b.length) {
            throw new MatrixNotMatchException(a.length + "," + b.length);
        }
        double[] c = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            c[i] = a[i] + b[i];
        }
        return c;
    }

    /**
     * 矩阵相减
     *
     * @param a
     * @param b
     * @return
     * @throws MatrixNotMatchException
     */
    public static final double[][] minus(double[][] a, double[][] b) throws MatrixNotMatchException {
        if (a == null || b == null) {
            throw new NullPointerException();
        }
        if (a.length != b.length) {
            throw new MatrixNotMatchException(a.length + "," + b.length);
        }
        double[][] c = new double[a.length][];
        for (int i = 0; i < a.length; i++) {
            if (a[i] == null || b[i] == null) {
                throw new NullPointerException();
            }
            if (a[i].length != b[i].length) {
                throw new MatrixNotMatchException("a[" + i + "].length=" + a.length + ",b[" + i + "].length=" + b.length);
            }
            c[i] = new double[a[i].length];
            for (int j = 0; j < a[i].length; j++) {
                c[i][j] = a[i][j] - b[i][j];
            }
        }
        return c;
    }

    /**
     * 矩阵相减
     *
     * @param a
     * @param b
     * @return
     * @throws MatrixNotMatchException
     */
    public static final double[] minus(double[] a, double[] b) throws MatrixNotMatchException {
        if (a == null || b == null) {
            throw new NullPointerException();
        }
        if (a.length != b.length) {
            throw new MatrixNotMatchException(a.length + "," + b.length);
        }
        double[] c = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            c[i] = a[i] - b[i];
        }
        return c;
    }

    public static final double[] minus(int[] a, double[] b) throws MatrixNotMatchException {
        if (a == null || b == null) {
            throw new NullPointerException();
        }
        double[] c = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            c[i] = a[i];
        }
        return minus(c, b);
    }

    public static final double[] minus(double[] a, int[] b) throws MatrixNotMatchException {
        if (a == null || b == null) {
            throw new NullPointerException();
        }
        double[] c = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            c[i] = b[i];
        }
        return minus(a, c);
    }

    public static final double[] minus(double a, double[] b) throws MatrixNotMatchException {
        if (b == null) {
            throw new NullPointerException();
        }
        double[] c = new double[b.length];
        for (int i = 0; i < b.length; i++) {
            c[i] = a - b[i];
        }
        return c;
    }

    public static final double[] minus(int a, double[] b) throws MatrixNotMatchException {
        if (b == null) {
            throw new NullPointerException();
        }
        double[] c = new double[b.length];
        for (int i = 0; i < b.length; i++) {
            c[i] = a - b[i];
        }
        return c;
    }

    /**
     * 矩阵相乘
     *
     * @param a
     * @param b
     * @return
     * @throws MatrixNotMatchException
     */
    public static final double[][] multi(double[][] a, double[][] b) throws MatrixNotMatchException {
        if (a == null || b == null) {
            throw new NullPointerException();
        }
        int am = getM(a), an = getN(a), bm = getM(b), bn = getN(b);

        if (an != bm) {
            throw new MatrixNotMatchException("a.shape=[" + am + "," + an + "],b.shape=[" + bm + "," + bn + "]");
        }

        double[][] c = new double[am][];
        for (int i = 0; i < am; i++) {
            c[i] = new double[bn];
            for (int j = 0; j < bn; j++) {
                c[i][j] = 0;
                for (int k = 0; k < an; k++) {
                    c[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        return c;
    }

    public static final double[] multi(double[][] a, double[] b) throws MatrixNotMatchException {
        if (a == null || b == null) {
            throw new NullPointerException();
        }
        int am = getM(a), an = getN(a), bm = b.length, bn = 1;

        if (an != bm) {
            throw new MatrixNotMatchException("a.shape=[" + am + "," + an + "],b.shape=[" + bm + "," + bn + "]");
        }

        double[] c = new double[am];
        for (int i = 0; i < am; i++) {
            for (int j = 0; j < an; j++) {
                c[i] += a[i][j] * b[j];
            }
        }
        return c;
    }

    /**
     * 矩阵相乘
     *
     * @param a
     * @param b
     * @return
     * @throws MatrixNotMatchException
     */
    public static final double[] multi(double[] a, double[][] b) throws MatrixNotMatchException {
        if (a == null || b == null) {
            throw new NullPointerException();
        }
        int am = getM(a), an = getN(a), bm = getM(b), bn = getN(b);

        if (an != bm) {
            throw new MatrixNotMatchException("a.shape=[" + am + "," + an + "],b.shape=[" + bm + "," + bn + "]");
        }

        double[] c = new double[bn];
        for (int i = 0; i < bn; i++) {
            for (int j = 0; j < an; j++) {
                c[i] += a[j] * b[j][i];
            }
        }
        return c;
    }

    /**
     * 矩阵相乘
     *
     * @param a
     * @param b
     * @return
     * @throws MatrixNotMatchException
     */
    public static final double multi(double[] a, double[] b) throws MatrixNotMatchException {
        if (a == null || b == null) {
            throw new NullPointerException();
        }
        int an = getN(a), bn = getN(b);

        if (an != bn) {
            throw new MatrixNotMatchException("a.shape=[" + bn + "," + an + "],b.shape=[" + bn + "," + bn + "]");
        }

        double c = 0.0;
        for (int i = 0; i < an; i++) {
            c += a[i] * b[i];
        }
        return c;
    }

    public static final double[][] multi(double[] a, double[] b, boolean T) throws MatrixNotMatchException {
        if (a == null || b == null) {
            throw new NullPointerException();
        }
        int am = a.length, bn = b.length;

        double[][] c = new double[am][bn];
        for (int i = 0; i < am; i++) {
            for (int j = 0; j < bn; j++) {
                c[i][j] = a[i] * b[j];
            }
        }
        return c;
    }

    public static final double[][] multi(double a, double[][] b) throws MatrixNotMatchException {
        if (b == null) {
            throw new NullPointerException();
        }

        int bm = getM(b), bn = getN(b);
        double[][] c = new double[bm][bn];
        for (int i = 0; i < bm; i++) {
            for (int j = 0; j < bn; j++) {
                c[i][j] = a * b[i][j];
            }
        }
        return c;
    }

    public static final double[][] multi(double[][] a, double b) throws MatrixNotMatchException {
        return multi(b, a);
    }

    public static final double[] multi(double a, double[] b) throws MatrixNotMatchException {
        if (b == null) {
            throw new NullPointerException();
        }

        double[] c = new double[b.length];
        for (int i = 0; i < b.length; i++) {
            c[i] = a * b[i];
        }
        return c;
    }

    public static final double[] multi(double[] a, double b) throws MatrixNotMatchException {
        return multi(b, a);
    }

    /**
     * 矩阵点乘（对应位置相乘）
     *
     * @param a
     * @param b
     * @return
     * @throws MatrixNotMatchException
     */
    public static final double[][] dot(double[][] a, double[][] b) throws MatrixNotMatchException {
        if (a == null || b == null) {
            throw new NullPointerException();
        }
        int am = getM(a), an = getN(a), bm = getM(b), bn = getN(b);

        if (am != bm || an != bn) {
            throw new MatrixNotMatchException("a.shape=[" + am + "," + an + "],b.shape=[" + bm + "," + bn + "]");
        }

        double[][] c = new double[am][];
        for (int i = 0; i < am; i++) {
            c[i] = new double[an];
            for (int j = 0; j < an; j++) {
                c[i][j] = a[i][j] * b[i][j];
            }
        }
        return c;

    }

    public static final double[] dot(double[] a, double[] b) throws MatrixNotMatchException {
        if (a == null || b == null) {
            throw new NullPointerException();
        }
        int an = getN(a), bn = getN(b);

        if (an != bn) {
            throw new MatrixNotMatchException("a.shape=[" + an + "," + an + "],b.shape=[" + an + "," + bn + "]");
        }

        double[] c = new double[an];
        for (int i = 0; i < an; i++) {
            c[i] = a[i] * b[i];
        }
        return c;
    }

    /**
     * 返回指定结构的矩阵
     *
     * @param m
     * @param n
     * @return
     */
    public static final double[][] getMatrix(int m, int n) {
        if (m <= 0 || n <= 0) {
            new IllegalMatrixMNException("" + m + "," + n);
        }
        double[][] c = new double[m][n];
        return c;
    }

    /**
     * 返回指定结构的全零矩阵
     *
     * @param m
     * @param n
     * @return
     */
    public static final double[][] getZeroMatrix(int m, int n) {
        double[][] c = getMatrix(m, n);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                c[i][j] = 0;
            }
        }
        return c;
    }

    /**
     * 返回指定结构的全1矩阵
     *
     * @param m
     * @param n
     * @return
     */
    public static final double[][] getOnesMatrix(int m, int n) {
        double[][] c = getMatrix(m, n);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                c[i][j] = 1;
            }
        }
        return c;
    }

    /**
     * 返回指定结构的随机矩阵
     *
     * @param m
     * @param n
     * @return
     */
    public static final double[][] getRandomMatrix(int m, int n) {
        return getRandomMatrix(m, n, 1);
    }

    /**
     * 返回指定结构的随机矩阵
     *
     * @param m
     * @param n
     * @param range 随机值范围
     * @return
     */
    public static final double[][] getRandomMatrix(int m, int n, double range) {
        double[][] c = getMatrix(m, n);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                c[i][j] = Math.random() * range;
            }
        }
        return c;
    }

    /**
     * 取得矩阵的行数
     *
     * @param a
     * @return
     */
    public static final int getM(double[][] a) {
        if (a == null) {
            return -1;
        }
        return a.length;
    }

    /**
     * 取得矩阵的行数
     *
     * @param a
     * @return
     */
    public static final int getM(double[] a) {
        if (a == null) {
            return -1;
        }
        return 1;
    }

    /**
     * 取得矩阵的列数
     *
     * @param a
     * @return
     * @throws MatrixNotMatchException
     */
    public static final int getN(double[][] a) throws MatrixNotMatchException {
        if (a == null) {
            return -1;
        }
        int n = 0;
        for (int i = 0; i < a.length; i++) {
            if (a[i] == null) {
                throw new NullPointerException();
            }
            if (i == 0) {
                n = a[i].length;
            } else if (n != a[i].length) {
                printMatrix(a);
                throw new MatrixNotMatchException("a is not a Matrix");
            }
        }
        return n;
    }

    /**
     * 取得矩阵的列数
     *
     * @param a
     * @return
     * @throws MatrixNotMatchException
     */
    public static final int getN(double[] a) throws MatrixNotMatchException {
        if (a == null) {
            return -1;
        }
        return a.length;
    }

    /**
     * 输出矩阵
     *
     * @param a
     */
    public static final void printMatrix(double[][] a, String matrixName) {
        System.out.println(matrixName);
        printMatrix(a);
    }

    public static final void printMatrix(double[][] a) {
        if (a == null) {
            System.out.println("Matrix is null");
        }
        System.out.print("[");
        for (int i = 0; i < a.length; i++) {
            if (i > 0) {
                System.out.println("");
            }
            System.out.print("[");
            for (int j = 0; j < a[i].length; j++) {
                System.out.print(a[i][j] + " ");
            }
            System.out.print("]");
        }
        System.out.println("]");
    }

    /**
     * 输出矩阵
     *
     * @param a
     */
    public static final void printMatrix(double[] a, String matrixName) {
        System.out.println(matrixName);
        printMatrix(a);
    }

    public static final void printMatrix(double[] a) {
        if (a == null) {
            System.out.println("Matrix is null");
        }
        System.out.print("[");
        for (int i = 0; i < a.length; i++) {
            System.out.print(a[i] + " ");
        }
        System.out.println("]");
    }

    public static final void printMatrix(int[] a, String matrixName) {
        System.out.println(matrixName);
        printMatrix(a);
    }

    public static final void printMatrix(int[] a) {
        if (a == null) {
            System.out.println("Matrix is null");
        }
        System.out.print("[");
        for (int i = 0; i < a.length; i++) {
            System.out.print(a[i] + " ");
        }
        System.out.println("]");
    }

    /**
     * 复制矩阵，采用深拷贝方法
     *
     * @param src
     * @param target
     * @return
     * @throws MatrixNotMatchException
     */
    public static final int copyMatric(double[][] src, double[][] target) throws MatrixNotMatchException {
        if (src == null || target == null) {
            return -1;
        }
        if (src.length != target.length) {
            throw new MatrixNotMatchException("src.length=" + src.length + " target.length=" + target.length);
        }
        for (int i = 0; i < src.length; i++) {
            target[i] = new double[src[i].length];
            System.arraycopy(src[i], 0, target[i], 0, src[i].length);
        }
        return 1;
    }

    public static final int copyMatric(double[] src, double[] target) throws MatrixNotMatchException {
        if (src == null || target == null) {
            return -1;
        }
        if (src.length != target.length) {
            throw new MatrixNotMatchException("src.length=" + src.length + " target.length=" + target.length);
        }
        System.arraycopy(src, 0, target, 0, src.length);
        return 1;
    }

    public static void main(String args[]) {
        double[][] a = new double[2][3];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[i].length; j++) {
                a[i][j] = i + j;
            }
        }
        double[][] b = new double[3][2];
        for (int i = 0; i < b.length; i++) {
            for (int j = 0; j < b[i].length; j++) {
                b[i][j] = i + j;
            }
        }
        MatrixOperate.printMatrix(a);
        try {
            MatrixOperate.printMatrix(multi(a, b));
            MatrixOperate.printMatrix(dot(a, a));
        } catch (MatrixNotMatchException e) {
            e.printStackTrace();
        }

        try {
            MatrixOperate.printMatrix(multi(MatrixOperate.getOnesMatrix(3, 4), MatrixOperate.getRandomMatrix(4, 2)));
        } catch (MatrixNotMatchException e) {
            e.printStackTrace();
        }


    }
}

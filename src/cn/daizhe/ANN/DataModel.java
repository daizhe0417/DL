package cn.daizhe.ANN;

/**
 * 神经网络的数据模型，x是输入数据，y是输出数据，对于分类问题，y的取值是0或1
 * Created by venice on 2017/5/29.
 */
public class DataModel {
    public double[] x;
    public int[] y;

    public DataModel() {
    }

    public DataModel(double[] x, int[] y) {
        this.x = new double[x.length];
        System.arraycopy(x, 0, this.x, 0, x.length);
        this.y = new int[y.length];
        System.arraycopy(y, 0, this.y, 0, y.length);
    }

    public DataModel(int[] x, int[] y) {
        this.x = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            this.x[i]=x[i];
        }
        this.y = new int[y.length];
        System.arraycopy(y, 0, this.y, 0, y.length);
    }

    public static void main(String args[]) {
        double[] x = {1, 2, 3};
        //拷贝之前必须new
        double[] y = new double[x.length];
        //arraycopy深拷贝，即新数组的修改不影响旧数组，只复制了值
        System.arraycopy(x, 0, y, 0, x.length);
        y[0] = 4;
        System.out.println(x[0] + "====" + y[0]);
        int[] yi = {1, 2, 3};
        new DataModel(x, yi);
        new DataModel(new int[]{1, 2, 3}, new int[]{2});
    }
}

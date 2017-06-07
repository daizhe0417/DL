package cn.daizhe.oper.exception;

/**
 * Created by venice on 2017/5/29.
 */
public class MatrixNotMatchException extends Exception {
    public MatrixNotMatchException(String message) {
        super("矩阵长度不匹配" + message);
    }

}

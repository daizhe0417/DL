package cn.daizhe.oper.exception;

/**
 * Created by venice on 2017/5/29.
 */
public class IllegalMatrixMNException extends Exception {
    public IllegalMatrixMNException(String message) {
        super("矩阵长度错误" + message);
    }
}

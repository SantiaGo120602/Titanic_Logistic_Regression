#include "regressionlogistic.h"
#include <iostream>
#include <fstream>
#include <eigen3/Eigen/Dense>
#include <vector>
#include <list>

/*Primera función miembro: la función sigmoid*/

Eigen::MatrixXd  RegressionLogistic::Sigmoid(Eigen::MatrixXd Z){
    /*Función sigmoid que retorna la matriz con el cálculo de la función:
    */
    return 1/(1+(-Z.array()).exp());
}

/*Segunda función miembro: función de propagación, la cual contiene el tratamiento
 * de la función de costo (cross entropy), y sus correspondientes derivadas.*/

std::tuple<Eigen::MatrixXd, double, double> RegressionLogistic::Propagation(Eigen::MatrixXd W,
                                                                             Eigen::MatrixXd X, double b,
                                                                            Eigen::MatrixXd y, double lambda){
    /*Sobre la base de la presentación de regresión logística*/
    int m = y.rows();
    /*Se obtiene la matriz eigen Z*/
    Eigen::MatrixXd Z = (W.transpose()* X.transpose()).array()+b;
    Eigen::MatrixXd A = Sigmoid(Z);
    /*Se crea una función para la entropía cruzada:
     * NO sabemos que valor se va a retornar.*/
    auto cross_entropy = -(y.transpose()*(Eigen::VectorXd)A.array().log().transpose()+((Eigen::VectorXd)(1-y.array())).transpose()*(Eigen::VectorXd)(1-A.array()).log().transpose())/m;
    /* Función para reducir la varianza del modelo:
    * usando la regularización. */
    double l2_reg_costo = W.array().pow(2).sum()*(lambda/(2*m));
    /* Función para el cálculo del costo, usando la entropía cruzada
     * con el ajuste de regularización:
     * se hace uso de static_cast, debido a que la función debe retornar un double,
     * pero va a estar compuesta de tipos de datos definidos pro el usuario (auto) */
    double costo = static_cast<const double>((cross_entropy.array()[0]) + l2_reg_costo);
    /*Se calcula la derivada de las matrices en función de los pesos*/
    Eigen::MatrixXd dw = ((Eigen::MatrixXd)(A-y.transpose())*X/m)+(Eigen::MatrixXd)(lambda/m*W).transpose();

    /*Se calculo la derivada en función  del bias (punto de corte). */
    double db = ((A-y.transpose()).array().sum())/m;

    /* Se retorna de la función de propagación la derivada de los pesos (dw), la derivada del bias (db) y se retorna el costo;
     * el retorno es en una tupla comprimida. */
    return std::make_tuple(dw, db, costo);

}

/* Se crea la función de optimización: Se crea una lista en donde se va a almacenar cada uno de los valores de la función
 * de costo hasta converger. Esta actualización, se almacenará en un fichero para posteriormente ser visualizada. La actualización
 * se ve representada en una de las diapositivas de las presentaciones. Se observa en la presentación de regresión logística.
 Se  pasa una bandera a la función, para imprimir, si se quiere, el valor del costo cada 100 iteraciones. */
std::tuple<Eigen::MatrixXd, double, Eigen::MatrixXd, double, std::list<double>> RegressionLogistic::Optimization(Eigen::MatrixXd W, double b, Eigen::MatrixXd X, Eigen::MatrixXd y, int num_iter, double learningRate, double lambda, bool log_costo){
    /*Se crea la lista a entregar para ca*/
    std::list<double> lista_costo;
    Eigen::MatrixXd dw;
    double db, costo;
    /*Se hace la iteración*/

    for (int i =0; i<num_iter; i++){
        std::tuple<Eigen::MatrixXd, double, double> propagation = Propagation(W,X,b,y,lambda);
        std::tie(dw, db, costo)= propagation;
        /* Se actualizan los valores (W y b), que representan los weights and biases*/

        W = W -(learningRate * dw).transpose();
        b = b - (learningRate  * db);//.transpose();

        /*Según la bandera, se guarda cada 100 pasos el valor del costo.*/
        if (i%100 == 0){
            lista_costo.push_back(costo);
        }

        /*Se imprime por pantalla según la bandera*/
        if (log_costo && i%100 == 0){
            std::cout<<"Costo después de actualizar "<<i<<": "<<costo<<std::endl;
        }

    }

    return std::make_tuple(W, b, dw, db, lista_costo);
}

/* Función de predicción: La función estimará (predicción) la etiqueta de salida sí corresponde a 0 o 1
 * La idea es calcular y_hat (y estimado) usando los parámetros de regresión (W y b) aprendidos. Se convierten
 * las entradas a 0, si la función de activación es inferior o igual a 0.5.
 * Se convierten las entradas a 1, si la función de activación es superior a 0.5.
 */

Eigen::MatrixXd RegressionLogistic::Prediction(Eigen::MatrixXd W, double b, Eigen::MatrixXd X){
    /* Se calcula la cantidad de valores o registros (m) */
    int m = X.rows();
    /*Se crea una matriz con valores del vector de ceros, del tamaño de la matriz de entrada (X)*/
    Eigen::MatrixXd y_pred = Eigen::VectorXd::Zero(m).transpose();

    /*Se crea una matriz para almacenar los valores de Z (calculados)*/
    Eigen::MatrixXd Z = (W.transpose() * X.transpose()).array() + b;

    /*Se calcula la función sigmoid en la matriz A*/
    Eigen::MatrixXd A = Sigmoid(Z);

    /* Se calcula el valor estimado (Etiquetas 0 o 1) según la función de activación, para cada uno de los registros
     * (matriz X)*/

    for (int i = 0; i < A.cols(); i++) {
        if(A(0,i)<=0.5){
            y_pred(0, i)=0;
        }else{
            y_pred(0, i)=1;
        }
    }

    return y_pred.transpose();
}



//pedro.cardenas2102a@gmail.com








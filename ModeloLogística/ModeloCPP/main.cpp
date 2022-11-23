#include <iostream>
#include "Extraction/extraction.h"
#include "RegressionLogistic/regressionlogistic.h"
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <boost/algorithm/string.hpp>
#include <vector>
#include <string.h>
#include <list>
/*********************************************************
 * Fecha: 03-11-2022
 * Autor: Santiago Vivas
 * Materia: HPC-2
 * Tema: Regresión logística
 * ******************************************************/


/*
El método main, captura los argumentos de entrada: lugar en donde se encuentra el dataset, el separador/delimitador del dataset
y el header (tiene o no cabecera)

*/
int main(int argc, char *argv[])
{
    /* Se crea un objeto del tipo Extraer
         * para incluir los 3 argumentos que necesita
         * el objeto. */
    Extraction extraerData(argv[1], argv[2], argv[3]);
        /* Se requiere probar la lectura del fichero y
             * luego se requiere observar el dataset como
             * un objeto de matriz tipo dataframe. */
    std::vector<std::vector<std::string>> dataSET = extraerData.ReadCSV();
    int filas = dataSET.size()+1;
    int columnas = dataSET[0].size();
    Eigen::MatrixXd MatrizDataF = extraerData.CSVToEigen(
                        dataSET, filas, columnas);



    //Normalización de los datos
    MatrizDataF = extraerData.Normalizador(MatrizDataF, false);

    //Validación del dataset
    std::cout<<"Número de filas: "<<filas-1<<std::endl;
    std::cout<<"Número de columnas: "<<columnas<<std::endl;
    std::cout<<"Media por columna: "<<extraerData.Promedio(MatrizDataF)<<std::endl<<std::endl;

    //std::cout<<MatrizDataF<<std::endl;

    /*Se procede a dividir en 4 conjuntos los datos:
    X_Train
    y_Train
    X_Test
    y_Test*/

    Eigen::MatrixXd X_Train, y_Train, X_Test, y_Test;
    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> datos_divididos = extraerData.TrainTestSplit(MatrizDataF, 0.80);
    /*datos_divididos presenta la tupla comprimida con 4 matrices. A continuación, se deb descomprimir
    para las 4 matrices.*/

    std::tie(X_Train, y_Train, X_Test, y_Test) = datos_divididos;

    /*A continuación se instancia el objeto regresión logística*/

    RegressionLogistic modelo_lr;

    /*Se ajustan los parámetros*/
    int dimension = X_Train.cols();
    Eigen::MatrixXd W = Eigen::VectorXd::Zero(dimension);
    double b = 0;
    double lambda = 0.0;
    bool banderita = true;
    double learning_rate=0.01;
    int num_iter = 10000;
    Eigen::MatrixXd dw;
    double db;
    std::list<double> lista_costos;


    std::tuple<Eigen::MatrixXd, double, Eigen::MatrixXd, double, std::list<double>> optimo=
            modelo_lr.Optimization(W, b ,X_Train,y_Train,num_iter,learning_rate,lambda,banderita);

    /*Se desempaqueta el conjunto de valores de óptimo*/
    std::tie(W, b, dw, db, lista_costos) = optimo;


    /*Se crean las matrices de predicción, (prueba y entrenamiento)*/

    Eigen::MatrixXd y_pred_test = modelo_lr.Prediction(W,b,X_Test);
    //Eigen::MatrixXd y_pred_train = modelo_lr.Prediction(W,b,X_Train);

    //std::cout<<y_pred_test<<std::endl;
    /* A continuación se calcula la métrica de accuarcy para revisar la
     * precisión del modelo*/

    //auto train_accuracy = (100-((y_pred_train - y_Train).cwiseAbs().mean()*100))/100;

    auto test_accuracy = (100-((y_pred_test - y_Test).cwiseAbs().mean()*100))/100;


    //std::cout<<"Accuarcy de entrenamiento: "<<train_accuracy<<std::endl;

    std::cout<<"Accuarcy de prueba: "<<test_accuracy<<std::endl;
    std::cout<<"F1_score de prueba: "<<extraerData.f1_score(y_pred_test,y_Test)<<std::endl;





    /*std::vector<double> lista_salida;
    for (double const &c: lista_costos) {
        lista_salida.push_back(c);
    }

    extraerData.vector_to_file(lista_salida, "costo.txt");
    */



    return EXIT_SUCCESS;
}

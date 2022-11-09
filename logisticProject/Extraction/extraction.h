#ifndef EXTRACTION_H
#define EXTRACTION_H

#include <iostream>
#include <fstream>
#include <eigen3/Eigen/Dense>
#include <vector>
/*La clase extracción se compone de las funciones
o métodos para manipular el dataset.
Se presentan funciones para:
-LecturaCSV.
-Promedios.
-Normalización de datos.
-Desviación estándar.
La clase recibe como parámetros de entrada:
-El dataset(path del .csv)
-El delimitador, separador entre columnas
-Sí el dataset tiene o no cabecera.
*/

class Extraction
{

    //Se presenta el constructor para los argumentos de entrada a la clase: nombre_dataset, delimitador, header
    std::string setDatos;
    std::string delimitador;
    bool header;
public:
    Extraction(std::string datos,
            std::string separador,
            bool head):
        setDatos(datos),
        delimitador(separador),
        header(head){}
    /*Se presenta el prototipo de las funciones*/
    std::vector<std::vector<std::string>> ReadCSV();
    Eigen::MatrixXd CSVToEigen(
                std::vector<std::vector<std::string>>  SETdatos,
                int filas, int columnas);
    auto Promedio(Eigen::MatrixXd datos) -> decltype(datos.colwise().mean());
    auto DesvStand(Eigen::MatrixXd data) -> decltype((((data.array().square().colwise().sum())/(data.rows()-1)).sqrt()));
    Eigen::MatrixXd Normalizador(Eigen::MatrixXd datos, bool normalTarget);
    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> TrainTestSplit(Eigen::MatrixXd data, float train_size);
    void vector_to_file(std::vector<double> vector, std::string nombre_file);
    void eigen_to_file(Eigen::MatrixXd datos, std::string nombre_fichero);
    double f1_score(Eigen::MatrixXd y_pred_test, Eigen::MatrixXd y_Test);
};

#endif // EXTRACTION_H

#include "extraction.h"
#include <iostream>
#include <fstream>
#include <eigen3/Eigen/Dense>
#include <vector>
#include <boost/algorithm/string.hpp>

/*Impolementación de los métodos de la clase extracción*/


/* Primer funcion miembro: Lectura de fichero csv.
 * Se presenta como un vector de vectores del
 * tipo string.
 * La idea es leer linea por linea y almacenar
 * cada una en un vector de vectores del
 * tipo string. */
std::vector<std::vector<std::string>> Extraction::ReadCSV(){
    /* Abrir el fichero para lectura solamente */
    std::fstream Fichero(setDatos);

    /* Vector de vectores tipo string a entregar por
     * parte de la funcion */
    std::vector<std::vector<std::string>> datosString;

    /* Se itera a traves de cada linea, y se divide
     * el contenido dado por el separador( argumento
     * de entrada) provisto por el constructor */

    std::string linea = ""; // Almacenar cada linea
    while(getline(Fichero, linea)){
       /* Se crea un vector para almacenar la fila */
        std::vector<std::string> vectorFila;

       /* Se separa segun el delimitador */
        boost::algorithm::split(vectorFila,
                                linea,
                                boost::is_any_of(delimitador));
        datosString.push_back(vectorFila);
    }

    /* Se cierra el fichero .csv */
    Fichero.close();

    /* Se retorna el vector de
     * vectores del tipo string */
    return datosString;
}

/* Se implementa la segunda funcion miembro
 * la cual tiene como mision transformar el
 * vector de vectores del tipo String, en
 * una matrix Eigen. La idea es simular un
 * objeto DATAFRAME de pandas para poder
 * manipular los datos */

Eigen::MatrixXd Extraction::CSVToEigen(
        std::vector<std::vector<std::string>>  SETdatos,
        int filas, int columnas){
    /* Se hace la pregunta si tiene cabezera o no
     * el vector de vectores del tipo String.
     * Si tiene cabecera, se debe eliminar */
    if(header == true){
        filas = filas - 1;
    }

    /* Se itera sobre cada registro del fichero,
     * a la vez que se almacena en una matrixXd,
     * de dimension filas por columnas. Principalmente,
     * se almacenara Strings (porque llega un vector de
     * vectores del tipo String. La idea es
     * hacer un casting de String a float. */
    Eigen::MatrixXd MatrizDF(columnas, filas);
    for (int i = 0; i < filas; i++){
        for(int j = 0; j < columnas; j++){
            MatrizDF(j, i) = atof(SETdatos[i][j].c_str());
        }
    }
    //return MatrizDF;
    /* Se transpone la matriz, dado que viene
     * columnas por filas, para retornarla */
    return MatrizDF.transpose();
}

/* Funcion para calcular el promedio
 * En C++ la herencia del tipo de dato
 * no es directa (sobre todo si es a partir
 * de funciones dadas por otras interfaces/clases/
 * biblioteclas: EIGEN, shrkml, etc...). Entonces
 * se declara el tipo en una expresion "decltype"
 * con el fin de tener seguridad de que tipo de dato
 * retornara la funcion */
// En caso de no saber que dato encontrar usar auto y decltype (declarative type)
auto Extraction::Promedio(Eigen::MatrixXd datos) -> decltype(datos.colwise().mean()){
    return datos.colwise().mean();
}
auto Extraction::DesvStand(Eigen::MatrixXd data) -> decltype((((data.array().square().colwise().sum())/(data.rows()-1)).sqrt())){
    return (((data.array().square().colwise().sum())/(data.rows()-1)).sqrt());
}
/* Acto seguido se procede a hacer el cálculo o la función de normalización: La idea
* es vitar los cambios en orden de magnitud. Lo anterior representa un deterioro para
* la prediccion sobre la base de cualquier modelo de machine laearning. Evita los outliers
   Se le agrega un argumento bool para pasar como bandera si se quiere o no normalizar la variable target*/
Eigen::MatrixXd Extraction::Normalizador(Eigen::MatrixXd datos, bool normalTarget){
/* Normalización:
* MatrixNorm = xi - x.mean() / desviacionEstandar */

    /*Se condiciona si el target es normalizado o no*/
    Eigen::MatrixXd dataNorm;
    if (normalTarget==true){
        dataNorm=datos;
    }else{
        dataNorm = datos.leftCols(datos.cols()-1);
    }
    Eigen::MatrixXd DataEscalado = dataNorm.rowwise() - Promedio(dataNorm);


    Eigen::MatrixXd matrixNorm = DataEscalado.array().rowwise()/DesvStand(DataEscalado);

    if (normalTarget==false){
        matrixNorm.conservativeResize(matrixNorm.rows(), matrixNorm.cols()+1);
        matrixNorm.col(matrixNorm.cols()-1) = datos.rightCols(1);
    }

    /* Se retorna cada dato escalado */
    return matrixNorm;
}

/*Función para dividir conjunto sde entrenamiento y conjunto de pruebas:
Se crean 4 matrcies que representan los 4 conjuntos:
-variables inedpendientes de entrenamiento
-variables dependientes de entrenamiento
-variables inedpendientes de prueba
-variables dependientes de prueba
La función recibe como argumento la matriz normalizada y el tamaño a dividir los conjuntos.*/

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> Extraction::TrainTestSplit(Eigen::MatrixXd data, float train_size){
    /*Cantidad de filas totales de data*/
    int filas_totales = data.rows();

    /*Cantidad de filas para entrenamiento*/
    int filas_train = round(filas_totales * train_size);

    /*Cantidad de filas para prueba*/
    int filas_test = round(filas_totales - filas_train);

    Eigen::MatrixXd Train = data.topRows(filas_train);
    Eigen::MatrixXd X_Train = Train.leftCols(data.cols()-1);
    Eigen::MatrixXd y_Train = Train.rightCols(1);


    Eigen::MatrixXd Test = data.bottomRows(filas_test);
    Eigen::MatrixXd X_Test = Test.leftCols(data.cols()-1);
    Eigen::MatrixXd y_Test = Test.rightCols(1);

    /*Se retorna la tupla comprimirda*/
    return std::make_tuple(X_Train, y_Train, X_Test, y_Test);
}

/*Función para pasar de vector iostream a fichero de texto plano, para visualizar gráficamente.
La función recibe el vector y el nombre del fichero a exportar.
*/
void Extraction::vector_to_file(std::vector<double> vector, std::string nombre_file){
    /*Se nombra el fichero*/
    std::ofstream fichero_salida(nombre_file);
    /*Se itera por cada vector, para imprimirlo en el fichero*/
    std::ostream_iterator<double> iterador_salida(fichero_salida, "\n");
    std::copy(vector.begin(), vector.end(), iterador_salida);

    /* Se entrega la copia a un fichero*/
}

/* Función para llevar una matriz a un fichero*/
void Extraction::eigen_to_file(Eigen::MatrixXd datos, std::string nombre_fichero){
    /*Se nombra el fichero*/
    std::ofstream fichero_salida(nombre_fichero);

    /*Si el fichero está ABIERTO, guarda datos*/
    if (fichero_salida.is_open()){
        fichero_salida<< datos <<"\n";
    }
}

// Creación de las Función que calcula el f1 score
double Extraction::f1_score(Eigen::MatrixXd y_pred_test, Eigen::MatrixXd y_Test){
    /* Se crean las variables que van a almacenar el número de predicciones verdaderas positivas, verdaderas negativas,
     * falsas positivas, y falsas negativas*/
    double true_positives = 0;
    double true_negatives = 0;
    double false_positives = 0;
    double false_negatives = 0;
    /* Se itera sobre las matrices de y_Test y y_pred_test y se categoriza cada predicción según corresponda.*/
    for (int i =0; i<y_pred_test.rows(); i++){
        if ((y_pred_test(i, 0) == 1) && (y_Test(i, 0) == 1)){
            true_positives++;
        }
        if ((y_pred_test(i, 0) == 0) && (y_Test(i, 0) == 0)){
            true_negatives++;
        }
        if ((y_pred_test(i, 0) == 1) && (y_Test(i, 0) == 0)){
            false_positives++;
        }
        if ((y_pred_test(i, 0) == 0) && (y_Test(i, 0) == 1)){
            false_negatives++;
        }
    }
    /*Se calcula la "precision" del modelo*/
    double precision = true_positives/(true_positives+false_positives);
    /*Se calcula el "recall" del modelo*/
    double recall = true_positives/(true_positives+false_negatives);
    /* Se calcula y devuelve el f1 score*/
    double f1Score = 2*((precision*recall)/(precision+recall));
    return (f1Score);
}

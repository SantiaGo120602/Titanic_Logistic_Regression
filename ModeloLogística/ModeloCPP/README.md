# **Programa de regresión logística en C++**

* En esta carpeta se encuentra el código fuente de un programa capaz de generar un modelo de regresión lineal para un archivo csv (dataset) usando C++.

* El programa contiene una carpeta llamad build, con el binario "ModeloCpp" sin embargo es posible que intentar ejecutar este binario no sea posible dependiendo
de la arquitectura y el sistema operativo de la máquina que se este usando. Por esto, a continuación se mostrará como construir el proyecto usando CMake.

# **Contrucción del proyecto**

* Lo siguiente es una guía para construir uel proyecto desde su computadora.

* Requerimientos: CMake, MAKE, g++ u otro compilador de C++

* Descargue el proyecto a su computadora. Para ello copie y pegue la dirección de este repositorio en: https://download-directory.github.io/ y presione enter.

* Entre a una terminal y dirigase a la ubicación del proyecto, ingrese a la carpeta que se llama build. Por ejemplo se puede usar:

```
cd ModeloCPP/build
```

* Una vez dentro, elimine todos los archivos existentes:

```
rm -r *
```

* use CMake para generar el archimo make:

```
cmake ..
```
* use make para generar el archivo binario:

```
make
```
* Ahora debería haber un binario dentro de esa misma carpeta llamado ModeloCPP

# **Ejecución del programa**

* Una vez creado el programa, Se procede a mostrar la ejecución. El programa acepta 3 argumentos de linea de consola. El primero es la dirección en la que se encuentra el archvo csv, el segundo es el delimitador que indicará donde se separa cada columna dentro de una fila, el último es un booleano que indica si el archivo contiene o no una cabecera.
* A continuación se muestra un ejemplo de ejecución para el dataset de Titanic_dataset.csv, se recomienda guardar los datasets en la carpeta "Datasets" que se encontraba dentro del proyecto.
```
./ModeloCPP "/home/santiago/ModeloCPP/Datasets/Titanic_dataset.csv" , false
```

* Cabe aclarar que al programa se le pueden agregar cambios comentando o descomentando lineas de código dentro del archivo main.cpp. De este modo se puede ocultar algunas impresiones y guardar los datos obtenidos a archivos de texto plano. Dichos cambios se deben realizar dentro del código fuente, de manera manual.

#**Programa de regresión logística en C++**

* En esta carpeta se encuentra el código fuente de un programa capaz de generar un modelo de regresión lineal para un archivo csv (dataset) usando C++.

* El programa contiene una carpeta llamad build, con el binario "ModeloCpp" sin embargo es posible que intentar ejecutr este binario no sea posible dependiendo
de la arquitectura y el sistema operativo de la máquina que se este usando. Por esto, a continuación se mostrará como construir el proyecto usando CMake.

#**Construcción del proyecto**

* Lo siguiente es una guía para construir uel proyecto desde su computadora.

* Requerimientos: CMake, MAKE, g++ u otro compilador de C++

* Descargue el proyecto a su computadora.

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

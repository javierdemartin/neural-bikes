# Steps

Pasos realizados para entrenar la red neuronal
## Reading Dataset


Reading dataset, data gathered every ten minutes.

```
    datetime   time weekday            station  free_bikes
0        306  00:00  FRIDAY            LEVANTE          12
1        306  00:00  FRIDAY         BLAS OTERO          27
2        306  00:00  FRIDAY       AYUNTAMIENTO          16
3        306  00:00  FRIDAY            ARRIAGA          41
4        306  00:00  FRIDAY         ASKATASUNA          13
5        306  00:00  FRIDAY            REKALDE          15
6        306  00:00  FRIDAY           AREILTZA          16
7        306  00:00  FRIDAY           TERMIBUS          23
8        306  00:00  FRIDAY          ASTILLERO          12
9        306  00:00  FRIDAY           EGUILLOR          15
10       306  00:00  FRIDAY      ANSELMO CLAVE          17
11       306  00:00  FRIDAY           INDAUTXU          22
12       306  00:00  FRIDAY           LEIZAOLA          22
13       306  00:00  FRIDAY          IBAIZABAL          16
14       306  00:00  FRIDAY  PLAZA ENCARNACIÓN          15
15       306  00:00  FRIDAY          SAN PEDRO          15
16       306  00:00  FRIDAY            BOLUETA          17
17       306  00:00  FRIDAY         OTXARKOAGA          17
18       306  00:00  FRIDAY            SARRIKO          24
19       306  00:00  FRIDAY              HEROS          15
20       306  00:00  FRIDAY              EGAÑA          19
21       306  00:00  FRIDAY         ETXEBARRIA           8
22       306  00:00  FRIDAY     GABRIEL ARESTI          20
23       306  00:00  FRIDAY             ABANDO          20
24       306  00:00  FRIDAY    ESTRADA CALEROS          19
25       306  00:00  FRIDAY             EPALZA          22
26       306  00:00  FRIDAY           AMETZOLA          13
27       306  00:00  FRIDAY       SABINO ARANA          15
28       306  00:00  FRIDAY      CORAZÓN MARIA          19
29       306  00:00  FRIDAY            KARMELO          17
30       306  00:10  FRIDAY            LEVANTE          12
31       306  00:10  FRIDAY         BLAS OTERO          27
32       306  00:10  FRIDAY       AYUNTAMIENTO          14
33       306  00:10  FRIDAY            ARRIAGA          41
34       306  00:10  FRIDAY         ASKATASUNA          14
35       306  00:10  FRIDAY            REKALDE          15
36       306  00:10  FRIDAY           AREILTZA          16
37       306  00:10  FRIDAY           TERMIBUS          23
38       306  00:10  FRIDAY          ASTILLERO          12
39       306  00:10  FRIDAY           EGUILLOR          15
```

## Encoding Data


Encode each column as integers
Got list of 36 stations before encoding

```
['ABANDO', 'AMETZOLA', 'ANSELMO CLAVE', 'ARANGOIT', 'ARANGOITI', 'AREILTZA', 'ARRIAGA', 'ASKATASUNA', 'ASTILLERO', 'AYUNTAMIENTO', 'BLAS OTERO', 'BOLUETA', 'CORAZÓN MARIA', 'EGAÑA', 'EGUILLOR', 'EPALZA', 'ESKURTZE', 'ESTRADA CALEROS', 'ETXEBARRIA', 'GABRIEL ARESTI', 'HEROS', 'IBAIZABAL', 'INDAUTXU', 'KARMELO', 'LEIZAOLA', 'LEVANTE', 'MERCADO RIBERA', 'OLABEAGA', 'OTXARKOAGA', 'PLAZA ENCARNACIÓN', 'POLIDEPORTIVO ZORROZA', 'REKALDE', 'SABINO ARANA', 'SAN PEDRO', 'SARRIKO', 'TERMIBUS']
```

## Creating Label Encoders and then encoding the previously read dataset


Hour Encoder (144 values)

```
['00:00' '00:10' '00:20' '00:30' '00:40' '00:50' '01:00' '01:10' '01:20'
 '01:30' '01:40' '01:50' '02:00' '02:10' '02:20' '02:30' '02:40' '02:50'
 '03:00' '03:10' '03:20' '03:30' '03:40' '03:50' '04:00' '04:10' '04:20'
 '04:30' '04:40' '04:50' '05:00' '05:10' '05:20' '05:30' '05:40' '05:50'
 '06:00' '06:10' '06:20' '06:30' '06:40' '06:50' '07:00' '07:10' '07:20'
 '07:30' '07:40' '07:50' '08:00' '08:10' '08:20' '08:30' '08:40' '08:50'
 '09:00' '09:10' '09:20' '09:30' '09:40' '09:50' '10:00' '10:10' '10:20'
 '10:30' '10:40' '10:50' '11:00' '11:10' '11:20' '11:30' '11:40' '11:50'
 '12:00' '12:10' '12:20' '12:30' '12:40' '12:50' '13:00' '13:10' '13:20'
 '13:30' '13:40' '13:50' '14:00' '14:10' '14:20' '14:30' '14:40' '14:50'
 '15:00' '15:10' '15:20' '15:30' '15:40' '15:50' '16:00' '16:10' '16:20'
 '16:30' '16:40' '16:50' '17:00' '17:10' '17:20' '17:30' '17:40' '17:50'
 '18:00' '18:10' '18:20' '18:30' '18:40' '18:50' '19:00' '19:10' '19:20'
 '19:30' '19:40' '19:50' '20:00' '20:10' '20:20' '20:30' '20:40' '20:50'
 '21:00' '21:10' '21:20' '21:30' '21:40' '21:50' '22:00' '22:10' '22:20'
 '22:30' '22:40' '22:50' '23:00' '23:10' '23:20' '23:30' '23:40' '23:50']
```

Weekday Encoder (7 values)

```
['MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY', 'FRIDAY', 'SATURDAY', 'SUNDAY']
```

Station Encoder (36 values)

```
['ABANDO', 'AMETZOLA', 'ANSELMO CLAVE', 'ARANGOIT', 'ARANGOITI', 'AREILTZA', 'ARRIAGA', 'ASKATASUNA', 'ASTILLERO', 'AYUNTAMIENTO', 'BLAS OTERO', 'BOLUETA', 'CORAZÓN MARIA', 'EGAÑA', 'EGUILLOR', 'EPALZA', 'ESKURTZE', 'ESTRADA CALEROS', 'ETXEBARRIA', 'GABRIEL ARESTI', 'HEROS', 'IBAIZABAL', 'INDAUTXU', 'KARMELO', 'LEIZAOLA', 'LEVANTE', 'MERCADO RIBERA', 'OLABEAGA', 'OTXARKOAGA', 'PLAZA ENCARNACIÓN', 'POLIDEPORTIVO ZORROZA', 'REKALDE', 'SABINO ARANA', 'SAN PEDRO', 'SARRIKO', 'TERMIBUS']
```

columns used in the training set

```
['datetime', 'time', 'weekday', 'station', 'free_bikes']
```

Encoded dataset

```
   datetime time weekday station free_bikes
0       306    0       4      25         12
1       306    0       4      10         27
2       306    0       4       9         16
3       306    0       4       6         41
4       306    0       4       7         13
5       306    0       4      31         15
6       306    0       4       5         16
7       306    0       4      35         23
8       306    0       4       8         12
9       306    0       4      14         15
10      306    0       4       2         17
11      306    0       4      22         22
12      306    0       4      24         22
13      306    0       4      21         16
14      306    0       4      29         15
15      306    0       4      33         15
16      306    0       4      11         17
17      306    0       4      28         17
18      306    0       4      34         24
19      306    0       4      20         15
```

## Finding holes in dataset


Los datos son recogidos cada 10' en el servidor y puede que en algunos casos no funcione correctamente y se queden huecos, arreglarlo inventando datos en esos huecos.

| Estación | Missing Samples | Missing Whole Days
| --- | --- | --- |
 | ABANDO | 585 | 0 | 
 | AMETZOLA | 166 | 0 | 
 | ANSELMO CLAVE | 219 | 0 | 
 | ARANGOIT | 28 | 0 | 
 | ARANGOITI | 189 | 0 | 
 | AREILTZA | 279 | 0 | 
 | ARRIAGA | 246 | 0 | 
 | ASKATASUNA | 219 | 0 | 
 | ASTILLERO | 177 | 0 | 
 | AYUNTAMIENTO | 297 | 0 | 
 | BLAS OTERO | 231 | 0 | 
 | BOLUETA | 219 | 0 | 
 | CORAZÓN MARIA | 225 | 0 | 
 | EGAÑA | 313 | 0 | 
 | EGUILLOR | 268 | 0 | 
 | EPALZA | 234 | 0 | 
 | ESKURTZE | 80 | 0 | 
 | ESTRADA CALEROS | 268 | 0 | 
 | ETXEBARRIA | 226 | 0 | 
 | GABRIEL ARESTI | 183 | 0 | 
 | HEROS | 209 | 0 | 
 | IBAIZABAL | 235 | 0 | 
 | INDAUTXU | 184 | 0 | 
 | KARMELO | 245 | 0 | 
 | LEIZAOLA | 217 | 0 | 
 | LEVANTE | 287 | 0 | 
 | MERCADO RIBERA | 91 | 0 | 
 | OLABEAGA | 66 | 0 | 
 | OTXARKOAGA | 285 | 0 | 
 | PLAZA ENCARNACIÓN | 208 | 0 | 
 | POLIDEPORTIVO ZORROZA | 231 | 0 | 
 | REKALDE | 238 | 0 | 
 | SABINO ARANA | 245 | 0 | 
 | SAN PEDRO | 244 | 0 | 
 | SARRIKO | 190 | 0 | 
 | TERMIBUS | 339 | 0 | 



## Scaling dataset


| Values | datetime | time | weekday | station | free_bikes |
| --- | --- | --- | --- | --- | --- |
| Minimum Values | -0.0027472527472527475 | 0.006944444444444444 | 0.0 | 0.0 | 0.0 | 
| Data Max | 365.0 | 143.0 | 6.0 | 35.0 | 42.0 | 
| Data Min | 1.0 | -1.0 | 0.0 | 0.0 | 0.0 | 
| Data Range | 364.0 | 144.0 | 6.0 | 35.0 | 42.0 | 
| Scale | 0.0027472527472527475 | 0.006944444444444444 | 0.16666666666666666 | 0.02857142857142857 | 0.023809523809523808 | 


## Supervised Learning


| Station | Days | 
| --- | --- |
| ABANDO | 33 | 
| AMETZOLA | 30 | 
| ANSELMO CLAVE | 44 | 
| ARANGOIT | 1 | 
| ARANGOITI | 22 | 
| AREILTZA | 31 | 
| ARRIAGA | 28 | 
| ASKATASUNA | 24 | 
| ASTILLERO | 29 | 
| AYUNTAMIENTO | 52 | 
| BLAS OTERO | 38 | 
| BOLUETA | 43 | 
| CORAZÓN MARIA | 25 | 
| EGAÑA | 65 | 
| EGUILLOR | 49 | 
| EPALZA | 25 | 
| ESKURTZE | 24 | 
| ESTRADA CALEROS | 55 | 
| ETXEBARRIA | 38 | 
| GABRIEL ARESTI | 21 | 
| HEROS | 36 | 
| IBAIZABAL | 38 | 
| INDAUTXU | 37 | 
| KARMELO | 43 | 
| LEIZAOLA | 38 | 
| LEVANTE | 79 | 
| MERCADO RIBERA | 24 | 
| OLABEAGA | 17 | 
| OTXARKOAGA | 54 | 
| PLAZA ENCARNACIÓN | 43 | 
| POLIDEPORTIVO ZORROZA | 19 | 
| REKALDE | 25 | 
| SABINO ARANA | 44 | 
| SAN PEDRO | 29 | 
| SARRIKO | 36 | 
| TERMIBUS | 81 | 


## Split datasets


Dividing whole dataset into training 80.0%, validation 15.0% & test 5.0%

| Dataset | Percentage | Samples |
| --- | --- | --- |
| Training | 80.0 | 1056 | 
| Validation | 15.0 | 198 | 
| Test | 5.0 | 66 | 


## Neural Network Training


![Model Shape](model/model.png)

* [20] epochs
* [100] batch size

![Training Acc](plots/training_acc.png)

![Training Loss](plots/training_loss.png)

![Training MAPE](plots/training_mape.png)

![Training MSE](plots/training_mse.png)

![Prediction Sample 1](plots/1.png)

![Prediction Sample 2](plots/2.png)

![Prediction Sample 3](plots/3.png)

More prediction samples in [plots/](https://github.com/javierdemartin/neural-bikes/tree/master/plots).

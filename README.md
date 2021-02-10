## Введение

С развитием области компьютерного зрения разрабатываются всё более и более сложные модели требующие большое количество вычислительных ресурсов. Так например, созданная в 1998 году Яном Лекуном модель LeNet-5 содержала менее 1 миллиона параметров и решала задачу классификации рукописных цифр. В 2012 году сеть победившая на соревновании распознованию изображения содержала уже 60 миллионов параметров. Для задач распознавания человеческих лиц требуется уже более 120 миллионов параметров. Ьаким образом, возникает необходимость способов уменьшения числа параметров без существенной потери качества предсказаний. 

В связи с этим актуальной задачей становится сжатие свёрточных нейронных сетей и одним из методов успешно применяющися для решения этой задачи является pruning. Этот метод представляет собой удаление параметров нейронной сети в соответсвии с определённым алгоритмом. Сначала нейронная сеть обучается, после чего удаляются параметры не влияющие на процесс обучения (т. е. ближкие к нулю веса нейронной сети). После этого уже сжатая модель снова обучается для получения более точного значения параметров. В работе [Han2015_Learning bothWeights and Connections for Efficient] реализованы эксперименты по сжатию нейронных сетей AlexNet и VGG-16. Авторы показали, что 

Выделяют два метода prunning:
* <b>Prunning весов (Weight pruning)</b>
* <b>Unit/Neuron pruning</b>

В первом методе определённые веса в матрице весов принимаются равными нулю, что означает удаление связей между нейронами (см. рис. 1). Для того чтобы определить какие связи между нейронами нужно удалить для достижения разреженности матрицы k% мы располагаем веса из матрицы весов в порядке возрастания (по абсолютному значению)и заменяем на "0" k% наименьших весов.

В случае Unit/Neuron pruning мы устанавливаем равными нулю столбцы матрицы весов тем самым фактически удаляя соответствующий выходной нейрон. В это случаедля достижения разреженности k% (т. е. удаления k% узлов) мы ранжируем столбцы матрицы весов в соответствии с их $L_2$ нормой и удаляем наименьшие k% столбцов. Схематически описанные два метода показаны на рис. 1.

![prunning.png](https://github.com/Svetlana19/Prunning/blob/main/images/prunning.png) Рис. 1. Методы prunning. Источник: https://medium.com/@yasersakkaf123/pruning-neural-networks-using-pytorch-3bf03d16a76e

Если проводить аналогию с нейроннами головного мозга, то Prunning весов можно сравнить с удаление синапсов, а Unit/Neuron pruning с удалением самих нейронов.

## Построение модели

В качестве модели для обучения в соответствии с заданием была выбрана архитектура ResNet20. В ходе обучения были использованы оптимизаторы Adam, SGD и Adamax.  Результаты обучения существенно не различаются. ВО всех случаях заметна необходимость понижения rearning rate и увеличения числа эпох обучения.
![resnet20_Adam_lr%3D0.0005.png](https://github.com/Svetlana19/Prunning/blob/main/results/resnet20_Adam_lr%3D0.0005.png)
![resnet20_Adamax_lr%3D0.002_betas%3D(0.9%2C%200.999)_eps%3D1e-08_weight_decay%3D0.png](https://github.com/Svetlana19/Prunning/blob/main/results/resnet20_Adamax_lr%3D0.002_betas%3D(0.9%2C%200.999)_eps%3D1e-08_weight_decay%3D0.png)
![resnet20_SGD_lr%3D0.01_momentum%3D0.9.png](https://github.com/Svetlana19/Prunning/blob/main/results/resnet20_SGD_lr%3D0.01_momentum%3D0.9.png)

## Результаты

К сожалению, результаты для анализа по prunning c использованием кластеризации не получены.

Предложения по поводу алгоритмов prunning:

* Выбор весов из матрицы исользуя некую аналогию с пуллингом (выбираются N весов, например, 4 выбирается наибольшее значение оно остаётся, а остальные удаляются). Это будет быстрее чем классеризация, но возможно ухудшение результата будет незначительным.

* Удаление весов из матрицы исходя из статистического распределения весов (здесь дополнительно потребуется округление значений весов до целых или десятых). 

* Использание одновременно prunning весов и Unit/Neuron pruning.

* Использование кластеризации в самой нейронной сети по аналогии с пуллингом. Но возможно, это будет слишком затратным вариантом.

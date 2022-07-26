# Ссылка на оригинальный репозиторий

https://github.com/seorim0/DCCRN-with-various-loss-functions.git

# Клонирование репозитория
```
git clone https://github.com/ShJacub/DCCRN.git
```

# Создание и запуск docker-контейнера
```
docker create -v absolute_DCCRN_directory:/DCCRN -v absolute_path_to_dataset:/datasets --runtime=nvidia -it -p 1111:1111 --ipc=host --name=My_Docker floydhub/pytorch:1.3.0-gpu.cuda10cudnn7-py3.52 /bin/bash

docker start My_Docker

docker exec -it My_Docker /bin/bash
```

# Установка библиотек
```
cd /DCCRN
pip install -r requirements.txt
```

# Перевод набора данных в wav формат
```
cd /DCCRN

python MelToWav.py
```

# Обучение
```
python trainer.py
```

# Подсчёт метрики MSE
```
cd /DCCRN

python my_tester.py path_to_weights

python MSE_calc.py 
```
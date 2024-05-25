# Treinamento de Deep Learning - Detecção de Objetos - Yolo V8

## Códigos Base

- [Divisão do Dataset](/Códigos%20Originais%20e%20Dados/split_data.py) <br>
- [Transformação de Formato](/Códigos%20Originais%20e%20Dados/coco2yolo.py) <br>
- [Treinamento]() <br>
- [Embarcado na Jetson](/Códigos%20Originais%20e%20Dados/detector_infer_jetson.py) <br>
- [Visualização do Processo](/Códigos%20Originais%20e%20Dados/vision_setup.sh) <br>

## Dados

- [Dataset](https://drive.google.com/drive/folders/1zUIphpnM9JvzkhCJYgjFE5hVZhL5BLSh?usp=drive_link)

## Aulas Desenvolvidas

- [Aula 1: Divisão do Dataset](/Aulas/Aula%201.ipynb) <br>
- [Aula 2: Transformação do Formato - COCO para Yolo](/Códigos%20Originais%20e%20Dados/coco2yolo.py) <br>
- [Aula 3: Treinamento](/Aulas/Aula%203.ipynb) <br>
- [Aula 4: Embarcando na Jetson](/Aulas/Aula%204.ipynb) <br>
- [Aula 5: Visualização do processo](/Aulas/Aula%205.ipynb) <br>

## Requesitos

Para executar as aulas, basta criar um ambiente conda com as seguintes dependências.

```bash 
conda create --name object_detection_amb python=3.x  # specify the Python version you need
conda activate object_detection_amb
pip install -r requirements.txt
```



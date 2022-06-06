# potato-leaf-diseases-detection
SCC0276 - Aprendizado de Máquina
Trabalho Final - Entrega 2

Alunos:
Marianna Karenina de A. Flôres - 10821144
Rodrigo Bragato Piva - 10684573
Vinícius Ribeiro da Silva - 10828141

Descrição dos Arquivos:

bwTrain.py
-> Arquivo que carrega as imagens no formato Preto e Branco e cria o modelo de Rede Neural;

evaluate.py
-> Carrega os modelos KNN e Redes densas, mostrando as respectivas avaliações;

extractFeatures.py
-> Extrai as Features dos dados de imagens usando a rede VGG16 treinada em imagenet;

LearningWithFeatures.py
-> Aprendizado de uma rede neural densa e do modelo KNN, com as Features;

remoce_repeats.py
-> Remoção das imagens duplicadas e armazenamento como imagens RGB e Preto e Branco;

Descrição das pastas:

Pasta Imgs 
-> Armazena as imagens pré-processadas
subdivisão Imgs/BW -> Armazena as imagens Preto e Branco;
subdivisão Imgs/RGB -> Armazena as imagensno formato RGB;
subdivisão Imgs/VGG16 -> Armazena Features das imagens;

Pasta Modes
-> Armazena os arquivos das redes criadas;



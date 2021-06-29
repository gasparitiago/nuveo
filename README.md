# Testes Práticos Nuveo

## 01-WheresWally

Para este teste, foram propostas duas soluções diferentes, ambas implementadas dentro do diretório `01-WheresWally`, uma delas baseada em técnicas tradicionais de visão computacional e outra baseada em redes neurais (como definido no enunciado).

Particularmente, eu resolveria este problema utilizando uma abordagem tradicional, sem utilização de redes neurais, por alguns motivos explicados abaixo.

### Solução utilizando *Feature Maching*

Para a solução deste problema, eu não utilizaria redes neurais, uma solução elegante, simples e robusta para este problema é utilizar métodos de casamento de características, conhecidos como *feature maching*, no qual pontos da imagem de referência são buscados em novas imagens para identificar onde a imagem de referência se encontra. Esse tipo de método é bastante utilizado para resolver o problema em questão, onde a imagem buscada é sempre a mesma, com apenas alterações na posição, rotação, escala e mudanças de perspectivas (transformações rígidas).

Vantagens desta solução em relação à utilização de redes neurais:
- É uma abordagem mais simples, mais rápida de ser implementada e mais rápida de ser executada.
- Não depende de um dataset de treinamento, nem de modelos pré-treinados, apenas de uma imagem de referência.
- Pode ser implementada facilmente em linguagens de mais baixo nível, como C++, para *deploy* em dispositivos móveis ou sistemas embarcados, considerando poder de processamento e memória limitados.
- Como a imagem de referência possui *features* facilmente detectáveis (bordas bem definidas e com bom contraste), essa abordagem produz bons resultados quando aplicadas as técnicas corretas.
- É possível implementar a solução utilizando técnicas de código aberto, com licenças permissivas e que podem ser utilizadas em sistemas comerciais.

A solução foi implementada utilizando OpenCV, escrita em Python3, no script `find_wally_feature_matching.py`. Os métodos não precisaram ser tunados e devido ao pouco tempo para implementação, os parâmetros padrão de um *matcher* de força bruta com RANSAC foi utilizado.

O resultado desta abordagem pode ser encontrado no arquivo `results_feature_matching.py`.

Essa solução só é possível pois sempre o mesmo *template* é utilizado. Caso o problema fosse reconhecer imagens como nos livros tradicionais de "Onde está o Wally?", onde o Wally fica em diferentes poses, em cenários onde outros personagens possuem características semelhantes ao Wally buscado, uma abordagem utilizando redes neurais é mais interessante.

Como o enunciado pede que a solução para esse problema utilize redes neurais, realizei a implementação da solução desta forma também, mesmo acreditando que esse tipo de abordagem, para a resolução deste problema específico não seja necessária.

### Solução utilizando redes neurais*

A segunda solução para este problema foi utilizando redes neurais, mais precisamente uma rede de segmentação de imagens e detecção de objetos, que utiliza uma arquitetura Mask R-CNN e *transfer learning*.

Para utilização desta abordagem, o conjunto de treinamento disponibilizado foi utilizado e esses dados passaram por 2 processamentos diferentes: a preparação/limpeza do dataset e o treinamento de um modelo customizado para resolver o problema.

A preparação do dataset é realizada pelo script `prepare_dataset.py` e é responsável tanto pela limpeza do dataset (removendo anotações incompletas) e também é responsável por facilitar a visualização das demais anotações.

Após a limpeza das anotações incompletas, eu analisei as imagens manualmente para remover possíveis anotações incorretas. Nesse processo manual, selecionei apenas uma imagem do conjunto de treinamento para remoção. Após remoção, o novo dataset com as imagens que serão de fato utilizadas foi criado e organizado no diretório `TrainingSetClean` que também está disponibilizado neste repositório.

Após isso, uma rede baseada na arquitetura Mask RCNN foi treinada utilizando a interface Pytorch TorchVision. Os detalhes de implementação estão descritos no arquivo `Wally_PyTorch.ipynb` que pode ser executado utilizando Google Colab ou localmente.

Por se tratar de um problema simples, não foram exercitados melhorias no modelo proposto inicialmente, não foram utilizadas estratégias de *data augmentation* e de busca pelos melhores parâmetros durante a etapa de treinamento, visto que a acurácia do modelo durante a validação foi acima de 97% na métrica IOU.

Os resultados da inferência deste modelo no conjunto de testes está no arquivo `results_cnn_test.csv`.

Resolvi utilizar Pytorch pois é uma ferramenta que eu nunca havia utilizado para realizar esse tipo de treinamento (apenas Tensorflow e Keras), mas foi algo bem simples de aprender.

## 02-SMSSpamDetection

Foram criados dois arquivos: `spam_detector.py`, contendo a classe principal solicitada no enunciado e `unit_test` que contém testes unitários que utilizam a classe criada.

Para executar os unit_tests, é necessário executar o script:
```
    python3 -m unittest unit_test.py
```

#### Opções de limiares para considerar uma mensagem como SPAM ou não

No enunciado, há a seguinte instrução:

> Para construção do método is_spam , considere um cenário real em que os usuários finais possa escolher uma entre duas opções:
1- que nenhuma mensagem de spam chegue em sua caixa de entrada, mesmo com algumas ham sendo classificadas como spam
2- que nenhuma mensagem ham vá para caixa de spam, mesmo que algumas spam cheguem na caixa de entrada

A única forma de assegurar com 100% de certeza, de que as opções 1 e 2 estão, de fato, sendo atendidas é:
- 1: Classificar todas as mensagens como SPAM.
- 2: Classificar todas as mensagens como HAM.

Logicamente, essas soluções não podem ser utilizadas em uma solução deste tipo, pois o modelo nem está sendo utilizado.

O que pode ser feito para que essas duas opções sejam atendidas com uma melhor acurácia que o modelo original, é a utilização de um threshold dentro da função is_spam para considerar diferentes modos e retornar que uma mensagem é considerada SPAM a partir deste threshold (apenas quando a probabilidade é muito alta), e vice-versa.

Para a obtenção destes 2 thresholds (um para evitar falsos positivos e um para evitar falsos negativos), eu fiz uma análise da curva de precision e recall, que está disponível no arquivo `SMS_PrecisionRecall`. Em outras palavras, esse script foi apenas utilizado para determinar os thresholds e esses valores foram utilizados dentro da classe criada anteriormente.

Como esse funcionamento depende apenas de um threshold, não é necessário retreinar um modelo novo e nem utilizar múltiplos modelos para as diferentes opções solicitadas.
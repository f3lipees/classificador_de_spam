# Funcionalidades Principais

- Classificação precisa de mensagens em spam ou ham com base em texto livre.
- Pré-processamento linguístico com remoção de caracteres especiais e stopwords específicas do português brasileiro.
- Vetorização otimizada para capturar nuances lexicais e semânticas do idioma.
- Treinamento com divisão estratificada dos dados para evitar viés.

# Requisitos

- Python 3.8 ou superior
- Corpus de stopwords do NLTK para português (nltk.download('stopwords'))

# Instalação

- Clone o repositório:

      git clone https://github.com/f3lipees/classificador_de_spam.git

cd classificador_de_spam

- Instale as dependências:

      pip install -r requirements.txt

- Baixe as stopwords do NLTK:
  
      import nltk
      nltk.download('stopwords')

- Próximo passo é preparar seu arquivo CSV com mensagens para treinamento, contendo duas colunas, a ***label (com valores ham ou spam)*** e ***message (texto da mensagem).***

- Feito isso, execute o script principal para treinar o modelo e realizar classificações:

                 python classificador.py



Este projeto foi inspirado em [ferramenta](https://github.com/NisaarAgharia/SMS-Spam-Classification)

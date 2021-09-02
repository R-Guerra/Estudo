# -*- coding: utf-8 -*-
"""Criação e Manipulação de uma Planilha do Excel com a Biblioteca Pandas.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Wi8eyoSFyv2xEtpLC-sO-rCHphQlsiMm

Criação e Manipulação de uma Planilha do Excel por meio da Biblioteca Pandas.

A) Por meio de um notebook, ou do prompt do Python (é necessário já ter instalado o Python no seu computador) vamos importar a biblioteca Pandas, dica use o comando “import pandas as pd”, por exemplo;
"""

import pandas as pd

"""B) Agora com a planilha salva em uma pasta conhecida do seu computador vamos ler os dados para uma variável que podemos chamar de df, por exemplo, em referência ao seu tipo, Data Frame, ou outro nome a sua escolha, dica, use o comando “df = read_excel (<pasta_e_nome_do_arquivo>)”; 
-- Tive problemas para carregar o arquivo dessa forma, então fiz o upload de uma forma diferente --
"""

from google.colab import files
uploaded = files.upload()

import io
df = pd.read_excel(io.BytesIO(uploaded['Tabela Trabalho.xlsx']))

"""C) Agora vamos ler as 5 primeiras e as 3 últimas linhas da planilha com usando os métodos head e tail do Pandas;"""

df.head(5)

df.tail(3)

"""D) Descubra o método que nos mostra um relatório resumido da nossa planilha, neste momento, do nosso data frame e anote a saída em seu relatório;"""

df.count

"""E) Agora vamos adicionar o próprio data frame a atual data frame, e vamos conferir que seu tamanho realmente duplicou como de esperado, após relatar isso, podemos eliminar as duplicações, dica de comando, vejam: append, shape e drop_duplicates;"""

df.shape

df.append(df)

df.drop_duplicates()

"""F) Vamos agora imprimir os nomes atuais das colunas, com o comando columns, e depois renomear algumas destas colunas, com o comando rename, e imprimir novamente os nomes das colunas para termos a certeza que foram alterados com o comando da forma que queríamos;"""

df.columns

df.rename(columns={
    'Cadastro' : 'Cad.',
    'Telefone' : 'Tel.'
}, inplace=True)

df.columns

"""G) Agora seguindo a apostila acrescente mais três comandos usando métodos do Pandas e anote as saídas no relatório que deverá ser enviado nesta tarefa."""

df.columns = [col.upper() for col in df]

df.columns

df.info()

df.count()

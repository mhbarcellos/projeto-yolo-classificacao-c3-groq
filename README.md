# Reconhecimento de Objetos com YOLO, OpenCV e Análise Generativa com Groq

Este projeto implementa uma aplicação de visão computacional para reconhecimento de objetos em tempo real, utilizando **Python**, **YOLO**, **OpenCV** e **CustomTkinter**. Na versão C3, o resultado gerado pela C2 também é enviado a uma LLM pela plataforma **Groq**, que devolve uma análise interpretativa ao usuário.

A solução foi desenvolvida como parte de uma atividade acadêmica de reconhecimento de objetos, com o objetivo de demonstrar a implementação prática de um algoritmo de detecção, o funcionamento em tempo real e a organização técnica do projeto.

## Objetivo

O objetivo do projeto é reconhecer objetos em tempo real a partir da câmera do computador, exibindo as detecções na tela com caixas delimitadoras, nome da classe identificada, confiança da predição e estatísticas da sessão.

Além da detecção em si, o sistema também gera evidências da execução, como prints dos frames processados, logs em CSV e resumo técnico em JSON. Esse resumo técnico é serializado e usado como entrada para a análise generativa da C3.

## Algoritmo escolhido

O algoritmo escolhido foi o **YOLO**, sigla para **You Only Look Once**.

O YOLO é um algoritmo de detecção de objetos muito utilizado em aplicações de tempo real, pois processa a imagem em uma única passagem pela rede neural. A partir de cada frame capturado pela câmera, o modelo retorna:

- classe do objeto detectado;
- confiança da predição;
- coordenadas da caixa delimitadora;
- localização do objeto na imagem.

Essa abordagem torna o YOLO adequado para aplicações com câmera, monitoramento, automação visual e demonstrações em tempo real.

## Tecnologias utilizadas

- Python
- OpenCV
- Ultralytics YOLO
- NumPy
- Pillow
- CustomTkinter
- Groq API
- python-dotenv

## Funcionalidades

O projeto possui as seguintes funcionalidades:

- detecção de objetos em tempo real;
- captura de vídeo pela webcam;
- suporte à execução via terminal;
- app desktop com interface gráfica;
- exibição de bounding boxes;
- exibição da classe detectada;
- exibição da confiança da predição;
- mini-dashboard com estatísticas da sessão;
- contagem de frames processados;
- cálculo de FPS;
- contagem acumulada por classe;
- salvamento de prints;
- geração de logs em CSV;
- geração de resumo técnico em JSON;
- envio do resumo da C2 para uma LLM via Groq;
- construção de prompt com system prompt e user prompt;
- salvamento do prompt enviado à LLM;
- salvamento da análise interpretativa em JSON;
- registro de latência e tokens consumidos quando disponíveis;
- cache de respostas para entradas repetidas;
- tratamento de erros;
- log de debug;
- geração de executável para Windows.

## Estrutura do projeto

```text
projeto_yolo_reconhecimento_objetos/
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── dashboard.py
│   ├── desktop_app.py
│   ├── detector.py
│   ├── llm_analyzer.py
│   ├── session_logger.py
│   ├── utils.py
│   └── video_stream.py
├── assets/
│   └── sample_objects.txt
├── outputs/
│   ├── logs/
│   └── prints/
├── main.py
├── run_app.py
├── analisar_ultima_sessao.py
├── build_exe.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Organização dos módulos

O projeto foi organizado de forma modular, separando responsabilidades entre os arquivos.

### `app/detector.py`

Responsável pelo carregamento do modelo YOLO e pela execução da inferência nos frames capturados.

### `app/video_stream.py`

Responsável pela abertura da câmera ou fonte de vídeo e pela leitura dos frames.

### `app/dashboard.py`

Responsável por desenhar as informações visuais no frame, incluindo caixas delimitadoras, labels, FPS, tempo de sessão, total de detecções e classes mais detectadas.

### `app/session_logger.py`

Responsável por registrar as detecções em CSV, salvar prints e gerar o resumo técnico da sessão em JSON.

### `app/llm_analyzer.py`

Responsável pela camada da C3. Esse módulo lê o resumo JSON produzido pela C2, padroniza os dados em um payload estruturado, constrói o prompt, chama a API do Groq, trata falhas e salva a análise final em `outputs/logs`.

### `app/desktop_app.py`

Responsável pela interface gráfica desktop desenvolvida com CustomTkinter.

### `app/utils.py`

Contém funções auxiliares utilizadas pelo sistema.

### `main.py`

Permite executar o sistema pelo terminal.

### `run_app.py`

Inicializa o app desktop.

### `build_exe.py`

Gera o executável Windows do projeto.

## Classes reconhecidas

O projeto utiliza um modelo YOLO pré-treinado no dataset COCO. Por padrão, foram utilizadas classes comuns e fáceis de demonstrar, como:

- pessoa;
- garrafa;
- celular;
- notebook;
- livro;
- copo;
- cadeira;
- teclado;
- mouse;
- mochila;
- controle remoto.

O mouse é uma classe nativa do dataset COCO e pode ser detectado diretamente.

Tablet e Kindle não são classes nativas do modelo COCO utilizado. Por isso, esses objetos podem ser confundidos com celular, notebook ou livro, dependendo do formato e do ângulo. Para identificar tablet e Kindle especificamente, seria necessário treinar um modelo customizado com imagens rotuladas ou utilizar outro tipo de modelo.

## Instalação

Clone o repositório ou abra a pasta do projeto no terminal.

Crie um ambiente virtual:

```powershell
python -m venv .venv
```

Ative o ambiente virtual:

```powershell
.\.venv\Scripts\activate
```

Instale as dependências:

```powershell
pip install -r requirements.txt
```

Configure a chave da Groq:

```powershell
copy .env.example .env
```

Depois edite o arquivo `.env` e preencha:

```text
GROQ_API_KEY=gsk_sua_chave_aqui
GROQ_MODEL=llama-3.3-70b-versatile
```

O arquivo `.env` não deve ser enviado ao GitHub.

## Execução pelo terminal

Para executar o projeto pelo terminal:

```powershell
python main.py
```

Também é possível definir classes específicas:

```powershell
python main.py --classes "person,bottle,cell phone,laptop,book,cup,chair,mouse"
```

Para alterar a confiança mínima:

```powershell
python main.py --conf 0.35
```

Para detectar todas as classes disponíveis no modelo:

```powershell
python main.py --show-all
```

Para executar sem chamar a LLM ao encerrar:

```powershell
python main.py --no-llm
```

Para escolher outro modelo Groq na execução:

```powershell
python main.py --llm-model llama-3.1-8b-instant
```

Para visualizar os argumentos disponíveis:

```powershell
python main.py --help
```

## Execução pelo app desktop

Para abrir a interface gráfica:

```powershell
python run_app.py
```

Para abrir em modo debug:

```powershell
python run_app.py --debug
```

No app desktop, é possível:

- iniciar a câmera;
- parar a execução;
- salvar print;
- ligar ou desligar logs;
- abrir a pasta de saída;
- acompanhar estatísticas em tempo real;
- encerrar a aplicação.

## Atalhos

Durante a execução do app desktop, os atalhos disponíveis são:

```text
S    salvar print
L    ligar/desligar logs
Q    sair
ESC  sair
```

## Arquivos gerados

Durante a execução, o projeto gera arquivos na pasta `outputs`.

### Prints

Os prints dos frames processados são salvos em:

```text
outputs/prints/
```

### Logs CSV

As detecções são registradas em:

```text
outputs/logs/detections.csv
```

O CSV contém informações como:

```text
data_hora;frame;classe_original;classe_pt;confianca;x1;y1;x2;y2
```

### Resumo JSON

Ao encerrar a sessão, o sistema gera:

```text
outputs/logs/session_summary.json
```

Esse arquivo contém informações como:

- data e hora de início;
- data e hora de fim;
- duração da sessão;
- modelo utilizado;
- fonte de vídeo;
- confiança mínima;
- total de frames processados;
- FPS médio;
- total de detecções;
- classes detectadas;
- quantidade por classe;
- confiança média;
- prints salvos;
- status dos logs.

### Análise LLM da C3

Ao encerrar a execução, o resumo bruto da C2 é usado para gerar uma análise interpretativa por LLM. Os arquivos são salvos em:

```text
outputs/logs/llm_prompt_<hash>.txt
outputs/logs/llm_analysis_<data_hora>.json
```

O arquivo de análise contém:

- status da chamada;
- modelo utilizado;
- latência;
- tokens consumidos, quando a API retornar essa informação;
- texto final produzido pela LLM;
- payload enviado para análise;
- erro controlado, caso a chamada falhe.

Também é possível analisar a última sessão sem abrir a câmera novamente:

```powershell
python analisar_ultima_sessao.py
```

### Debug log

Em modo debug, o sistema registra eventos em:

```text
outputs/logs/debug_app.log
```

Esse arquivo auxilia na identificação de problemas de câmera, interface, logs ou encerramento da aplicação.

## Integração da C3 com a LLM

A camada generativa foi acoplada ao final do pipeline da C2. O fluxo implementado é:

1. O YOLO processa os frames e registra as detecções.
2. O `SessionLogger` consolida os resultados em `outputs/logs/session_summary.json`.
3. O módulo `llm_analyzer.py` lê esse JSON e cria um payload padronizado com contexto, métricas e resultado bruto.
4. O payload é serializado com `json.dumps(..., ensure_ascii=False, indent=2)`.
5. O sistema constrói um `system prompt` com o papel da LLM e um `user prompt` com os dados da C2.
6. A API do Groq é chamada com temperatura baixa e limite de tokens.
7. A resposta da LLM é exibida no terminal ou no log da interface desktop e também é salva em JSON.

O prompt solicita uma resposta com cinco partes: resumo da execução, padrões observados, anomalias ou limitações, recomendações técnicas e conclusão objetiva. A temperatura padrão é `0.2`, adequada para uma análise mais controlada e menos criativa.

A chamada à LLM é protegida por tratamento de erros. Se a chave `GROQ_API_KEY` estiver ausente, se a biblioteca `groq` não estiver instalada ou se ocorrer erro de rede, o sistema principal não quebra. Nesses casos, o projeto salva um arquivo de análise com status de falha controlada.

## Geração do executável

Para gerar o executável Windows:

```powershell
python build_exe.py
```

Após a geração, os arquivos ficam em:

```text
dist/
```

Executável principal:

```text
dist/ReconhecimentoYOLO.exe
```

Executável com console de debug:

```text
dist/ReconhecimentoYOLO_DEBUG.exe
```

## Como demonstrar o funcionamento

Para demonstrar o projeto em vídeo:

1. Abrir o projeto no editor de código.
2. Mostrar rapidamente a estrutura dos arquivos.
3. Abrir o app desktop.
4. Iniciar a câmera.
5. Mostrar objetos como pessoa, garrafa, celular, notebook, livro ou mouse.
6. Exibir as caixas delimitadoras e a confiança da predição.
7. Salvar um print.
8. Encerrar a sessão para gerar o resumo JSON.
9. Mostrar a análise interpretativa retornada pela LLM no terminal ou no log do app.
10. Abrir a pasta `outputs`.
11. Mostrar os prints, o CSV, o resumo JSON da C2, o prompt salvo e o JSON da análise LLM.

O vídeo da entrega pode ser gravado externamente, usando OBS, Xbox Game Bar ou outro gravador de tela.

## Relação com a proposta acadêmica

O projeto atende aos principais critérios da atividade.

### Funcionamento do algoritmo

A solução executa detecção de objetos em tempo real usando YOLO e reconhece múltiplas classes do dataset COCO.

### Qualidade técnica

O código foi organizado em módulos, separando responsabilidades entre detecção, captura de vídeo, dashboard, logs, interface e utilitários.

### Demonstração prática

A aplicação permite visualizar a detecção em tempo real, salvar prints e gerar evidências da execução.

### Documentação

O projeto possui README com instruções de instalação, execução, estrutura do projeto, funcionalidades, limitações, integração com Groq, gestão de credenciais e forma de demonstração.

### Integração generativa

A saída bruta da C2 é enviada a uma LLM via Groq, e a análise gerada é integrada de volta ao sistema por terminal, interface desktop e arquivo JSON.

## Limitações

O modelo utilizado é pré-treinado no dataset COCO. Dessa forma, ele reconhece apenas classes presentes nesse dataset.

A qualidade da detecção pode variar de acordo com:

- iluminação do ambiente;
- qualidade da câmera;
- distância do objeto;
- ângulo de visualização;
- oclusão parcial do objeto;
- similaridade entre objetos;
- classe estar ou não presente no dataset COCO.

## Melhorias futuras

Algumas melhorias possíveis para versões futuras:

- treinar um modelo customizado com classes específicas;
- incluir classes como tablet e Kindle por meio de dataset próprio;
- utilizar GPU para melhorar desempenho;
- integrar os registros a um banco de dados;
- criar relatórios automáticos;
- aplicar a solução em monitoramento industrial, educacional ou portuário;
- disponibilizar uma versão web ou embarcada.

## Observação sobre arquivos ignorados

Arquivos gerados automaticamente, ambientes virtuais, executáveis, pesos do modelo e vídeos de gravação de tela não devem ser versionados no GitHub.

Exemplos de arquivos que devem permanecer fora do repositório:

```text
.venv/
outputs/
dist/
build/
*.spec
*.pt
*.mp4
*.avi
*.mov
*.mkv
```

O vídeo de apresentação deve ser enviado separadamente na plataforma indicada pelo professor, não dentro do repositório.

## Conclusão

O projeto não se limita a executar um modelo YOLO pré-treinado. Ele transforma o algoritmo em uma aplicação completa de visão computacional, com interface desktop, dashboard, logs, prints, resumo técnico da sessão e tratamento de limitações.

A solução demonstra o funcionamento prático de reconhecimento de objetos em tempo real e pode servir como base para aplicações mais avançadas de monitoramento, automação e análise visual.
# Relatório técnico da C3: integração do YOLO com LLM via Groq

## 1. Objetivo

O projeto da C3 adiciona uma camada de inteligência generativa ao sistema desenvolvido na C2. A C2 realiza reconhecimento de objetos com YOLO, registra as detecções e gera um resumo técnico da sessão. Na C3, esse resumo é enviado a uma LLM pela API da Groq, e a resposta gerada é incorporada ao sistema como uma análise interpretativa para o usuário.

## 2. Pipeline implementado

O fluxo completo ficou organizado da seguinte forma:

1. O usuário executa o sistema pelo terminal (`main.py`) ou pela interface desktop (`run_app.py`).
2. O YOLO processa os frames da câmera, vídeo ou URL.
3. As detecções são registradas em CSV pelo módulo `SessionLogger`.
4. Ao encerrar a execução, o sistema consolida as métricas em `outputs/logs/session_summary.json`.
5. O módulo `app/llm_analyzer.py` lê esse JSON e cria um payload estruturado.
6. O payload é serializado em JSON com `json.dumps(..., ensure_ascii=False, indent=2)`.
7. O sistema monta o prompt da LLM com um `system prompt` e um `user prompt`.
8. A API do Groq é chamada com temperatura baixa e limite de tokens.
9. A análise gerada é exibida ao usuário e salva em `outputs/logs/llm_analysis_<data_hora>.json`.
10. O prompt usado também é salvo em `outputs/logs/llm_prompt_<hash>.txt`.

## 3. Resultado da C2 enviado à LLM

A saída da C2 usada pela C3 é o arquivo `session_summary.json`, que contém dados como:

- data e hora de início e fim;
- duração da sessão;
- modelo YOLO utilizado;
- fonte de vídeo;
- confiança mínima configurada;
- classes configuradas;
- total de frames processados;
- FPS médio;
- total de detecções;
- classes detectadas;
- quantidade por classe;
- confiança média;
- prints salvos;
- vídeo gerado, quando houver;
- status dos logs.

Antes de ser enviada à LLM, essa saída é encapsulada em um payload com contexto adicional, informando que a origem dos dados é um sistema de reconhecimento de objetos com YOLO e que as classes são limitadas ao modelo pré-treinado utilizado.

## 4. Engenharia de prompt

A integração usa dois tipos de prompt.

O `system prompt` define o papel da LLM como especialista em visão computacional, robótica e sistemas inteligentes. Ele também impõe uma restrição importante: a LLM deve usar apenas os dados enviados e não deve inventar medições, classes ou eventos.

O `user prompt` contém o payload JSON da C2 e solicita uma resposta com o seguinte formato:

- Resumo da execução
- Padrões observados
- Anomalias ou limitações
- Recomendações técnicas
- Conclusão objetiva

A temperatura padrão é `0.2`, pois a tarefa é analítica e exige uma resposta mais controlada. O limite padrão de resposta é `900` tokens, evitando análises longas demais.

## 5. Integração com Groq

A chamada à API está implementada no arquivo `app/llm_analyzer.py`, na função `analyze_summary_with_groq()`.

O modelo padrão é:

```text
llama-3.3-70b-versatile
```

Esse modelo pode ser alterado por variável de ambiente:

```text
GROQ_MODEL=llama-3.1-8b-instant
```

Também pode ser alterado na execução pelo terminal:

```powershell
python main.py --llm-model llama-3.1-8b-instant
```

A chave da API é lida pela variável de ambiente `GROQ_API_KEY`, carregada a partir do arquivo `.env` com `python-dotenv`.

## 6. Tratamento de erros

A chamada à LLM foi implementada para não quebrar o sistema principal. Se ocorrer algum problema, a aplicação continua funcionando e registra a falha de forma controlada.

Situações tratadas:

- ausência da variável `GROQ_API_KEY`;
- biblioteca `groq` não instalada;
- erro de leitura do JSON da C2;
- falha na chamada à API;
- erro de rede ou timeout;
- resposta vazia da LLM.

Quando há falha, o sistema salva um arquivo `llm_analysis_<data_hora>.json` com `success=false`, status da falha e mensagem explicativa.

## 7. Cache

O sistema calcula um hash SHA-256 do payload enviado à LLM. Se a mesma entrada for analisada novamente com o mesmo modelo, a resposta pode ser reaproveitada a partir do cache salvo em `outputs/logs/llm_cache`. Isso reduz custo, evita chamadas repetidas e melhora a velocidade em testes.

## 8. Apresentação ao usuário

A análise da LLM é apresentada de três formas:

1. No terminal, ao executar `python main.py`.
2. No log da interface desktop, ao executar `python run_app.py`.
3. Em arquivo JSON salvo na pasta `outputs/logs`.

Também é possível gerar a análise da última sessão sem abrir a câmera novamente:

```powershell
python analisar_ultima_sessao.py
```

## 9. Gestão de credenciais

O projeto usa `.env` para armazenar a chave da Groq. O arquivo `.env` está no `.gitignore` e não deve ser enviado ao GitHub.

O repositório contém apenas o arquivo `.env.example`, que serve como modelo:

```text
GROQ_API_KEY=gsk_sua_chave_aqui
GROQ_MODEL=llama-3.3-70b-versatile
```

## 10. Conclusão

A C3 foi integrada ao projeto da C2 sem substituir o pipeline original. O sistema continua realizando detecção de objetos com YOLO, mas agora também transforma o resumo técnico da execução em uma análise interpretativa gerada por LLM. A solução atende aos requisitos de serialização dos dados, engenharia de prompt, integração com Groq, tratamento de erros, apresentação ao usuário, uso de variáveis de ambiente e documentação da arquitetura.

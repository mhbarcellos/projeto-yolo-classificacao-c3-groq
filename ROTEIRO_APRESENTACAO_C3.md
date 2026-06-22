# Roteiro breve para apresentação da C3

1. Mostrar o objetivo: a C2 detectava objetos com YOLO; a C3 adicionou análise generativa via Groq.
2. Abrir a estrutura do projeto e destacar os arquivos novos:
   - `app/llm_analyzer.py`
   - `analisar_ultima_sessao.py`
   - `.env.example`
   - `RELATORIO_C3_GROQ.md`
3. Mostrar o `.env.example` e explicar que a chave real fica no `.env`, que não vai para o GitHub.
4. Executar o projeto:

```powershell
python run_app.py
```

ou, pelo terminal:

```powershell
python main.py
```

5. Iniciar a câmera e mostrar alguns objetos.
6. Encerrar a execução para gerar o `session_summary.json`.
7. Explicar que o resumo JSON da C2 é serializado e enviado à LLM.
8. Mostrar os arquivos gerados em `outputs/logs`:
   - `session_summary.json`
   - `llm_prompt_<hash>.txt`
   - `llm_analysis_<data_hora>.json`
9. Abrir o arquivo de análise e comentar os pontos: resumo, padrões, limitações e recomendações.
10. Finalizar explicando o tratamento de erro: se a chave da Groq estiver ausente ou a API falhar, o sistema não quebra.

Comando alternativo para testar só a análise da última sessão:

```powershell
python analisar_ultima_sessao.py
```

## Usa a imagem oficial e enxuta do Python 3.11 como base
FROM python:3.11-slim-bullseye

# Define o diretório de trabalho dentro do container
WORKDIR /app

# Cria um usuário sem privilégios de administrador (root) por segurança 
# e dá a ele a posse da pasta /app
RUN useradd -ms /bin/bash appuser && chown -R appuser:appuser /app

# Instala dependências de sistema necessárias para compilar certos pacotes do Python (como bibliotecas de ML)
RUN apt-get update && apt-get install -y build-essential libffi-dev && rm -rf /var/lib/apt/lists/*

# Muda a execução para o usuário seguro recém-criado e adiciona os scripts dele no PATH
USER appuser
ENV PATH="/home/appuser/.local/bin:$PATH"

# Atualiza o pip para a versão mais recente sem guardar arquivos inúteis de cache
RUN pip install --no-cache-dir --upgrade pip

# Copia APENAS o arquivo de dependências primeiro. 
# Boa Prática: Isso aproveita o cache do Docker e evita reinstalar bibliotecas pesadas se você mudar apenas o código.
COPY --chown=appuser:appuser requirements.txt .

# Instala as dependências Python do projeto
RUN pip install --no-cache-dir -r requirements.txt

# Copia o restante do código da aplicação
COPY --chown=appuser:appuser . .

# Informa ao Docker que o container vai se comunicar através da porta 8000
EXPOSE 8000

# Comando final que liga o servidor da API (Uvicorn) assim que o container iniciar
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "debug"]

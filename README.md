# MLOps Intent Classification API 🧠

Uma solução de **Processamento de Linguagem Natural (NLP) ponta a ponta** para classificação de intenções. O projeto recebe textos do usuário e detecta a intenção utilizando modelos de **Deep Learning (TensorFlow/Keras)**, unindo uma API em **FastAPI**, persistência no **MongoDB**, interface gráfica em **Streamlit** e monitoramento de Machine Learning na nuvem.

Este projeto foi desenvolvido como parte do curso **IMD3005 - MLOPS** (Prof.: [adelson.araujo@imd.ufrn.br](mailto:adelson.araujo@imd.ufrn.br)).

**Contato/Suporte:** [pedropjc500@gmail.com](mailto:pedropjc500@gmail.com)

---

## ⚙️ Configuração Inicial (`.env`)

Antes de rodar, crie um arquivo `.env` na raiz do projeto (Utilize o `.env.example`). Você pode optar por rodar **100% offline** (deixando as chaves do W&B vazias e apontando para arquivos locais) ou conectar à nuvem:

```env
# Ambiente e Banco de Dados
ENV="dev"
MONGO_URI="mongodb+srv://<usuario>:<senha>@<seu_cluster>.mongodb.net/?appName=Cluster0"
MONGO_DB="ml_app"
API_URL="http://backend:8000"

# --- Configurações de Modelos e MLOps (Weights & Biases) ---
# Para rodar OFFLINE: deixe as chaves da API vazias e use caminhos locais.
WANDB_API_KEY=""
WANDB_PROJECT=""

# Indique os modelos a serem carregados (separados por vírgula).
# Uso Local (Exemplo): "intent_classifier/models/confusion-v1.keras"
# Uso Nuvem (Exemplo): "seu_usuario/projeto/modelo:versao"
WANDB_MODELS="intent_classifier/models/confusion-v1.keras,intent_classifier/models/clair-v1.keras"
```

---

## 🐳 Como Rodar a Aplicação

A forma mais rápida e isolada de rodar o sistema completo é utilizando o **Docker**.

No terminal, dentro da pasta do projeto, execute:

```bash
docker-compose up --build
```

Assim que os containers subirem, acesse os serviços pelo navegador:

* **Interface Visual (Streamlit):** http://localhost:8501
* **API / Swagger UI:** http://localhost:8000/docs

---

## 🧠 Treinamento e Monitoramento (W&B)

O projeto possui um pipeline integrado para treinar a rede neural e enviar os resultados (**métricas e o arquivo `.keras`**) automaticamente para o **Weights & Biases**.

Para treinar um novo modelo, certifique-se de que a variável `WANDB_API_KEY` está preenchida, acesse a pasta do classificador e execute o comando via terminal:

```bash
cd intent_classifier

python intent_classifier.py train \
    --config="models/confusion-v2_config.yml" \
    --training_data="data/confusion_intents.yml" \
    --save_model="models/confusion-v2.keras" \
    --wandb_project="ml_app"
```

ou

```bash
cd intent_classifier

python intent_classifier.py train \
    --config="models/clair-v2_config.yml" \
    --training_data="data/clair_intents.yml" \
    --save_model="models/clair-v2.keras" \
    --wandb_project="ml_app"
```

> **Nota:** o script fará o parsing dos dados YAML, treinará a rede, salvará o modelo localmente e, em seguida, fará o upload como um **Artifact** no seu painel do W&B. 
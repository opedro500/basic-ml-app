import os
import sys
import pytest

from unittest.mock import MagicMock
from dotenv import load_dotenv

# Adiciona a raiz do projeto ao path para permitir a importação de 'app' e 'intent_classifier'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from fastapi import HTTPException
from app.app import app
from intent_classifier import IntentClassifier

# --- Fixtures (Preparações para os Testes) ---

@pytest.fixture(scope="function", autouse=True)
def mock_app_dependencies(monkeypatch, request):
    """
    Fixture ativada automaticamente para simular dependências externas nos testes unitários.
    Para testes de integração, ela simula apenas a coleção do banco de dados para não sujar o BD real.
    """
    mock_collection = MagicMock()
    # Simula a função de conexão do banco para garantir que o app use nossa coleção "falsa"
    monkeypatch.setattr("db.engine.get_mongo_collection", lambda name: mock_collection)

    # Se for um teste de integração, paramos por aqui (deixamos o resto rodar de verdade)
    if "integration" in request.node.keywords:
        yield mock_collection, None, None
        return

    # --- Simulações (Mocks) Completas para Testes Unitários ---
    # Cria um modelo de Machine Learning "de mentira" que sempre responde a mesma coisa
    mock_model = MagicMock(spec=IntentClassifier)
    mock_model.predict.return_value = ("mock_intent", {"mock_intent": 0.9, "other": 0.1})
    
    # Simula a função que baixa modelos da nuvem durante a inicialização do app
    mock_load = MagicMock(return_value={"mock-model": mock_model})
    monkeypatch.setattr("app.services.load_all_classifiers", mock_load)

    # Simula o validador de token de segurança
    mock_verify_token = MagicMock(return_value="mock_prod_user")
    monkeypatch.setattr("db.auth.verify_token", mock_verify_token)

    yield mock_collection, mock_model, mock_verify_token

@pytest.fixture(scope="function")
def client():
    """Fornece um TestClient para simular requisições HTTP para a API na memória."""
    with TestClient(app) as test_client:
        yield test_client

# --- Testes Unitários ---

def test_predict_dev_mode(client, monkeypatch, mock_app_dependencies):
    """Testa o endpoint POST /predict no modo 'dev', que deve pular a exigência de senha/token."""
    monkeypatch.setattr("db.auth.ENV", "dev")
    mock_collection, mock_model, mock_verify_token = mock_app_dependencies
    
    # IMPORTANTE: Agora enviamos os dados via 'json=' em vez de 'params='
    response = client.post("/predict", json={"text": "olá modo dev"})
    
    assert response.status_code == 200
    data = response.json()
    assert data["owner"] == "dev_user"
    assert "mock-model" in data["predictions"]
    
    # Verifica se a autenticação foi realmente ignorada e se o modelo e o banco foram chamados
    mock_verify_token.assert_not_called()
    mock_model.predict.assert_called_once_with("olá modo dev")
    mock_collection.insert_one.assert_called_once()

def test_predict_prod_mode_auth_success(client, monkeypatch, mock_app_dependencies):
    """Testa POST /predict no modo 'prod' (produção) passando um token correto."""
    monkeypatch.setattr("db.auth.ENV", "prod")
    mock_collection, mock_model, mock_verify_token = mock_app_dependencies

    # Enviando texto no JSON e o token no Header de Autorização
    response = client.post("/predict", json={"text": "olá prod"}, headers={"Authorization": "Bearer valid"})
    
    assert response.status_code == 200
    assert response.json()["owner"] == "mock_prod_user"
    
    mock_verify_token.assert_called_once()
    mock_model.predict.assert_called_once_with("olá prod")
    mock_collection.insert_one.assert_called_once()

def test_predict_prod_mode_auth_fail(client, monkeypatch, mock_app_dependencies):
    """Testa POST /predict no modo 'prod' enviando um token falso/inválido."""
    monkeypatch.setattr("db.auth.ENV", "prod")
    mock_collection, mock_model, mock_verify_token = mock_app_dependencies
    # Força a função de validar token a dar um erro 401
    mock_verify_token.side_effect = HTTPException(status_code=401, detail="Token Inválido")

    response = client.post("/predict", json={"text": "nao vai funcionar"}, headers={"Authorization": "Bearer invalid"})
    
    # Garante que a API barrou o usuário
    assert response.status_code == 401
    assert "Token Inválido" in response.json()["detail"]
    
    # Garante que, como o token falhou, o modelo e o banco não foram acessados
    mock_model.predict.assert_not_called()
    mock_collection.insert_one.assert_not_called()

def test_predict_no_models_loaded(client, monkeypatch, mock_app_dependencies):
    """Testa a situação extrema (edge case) onde nenhum modelo conseguiu ser carregado da nuvem."""
    monkeypatch.setattr("app.app.ENV", "dev")
    # Sobrescreve a simulação para retornar um dicionário de modelos vazio
    monkeypatch.setattr("app.services.load_all_classifiers", lambda urls: {})
    mock_collection, _, _ = mock_app_dependencies
    
    # Recria o client para forçar o evento de 'lifespan' (início da API) com a nova simulação
    with TestClient(app) as test_client:
        response = test_client.post("/predict", json={"text": "sem modelos"})
    
    assert response.status_code == 200
    assert response.json()["predictions"] == {}
    mock_collection.insert_one.assert_called_once()


# --- Testes de Integração ---

@pytest.mark.integration
def test_integration_real_model_predict(monkeypatch, mock_app_dependencies):
    """
    Teste de Integração: Verifica o fluxo completo conectando as peças reais.
    Ele baixa de verdade um modelo do W&B na inicialização do app.
    """
    load_dotenv() 
    if not os.getenv("WANDB_API_KEY") or not os.getenv("WANDB_MODELS"):
        pytest.skip("WANDB_API_KEY ou WANDB_MODELS não estão configurados. Pulando teste.")

    monkeypatch.setattr("app.app.ENV", "dev")
    
    # 1. Configura o app para carregar apenas o primeiro modelo do arquivo .env (para ser mais rápido)
    first_model_url = os.getenv("WANDB_MODELS").split(',')[0].strip()
    model_name = first_model_url.split('/')[-1].split(':')[0]
    monkeypatch.setattr("app.app.get_model_urls", lambda: first_model_url)
    
    # 2. Cria o TestClient, o que aciona o evento lifespan para baixar o modelo real
    with TestClient(app) as client:
        mock_collection, _, _ = mock_app_dependencies

        # 3. Faz uma requisição real de predição
        test_text = "wait what?" # Assume que o primeiro modelo treinado foi o classificador de 'confusion'
        response = client.post("/predict", json={"text": test_text})
        
        # 4. Validações (Assertions)
        assert response.status_code == 200
        data = response.json()
        
        assert model_name in data["predictions"]
        prediction = data["predictions"][model_name]["top_intent"]
        assert prediction == "confusion"
        
        mock_collection.insert_one.assert_called_once()
    
    print("\n[Teste de Integração] Passou: Modelo real baixado e predição feita corretamente.")

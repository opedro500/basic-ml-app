import fire
import pymongo
import sys

from pymongo.errors import ConnectionFailure, OperationFailure

def test_mongo(connection_string: str):
    """
    Testa uma string de conexão do MongoDB tentando enviar um "ping" para o servidor.
    Parâmetros:
        connection_string: A URL completa de conexão do MongoDB (URI).
    """
    print(f"Tentando conectar a: {connection_string[:50]}...")
    client = None
    
    try:
        client = pymongo.MongoClient(connection_string, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        
        print("\n✅ Conexão bem-sucedida!")
    except OperationFailure as e:
        # Este erro acontece se o nome de usuário ou a senha estiverem incorretos
        print("\n❌ Falha na autenticação:")
        print(e.details)

        sys.exit(1)
    except ConnectionFailure as e:
        # Este erro acontece se o servidor estiver inacessível. 
        # Causa muito comum: Seu IP atual não está na "Whitelist" do MongoDB Atlas (Network Access).
        print("\n❌ Falha de conexão (Erro de rede ou IP não autorizado?):")
        print(e)

        sys.exit(1)
    except Exception as e:
        print(f"\nOcorreu um erro inesperado: {e}")
        
        sys.exit(1)
    finally:
        if client:
            client.close()

if __name__ == "__main__":
    fire.Fire(test_mongo)

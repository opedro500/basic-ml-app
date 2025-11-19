import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime

# --- CONFIGURAÇÃO ---
st.set_page_config(
    page_title="Análise de Intenção", 
    page_icon="🧠",
    layout="wide" # Usa a tela inteira para melhor visualização
)

API_URL = "http://localhost:8000/predict"

# --- FUNÇÕES AUXILIARES ---
def format_label(label):
    """Remove underscores e capitaliza para exibição bonita."""
    return label.replace("_", " ").capitalize()

def plot_probabilities(probs_dict):
    """Gera um gráfico de barras bonito com Plotly."""
    # Converte dicionário para DataFrame
    df = pd.DataFrame(list(probs_dict.items()), columns=['Intenção', 'Confiança'])
    
    # Tratamento de dados
    df['Confiança'] = df['Confiança'].astype(float)
    df['Intenção Formatada'] = df['Intenção'].apply(format_label)
    df = df.sort_values(by='Confiança', ascending=True) # Ordenar para o gráfico

    # Criação do Gráfico
    fig = px.bar(
        df, 
        x='Confiança', 
        y='Intenção Formatada', 
        orientation='h',
        text_auto='.1%', # Mostra porcentagem na barra
        color='Confiança',
        color_continuous_scale='Blugrn', # Escala de cor azul-verde moderna
    )
    
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis_title=None,
        yaxis_title=None,
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
        height=300
    )
    fig.update_xaxes(showticklabels=False, range=[0, 1.1]) # Esconde eixo X e garante margem
    
    return fig

# --- INTERFACE PRINCIPAL ---
def main():
    # Cabeçalho Estilizado
    col1, col2 = st.columns([1, 6])
    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=80) # Ícone genérico de AI
    with col2:
        st.title("Classificador de Intenções Inteligente")
        st.markdown("Analise o sentimento e a intenção do cliente em tempo real.")

    st.divider()

    # Área de Input Centralizada
    text_input = st.text_area(
        "Digite a mensagem do cliente:", 
        height=100, 
        placeholder="Ex: Olá, meu pedido está atrasado e eu gostaria de saber onde ele está.",
        help="Digite o texto completo para obter a melhor precisão."
    )

    col_btn, col_info = st.columns([1, 4])
    with col_btn:
        analyze_btn = st.button("🔍 Analisar Texto", type="primary", use_container_width=True)

    # Lógica de Processamento
    if analyze_btn:
        if not text_input.strip():
            st.warning("⚠️ Por favor, digite algum texto para analisar.")
            return

        with st.status("🤖 Processando...", expanded=True) as status:
            try:
                st.write("Conectando à API...")
                payload = {"text": text_input}
                response = requests.post(API_URL, params=payload)
                response.raise_for_status()
                
                data = response.json()
                st.write("Interpretando resultados...")
                status.update(label="Análise concluída!", state="complete", expanded=False)

                # --- RENDERIZAÇÃO DOS RESULTADOS ---
                
                # 1. Metadados na Sidebar (Baseado no JSON da sua imagem)
                with st.sidebar:
                    st.header("ℹ️ Metadados da Requisição")
                    st.info(f"**ID:** `{data.get('id', 'N/A')}`")
                    st.text(f"Owner: {data.get('owner', 'N/A')}")
                    
                    ts = data.get('timestamp')
                    if ts:
                        # Tenta converter timestamp se for numérico, senão mostra como está
                        try:
                            dt_object = datetime.fromtimestamp(ts)
                            st.text(f"Data: {dt_object.strftime('%d/%m/%Y %H:%M')}")
                        except:
                            st.text(f"Timestamp: {ts}")
                    
                    with st.expander("Ver JSON Bruto"):
                        st.json(data)

                # 2. Processamento das Predições
                predictions = data.get("predictions", {})

                if not predictions:
                    st.error("A API não retornou predições no formato esperado.")
                    return

                st.subheader("📊 Resultados da Análise")

                # Se houver mais de um modelo (ex: confusion-clf E clair-clf), cria abas
                model_names = list(predictions.keys())
                tabs = st.tabs([name.replace("-", " ").upper() for name in model_names])

                for i, model_key in enumerate(model_names):
                    with tabs[i]:
                        model_data = predictions[model_key]
                        
                        # Extrai dados principais
                        top_intent = model_data.get("top_intent", "Desconhecido")
                        probs = model_data.get("all_probs", {})
                        
                        # Pega a pontuação da top intent
                        top_score = probs.get(top_intent, 0.0)

                        # Exibe Cartão de Destaque
                        c1, c2 = st.columns([1, 2])
                        with c1:
                            st.markdown("#### Intenção Detectada")
                            st.markdown(
                                f"""
                                <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid #4CAF50;">
                                    <h2 style="color: #1f1f1f; margin:0;">{format_label(top_intent)}</h2>
                                    <p style="color: #555; margin:0;">Confiança: <b>{top_score:.1%}</b></p>
                                </div>
                                """, 
                                unsafe_allow_html=True
                            )
                        
                        with c2:
                            st.markdown("#### Distribuição de Probabilidades")
                            if probs:
                                fig = plot_probabilities(probs)
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning("Sem dados de probabilidade detalhados.")

            except requests.exceptions.ConnectionError:
                status.update(label="Erro de Conexão", state="error")
                st.error(f"❌ Não foi possível conectar em `{API_URL}`. Verifique se o backend está rodando.")
            except Exception as e:
                status.update(label="Erro Interno", state="error")
                st.error(f"❌ Ocorreu um erro: {str(e)}")

if __name__ == "__main__":
    main()
"""
Este script funciona como um módulo e como uma ferramenta de linha de comando (CLI).

Para usá-lo como um módulo, você pode fazer:
::

    from intent_classifier import IntentClassifier

    # treinar um modelo
    classifier = IntentClassifier(config="models/confusion_config.yml", training_data="data/confusion_intents.yml")
    classifier.train(save_model="models/confusion-clf/")
    
    # ou carregar um modelo do W&B
    classifier = IntentClassifier(config="models/confusion_config.yml", load_model="adaj/intent-classifier-2025-2/confusion-clf:v1")
    
    # prever um novo texto
    classifier.predict(input_text="oi")
    
    # validação cruzada do modelo
    classifier.cross_validation(n_splits=5)

Ou você pode usá-lo como uma ferramenta CLI:
::

    cd intent_classifier

    python intent_classifier.py train \
        --config="models/confusion_config.yml" \
        --training_data="data/confusion_intents.yml" \
        --save_model="models/confusion.keras" \
        --wandb_project="intent-classifier"

    python intent_classifier.py predict \
        --load_model="models/confusion.keras" \
        --input_text="teste teste" \
        --wandb_project="intent-classifier"

    python intent_classifier.py cross_validation \
        --config="models/confusion_config.yml" \
        --training_data="data/confusion_intents.yml" \
        --n_splits=5 \
        --wandb_project="intent-classifier"
"""

import os
import logging
import yaml
import re
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_text
import tensorflow_hub as hub
import wandb
import dotenv

from pathlib import Path
from typing import List, Optional, Union, Tuple, Dict, Any
from dataclasses import dataclass
from pprint import pprint
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, cohen_kappa_score
from tensorflow.keras import regularizers
from tensorflow.keras.saving import register_keras_serializable
from wandb.integration.keras import WandbMetricsLogger, WandbEvalCallback

dotenv.load_dotenv()

logger = logging.getLogger(__name__)

@register_keras_serializable()
class HubLayer(tf.keras.layers.Layer):
    """
    Uma camada Keras customizada para carregar e usar um módulo do TensorFlow Hub.
    Esta camada carrega um modelo pré-treinado a partir de uma URL do TensorFlow Hub
    e o integra em um modelo Keras. Pode ser configurada como treinável ou congelada.

    :param hub_url: A URL do módulo TensorFlow Hub a ser carregado.
    :type hub_url: str
    :param trainable: Se o módulo carregado deve ser treinável (fine-tuning).
    :type trainable: bool, opcional
    """
    def __init__(self, hub_url, trainable=False, **kwargs):
        """Inicializa a camada HubLayer."""
        super(HubLayer, self).__init__(**kwargs)
        self.hub_module = hub.load(hub_url)
        self.hub_module.trainable = trainable

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Executa o passo de inferência (forward pass) da camada.

        :param inputs: O(s) tensor(es) de entrada para o módulo Hub.
        :type inputs: tf.Tensor
        :return: O(s) tensor(es) de saída do módulo Hub.
        :rtype: tf.Tensor
        """
        return self.hub_module(inputs)

@dataclass
class Config:
    """
    Um dataclass para armazenar todos os parâmetros de configuração do IntentClassifier.
    Este objeto armazena as configurações relacionadas ao conjunto de dados, arquitetura 
    do modelo, processo de treinamento e logs.
    """
    dataset_name: str = "undefined"
    """Nome do dataset, usado para logs e nomenclatura do modelo."""
    codes : List[str] = None
    """Uma lista de códigos de intenção (classes). Povoado automaticamente a partir dos dados se não fornecido."""
    architecture: str = "v0.1.5"
    """Tag de versão para a arquitetura do modelo."""
    task: str = "undefined"
    """A tarefa atual sendo executada (ex: 'train', 'predict')."""
    stop_words_file: Optional[str] = None
    """Caminho para um arquivo de texto contendo stopwords, uma por linha."""
    min_words: int = 1
    """Número mínimo de palavras exigidas em uma frase para processamento. Entradas menores sofrem padding."""
    embedding_model: Union[str, List[str]] = 'https://www.kaggle.com/models/google/universal-sentence-encoder/tensorFlow2/multilingual/2'
    """URL ou caminho para o modelo de embeddings do TensorFlow Hub."""
    sent_hl_units: Union[int, List[int]] = 32
    """Número de neurônios (units) na camada oculta."""
    sent_dropout: Union[float, List[float]] = 0.1
    """Taxa de Dropout aplicada após a camada oculta."""
    l1_reg: float = 0.01
    """Fator de regularização L1 para o kernel da camada oculta."""
    l2_reg: float = 0.01
    """Fator de regularização L2 para o kernel da camada oculta."""
    epochs: int = 500
    """Número máximo de épocas para o treinamento."""
    callback_patience: int = 20
    """Número de épocas sem melhoria para aguardar antes do Early Stopping (Parada Antecipada)."""
    learning_rate: Union[float, List[float]] = 5e-3
    """Taxa de aprendizado inicial para o otimizador."""
    validation_split: float = 0.2
    """Fração dos dados de treinamento a ser usada como validação."""

def remove_duplicate_words(text: str) -> str:
    """
    Remove palavras duplicadas consecutivas de uma string.

    :param text: A string de entrada.
    :type text: str
    :return: O texto sem palavras duplicadas consecutivas.
    :rtype: str
    """
    words = text.split()
    seen = set()
    result = []
    for word in words:
        if word not in seen:
            seen.add(word)
            result.append(word)
    return ' '.join(result)

def fetch_artifact_from_wandb(model_full_name: str) -> Tuple[str, str]:
    """
    Faz o download de um artefato de modelo do Weights & Biases (W&B) e retorna 
    os caminhos locais para o modelo e para os arquivos de configuração.

    :param model_full_name: O nome completo do artefato no W&B (ex: "adaj/intent-classifier/confusion-clf:v1").
                           Deve seguir o formato: "entidade/projeto/nome_artefato:versão"
    :type model_full_name: str
    :return: Uma tupla contendo o caminho local do arquivo de modelo do Keras e o arquivo de config.
    :rtype: tuple[str, str]
    :raises ValueError: Se o formato for inválido ou se os arquivos não forem encontrados.
    """
    parts = model_full_name.split("/")
    if len(parts) != 3 or ":" not in parts[2]:
        raise ValueError(
            f"Formato de model_full_name inválido: '{model_full_name}'. "
            f"Formato esperado: 'entidade/projeto/nome_artefato:versao'"
        )
    
    try:
        api = wandb.Api()
        artifact = api.artifact(model_full_name, type='model')
    except wandb.errors.CommError as e:
        raise ValueError(f"Não foi possível buscar o artefato '{model_full_name}' no W&B. Erro original: {e}")

    models_dir = Path(os.path.dirname(__file__)) / "models"
    models_dir.mkdir(exist_ok=True)
    
    download_path = artifact.download(root=models_dir)
    
    model_file, config_file = None, None
    for f in artifact.files():
        if f.name.endswith((".keras", ".h5")):
            model_file = os.path.join(download_path, f.name)
        elif f.name.endswith("_config.yml"):
            config_file = os.path.join(download_path, f.name)
            
    if not model_file:
        raise ValueError(f"Arquivo do modelo (.keras ou .h5) não encontrado no artefato '{model_full_name}'.")
    if not config_file:
        raise ValueError(f"Arquivo de config (_config.yml) não encontrado no artefato '{model_full_name}'.")
        
    return model_file, config_file

class IntentClassifier:
    """
    Uma classe para treinar, avaliar e fazer predições de intenções usando um modelo Keras.

    Este classificador empacota todo o pipeline de MLOps, incluindo carregamento de dados,
    pré-processamento, construção do modelo (com embeddings do TensorFlow Hub), treinamento,
    log de métricas no W&B e inferência.

    :param config: Um caminho para um arquivo config YAML, um objeto Config ou None.
                   Se None, a config é inferida a partir de `load_model`.
    :type config: str, Config, opcional
    :param load_model: Caminho para um modelo salvo `.keras` ou URL de um artefato W&B.
                       Se fornecido, o modelo e sua config serão carregados.
    :type load_model: str, opcional
    :param training_data: Caminho para um arquivo YAML com os exemplos de treinamento.
                          Obrigatório para treinar ou realizar validação cruzada.
    :type training_data: str, opcional
    """

    def __init__(self, config: Optional[Union[str, Config]] = None,
                 load_model: Optional[str] = None,
                 training_data: Optional[str] = None,
                 wandb_project: Optional[str] = None):
        """Inicializa o IntentClassifier."""
        self.model = None
        local_model_path = None
        
        self.wandb_project = wandb_project or os.environ.get("WANDB_PROJECT") or "intent-classifier"

        if load_model:
            if os.path.exists(load_model):
                local_model_path = load_model
            else:
                local_model_path, config = fetch_artifact_from_wandb(load_model)
            
            self.model = tf.keras.models.load_model(local_model_path)
            print(f"Modelo Keras carregado de {local_model_path}.")

        self._load_config(config)
        self._load_intents(training_data)
        
        if self.model:
            self._validate_model_config_compatibility()
            
        self._load_stop_words(self.config.stop_words_file)
        self._setup_onehot_encoder()
        
        if self.wandb_project:
            print(f"Configurando o projeto no W&B: {self.wandb_project}")
            wandb_key = os.environ.get("WANDB_API_KEY")
            if wandb_key:
                wandb.login(key=wandb_key)
            else:
                os.environ["WANDB_MODE"] = "disabled"
            self.wandb_run = wandb.init(project=self.wandb_project, config=self.config.__dict__)
            if self.training_data:
                artifact = wandb.Artifact(Path(self.training_data).name, type="dataset")
                artifact.add_file(self.training_data)
                self.wandb_run.log_artifact(artifact)
        else:
            self.wandb_run = None
            print("W&B project não definido. Nenhuma execução será registrada no W&B.")

    def _load_config(self, config: Optional[Union[str, Config]]) -> None:
        """
        Carrega a configuração a partir de um arquivo ou objeto Config.

        :param config: Um caminho para arquivo YAML ou um objeto Config.
        :type config: str, Config, opcional
        :raises ValueError: Se config não for fornecida.
        """
        if isinstance(config, str):
            with open(config, 'r') as f:
                self.config = Config(**yaml.safe_load(f))
            print(f"Configuração carregada de {config}.")
        elif isinstance(config, Config):
            self.config = config
        elif config is None:
            raise ValueError(
                "A 'config' deve ser informada como arquivo de texto, objeto Config ou ser baixada via W&B."
            )
        else:
            raise TypeError(f"Tipo de configuração não suportada: {type(config)}")
    
    def _load_intents(self, training_data: Optional[str]) -> np.ndarray:
        """
        Carrega e pré-processa as intenções a partir de um arquivo YAML.
        
        Se `training_data` for passado, extrai as frases e labels, os embaralha 
        e armazena em self.input_text e self.labels. 

        :param training_data: Caminho do arquivo YAML com exemplos.
        :type training_data: str, opcional
        :return: Um array contendo os códigos (labels) únicos de intenção.
        :rtype: np.ndarray
        """
        self.training_data = training_data
        if training_data is not None:
            pprint(f"Carregando intenções de {training_data}...")
            with open(training_data, 'r') as f:
                self.intents_data = yaml.safe_load(f)
            
            input_text = []
            labels = []
            for i in self.intents_data:
                input_text += i['examples']
                labels += [i['intent']]*len(i['examples'])
            input_text = np.array(input_text)
            labels = np.array(labels)
            
            indices = np.arange(len(labels))
            np.random.shuffle(indices)
            self.input_text = input_text[indices]
            self.input_text = tf.convert_to_tensor(self.input_text, dtype=tf.string)
            self.labels = labels[indices]
            self.codes = np.unique(self.labels)
            self.config.codes = self.codes.tolist()
        else:
            self.codes = self.config.codes

        return self.codes
    
    def _load_stop_words(self, stop_words_file: Optional[str]) -> 'IntentClassifier':
        """
        Carrega as stopwords do arquivo.

        :param stop_words_file: Caminho do arquivo txt com stopwords.
        :type stop_words_file: str, opcional
        :return: A instância do IntentClassifier.
        :rtype: IntentClassifier
        """
        if stop_words_file is None:
            self.stop_words = []
            return self
        with open(stop_words_file, 'r', encoding='utf-8') as f:
            self.stop_words = f.read().split('\n')
        print(f"{len(self.stop_words)} stop words carregadas de {stop_words_file}.")

        return self

    def _validate_model_config_compatibility(self) -> None:
        """
        Valida se o tamanho de saída do modelo corresponde ao número de categorias
        definidas na configuração, garantindo compatibilidade.

        :raises ValueError: Se houver divergência entre o modelo e a config.
        """
        if self.model is None:
            return
            
        model_output_size = self.model.output_shape[-1]
        config_codes_count = len(self.config.codes) if self.config.codes else 0
        
        if model_output_size != config_codes_count:
            error_msg = (
                f"Incompatibilidade detectada: O modelo foi treinado com "
                f"{model_output_size} categorias, mas o arquivo de config tem "
                f"{config_codes_count} categorias (codes: {self.config.codes}).\n\n"
                f"A camada de saída do modelo deve bater com o encoder (one-hot).\n\n"
            )
            raise ValueError(error_msg)
    
    def _setup_onehot_encoder(self) -> OneHotEncoder:
        """
        Inicializa e treina o OneHotEncoder baseado nas classes de intenção.

        :return: A instância treinada do OneHotEncoder.
        :rtype: OneHotEncoder
        """
        assert self.codes is not None, "Os códigos de intenção precisam ser carregados antes do encoder."
        self.onehot_encoder = OneHotEncoder(categories=[self.codes],)\
                                  .fit(np.array(self.codes).reshape(-1, 1))
                                  
        return self.onehot_encoder

    def _get_callbacks(self) -> list:
        """
        Gera uma lista de callbacks do Keras com base nas configurações,
        como EarlyStopping, Logs do W&B e Agendador de Taxa de Aprendizado (Decay).

        :return: Uma lista de instâncias Callback do Keras.
        :rtype: list
        """
        callbacks = []
        if self.config.callback_patience > 0:
            callbacks.append(
                tf.keras.callbacks.EarlyStopping(monitor='val_f1_score',
                    patience=self.config.callback_patience,
                    restore_best_weights=True)
            )
        if self.wandb_project:
            callbacks.append(WandbMetricsLogger())
        
        if self.config.learning_rate is not None and not isinstance(self.config.learning_rate, str):
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=self.config.learning_rate,
                decay_steps=1000,
                decay_rate=0.96,
                staircase=False
            )
            
            def lr_scheduler(epoch, lr):
                """Função interna do Scheduler de Learning Rate."""
                return lr_schedule(epoch).numpy().astype(float)
            
            lr_scheduler_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)
            callbacks.append(lr_scheduler_callback)

        return callbacks
    
    def finish_wandb(self):
        """Finaliza a execução ativa no Weights & Biases."""
        if self.wandb_project and self.wandb_run:
            self.wandb_run.finish()

    def preprocess_text(self, text: tf.Tensor) -> tf.Tensor:
        """
        Aplica etapas de pré-processamento em um tensor de texto cru.
        
        Etapas:
        1. Minúsculas (Lowercasing).
        2. Remoção de Stopwords (se configurado).
        3. Padding com tokens "<>" se a frase for menor que `min_words`.

        :param text: Tensor escalar (0-D string tensor) do texto bruto.
        :type text: tf.Tensor
        :return: Tensor pre-processado escalar.
        :rtype: tf.Tensor
        """
        text = tf.strings.lower(text)
        if self.stop_words:
            words = tf.strings.split(text)
            words = tf.boolean_mask(words, tf.reduce_all(tf.not_equal(words[:, None], tf.constant(self.stop_words)), axis=1))
            text = tf.strings.reduce_join(words, separator=' ')
        
        if self.config.min_words:
            words = tf.strings.split(text)
            words = tf.boolean_mask(words, tf.reduce_all(tf.not_equal(words[:, None], tf.constant(["?", ".", ",", "!"])), axis=1))
            num_words = tf.shape(words)[0]
            
            if tf.less_equal(num_words, self.config.min_words):
                padding = tf.strings.join(["<>"] * (self.config.min_words + 1), separator=' ')
                text = padding

        for p, t in {"?": "QUESTION_MARK", ".": "PERIOD", ",": "COMMA", "!": "EXCLAMATION_MARK"}.items():
            espaped_p = re.escape(p)
            text = tf.strings.regex_replace(text, espaped_p, f" {t} ")
        text = tf.strings.regex_replace(text, r"\s+", " ")
        text = tf.strings.strip(text)
        
        return tf.strings.as_string(text)

    def make_model(self, config: Config) -> tf.keras.Model:
        """
        Constrói e retorna um novo modelo Keras baseado na configuração dada.
        A arquitetura consiste em:
        1. Camada de entrada (Input layer) para string tensors.
        2. Um HubLayer para aplicar embeddings.
        3. Uma camada Dense Oculta com Normalização de Lote, ativação ReLU e Dropout.
        4. Camada de saída (Dense Output Layer) com softmax para a classificação.

        :param config: O objeto contendo os hiperparâmetros.
        :type config: Config
        :return: Um modelo Keras compilado.
        :rtype: tf.keras.Model
        """
        seed = 42
        tf.random.set_seed(seed)  
        
        sent_hl_units, sent_dropout = config.sent_hl_units, config.sent_dropout
        l1_reg, l2_reg = config.l1_reg, config.l2_reg
        output_size = len(self.codes)
    
        initializer = tf.keras.initializers.GlorotUniform(seed=seed)  
        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name="inputs")
        
        encoder = HubLayer(config.embedding_model, trainable=False, name="sent_encoder")(text_input)
        
        sent_hl = tf.keras.layers.Dense(sent_hl_units,
                                        kernel_initializer=initializer,
                                        kernel_regularizer=regularizers.l1_l2(l1=l1_reg, l2=l2_reg),
                                        activation=None,  
                                        name='sent_hl')(encoder)
        sent_hl_norm = tf.keras.layers.BatchNormalization()(sent_hl)  
        sent_hl_activation = tf.keras.layers.Activation('relu')(sent_hl_norm)  
        sent_hl_dropout = tf.keras.layers.Dropout(sent_dropout, seed=seed)(sent_hl_activation)  
        
        sent_output = tf.keras.layers.Dense(output_size,
                                            kernel_initializer=initializer,
                                            activation='softmax',
                                            name="sent_output")(sent_hl_dropout)
        model = tf.keras.Model(inputs=text_input, outputs=sent_output)
        return model

    def train(self, save_model: Optional[str] = None, tf_verbosity: int = 1) -> tf.keras.Model:
        """
        Treina o modelo com base nos exemplos informados.
        Realiza a quebra dos dados, pré-processamento e o laço de treinamento.
        Pode salvar o resultado final no disco, caso seja solicitado.

        :param save_model: Caminho para salvar a arquitetura final (`.keras`).
        :type save_model: str, opcional
        :param tf_verbosity: Nível de detalhamento do treinamento (0, 1, 2).
        :type tf_verbosity: int, opcional
        :return: O modelo Keras treinado.
        :rtype: tf.keras.Model
        """
        pprint(self.config.__dict__)
        self.config.task = "train"
        assert self.training_data is not None, "training_data é obrigatório ao instanciar IntentClassifier para treino."
        
        labels_ohe = self.onehot_encoder\
                            .transform(self.labels.reshape(-1, 1))\
                            .toarray()
                            
        X_train_text, X_val_text, y_train, y_val = train_test_split(
            self.input_text.numpy(), labels_ohe, 
            test_size=self.config.validation_split,
            stratify=labels_ohe,      
            random_state=42           
        )
        
        X_train = tf.map_fn(self.preprocess_text, tf.constant(X_train_text), dtype=tf.string)
        X_val = tf.map_fn(self.preprocess_text, tf.constant(X_val_text), dtype=tf.string)

        epochs = self.config.epochs
        
        self.model = self.make_model(self.config)
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=tf.keras.optimizers.Adam(), 
            metrics=[tf.keras.metrics.F1Score(average='macro')])
            
        self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            shuffle=True,
            epochs=epochs,
            verbose=tf_verbosity,
            callbacks=self._get_callbacks()
        )
        
        if save_model is not None:
            self.save_model(path=save_model)
        return self.model

    def save_model(self, path: str):
        """
        Salva o modelo atual e seu arquivo YAML de configuração.
        O modelo ganha extensão `.keras` e a config ganha o sufixo `_config.yml`.
        Também envia para o W&B Artifacts caso esteja rodando na nuvem.

        :param path: Caminho destino para guardar (ex: "models/meu_modelo.keras").
        :type path: str
        """
        Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
        if path.endswith('/'):
            path = path.rstrip('/')
            
        self.model.save(path)
        
        config_path = path.replace(".keras", "_config.yml") 
        with open(config_path, 'w') as f:
            f.write(yaml.dump(self.config.__dict__))
        print(f"Modelo salvo com sucesso em {path}.")
        
        if self.wandb_project:
            artifact = wandb.Artifact(
                name=f"{self.config.dataset_name}-clf",
                type="model",
                description="Modelo Keras para classificação de intenção"
            )
            artifact.add_file(path)
            artifact.add_file(config_path) 
            self.wandb_run.log_artifact(artifact)
            self.finish_wandb() 

    def predict(self, input_text: Union[str, List[str]],
                true_labels: Optional[List[str]] = None,
                log_to_wandb: bool = False) -> Union[Tuple[str, Dict[str, float]], List[Tuple[str, Dict[str, float]]]]:
        """
        Prevê a intenção de um texto individual ou uma lista de textos.

        :param input_text: Texto (str) ou lista de textos (List[str]).
        :type input_text: str or list[str]
        :param true_labels: Lista opcional de labels verdadeiras para enviar ao W&B.
        :type true_labels: list[str], opcional
        :param log_to_wandb: Se True, envia tudo para os relatórios do W&B.
        :type log_to_wandb: bool, opcional
        :return: Tupla com intenção principal e dicionário de probabilidades. 
                 Se a entrada for lista, devolve uma lista dessas tuplas.
        :rtype: tuple(str, dict(str, float)) ou list[tuple(str, dict(str, float))]
        """
        self.config.task = "predict"  
        original_input_is_string = isinstance(input_text, str)
        if original_input_is_string:
            input_text_list = [input_text]  
        else:
            input_text_list = input_text
        
        preprocessed_texts = tf.map_fn(self.preprocess_text, tf.constant(input_text_list), dtype=tf.string)

        all_probs = self.model.predict(preprocessed_texts)
        results = []
        predicted_labels_for_log = []
        
        for i in range(all_probs.shape[0]):
            current_probs = all_probs[i] 
            highest_prob_idx = np.argmax(current_probs)
            highest_prob_intent_name = self.codes[highest_prob_idx]
            predicted_labels_for_log.append(highest_prob_intent_name)
            
            probs_dict = {code: float(current_probs[j]) for j, code in enumerate(self.codes)}
            results.append((highest_prob_intent_name, probs_dict))
        
        if log_to_wandb and self.wandb_project:
            run_id = wandb.run.id if wandb.run else wandb.util.generate_id()
            with wandb.init(project=self.wandb_project, id=run_id, resume="allow"):
                wandb.log({
                    "inputs": input_text_list, 
                    "true_labels": true_labels,
                    "predictions": predicted_labels_for_log 
                })
        
        if original_input_is_string:
            return results[0]
        return results

    def cross_validation(self, n_splits: int = 3) -> List[Dict[str, Any]]:
        """
        Roda a validação cruzada do tipo K-Fold (Stratified).
        Treina e avalia o modelo em dobras dos dados, enviando a média e
        os resultados globais das métricas para o Weights & Biases.

        :param n_splits: Número de iterações/dobras da divisão do conjunto de dados.
        :type n_splits: int, opcional
        :return: Uma lista de dicionários contendo os relatórios (sklearn) de cada dobra.
        :rtype: list[dict(str, Any)]
        :raises AssertionError: Se `training_data` for nulo na inicialização.
        """
        assert self.training_data is not None, "training_data é obrigatório ao instanciar IntentClassifier."
        
        self.config.task = "cross_validation"
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        preprocessed_input_text = tf.map_fn(self.preprocess_text, self.input_text, dtype=tf.string)
        
        labels_ohe = self.onehot_encoder.transform(self.labels.reshape(-1, 1)).toarray()
        
        results = []
        
        for i, (train_index, test_index) in enumerate(kf.split(preprocessed_input_text.numpy(), self.labels)):
            print(f"Dobra (Fold) {i+1}/{n_splits}")
            
            run_name = f"cv_fold_{i+1}"
            with wandb.init(project=self.wandb_project, config=self.config.__dict__, 
                            group="cross_validation", name=run_name, reinit=True, 
                            job_type=f"fold_{i+1}"):
                
                model = self.make_model(self.config)
                model.compile(
                    loss='categorical_crossentropy',
                    optimizer=tf.keras.optimizers.Adam(), 
                    metrics=[tf.keras.metrics.F1Score(average='macro')])
                
                X_train, X_test = preprocessed_input_text[train_index], preprocessed_input_text[test_index]
                y_train_ohe, y_test_ohe = labels_ohe[train_index], labels_ohe[test_index]

                model.fit(X_train, y_train_ohe,
                          epochs=self.config.epochs, verbose=0,
                          validation_data=(X_test, y_test_ohe), 
                          callbacks=self._get_callbacks()) 
                
                preds_probs = model.predict(X_test)
                preds = self.onehot_encoder.inverse_transform(preds_probs)
                labels = self.onehot_encoder.inverse_transform(y_test_ohe)
                
                res = classification_report(labels, preds, output_dict=True, zero_division=0)
                res['kappa'] = cohen_kappa_score(labels, preds)
                results.append(res)
                
                wandb.log({"fold_results": res, "val_f1_macro": res["macro avg"]["f1-score"], "val_kappa": res['kappa']})
        
        avg_f1 = np.mean([r['macro avg']['f1-score'] for r in results])
        avg_kappa = np.mean([r['kappa'] for r in results])
        print(f"F1-score Médio: {avg_f1}")
        print(f"Kappa Médio: {avg_kappa}")
        
        with wandb.init(project=self.wandb_project, config=self.config.__dict__, 
                        group="cross_validation", name="cv_summary", reinit=True, 
                        job_type="summary"):
            wandb.log({"avg_f1_macro": avg_f1, "avg_kappa": avg_kappa})
            
        self.finish_wandb()

        return results

if __name__ == "__main__":
    import fire

    def train(config: str, training_data: str, save_model: str, wandb_project: str = None):
        """
        Treina o modelo com a configuração e exemplos especificados pelo CLI.
        
        :param config: Caminho para o arquivo YAML.
        :type config: str
        :param training_data: Caminho para os exemplos YAML.
        :type training_data: str
        :param save_model: Local para salvar a estrutura gerada (ex: modelo.keras)
        :type save_model: str   
        :param wandb_project: Projeto no W&B para envio de relatórios.
        :type wandb_project: str
        """
        classifier = IntentClassifier(config=config, training_data=training_data, wandb_project=wandb_project)
        classifier.train(save_model=save_model)
        print("Treinamento concluído com sucesso!")

    def predict(load_model: str, input_text: str, wandb_project: str = None):
        """
        Realiza predições utilizando um modelo já treinado.
        
        :param load_model: Caminho do disco ou URL W&B do arquivo .keras.
        :type load_model: str
        :param input_text: Frase de texto para avaliação.
        :type input_text: str
        :param wandb_project: Projeto no W&B para envio de métricas.
        :type wandb_project: str
        """
        classifier = IntentClassifier(load_model=load_model, wandb_project=wandb_project)
        predictions = classifier.predict(input_text)
        print(f"Predições: {predictions}")

    def cross_validation(config: str, training_data: str, n_splits: int = 3, wandb_project: str = None):
        """
        Executa a validação cruzada do modelo atual via terminal.
        
        :param config: Caminho para a configuração .yml.
        :type config: str
        :param training_data: Caminho para as intenções do dataset em .yml.
        :type training_data: str
        :param n_splits: Dobras (folds) a serem usadas para k-Fold.
        :type n_splits: int, opcional
        :param wandb_project: Projeto no W&B de destino.
        :type wandb_project: str
        """
        classifier = IntentClassifier(config=config, training_data=training_data, wandb_project=wandb_project)
        results = classifier.cross_validation(n_splits=n_splits)
        print("Validação Cruzada finalizada com sucesso!")
        pprint(results)

    fire.Fire({
        'train': train,
        'predict': predict,
        'cross_validation': cross_validation
    }, serialize=False)

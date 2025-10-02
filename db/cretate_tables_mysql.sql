-- ------------------------------------------------------------
-- Estufa Inteligente - MySQL DDL
-- ------------------------------------------------------------
SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- =========================
-- Tabelas "pais" primeiro
-- =========================

CREATE TABLE estufas (
  id                INT NOT NULL AUTO_INCREMENT,
  nome              VARCHAR(200) NOT NULL,
  localizacao       VARCHAR(200) NOT NULL,
  latitude          DECIMAL(10,8),
  longitude         DECIMAL(11,8),
  area_m2           DECIMAL(10,2),
  tipo_cultura      VARCHAR(50),
  data_instalacao   DATETIME NOT NULL,
  status            VARCHAR(20) DEFAULT 'ativa',
  observacoes       LONGTEXT,
  created_at        DATETIME(6),
  updated_at        DATETIME(6),
  PRIMARY KEY (id),
  KEY idx_status (status),
  KEY idx_tipo_cultura (tipo_cultura)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE tipos_sensores (
  id             INT NOT NULL AUTO_INCREMENT,
  codigo         VARCHAR(20) NOT NULL,
  nome           VARCHAR(50) NOT NULL,
  unidade_medida VARCHAR(20) NOT NULL,
  valor_min      DECIMAL(10,2),
  valor_max      DECIMAL(10,2),
  precisao       DECIMAL(5,2),
  descricao      LONGTEXT,
  PRIMARY KEY (id),
  UNIQUE KEY uk_tipos_sensores_codigo (codigo)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE configuracoes_sistema (
  id         INT NOT NULL AUTO_INCREMENT,
  chave      VARCHAR(50) NOT NULL,
  valor      LONGTEXT NOT NULL,
  tipo_data  VARCHAR(20) DEFAULT 'string',
  descricao  LONGTEXT,
  categoria  VARCHAR(50),
  editavel   CHAR(1) DEFAULT 'Y',
  updated_at DATETIME(6),
  PRIMARY KEY (id),
  UNIQUE KEY uk_configuracoes_sistema_chave (chave),
  KEY idx_chave (chave),
  KEY idx_categoria (categoria)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- =========================
-- Demais tabelas
-- =========================

CREATE TABLE sensores (
  id                         INT NOT NULL AUTO_INCREMENT,
  estufa_id                  INT NOT NULL,
  tipo_sensor_id             INT NOT NULL,
  codigo_sesnor              VARCHAR(50) NOT NULL,
  pino_gpio                  INT,
  localizacao_sensor         VARCHAR(100),
  data_instalacao            DATETIME NOT NULL,
  data_ultima_calibracao     DATETIME,
  intervalo_leitura_segundos INT DEFAULT 60,
  status                     VARCHAR(20) DEFAULT 'ativo',
  parametros_calibracao      LONGTEXT,
  PRIMARY KEY (id),
  UNIQUE KEY uk_sensores_codigo (codigo_sesnor),
  KEY idx_estufa (estufa_id),
  KEY idx_tipo (tipo_sensor_id),
  KEY idx_status_sensor (status),
  CONSTRAINT fk_sensores_estufas
    FOREIGN KEY (estufa_id) REFERENCES estufas (id)
    ON UPDATE RESTRICT ON DELETE RESTRICT,
  CONSTRAINT fk_sensores_tipos_sensores
    FOREIGN KEY (tipo_sensor_id) REFERENCES tipos_sensores (id)
    ON UPDATE RESTRICT ON DELETE RESTRICT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE leituras (
  id                INT NOT NULL AUTO_INCREMENT,
  sensor_id         INT NOT NULL,
  valor             DECIMAL(10,4) NOT NULL,
  timestamp_leitura DATETIME(6) NOT NULL,
  qualidade_sinal   INT,
  status_leitura    VARCHAR(20) DEFAULT 'normal',
  raw_data          LONGTEXT,
  created_at        DATETIME(6),
  PRIMARY KEY (id),
  KEY idx_sensor_timestamp (sensor_id, timestamp_leitura DESC),
  KEY idx_timestamp (timestamp_leitura DESC),
  KEY idx_status_leitura (status_leitura),
  CONSTRAINT fk_leituras_sensores
    FOREIGN KEY (sensor_id) REFERENCES sensores (id)
    ON UPDATE RESTRICT ON DELETE RESTRICT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE alertas (
  id                  INT NOT NULL AUTO_INCREMENT,
  leitura_id          INT NOT NULL,
  tipo_alerta         VARCHAR(50) NOT NULL,
  severidade          VARCHAR(20) NOT NULL,
  valor_lido          DECIMAL(10,4) NOT NULL,
  valor_esperado_min  DECIMAL(10,4),
  valor_esperado_max  DECIMAL(10,4),
  mensagem            LONGTEXT,
  timestamp_alerta    DATETIME(6) NOT NULL,
  timestamp_resolucao DATETIME(6),
  resolvido           CHAR(1) DEFAULT 'N',
  acao_tomada         LONGTEXT,
  usuario_responsavel VARCHAR(100),
  PRIMARY KEY (id),
  KEY idx_leitura (leitura_id),
  KEY idx_tipo_alerta (tipo_alerta),
  KEY idx_severidade (severidade),
  KEY idx_resolvido (resolvido),
  KEY idx_timestamp_alerta (timestamp_alerta DESC),
  CONSTRAINT fk_alertas_leituras
    FOREIGN KEY (leitura_id) REFERENCES leituras (id)
    ON UPDATE RESTRICT ON DELETE RESTRICT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE acoes_controle (
  id                 INT NOT NULL AUTO_INCREMENT,
  estufa_id          INT NOT NULL,
  alerta_id          INT NOT NULL,
  tipo_acao          VARCHAR(50) NOT NULL,
  comando_enviado    VARCHAR(200) NOT NULL,
  parametros_comando LONGTEXT,
  timestamp_execucao DATETIME(6) NOT NULL,
  duracao_segundos   INT,
  status_execucao    VARCHAR(20) NOT NULL,
  resultado          LONGTEXT,
  PRIMARY KEY (id),
  KEY idx_estufa_acao (estufa_id),
  KEY idx_alerta_acao (alerta_id),
  KEY idx_timestamp_exec (timestamp_execucao DESC),
  CONSTRAINT fk_acoes_controle_estufas
    FOREIGN KEY (estufa_id) REFERENCES estufas (id)
    ON UPDATE RESTRICT ON DELETE RESTRICT,
  CONSTRAINT fk_acoes_controle_alertas
    FOREIGN KEY (alerta_id) REFERENCES alertas (id)
    ON UPDATE RESTRICT ON DELETE RESTRICT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE manutencoes (
  id                  INT NOT NULL AUTO_INCREMENT,
  sensor_id           INT NOT NULL,
  tipo_manutencao     VARCHAR(50) NOT NULL,
  data_manutencao     DATETIME NOT NULL,
  hora_inicio         VARCHAR(8),
  hora_fim            VARCHAR(8),
  descricao           LONGTEXT NOT NULL,
  pecas_substituidas  LONGTEXT,
  custo               DECIMAL(10,2),
  tecnico_responsavel VARCHAR(100) NOT NULL,
  status              VARCHAR(20) DEFAULT 'agendada',
  observacoes         LONGTEXT,
  PRIMARY KEY (id),
  KEY idx_sensor_manut (sensor_id),
  KEY idx_data_manut (data_manutencao DESC),
  KEY idx_tipo_manut (tipo_manutencao),
  CONSTRAINT fk_manutencoes_sensores
    FOREIGN KEY (sensor_id) REFERENCES sensores (id)
    ON UPDATE RESTRICT ON DELETE RESTRICT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE parametros_culturas (
  id                INT NOT NULL AUTO_INCREMENT,
  tipo_cultura      VARCHAR(50) NOT NULL,
  tipo_sensor_id    INT NOT NULL,
  valor_min_ideal   DECIMAL(10,2) NOT NULL,
  valor_max_ideal   DECIMAL(10,2) NOT NULL,
  valor_min_critico DECIMAL(10,2),
  valor_max_critico DECIMAL(10,2),
  periodo           VARCHAR(20) DEFAULT 'ambos',
  estacao           VARCHAR(20) DEFAULT 'todas',
  PRIMARY KEY (id),
  KEY idx_cultura (tipo_cultura),
  KEY idx_tipo_sensor_param (tipo_sensor_id),
  CONSTRAINT fk_parametros_culturas_tipos_sensores
    FOREIGN KEY (tipo_sensor_id) REFERENCES tipos_sensores (id)
    ON UPDATE RESTRICT ON DELETE RESTRICT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE predicoes_ml (
  id                  INT NOT NULL AUTO_INCREMENT,
  estufa_id           INT NOT NULL,
  tipo_predicao       VARCHAR(50) NOT NULL,
  timestamp_predicao  DATETIME(6) NOT NULL,
  horizonte_horas     INT NOT NULL,
  valor_predito       VARCHAR(100) NOT NULL,
  confianca           DECIMAL(5,2),
  features_utilizadas LONGTEXT,
  modelo_versao       VARCHAR(20),
  acuracia_modelo     DECIMAL(5,2),
  PRIMARY KEY (id),
  KEY idx_estufa_pred (estufa_id),
  KEY idx_timestamp_pred (timestamp_predicao DESC),
  KEY idx_tipo_pred (tipo_predicao),
  CONSTRAINT fk_predicoes_ml_estufas
    FOREIGN KEY (estufa_id) REFERENCES estufas (id)
    ON UPDATE RESTRICT ON DELETE RESTRICT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

SET FOREIGN_KEY_CHECKS = 1;

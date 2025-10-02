-- Gerado por Oracle SQL Developer Data Modeler 24.3.1.351.0831
--   em:        2025-08-31 16:53:00 BRT
--   site:      Oracle Database 21c
--   tipo:      Oracle Database 21c



-- predefined type, no DDL - MDSYS.SDO_GEOMETRY

-- predefined type, no DDL - XMLTYPE

CREATE TABLE acoes_controle 
    ( 
     id                 INTEGER  NOT NULL , 
     estufa_id          INTEGER  NOT NULL , 
     alerta_id          INTEGER  NOT NULL , 
     tipo_acao          VARCHAR2 (50)  NOT NULL , 
     comando_enviado    VARCHAR2 (200)  NOT NULL , 
     parametros_comando CLOB , 
     timestamp_execucao TIMESTAMP WITH LOCAL TIME ZONE  NOT NULL , 
     duracao_segundos   INTEGER , 
     status_execucao    VARCHAR2 (20)  NOT NULL , 
     resultado          CLOB 
    ) 
;
CREATE INDEX idx_estufa_acao ON acoes_controle 
    ( 
     estufa_id ASC 
    ) 
;
CREATE INDEX "idx_alerta_acao " ON acoes_controle 
    ( 
     alerta_id ASC 
    ) 
;
CREATE INDEX idx_timestamp_exec ON acoes_controle 
    ( 
     timestamp_execucao DESC 
    ) 
;

ALTER TABLE acoes_controle 
    ADD CONSTRAINT acoes_controle_PK PRIMARY KEY ( id ) ;

CREATE TABLE alertas 
    ( 
     id                  INTEGER  NOT NULL , 
     leitura_id          INTEGER  NOT NULL , 
     tipo_alerta         VARCHAR2 (50)  NOT NULL , 
     severidade          VARCHAR2 (20)  NOT NULL , 
     valor_lido          NUMBER (10,4)  NOT NULL , 
     valor_esperado_min  NUMBER (10,4) , 
     valor_esperado_max  NUMBER (10,4) , 
     mensagem            CLOB , 
     timestamp_alerta    TIMESTAMP WITH LOCAL TIME ZONE  NOT NULL , 
     timestamp_resolucao TIMESTAMP WITH LOCAL TIME ZONE , 
     resolvido           CHAR (1) DEFAULT 'N' , 
     acao_tomada         CLOB , 
     usuario_responsavel VARCHAR2 (100) 
    ) 
;
CREATE INDEX idx_leitura ON alertas 
    ( 
     leitura_id ASC 
    ) 
;
CREATE INDEX idx_tipo_alerta ON alertas 
    ( 
     tipo_alerta ASC 
    ) 
;
CREATE INDEX idx_severidade ON alertas 
    ( 
     severidade ASC 
    ) 
;
CREATE INDEX idx_resolvido ON alertas 
    ( 
     resolvido ASC 
    ) 
;
CREATE INDEX idx_timestamp_alerta ON alertas 
    ( 
     timestamp_alerta DESC 
    ) 
;

ALTER TABLE alertas 
    ADD CONSTRAINT alertas_PK PRIMARY KEY ( id ) ;

CREATE TABLE configuracoes_sistema 
    ( 
     id         INTEGER  NOT NULL , 
     chave      VARCHAR2 (50)  NOT NULL , 
     valor      CLOB  NOT NULL , 
     tipo_data  VARCHAR2 (20) DEFAULT 'string' , 
     descricao  CLOB , 
     categoria  VARCHAR2 (50) , 
     editavel   CHAR (1) DEFAULT 'Y' , 
     updated_at TIMESTAMP WITH LOCAL TIME ZONE 
    ) 
;
CREATE INDEX idx_chave ON configuracoes_sistema 
    ( 
     chave ASC 
    ) 
;
CREATE INDEX idx_categoria ON configuracoes_sistema 
    ( 
     categoria ASC 
    ) 
;

ALTER TABLE configuracoes_sistema 
    ADD CONSTRAINT configuracoes_sistema_PK PRIMARY KEY ( id ) ;

ALTER TABLE configuracoes_sistema 
    ADD CONSTRAINT configuracoes_sistema__UN UNIQUE ( chave ) ;

CREATE TABLE estufas 
    ( 
     id              INTEGER  NOT NULL , 
     nome            VARCHAR2 (200)  NOT NULL , 
     localizacao     VARCHAR2 (200)  NOT NULL , 
     latitude        NUMBER (10,8) , 
     longitude       NUMBER (11,8) , 
     area_m2         NUMBER (10,2) , 
     tipo_cultura    VARCHAR2 (50) , 
     data_instalacao DATE  NOT NULL , 
     status          VARCHAR2 (20) DEFAULT ON NULL 'ativa' , 
     observacoes     CLOB , 
     created_at      TIMESTAMP WITH LOCAL TIME ZONE , 
     updated_at      TIMESTAMP WITH LOCAL TIME ZONE 
    ) 
;
CREATE INDEX idx_status ON estufas 
    ( 
     status ASC 
    ) 
;
CREATE INDEX idx_tipo_cultura ON estufas 
    ( 
     tipo_cultura ASC 
    ) 
;

ALTER TABLE estufas 
    ADD CONSTRAINT estufas_PK PRIMARY KEY ( id ) ;

CREATE TABLE leituras 
    ( 
     id                INTEGER  NOT NULL , 
     sensor_id         INTEGER  NOT NULL , 
     valor             NUMBER (10,4)  NOT NULL , 
     timestamp_leitura TIMESTAMP WITH LOCAL TIME ZONE  NOT NULL , 
     qualidade_sinal   INTEGER , 
     status_leitura    VARCHAR2 (20) DEFAULT 'normal' , 
     raw_data          CLOB , 
     created_at        TIMESTAMP WITH LOCAL TIME ZONE 
    ) 
;
CREATE INDEX idx_sensor_timestamp ON leituras 
    ( 
     sensor_id ASC , 
     timestamp_leitura DESC 
    ) 
;
CREATE INDEX idx_timestamp ON leituras 
    ( 
     timestamp_leitura DESC 
    ) 
;
CREATE INDEX idx_status_leitura ON leituras 
    ( 
     status_leitura ASC 
    ) 
;

ALTER TABLE leituras 
    ADD CONSTRAINT leituras_PK PRIMARY KEY ( id ) ;

CREATE TABLE manutencoes 
    ( 
     id                  INTEGER  NOT NULL , 
     sensor_id           INTEGER  NOT NULL , 
     tipo_manutencao     VARCHAR2 (50)  NOT NULL , 
     data_manutencao     DATE  NOT NULL , 
     hora_inicio         VARCHAR2 (8) , 
     hora_fim            VARCHAR2 (8) , 
     descricao           CLOB  NOT NULL , 
     pecas_substituidas  CLOB , 
     custo               NUMBER (10,2) , 
     tecnico_responsavel VARCHAR2 (100)  NOT NULL , 
     status              VARCHAR2 (20) DEFAULT 'agendada' , 
     observacoes         CLOB 
    ) 
;
CREATE INDEX idx_sensor_manut ON manutencoes 
    ( 
     sensor_id ASC 
    ) 
;
CREATE INDEX idx_data_manut ON manutencoes 
    ( 
     data_manutencao DESC 
    ) 
;
CREATE INDEX idx_tipo_manut ON manutencoes 
    ( 
     tipo_manutencao ASC 
    ) 
;

ALTER TABLE manutencoes 
    ADD CONSTRAINT manutencoes_PK PRIMARY KEY ( id ) ;

CREATE TABLE parametros_culturas 
    ( 
     id                INTEGER  NOT NULL , 
     tipo_cultura      VARCHAR2 (50)  NOT NULL , 
     tipo_sensor_id    INTEGER  NOT NULL , 
     valor_min_ideal   NUMBER (10,2)  NOT NULL , 
     valor_max_ideal   NUMBER (10,2)  NOT NULL , 
     valor_min_critico NUMBER (10,2) , 
     valor_max_critico NUMBER (10,2) , 
     periodo           VARCHAR2 (20) DEFAULT 'ambos' , 
     estacao           VARCHAR2 (20) DEFAULT 'todas' 
    ) 
;
CREATE INDEX idx_cultura ON parametros_culturas 
    ( 
     tipo_cultura ASC 
    ) 
;
CREATE INDEX idx_tipo_sensor_param ON parametros_culturas 
    ( 
     tipo_sensor_id ASC 
    ) 
;

ALTER TABLE parametros_culturas 
    ADD CONSTRAINT parametros_culturas_PK PRIMARY KEY ( id ) ;

CREATE TABLE predicoes_ml 
    ( 
     id                  INTEGER  NOT NULL , 
     estufa_id           INTEGER  NOT NULL , 
     tipo_predicao       VARCHAR2 (50)  NOT NULL , 
     timestamp_predicao  TIMESTAMP WITH LOCAL TIME ZONE  NOT NULL , 
     horizonte_horas     INTEGER  NOT NULL , 
     valor_predito       VARCHAR2 (100)  NOT NULL , 
     confianca           NUMBER (5,2) , 
     features_utilizadas CLOB , 
     modelo_versao       VARCHAR2 (20) , 
     acuracia_modelo     NUMBER (5,2) 
    ) 
;
CREATE INDEX idx_estufa_pred ON predicoes_ml 
    ( 
     estufa_id ASC 
    ) 
;
CREATE INDEX idx_timestamp_pred ON predicoes_ml 
    ( 
     timestamp_predicao DESC 
    ) 
;
CREATE INDEX idx_tipo_pred ON predicoes_ml 
    ( 
     tipo_predicao ASC 
    ) 
;

ALTER TABLE predicoes_ml 
    ADD CONSTRAINT predicoes_ml_PK PRIMARY KEY ( id ) ;

CREATE TABLE sensores 
    ( 
     id                         INTEGER  NOT NULL , 
     estufa_id                  INTEGER  NOT NULL , 
     tipo_sensor_id             INTEGER  NOT NULL , 
     codigo_sesnor              VARCHAR2 (50)  NOT NULL , 
     pino_gpio                  INTEGER , 
     localizacao_sensor         VARCHAR2 (100) , 
     data_instalacao            DATE  NOT NULL , 
     data_ultima_calibracao     DATE , 
     intervalo_leitura_segundos INTEGER DEFAULT 60 , 
     status                     VARCHAR2 (20) DEFAULT 'ativo' , 
     parametros_calibracao      CLOB 
    ) 
;
CREATE INDEX idx_estufa ON sensores 
    ( 
     estufa_id ASC 
    ) 
;
CREATE INDEX idx_tipo ON sensores 
    ( 
     tipo_sensor_id ASC 
    ) 
;
CREATE INDEX idx_status_sensor ON sensores 
    ( 
     status ASC 
    ) 
;
CREATE UNIQUE INDEX idx_codigo_sensor ON sensores 
    ( 
     codigo_sesnor ASC 
    ) 
;

ALTER TABLE sensores 
    ADD CONSTRAINT sensores_PK PRIMARY KEY ( id ) ;

ALTER TABLE sensores 
    ADD CONSTRAINT sensores__UN UNIQUE ( codigo_sesnor ) ;

CREATE TABLE tipos_sensores 
    ( 
     id             INTEGER  NOT NULL , 
     codigo         VARCHAR2 (20)  NOT NULL , 
     nome           VARCHAR2 (50)  NOT NULL , 
     unidade_medida VARCHAR2 (20)  NOT NULL , 
     valor_min      NUMBER (10,2) , 
     valor_max      NUMBER (10,2) , 
     precisao       NUMBER (5,2) , 
     descricao      CLOB 
    ) 
;

ALTER TABLE tipos_sensores 
    ADD CONSTRAINT tipos_sensores_PK PRIMARY KEY ( id ) ;

ALTER TABLE tipos_sensores 
    ADD CONSTRAINT tipos_sensores__UN UNIQUE ( codigo ) ;

ALTER TABLE acoes_controle 
    ADD CONSTRAINT acoes_controle_alertas_FK FOREIGN KEY 
    ( 
     alerta_id
    ) 
    REFERENCES alertas 
    ( 
     id
    ) 
;

ALTER TABLE acoes_controle 
    ADD CONSTRAINT acoes_controle_estufas_FK FOREIGN KEY 
    ( 
     estufa_id
    ) 
    REFERENCES estufas 
    ( 
     id
    ) 
;

ALTER TABLE alertas 
    ADD CONSTRAINT alertas_leituras_FK FOREIGN KEY 
    ( 
     leitura_id
    ) 
    REFERENCES leituras 
    ( 
     id
    ) 
;

ALTER TABLE leituras 
    ADD CONSTRAINT leituras_sensores_FK FOREIGN KEY 
    ( 
     sensor_id
    ) 
    REFERENCES sensores 
    ( 
     id
    ) 
;

ALTER TABLE manutencoes 
    ADD CONSTRAINT manutencoes_sensores_FK FOREIGN KEY 
    ( 
     sensor_id
    ) 
    REFERENCES sensores 
    ( 
     id
    ) 
;

--  ERROR: FK name length exceeds maximum allowed length(30) 
ALTER TABLE parametros_culturas 
    ADD CONSTRAINT parametros_culturas_tipos_sensores_FK FOREIGN KEY 
    ( 
     tipo_sensor_id
    ) 
    REFERENCES tipos_sensores 
    ( 
     id
    ) 
;

ALTER TABLE predicoes_ml 
    ADD CONSTRAINT predicoes_ml_estufas_FK FOREIGN KEY 
    ( 
     estufa_id
    ) 
    REFERENCES estufas 
    ( 
     id
    ) 
;

ALTER TABLE sensores 
    ADD CONSTRAINT sensores_estufas_FK FOREIGN KEY 
    ( 
     estufa_id
    ) 
    REFERENCES estufas 
    ( 
     id
    ) 
;

ALTER TABLE sensores 
    ADD CONSTRAINT sensores_tipos_sensores_FK FOREIGN KEY 
    ( 
     tipo_sensor_id
    ) 
    REFERENCES tipos_sensores 
    ( 
     id
    ) 
;



-- Relatório do Resumo do Oracle SQL Developer Data Modeler: 
-- 
-- CREATE TABLE                            10
-- CREATE INDEX                            27
-- ALTER TABLE                             22
-- CREATE VIEW                              0
-- ALTER VIEW                               0
-- CREATE PACKAGE                           0
-- CREATE PACKAGE BODY                      0
-- CREATE PROCEDURE                         0
-- CREATE FUNCTION                          0
-- CREATE TRIGGER                           0
-- ALTER TRIGGER                            0
-- CREATE COLLECTION TYPE                   0
-- CREATE STRUCTURED TYPE                   0
-- CREATE STRUCTURED TYPE BODY              0
-- CREATE CLUSTER                           0
-- CREATE CONTEXT                           0
-- CREATE DATABASE                          0
-- CREATE DIMENSION                         0
-- CREATE DIRECTORY                         0
-- CREATE DISK GROUP                        0
-- CREATE ROLE                              0
-- CREATE ROLLBACK SEGMENT                  0
-- CREATE SEQUENCE                          0
-- CREATE MATERIALIZED VIEW                 0
-- CREATE MATERIALIZED VIEW LOG             0
-- CREATE SYNONYM                           0
-- CREATE TABLESPACE                        0
-- CREATE USER                              0
-- 
-- DROP TABLESPACE                          0
-- DROP DATABASE                            0
-- 
-- REDACTION POLICY                         0
-- 
-- ORDS DROP SCHEMA                         0
-- ORDS ENABLE SCHEMA                       0
-- ORDS ENABLE OBJECT                       0
-- 
-- ERRORS                                   1
-- WARNINGS                                 0

-- Gerado por Oracle SQL Developer Data Modeler 24.3.1.351.0831
--   em:        2025-08-31 16:53:43 BRT
--   site:      SQL Server 2012
--   tipo:      SQL Server 2012



CREATE TABLE acoes_controle 
    (
     id BIGINT NOT NULL , 
     estufa_id BIGINT NOT NULL , 
     alerta_id BIGINT NOT NULL , 
     tipo_acao VARCHAR (50) NOT NULL , 
     comando_enviado VARCHAR (200) NOT NULL , 
     parametros_comando VARCHAR (max) , 
     timestamp_execucao DATETIME NOT NULL , 
     duracao_segundos INTEGER , 
     status_execucao VARCHAR (20) NOT NULL , 
     resultado VARCHAR (max) 
    )
GO 

    


CREATE NONCLUSTERED INDEX 
    idx_estufa_acao ON acoes_controle 
    ( 
     estufa_id 
    ) 
GO 


CREATE NONCLUSTERED INDEX 
    "idx_alerta_acao " ON acoes_controle 
    ( 
     alerta_id 
    ) 
GO 


CREATE NONCLUSTERED INDEX 
    idx_timestamp_exec ON acoes_controle 
    ( 
     timestamp_execucao DESC 
    ) 
GO

ALTER TABLE acoes_controle ADD CONSTRAINT acoes_controle_PK PRIMARY KEY CLUSTERED (id)
     WITH (
     ALLOW_PAGE_LOCKS = ON , 
     ALLOW_ROW_LOCKS = ON )
GO

CREATE TABLE alertas 
    (
     id BIGINT NOT NULL , 
     leitura_id BIGINT NOT NULL , 
     tipo_alerta VARCHAR (50) NOT NULL , 
     severidade VARCHAR (20) NOT NULL , 
     valor_lido NUMERIC (10,4) NOT NULL , 
     valor_esperado_min NUMERIC (10,4) , 
     valor_esperado_max NUMERIC (10,4) , 
     mensagem VARCHAR (max) , 
     timestamp_alerta DATETIME NOT NULL , 
     timestamp_resolucao DATETIME , 
     resolvido CHAR (1) DEFAULT 'N' , 
     acao_tomada VARCHAR (max) , 
     usuario_responsavel VARCHAR (100) 
    )
GO 

    


CREATE NONCLUSTERED INDEX 
    idx_leitura ON alertas 
    ( 
     leitura_id 
    ) 
GO 


CREATE NONCLUSTERED INDEX 
    idx_tipo_alerta ON alertas 
    ( 
     tipo_alerta 
    ) 
GO 


CREATE NONCLUSTERED INDEX 
    idx_severidade ON alertas 
    ( 
     severidade 
    ) 
GO 


CREATE NONCLUSTERED INDEX 
    idx_resolvido ON alertas 
    ( 
     resolvido 
    ) 
GO 


CREATE NONCLUSTERED INDEX 
    idx_timestamp_alerta ON alertas 
    ( 
     timestamp_alerta DESC 
    ) 
GO

ALTER TABLE alertas ADD CONSTRAINT alertas_PK PRIMARY KEY CLUSTERED (id)
     WITH (
     ALLOW_PAGE_LOCKS = ON , 
     ALLOW_ROW_LOCKS = ON )
GO

CREATE TABLE configuracoes_sistema 
    (
     id BIGINT NOT NULL , 
     chave VARCHAR (50) NOT NULL , 
     valor VARCHAR (max) NOT NULL , 
     tipo_data VARCHAR (20) DEFAULT 'string' , 
     descricao VARCHAR (max) , 
     categoria VARCHAR (50) , 
     editavel CHAR (1) DEFAULT 'Y' , 
     updated_at DATETIME 
    )
GO 

    


CREATE NONCLUSTERED INDEX 
    idx_chave ON configuracoes_sistema 
    ( 
     chave 
    ) 
GO 


CREATE NONCLUSTERED INDEX 
    idx_categoria ON configuracoes_sistema 
    ( 
     categoria 
    ) 
GO

ALTER TABLE configuracoes_sistema ADD CONSTRAINT configuracoes_sistema_PK PRIMARY KEY CLUSTERED (id)
     WITH (
     ALLOW_PAGE_LOCKS = ON , 
     ALLOW_ROW_LOCKS = ON )
GO
ALTER TABLE configuracoes_sistema ADD CONSTRAINT configuracoes_sistema__UN UNIQUE NONCLUSTERED (chave)
GO

CREATE TABLE estufas 
    (
     id BIGINT NOT NULL , 
     nome VARCHAR (200) NOT NULL , 
     localizacao VARCHAR (200) NOT NULL , 
     latitude NUMERIC (10,8) , 
     longitude NUMERIC (11,8) , 
     area_m2 NUMERIC (10,2) , 
     tipo_cultura VARCHAR (50) , 
     data_instalacao DATE NOT NULL , 
     status VARCHAR (20) DEFAULT 'ativa' , 
     observacoes VARCHAR (max) , 
     created_at DATETIME , 
     updated_at DATETIME 
    )
GO 

    


CREATE NONCLUSTERED INDEX 
    idx_status ON estufas 
    ( 
     status 
    ) 
GO 


CREATE NONCLUSTERED INDEX 
    idx_tipo_cultura ON estufas 
    ( 
     tipo_cultura 
    ) 
GO

ALTER TABLE estufas ADD CONSTRAINT estufas_PK PRIMARY KEY CLUSTERED (id)
     WITH (
     ALLOW_PAGE_LOCKS = ON , 
     ALLOW_ROW_LOCKS = ON )
GO

CREATE TABLE leituras 
    (
     id BIGINT NOT NULL , 
     sensor_id BIGINT NOT NULL , 
     valor NUMERIC (10,4) NOT NULL , 
     timestamp_leitura DATETIME NOT NULL , 
     qualidade_sinal INTEGER , 
     status_leitura VARCHAR (20) DEFAULT 'normal' , 
     raw_data VARCHAR (max) , 
     created_at DATETIME 
    )
GO 

    


CREATE NONCLUSTERED INDEX 
    idx_sensor_timestamp ON leituras 
    ( 
     sensor_id , 
     timestamp_leitura DESC 
    ) 
GO 


CREATE NONCLUSTERED INDEX 
    idx_timestamp ON leituras 
    ( 
     timestamp_leitura DESC 
    ) 
GO 


CREATE NONCLUSTERED INDEX 
    idx_status_leitura ON leituras 
    ( 
     status_leitura 
    ) 
GO

ALTER TABLE leituras ADD CONSTRAINT leituras_PK PRIMARY KEY CLUSTERED (id)
     WITH (
     ALLOW_PAGE_LOCKS = ON , 
     ALLOW_ROW_LOCKS = ON )
GO

CREATE TABLE manutencoes 
    (
     id BIGINT NOT NULL , 
     sensor_id BIGINT NOT NULL , 
     tipo_manutencao VARCHAR (50) NOT NULL , 
     data_manutencao DATE NOT NULL , 
     hora_inicio VARCHAR (8) , 
     hora_fim VARCHAR (8) , 
     descricao VARCHAR (max) NOT NULL , 
     pecas_substituidas VARCHAR (max) , 
     custo NUMERIC (10,2) , 
     tecnico_responsavel VARCHAR (100) NOT NULL , 
     status VARCHAR (20) DEFAULT 'agendada' , 
     observacoes VARCHAR (max) 
    )
GO 

    


CREATE NONCLUSTERED INDEX 
    idx_sensor_manut ON manutencoes 
    ( 
     sensor_id 
    ) 
GO 


CREATE NONCLUSTERED INDEX 
    idx_data_manut ON manutencoes 
    ( 
     data_manutencao DESC 
    ) 
GO 


CREATE NONCLUSTERED INDEX 
    idx_tipo_manut ON manutencoes 
    ( 
     tipo_manutencao 
    ) 
GO

ALTER TABLE manutencoes ADD CONSTRAINT manutencoes_PK PRIMARY KEY CLUSTERED (id)
     WITH (
     ALLOW_PAGE_LOCKS = ON , 
     ALLOW_ROW_LOCKS = ON )
GO

CREATE TABLE parametros_culturas 
    (
     id BIGINT NOT NULL , 
     tipo_cultura VARCHAR (50) NOT NULL , 
     tipo_sensor_id BIGINT NOT NULL , 
     valor_min_ideal NUMERIC (10,2) NOT NULL , 
     valor_max_ideal NUMERIC (10,2) NOT NULL , 
     valor_min_critico NUMERIC (10,2) , 
     valor_max_critico NUMERIC (10,2) , 
     periodo VARCHAR (20) DEFAULT 'ambos' , 
     estacao VARCHAR (20) DEFAULT 'todas' 
    )
GO 

    


CREATE NONCLUSTERED INDEX 
    idx_cultura ON parametros_culturas 
    ( 
     tipo_cultura 
    ) 
GO 


CREATE NONCLUSTERED INDEX 
    idx_tipo_sensor_param ON parametros_culturas 
    ( 
     tipo_sensor_id 
    ) 
GO

ALTER TABLE parametros_culturas ADD CONSTRAINT parametros_culturas_PK PRIMARY KEY CLUSTERED (id)
     WITH (
     ALLOW_PAGE_LOCKS = ON , 
     ALLOW_ROW_LOCKS = ON )
GO

CREATE TABLE predicoes_ml 
    (
     id BIGINT NOT NULL , 
     estufa_id BIGINT NOT NULL , 
     tipo_predicao VARCHAR (50) NOT NULL , 
     timestamp_predicao DATETIME NOT NULL , 
     horizonte_horas INTEGER NOT NULL , 
     valor_predito VARCHAR (100) NOT NULL , 
     confianca NUMERIC (5,2) , 
     features_utilizadas VARCHAR (max) , 
     modelo_versao VARCHAR (20) , 
     acuracia_modelo NUMERIC (5,2) 
    )
GO 

    


CREATE NONCLUSTERED INDEX 
    idx_estufa_pred ON predicoes_ml 
    ( 
     estufa_id 
    ) 
GO 


CREATE NONCLUSTERED INDEX 
    idx_timestamp_pred ON predicoes_ml 
    ( 
     timestamp_predicao DESC 
    ) 
GO 


CREATE NONCLUSTERED INDEX 
    idx_tipo_pred ON predicoes_ml 
    ( 
     tipo_predicao 
    ) 
GO

ALTER TABLE predicoes_ml ADD CONSTRAINT predicoes_ml_PK PRIMARY KEY CLUSTERED (id)
     WITH (
     ALLOW_PAGE_LOCKS = ON , 
     ALLOW_ROW_LOCKS = ON )
GO

CREATE TABLE sensores 
    (
     id BIGINT NOT NULL , 
     estufa_id BIGINT NOT NULL , 
     tipo_sensor_id BIGINT NOT NULL , 
     codigo_sesnor VARCHAR (50) NOT NULL , 
     pino_gpio INTEGER , 
     localizacao_sensor VARCHAR (100) , 
     data_instalacao DATE NOT NULL , 
     data_ultima_calibracao DATE , 
     intervalo_leitura_segundos INTEGER DEFAULT 60 , 
     status VARCHAR (20) DEFAULT 'ativo' , 
     parametros_calibracao VARCHAR (max) 
    )
GO 

    


CREATE NONCLUSTERED INDEX 
    idx_estufa ON sensores 
    ( 
     estufa_id 
    ) 
GO 


CREATE NONCLUSTERED INDEX 
    idx_tipo ON sensores 
    ( 
     tipo_sensor_id 
    ) 
GO 


CREATE NONCLUSTERED INDEX 
    idx_status_sensor ON sensores 
    ( 
     status 
    ) 
GO 


CREATE UNIQUE NONCLUSTERED INDEX 
    idx_codigo_sensor ON sensores 
    ( 
     codigo_sesnor 
    ) 
GO

ALTER TABLE sensores ADD CONSTRAINT sensores_PK PRIMARY KEY CLUSTERED (id)
     WITH (
     ALLOW_PAGE_LOCKS = ON , 
     ALLOW_ROW_LOCKS = ON )
GO
ALTER TABLE sensores ADD CONSTRAINT sensores__UN UNIQUE NONCLUSTERED (codigo_sesnor)
GO

CREATE TABLE tipos_sensores 
    (
     id BIGINT NOT NULL , 
     codigo VARCHAR (20) NOT NULL , 
     nome VARCHAR (50) NOT NULL , 
     unidade_medida VARCHAR (20) NOT NULL , 
     valor_min NUMERIC (10,2) , 
     valor_max NUMERIC (10,2) , 
     precisao NUMERIC (5,2) , 
     descricao VARCHAR (max) 
    )
GO

ALTER TABLE tipos_sensores ADD CONSTRAINT tipos_sensores_PK PRIMARY KEY CLUSTERED (id)
     WITH (
     ALLOW_PAGE_LOCKS = ON , 
     ALLOW_ROW_LOCKS = ON )
GO
ALTER TABLE tipos_sensores ADD CONSTRAINT tipos_sensores__UN UNIQUE NONCLUSTERED (codigo)
GO

ALTER TABLE acoes_controle 
    ADD CONSTRAINT acoes_controle_alertas_FK FOREIGN KEY 
    ( 
     alerta_id
    ) 
    REFERENCES alertas 
    ( 
     id 
    ) 
    ON DELETE NO ACTION 
    ON UPDATE NO ACTION 
GO

ALTER TABLE acoes_controle 
    ADD CONSTRAINT acoes_controle_estufas_FK FOREIGN KEY 
    ( 
     estufa_id
    ) 
    REFERENCES estufas 
    ( 
     id 
    ) 
    ON DELETE NO ACTION 
    ON UPDATE NO ACTION 
GO

ALTER TABLE alertas 
    ADD CONSTRAINT alertas_leituras_FK FOREIGN KEY 
    ( 
     leitura_id
    ) 
    REFERENCES leituras 
    ( 
     id 
    ) 
    ON DELETE NO ACTION 
    ON UPDATE NO ACTION 
GO

ALTER TABLE leituras 
    ADD CONSTRAINT leituras_sensores_FK FOREIGN KEY 
    ( 
     sensor_id
    ) 
    REFERENCES sensores 
    ( 
     id 
    ) 
    ON DELETE NO ACTION 
    ON UPDATE NO ACTION 
GO

ALTER TABLE manutencoes 
    ADD CONSTRAINT manutencoes_sensores_FK FOREIGN KEY 
    ( 
     sensor_id
    ) 
    REFERENCES sensores 
    ( 
     id 
    ) 
    ON DELETE NO ACTION 
    ON UPDATE NO ACTION 
GO

ALTER TABLE parametros_culturas 
    ADD CONSTRAINT parametros_culturas_tipos_sensores_FK FOREIGN KEY 
    ( 
     tipo_sensor_id
    ) 
    REFERENCES tipos_sensores 
    ( 
     id 
    ) 
    ON DELETE NO ACTION 
    ON UPDATE NO ACTION 
GO

ALTER TABLE predicoes_ml 
    ADD CONSTRAINT predicoes_ml_estufas_FK FOREIGN KEY 
    ( 
     estufa_id
    ) 
    REFERENCES estufas 
    ( 
     id 
    ) 
    ON DELETE NO ACTION 
    ON UPDATE NO ACTION 
GO

ALTER TABLE sensores 
    ADD CONSTRAINT sensores_estufas_FK FOREIGN KEY 
    ( 
     estufa_id
    ) 
    REFERENCES estufas 
    ( 
     id 
    ) 
    ON DELETE NO ACTION 
    ON UPDATE NO ACTION 
GO

ALTER TABLE sensores 
    ADD CONSTRAINT sensores_tipos_sensores_FK FOREIGN KEY 
    ( 
     tipo_sensor_id
    ) 
    REFERENCES tipos_sensores 
    ( 
     id 
    ) 
    ON DELETE NO ACTION 
    ON UPDATE NO ACTION 
GO



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
-- CREATE DATABASE                          0
-- CREATE DEFAULT                           0
-- CREATE INDEX ON VIEW                     0
-- CREATE ROLLBACK SEGMENT                  0
-- CREATE ROLE                              0
-- CREATE RULE                              0
-- CREATE SCHEMA                            0
-- CREATE SEQUENCE                          0
-- CREATE PARTITION FUNCTION                0
-- CREATE PARTITION SCHEME                  0
-- 
-- DROP DATABASE                            0
-- 
-- ERRORS                                   0
-- WARNINGS                                 0

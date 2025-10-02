-- =====================================================
-- INSERIR DADOS INICIAIS - TIPOS DE SENSORES
-- =====================================================
INSERT INTO tipos_sensores (codigo, nome, unidade_medida, valor_min, valor_max, precisao, descricao) VALUES
('TEMP_AR', 'Temperatura do Ar', '°C', -10.0, 50.0, 0.5, 'Sensor DHT22 - Temperatura ambiente'),
('UMID_AR', 'Umidade do Ar', '%', 0.0, 100.0, 2.0, 'Sensor DHT22 - Umidade relativa do ar'),
('TEMP_SOLO', 'Temperatura do Solo', '°C', -10.0, 50.0, 0.5, 'Sensor DS18B20 - Temperatura do solo'),
('UMID_SOLO', 'Umidade do Solo', '%', 0.0, 100.0, 3.0, 'Sensor Capacitivo - Umidade do solo'),
('LUX', 'Luminosidade', 'lux', 0.0, 100000.0, 10.0, 'Sensor LDR - Intensidade luminosa'),
('CO2', 'Dióxido de Carbono', 'ppm', 350.0, 2000.0, 50.0, 'Sensor MQ-135 - Concentração de CO2');

-- =====================================================
-- INSERIR CONFIGURAÇÕES INICIAIS DO SISTEMA
-- =====================================================
INSERT INTO configuracoes_sistema (chave, valor, tipo_data, descricao, categoria) VALUES
('intervalo_leitura_padrao', '60', 'integer', 'Intervalo padrão entre leituras em segundos', 'sensores'),
('limite_alertas_por_hora', '10', 'integer', 'Número máximo de alertas por hora por sensor', 'alertas'),
('tempo_retencao_leituras_dias', '365', 'integer', 'Tempo de retenção de leituras em dias', 'manutencao'),
('modelo_ml_ativo', 'irrigation_predictor_v1', 'string', 'Versão do modelo ML em produção', 'ml'),
('threshold_confianca_ml', '75.0', 'decimal', 'Confiança mínima para aceitar predição', 'ml'),
('email_alertas_criticos', 'admin@estufa.com', 'string', 'Email para alertas críticos', 'notificacoes'),
('backup_automatico', 'true', 'boolean', 'Habilitar backup automático', 'manutencao'),
('hora_backup', '03:00', 'string', 'Horário do backup automático', 'manutencao');

-- =====================================================
-- INSERIR DADOS INICIAIS - ESTUFAS
-- =====================================================
INSERT INTO estufas (id, nome, localizacao, data_instalacao, status, created_at)
VALUES (1, 'Estufa A', 'Laboratório', NOW(), 'ativa', NOW())
ON DUPLICATE KEY UPDATE nome=VALUES(nome), localizacao=VALUES(localizacao);

-- =====================================================
-- INSERIR DADOS INICIAIS - SENSORES
-- =====================================================
INSERT INTO sensores (id, estufa_id, tipo_sensor_id, codigo_sesnor, pino_gpio, localizacao_sensor, data_instalacao, status)
VALUES
(101, 1, (SELECT id FROM tipos_sensores WHERE codigo='TEMP_AR') , 'DHT22_TEMP_AR_ESP32_01', 4 , 'Topo Estufa', NOW(), 'ativo'),
(102, 1, (SELECT id FROM tipos_sensores WHERE codigo='UMID_AR') , 'DHT22_UMID_AR_ESP32_01', 4 , 'Topo Estufa', NOW(), 'ativo'),
(103, 1, (SELECT id FROM tipos_sensores WHERE codigo='TEMP_SOLO'), 'DS18B20_TEMP_SOLO_01' , 5 , 'Solo Vaso A', NOW(), 'ativo'),
(104, 1, (SELECT id FROM tipos_sensores WHERE codigo='UMID_SOLO'), 'CAP_SOLO_ESP32_01'    , 35, 'Solo Vaso A', NOW(), 'ativo'),
(105, 1, (SELECT id FROM tipos_sensores WHERE codigo='LUX')     , 'LDR_LUX_ESP32_01'      , 34, 'Lateral', NOW(), 'ativo'),
(106, 1, (SELECT id FROM tipos_sensores WHERE codigo='CO2')     , 'MQ135_CO2_ESP32_01'    , 32, 'Centro', NOW(), 'ativo')
ON DUPLICATE KEY UPDATE codigo_sesnor=codigo_sesnor;

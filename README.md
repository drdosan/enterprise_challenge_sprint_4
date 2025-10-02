# FIAP - Faculdade de Informática e Administração Paulista 

<p align="center">
<a href= "https://www.fiap.com.br/"><img src="assets/logo-fiap.png" alt="FIAP - Faculdade de Informática e Admnistração Paulista" border="0" width=40% height=40%></a>
</p>

<br>

# 🌱 FASE 5 - ENTERPRISE CHALLENGE (SPRINT 4)

### ▶️ Vídeo de Evidência do Funcionamento do Projeto
👉 [Link do vídeo no YouTube (não listado)](https://youtube.com)

---

## 👨‍🎓 Integrantes
| Matrícula | Aluno                           |
|-----------|---------------------------------|
| RM 565497 | Vera Maria Chaves de Souza      |
| RM 565286 | Diogo Rebello dos Santos        |

## 👩‍🏫 Professores
- **Tutor(a):** Leonardo Ruiz Orabona  
- **Coordenador(a):** André Godoi Chiovato  

---

## 📜 Descrição

Estufa Inteligente com IoT, Banco de Dados, Machine Learning e Dashboard.

Este projeto integra:
- **IoT (ESP32 no Wokwi/PlatformIO)** para coleta de dados ambientais.
- **API em Flask** para ingestão e persistência dos dados.
- **Banco relacional (MySQL/Oracle/SQL Server)** para armazenamento.
- **Machine Learning (Scikit-Learn)** para previsão de irrigação.
- **Dashboard em Streamlit** para KPIs e alertas em tempo real.

Objetivo: simular uma solução **fim-a-fim** de Agricultura 4.0.

---

## 📁 Estrutura de Pastas

```
ENTERPRISE_FINAL/
│
├── assets/                  # Recursos visuais (diagramas, logos)
│
├── db/                      # Scripts SQL
│
├── docs/arquitetura/        # Documentação da arquitetura
│
├── ingest/                  # Camada de ingestão
│   ├── api/                 # API Flask
│   └── esp32/               # Simulação IoT no Wokwi
│
├── ml/                      # Machine Learning
│
├── dashboard/               # Visualização (Streamlit)
│
├── README.md                # Documentação principal
```

---

## ⚙️ 1. API (Flask)

### Como rodar:
```bash
cd ingest/api
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
pip install --no-cache-dir -r requirements.txt
python app.py
```

👉 Acesse em: http://localhost:5000/apidocs

**Principais endpoints:**
- `POST /leituras` → Envia leituras do ESP32
- `GET /leituras` → Consulta últimas leituras
- `GET /status` → Status da irrigação

---

## ⚙️ 2. Banco de Dados

Scripts em `/db`.

### Modelo Relacional:

<img src="assets/diagrama_er.png" alt="Simulação ESP32 no Wokwi" width="600"/>

### Exemplo (MySQL):
```bash
mysql -u root -p < db/create_tables_mysql.sql
mysql -u root -p < db/insert_sample_data.sql
```

---

## ⚙️ 3. ESP32 + Wokwi

<img src="assets/esp32.png" alt="Simulação ESP32 no Wokwi" width="600"/>

Na pasta `/ingest/esp32`.  
Inclui simulação com:
- 🌡️ DHT22 (Temp/Umidade do ar)  
- 🌱 CAPACITIVO (Umidade do solo)  
- 💡 LDR (Luminosidade)  
- 🌡️ LM35 (Temp do solo)  
- 🟢 MQ135 (Qualidade do ar / CO₂)  

---

## ⚙️ 4. Machine Learning

Na pasta `/ml`.

### Treino:
```bash
cd ml
pip install -r requirements.txt
python train_model.py
```

### Predição:
```bash
python predict_model.py
```

Gera modelo salvo em `ml/models/model.joblib`.

---

## ⚙️ 5. Dashboard (Streamlit)

### Como rodar:
```bash
cd dashboard
pip install -r requirements.txt
streamlit run app.py
```

👉 Acesse em: http://localhost:8501

Exibe:
- KPIs em tempo real
- Séries temporais
- Alerta de irrigação (umidade do solo < threshold)
- Predição do modelo ML em tempo real

---

## 📊 Arquitetura Final

<img src="docs/arquitetura/diagrama_projeto.png" alt="Arquitetura Final" width="700"/>

Fluxo:  
**ESP32 → API Flask → Banco → Machine Learning → Dashboard/Alertas**

---

## 📌 Tecnologias

- **IoT/Hardware**: ESP32, Wokwi, PlatformIO  
- **API**: Python Flask + Flasgger  
- **Banco**: MySQL / Oracle 21c / SQL Server  
- **ML**: Scikit-Learn + Joblib  
- **Dashboard**: Streamlit  
- **Docs**: Diagrams.net, Swagger, ER Diagram  

---

## 🚀 Execução ponta a ponta

1. Configure o banco e rode os scripts.  
2. Rode a API Flask.  
3. Simule o ESP32 no Wokwi.  
4. Treine o modelo em `/ml/train_model.py`.  
5. Abra o dashboard (`streamlit run dashboard/app.py`).  
6. Veja KPIs, alertas e predições funcionando.  

---

## 🗃 Histórico
* 0.2.0 - 02/10/2025 - Versão final (Sprint 4)
* 0.1.0 - 01/09/2025 - Protótipo (Sprint 3)

---

## 📋 Licença
MIT / FIAP Template

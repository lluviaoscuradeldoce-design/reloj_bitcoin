# 🚀 Sniper Bot Pro - Advanced Crypto Trading Bot

![Status](https://img.shields.io/badge/Status-Active-brightgreen)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Binance](https://img.shields.io/badge/API-Binance%20Futures-yellow)

Un bot de trading algorítmico avanzado para criptomonedas, optimizado para **Binance Futures**. Incluye simulación (Paper Trading), gestión de riesgo profesional, y un ecosistema completo de Machine Learning.

---

## 🔥 Características Principales

### 🧠 Estrategia Inteligente
- **Core:** Sistema de puntuación multi-indicador (-7 a +7).
- **Indicadores:** MACD, RSI, StochRSI, Bollinger Bands, CVD Momentum, Trend Alignment (1H/15M).
- **Señales:** Compra (Score ≥ 4) y Venta (Score ≤ -4).

### 🛡️ Gestión de Riesgo Profesional
- **Dynamic Position Sizing:** Arriesga solo el 1% de la cuenta por operación.
- **Trailing Stop Triple Nivel:**
  1.  ⚡ **Break Even:** 0.3 ATR de ganancia.
  2.  🔒 **Secure Profit:** Asegura el 25% al llegar a 1.0 ATR.
  3.  🎯 **Aggressive Trail:** Persigue el precio a 0.5 ATR.
- **Circuit Breaker:** Detiene el trading si la pérdida diaria supera los $500.

### 🤖 Ecosistema de Machine Learning (Nuevo)
- **`data_collector.py`**: Descarga y etiqueta datos históricos de Binance.
- **`ml_trainer.py`**: Entrena modelos de IA (Random Forest) para predecir el éxito de los trades.
- **`optimizer.py`**: Encuentra automáticamente la mejor configuración (Stop Loss, TP) mediante Grid Search.
- **`backtester.py`**: Simula meses de trading en segundos.

---

## 🛠️ Instalación y Uso

### Requisitos
```bash
pip install tk websocket-client numpy scikit-learn
```

### 1. Ejecutar el Bot (Paper Trading)
```bash
python crypto_widget.py
```
*El bot iniciará con $10,000 virtuales. Abre operaciones automáticamente.*

### 2. Optimizar la Estrategia
Encuentra los mejores parámetros para el mercado actual:
```bash
python optimizer.py --days 14
```

### 3. Entrenar la Inteligencia Artificial
```bash
# Paso 1: Recolectar datos
python data_collector.py --days 30

# Paso 2: Entrenar modelo
python ml_trainer.py
```

---

## 📊 Estructura del Proyecto

File | Descripción
---|---
`crypto_widget.py` | 🖥️ **Core:** Bot principal e Interfaz Gráfica.
`config.py` | ⚙️ **Configuración:** Pares, riesgo, claves API.
`backtester.py` | ⏪ **Simulación:** Motor de backtesting rápido.
`optimizer.py` | ⚡ **Optimización:** Buscador de hiperparámetros.
`ml_trainer.py` | 🧠 **IA:** Entrenador de modelos de predicción.
`data_collector.py` | 📥 **Datos:** Descarga historial de Binance.
`notifications.py` | 📱 **Alertas:** Integración con Telegram.

---

## ⚠️ Disclaimer
Este software es para fines educativos y de investigación. El trading de criptomonedas conlleva un alto riesgo de pérdida de capital. El autor no se hace responsable de ninguna pérdida financiera.

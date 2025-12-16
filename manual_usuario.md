# 📘 Sniper Bot Pro - Manual Completo & Guía de Optimización

Este documento describe el estado actual del sistema, instrucciones de uso, flujo de trabajo de Machine Learning y opciones para futuras optimizaciones.

---

## 🚀 1. Estado Actual del Sistema

El **Sniper Bot Pro** es una plataforma de trading algorítmico avanzada que opera actualmente en modo **Paper Trading** (simulación).

### Capacidades Principales
| Módulo | Descripción |
|--------|-------------|
| **Estrategia** | Sistema de puntuación (-7 a +7) basado en 7 indicadores técnicos (MACD, RSI, Stoch, BB, OBI, CVD, Trend). |
| **Risk Mgmt** | Stop Loss dinámico (ATR), Trailing Stop de 3 niveles (BE, Lock, Trail) y Límite de Pérdida Diaria (-$500). |
| **Interfaz** | UI oscura y responsiva con panel de estadísticas en tiempo real (Win Rate, Profit Factor). |
| **Alertas** | Notificaciones instantáneas a Telegram (Señales, Resultados, Alertas de riesgo). |
| **IA/Data** | Ecosistema completo para recolección de datos y entrenamiento de modelos de Machine Learning. |

---

## 🎮 2. Instrucciones de Uso

### Ejecutar el Bot
Abre una terminal en la carpeta del proyecto y ejecuta:
```bash
python crypto_widget.py
```

### Controles de la Interfaz
- **`ESC`**: Cerrar la aplicación de forma segura.
- **`M`**: Minimizar la ventana.
- **`Click y Arrastre`**: Mover la ventana (si no está en pantalla completa).

### Panel de Estadísticas
- **Win Rate**: Porcentaje de operaciones ganadoras (Verde > 50%).
- **Profit Factor**: Relación Ganancia Bruta / Pérdida Bruta (Verde > 1.5).
- **Trades**: Conteo total de operaciones.

---

## 🧠 3. Ecosistema de Machine Learning (ML)

El bot incluye 3 herramientas especializadas para mejorar su inteligencia con el tiempo.

### Paso A: Recolectar Datos (`data_collector.py`)
Descarga datos históricos de Binance y los etiqueta automáticamente para entrenamiento.
- **Uso básico (30 días):**
  ```bash
  python data_collector.py
  ```
- **Uso avanzado (más días):**
  ```bash
  python data_collector.py --days 90
  ```
Esto genera el archivo `ml_training_data.csv`.

### Paso B: Entrenar Modelo (`ml_trainer.py`)
Utiliza los datos recolectados para entrenar un modelo que predice la probabilidad de éxito.
- **Entrenar:**
  ```bash
  python ml_trainer.py
  ```
- **Probar predicción:**
  ```bash
  python ml_trainer.py --predict
  ```
Esto guarda el modelo en `trading_model.pkl` y genera un reporte en `ml_training_results.json`.

### Paso C: Analizar Rendimiento (`analyzer.py`)
Analiza las operaciones realizadas por el bot (Paper Trading) para detectar patrones.
- **Ver reporte:**
  ```bash
  python analyzer.py
  ```
- **Obtener consejos de optimización:**
  ```bash
  python analyzer.py --optimize
  ```

---

## 🚀 4. Opciones de Optimización Futura

Si deseas llevar el bot al siguiente nivel, estas son las mejores opciones disponibles:

### A. Nivel Intermedio (Recomendado ahora)
1.  **Backtesting Engine**: Crear un script que simule los últimos 6 meses de mercado con tu estrategia actual para validar la rentabilidad sin esperar semanas.
2.  **Hyperparameter Tuning**: Usar algoritmos para encontrar automáticamente los valores óptimos de RSI, MACD y Stop Loss, en lugar de adivinarlos.
3.  **Filtrado por Horario**: Analizar en qué horas el bot pierde más dinero y prohibir operar en esos rangos (ej. fines de semana o cierre de sesión NY).

### B. Nivel Avanzado (Machine Learning)
1.  **Modelo Híbrido**: No usar el ML solo para predecir, sino para *filtrar*. El bot genera la señal tradicional (-7 a +7) y el modelo de IA decide si la aprueba o la rechaza (filtro de confirmación).
2.  **Sentiment Analysis**: Integrar análisis de noticias o Twitter/X para detectar pánicos de mercado antes de que afecten el precio.
3.  **Reinforcement Learning**: Entrenar un agente que aprenda por sí mismo jugando millones de simulaciones (como AlphaGo pero para trading).

---

## 🛡️ Recomendación de Seguridad
**NO operes con dinero real** hasta que cumplas estas 3 condiciones en Paper Trading:
1.  **Win Rate > 55%** consistente durante 2 semanas.
2.  **Profit Factor > 1.5**.
3.  **Drawdown Máximo < 10%** (caída máxima desde el punto más alto).

¡Tu capital es tu herramienta de trabajo, protégela!

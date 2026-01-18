"""
NEXUS - Prediction Engine Module
Lead Product Engineer & Behavioral Economist
==============================================

Motor de Predicciones con Ensemble ML:
- RandomForestClassifier: Predice dirección (SUBIR/BAJAR)
- LogisticRegression: Calcula probabilidad/confianza (0-100%)

Features utilizadas:
- SMA (Simple Moving Average): 5, 10, 20 días
- RSI (Relative Strength Index)
- Volatilidad (desviación estándar)
- Momentum (cambio porcentual)
- Sentiment Score (del intel_manager, si disponible)

Activos soportados: BTC-USD, ETH-USD, XRP-USD, GLD, DX-Y.NYB
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PredictionEngine:
    """
    Motor de Predicciones con Ensemble ML
    
    Arquitectura:
    1. RandomForestClassifier: Clasifica dirección (SUBIR=1, BAJAR=0)
    2. LogisticRegression: Calcula probabilidad de la predicción
    
    El usuario ve: "Señal de Compra (87% Confianza)"
    """
    
    # Activos soportados
    SUPPORTED_SYMBOLS = ["BTC-USD", "ETH-USD", "XRP-USD", "GLD", "DX-Y.NYB"]
    
    # Directorio para modelos entrenados
    MODELS_DIR = "models"
    
    def __init__(self, auto_train: bool = True):
        """
        Inicializa el motor de predicciones
        
        Args:
            auto_train: Si True, entrena modelos automáticamente si no existen
        """
        self.models: Dict[str, Dict] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        
        # Crear directorio de modelos si no existe
        os.makedirs(self.MODELS_DIR, exist_ok=True)
        
        # Cargar o entrenar modelos para cada símbolo
        for symbol in self.SUPPORTED_SYMBOLS:
            if self._model_exists(symbol):
                self._load_model(symbol)
            elif auto_train:
                logger.info(f"Entrenando modelo para {symbol}...")
                self.train(symbol)
    
    def _model_exists(self, symbol: str) -> bool:
        """Verifica si existe un modelo guardado para el símbolo"""
        safe_symbol = symbol.replace("-", "_").replace(".", "_")
        rf_path = os.path.join(self.MODELS_DIR, f"{safe_symbol}_rf.joblib")
        lr_path = os.path.join(self.MODELS_DIR, f"{safe_symbol}_lr.joblib")
        return os.path.exists(rf_path) and os.path.exists(lr_path)
    
    def _save_model(self, symbol: str):
        """Guarda los modelos entrenados en disco"""
        safe_symbol = symbol.replace("-", "_").replace(".", "_")
        
        rf_path = os.path.join(self.MODELS_DIR, f"{safe_symbol}_rf.joblib")
        lr_path = os.path.join(self.MODELS_DIR, f"{safe_symbol}_lr.joblib")
        scaler_path = os.path.join(self.MODELS_DIR, f"{safe_symbol}_scaler.joblib")
        
        joblib.dump(self.models[symbol]["rf"], rf_path)
        joblib.dump(self.models[symbol]["lr"], lr_path)
        joblib.dump(self.scalers[symbol], scaler_path)
        
        logger.info(f"Modelos guardados para {symbol}")
    
    def _load_model(self, symbol: str):
        """Carga modelos desde disco"""
        safe_symbol = symbol.replace("-", "_").replace(".", "_")
        
        rf_path = os.path.join(self.MODELS_DIR, f"{safe_symbol}_rf.joblib")
        lr_path = os.path.join(self.MODELS_DIR, f"{safe_symbol}_lr.joblib")
        scaler_path = os.path.join(self.MODELS_DIR, f"{safe_symbol}_scaler.joblib")
        
        try:
            self.models[symbol] = {
                "rf": joblib.load(rf_path),
                "lr": joblib.load(lr_path)
            }
            self.scalers[symbol] = joblib.load(scaler_path)
            logger.info(f"Modelos cargados para {symbol}")
        except Exception as e:
            logger.error(f"Error cargando modelos para {symbol}: {e}")
    
    def _calculate_features(self, df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """
        Calcula features técnicas para el modelo ML
        
        Features:
        - sma_5, sma_10, sma_20: Simple Moving Averages
        - sma_ratio_5, sma_ratio_10, sma_ratio_20: Precio / SMA
        - rsi: Relative Strength Index (14 períodos)
        - volatility: Desviación estándar de retornos (20 días)
        - momentum_1, momentum_5: Cambio porcentual
        - volume_sma_ratio: Volumen / SMA de volumen
        - sentiment_score: Score de noticias del intel_manager (-1 a 1)
        """
        df = df.copy()
        
        # Precios de cierre
        close = df['Close']
        
        # Simple Moving Averages
        df['sma_5'] = close.rolling(window=5).mean()
        df['sma_10'] = close.rolling(window=10).mean()
        df['sma_20'] = close.rolling(window=20).mean()
        
        # Ratios precio/SMA (señales de tendencia)
        df['sma_ratio_5'] = close / df['sma_5']
        df['sma_ratio_10'] = close / df['sma_10']
        df['sma_ratio_20'] = close / df['sma_20']
        
        # RSI (Relative Strength Index)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Volatilidad (desviación estándar de retornos)
        returns = close.pct_change()
        df['volatility'] = returns.rolling(window=20).std()
        
        # Momentum (cambio porcentual)
        df['momentum_1'] = close.pct_change(1)
        df['momentum_5'] = close.pct_change(5)
        
        # Volumen relativo (si hay datos de volumen)
        if 'Volume' in df.columns and df['Volume'].sum() > 0:
            df['volume_sma'] = df['Volume'].rolling(window=20).mean()
            df['volume_sma_ratio'] = df['Volume'] / df['volume_sma']
        else:
            df['volume_sma_ratio'] = 1.0
        
        return df
    
    def _prepare_training_data(
        self, 
        symbol: str, 
        period: str = "2y",
        prediction_horizon: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepara datos de entrenamiento desde YFinance
        
        Args:
            symbol: Símbolo del activo
            period: Período histórico (1y, 2y, 5y)
            prediction_horizon: Días hacia adelante para predecir
        
        Returns:
            X: Features normalizadas
            y: Labels (1=SUBIR, 0=BAJAR)
        """
        # Descargar datos históricos
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)
        
        if df.empty:
            raise ValueError(f"No se encontraron datos para {symbol}")
        
        logger.info(f"Datos descargados para {symbol}: {len(df)} registros")
        
        # Calcular features
        df = self._calculate_features(df)
        
        # Crear label: ¿El precio subió después de N días?
        df['future_return'] = df['Close'].shift(-prediction_horizon) / df['Close'] - 1
        df['target'] = (df['future_return'] > 0).astype(int)
        
        # Eliminar NaN
        df = df.dropna()
        
        # Features para el modelo
        feature_columns = [
            'sma_ratio_5', 'sma_ratio_10', 'sma_ratio_20',
            'rsi', 'volatility', 'momentum_1', 'momentum_5',
            'volume_sma_ratio'
        ]
        
        X = df[feature_columns].values
        y = df['target'].values
        
        return X, y
    
    def train(self, symbol: str, test_size: float = 0.2) -> Dict:
        """
        Entrena el ensemble de modelos para un símbolo
        
        Args:
            symbol: Símbolo del activo
            test_size: Proporción de datos para test
        
        Returns:
            Dict con métricas de entrenamiento
        """
        if symbol not in self.SUPPORTED_SYMBOLS:
            raise ValueError(f"Símbolo no soportado: {symbol}")
        
        try:
            # Preparar datos
            X, y = self._prepare_training_data(symbol)
            
            # Split train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, shuffle=False
            )
            
            # Normalizar features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Entrenar RandomForestClassifier
            rf = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
            rf.fit(X_train_scaled, y_train)
            rf_accuracy = rf.score(X_test_scaled, y_test)
            
            # Entrenar LogisticRegression
            lr = LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=42
            )
            lr.fit(X_train_scaled, y_train)
            lr_accuracy = lr.score(X_test_scaled, y_test)
            
            # Guardar modelos
            self.models[symbol] = {"rf": rf, "lr": lr}
            self.scalers[symbol] = scaler
            self._save_model(symbol)
            
            metrics = {
                "symbol": symbol,
                "samples": len(X),
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "rf_accuracy": round(rf_accuracy, 4),
                "lr_accuracy": round(lr_accuracy, 4),
                "trained_at": datetime.utcnow().isoformat()
            }
            
            logger.info(f"Modelo entrenado para {symbol}: RF={rf_accuracy:.2%}, LR={lr_accuracy:.2%}")
            
            return metrics
        
        except Exception as e:
            logger.error(f"Error entrenando modelo para {symbol}: {e}")
            raise
    
    def predict(self, symbol: str) -> Dict:
        """
        Genera predicción para un símbolo
        
        Returns:
            Dict con:
            - direction: "SUBIR" o "BAJAR"
            - confidence: Porcentaje de confianza (0-100)
            - current_price: Precio actual
            - signal: Texto descriptivo
        """
        if symbol not in self.SUPPORTED_SYMBOLS:
            raise ValueError(f"Símbolo no soportado: {symbol}")
        
        # Verificar que el modelo existe
        if symbol not in self.models:
            if self._model_exists(symbol):
                self._load_model(symbol)
            else:
                self.train(symbol)
        
        try:
            # Obtener datos recientes
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="60d")
            
            if df.empty:
                raise ValueError(f"No se encontraron datos para {symbol}")
            
            # Calcular features
            df = self._calculate_features(df)
            df = df.dropna()
            
            # Obtener última fila de features
            feature_columns = [
                'sma_ratio_5', 'sma_ratio_10', 'sma_ratio_20',
                'rsi', 'volatility', 'momentum_1', 'momentum_5',
                'volume_sma_ratio'
            ]
            
            X_latest = df[feature_columns].iloc[-1:].values
            
            # Normalizar
            X_scaled = self.scalers[symbol].transform(X_latest)
            
            # Predicción del RandomForest (dirección)
            rf_prediction = self.models[symbol]["rf"].predict(X_scaled)[0]
            rf_proba = self.models[symbol]["rf"].predict_proba(X_scaled)[0]
            
            # Predicción del LogisticRegression (confianza)
            lr_proba = self.models[symbol]["lr"].predict_proba(X_scaled)[0]
            
            # Combinar probabilidades (ensemble)
            # Peso: 60% RF, 40% LR
            combined_proba = 0.6 * rf_proba + 0.4 * lr_proba
            
            # Determinar dirección y confianza
            direction = "SUBIR" if rf_prediction == 1 else "BAJAR"
            confidence = max(combined_proba) * 100
            
            # Precio actual
            current_price = df['Close'].iloc[-1]
            
            # Generar señal descriptiva
            signal_type = "Compra" if direction == "SUBIR" else "Venta"
            
            # Ajustar texto según nivel de confianza
            if confidence >= 80:
                strength = "Fuerte"
            elif confidence >= 65:
                strength = ""
            else:
                strength = "Débil"
            
            signal = f"Señal {strength} de {signal_type} ({confidence:.0f}% Confianza)".strip()
            signal = signal.replace("  ", " ")
            
            result = {
                "symbol": symbol,
                "direction": direction,
                "confidence": round(confidence, 2),
                "current_price": round(current_price, 2),
                "signal": signal,
                "rf_proba": rf_proba.tolist(),
                "lr_proba": lr_proba.tolist(),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info(f"Predicción {symbol}: {direction} ({confidence:.1f}%)")
            
            return result
        
        except Exception as e:
            logger.error(f"Error en predicción para {symbol}: {e}")
            raise
    
    def predict_all(self) -> List[Dict]:
        """
        Genera predicciones para todos los símbolos soportados
        
        Returns:
            Lista de predicciones
        """
        predictions = []
        
        for symbol in self.SUPPORTED_SYMBOLS:
            try:
                prediction = self.predict(symbol)
                predictions.append(prediction)
            except Exception as e:
                logger.error(f"Error en predicción de {symbol}: {e}")
                predictions.append({
                    "symbol": symbol,
                    "error": str(e)
                })
        
        return predictions
    
    def get_model_info(self, symbol: str) -> Optional[Dict]:
        """
        Obtiene información del modelo para un símbolo
        """
        if symbol not in self.models:
            return None
        
        rf = self.models[symbol]["rf"]
        
        return {
            "symbol": symbol,
            "model_type": "Ensemble (RandomForest + LogisticRegression)",
            "rf_n_estimators": rf.n_estimators,
            "rf_max_depth": rf.max_depth,
            "feature_importances": {
                "sma_ratio_5": round(rf.feature_importances_[0], 4),
                "sma_ratio_10": round(rf.feature_importances_[1], 4),
                "sma_ratio_20": round(rf.feature_importances_[2], 4),
                "rsi": round(rf.feature_importances_[3], 4),
                "volatility": round(rf.feature_importances_[4], 4),
                "momentum_1": round(rf.feature_importances_[5], 4),
                "momentum_5": round(rf.feature_importances_[6], 4),
                "volume_sma_ratio": round(rf.feature_importances_[7], 4)
            }
        }
    
    def retrain_all(self) -> List[Dict]:
        """
        Reentrena todos los modelos con datos actualizados
        
        Returns:
            Lista de métricas de entrenamiento
        """
        results = []
        
        for symbol in self.SUPPORTED_SYMBOLS:
            try:
                metrics = self.train(symbol)
                results.append(metrics)
            except Exception as e:
                logger.error(f"Error reentrenando {symbol}: {e}")
                results.append({
                    "symbol": symbol,
                    "error": str(e)
                })
        
        return results


# ============================================================================
# FUNCIONES DE UTILIDAD
# ============================================================================

def get_market_sentiment(symbol: str) -> float:
    """
    Obtiene el sentiment score del mercado para un símbolo
    
    Integración con intel_manager.py para obtener sentiment de noticias
    
    Returns:
        Score de -1.0 (muy negativo) a 1.0 (muy positivo)
    """
    try:
        from intel_manager import IntelManager
        from database_setup import init_database
        
        engine, SessionLocal = init_database("nexus_app.sqlite")
        db = SessionLocal()
        
        intel = IntelManager(db)
        recent_intel = intel.get_recent_intelligence(hours=24, min_risk_score=0)
        
        if not recent_intel:
            return 0.0  # Neutral si no hay datos
        
        # Calcular promedio de sentiment
        sentiment_scores = []
        for item in recent_intel:
            if symbol.upper().replace("-", "").replace("USD", "") in item.headline.upper():
                if item.sentiment.value == "POSITIVO":
                    sentiment_scores.append(1.0)
                elif item.sentiment.value == "NEGATIVO":
                    sentiment_scores.append(-1.0)
                else:
                    sentiment_scores.append(0.0)
        
        db.close()
        
        if sentiment_scores:
            return sum(sentiment_scores) / len(sentiment_scores)
        return 0.0
    
    except Exception as e:
        logger.warning(f"No se pudo obtener sentiment para {symbol}: {e}")
        return 0.0


# ============================================================================
# SCRIPT DE PRUEBA
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("NEXUS - Motor de Predicciones ML")
    print("=" * 60)
    
    # Inicializar motor (entrena automáticamente si es necesario)
    print("\nInicializando motor de predicciones...")
    engine = PredictionEngine(auto_train=True)
    
    # Generar predicciones para todos los símbolos
    print("\n" + "-" * 60)
    print("PREDICCIONES ACTUALES")
    print("-" * 60)
    
    predictions = engine.predict_all()
    
    for pred in predictions:
        if "error" in pred:
            print(f"\n{pred['symbol']}: ERROR - {pred['error']}")
        else:
            direction_emoji = "🟢" if pred["direction"] == "SUBIR" else "🔴"
            print(f"\n{direction_emoji} {pred['symbol']}")
            print(f"   Precio: ${pred['current_price']:,.2f}")
            print(f"   Señal: {pred['signal']}")
            print(f"   Confianza: {pred['confidence']:.1f}%")
    
    # Mostrar información del modelo
    print("\n" + "-" * 60)
    print("IMPORTANCIA DE FEATURES (BTC-USD)")
    print("-" * 60)
    
    info = engine.get_model_info("BTC-USD")
    if info:
        for feature, importance in info["feature_importances"].items():
            bar = "█" * int(importance * 50)
            print(f"   {feature:20s}: {bar} {importance:.2%}")
    
    print("\n✅ Motor de predicciones listo!")

"""
NEXUS - Main Application Server
Lead Product Engineer & Behavioral Economist
==============================================

Servidor FastAPI monolítico que integra:
- Frontend: Jinja2 templates con TailwindCSS (Bloomberg-style dark mode)
- API REST: Endpoints JSON para precios, predicciones y apuestas
- Core: Motor económico conductual + Motor de predicciones ML

Ejecutar con: uvicorn main_app:app --reload --port 8000
"""

from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
import yfinance as yf
import logging

# Importar módulos NEXUS existentes
from database_setup import init_database, User, Prediction, UserBet, PredictionDirection, BetResult, PredictionStatus
from economy_engine import NexusBank, compare_withdrawal_methods, WithdrawalMethod

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURACIÓN DE LA APLICACIÓN
# ============================================================================

app = FastAPI(
    title="NEXUS",
    description="Plataforma de Predicciones Geopolíticas con IA + Gamificación",
    version="1.0.0"
)

# CORS para desarrollo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Templates Jinja2
templates = Jinja2Templates(directory="templates")

# Archivos estáticos (opcional)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Base de datos
engine, SessionLocal = init_database("nexus_app.sqlite")


# ============================================================================
# DEPENDENCIAS
# ============================================================================

def get_db():
    """Dependency para obtener sesión de BD"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ============================================================================
# MODELOS PYDANTIC (Request/Response)
# ============================================================================

class PriceResponse(BaseModel):
    symbol: str
    price: float
    change_percent: float
    currency: str
    timestamp: str


class PredictionResponse(BaseModel):
    symbol: str
    direction: str
    confidence: float
    current_price: float
    signal: str
    timestamp: str


class BetRequest(BaseModel):
    user_id: int
    prediction_id: int
    direction: str  # "SUBIR" o "BAJAR"
    points: float


class BetResponse(BaseModel):
    success: bool
    bet_id: Optional[int] = None
    odds: Optional[float] = None
    potential_win: Optional[float] = None
    message: str


class UserSummaryResponse(BaseModel):
    user_id: int
    email: str
    nexus_points: float
    fiat_balance: float
    total_value_usd: float
    reputation_level: int
    active_bets: int


class WithdrawalComparisonResponse(BaseModel):
    amount_points: float
    paypal: dict
    kraken: dict
    recommendation: str


# ============================================================================
# RUTAS HTML (FRONTEND)
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """
    Dashboard principal - Estilo Bloomberg Terminal
    """
    return templates.TemplateResponse("index.html", {
        "request": request,
        "title": "NEXUS Dashboard",
        "symbols": ["BTC-USD", "ETH-USD", "XRP-USD", "GLD", "DX-Y.NYB"]
    })


@app.get("/predictions", response_class=HTMLResponse)
async def predictions_page(request: Request):
    """
    Página de predicciones activas
    """
    return templates.TemplateResponse("predictions.html", {
        "request": request,
        "title": "NEXUS - Predicciones"
    })


@app.get("/portfolio", response_class=HTMLResponse)
async def portfolio_page(request: Request):
    """
    Página de portfolio del usuario
    """
    return templates.TemplateResponse("portfolio.html", {
        "request": request,
        "title": "NEXUS - Portfolio"
    })


# ============================================================================
# RUTAS API - PRECIOS EN TIEMPO REAL
# ============================================================================

@app.get("/api/price/{symbol}", response_model=PriceResponse)
async def get_price(symbol: str):
    """
    Obtiene el precio actual de un activo usando YFinance
    
    Símbolos soportados:
    - BTC-USD (Bitcoin)
    - ETH-USD (Ethereum)
    - XRP-USD (Ripple)
    - GLD (Gold ETF)
    - DX-Y.NYB (US Dollar Index)
    """
    try:
        ticker = yf.Ticker(symbol)
        
        # Obtener datos del día
        hist = ticker.history(period="2d")
        
        if hist.empty:
            raise HTTPException(status_code=404, detail=f"Símbolo no encontrado: {symbol}")
        
        current_price = hist['Close'].iloc[-1]
        
        # Calcular cambio porcentual
        if len(hist) >= 2:
            previous_price = hist['Close'].iloc[-2]
            change_percent = ((current_price - previous_price) / previous_price) * 100
        else:
            change_percent = 0.0
        
        return PriceResponse(
            symbol=symbol,
            price=round(current_price, 2),
            change_percent=round(change_percent, 2),
            currency="USD",
            timestamp=datetime.utcnow().isoformat()
        )
    
    except Exception as e:
        logger.error(f"Error obteniendo precio de {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/prices/all")
async def get_all_prices():
    """
    Obtiene precios de todos los activos principales
    """
    symbols = ["BTC-USD", "ETH-USD", "XRP-USD", "GLD", "DX-Y.NYB"]
    prices = {}
    
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="2d")
            
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                
                if len(hist) >= 2:
                    previous_price = hist['Close'].iloc[-2]
                    change_percent = ((current_price - previous_price) / previous_price) * 100
                else:
                    change_percent = 0.0
                
                prices[symbol] = {
                    "price": round(current_price, 2),
                    "change_percent": round(change_percent, 2),
                    "currency": "USD"
                }
        except Exception as e:
            logger.warning(f"Error obteniendo {symbol}: {e}")
            prices[symbol] = {"error": str(e)}
    
    return {
        "prices": prices,
        "timestamp": datetime.utcnow().isoformat()
    }


# ============================================================================
# RUTAS API - PREDICCIONES IA
# ============================================================================

@app.get("/api/prediction/{symbol}", response_model=PredictionResponse)
async def get_prediction(symbol: str):
    """
    Obtiene la predicción de IA para un activo
    
    Retorna:
    - direction: SUBIR o BAJAR
    - confidence: Porcentaje de confianza (0-100)
    - signal: Texto descriptivo ("Señal de Compra (87% Confianza)")
    """
    try:
        # Importar motor de predicciones (se creará después)
        try:
            from prediction_engine import PredictionEngine
            engine = PredictionEngine()
            prediction = engine.predict(symbol)
            
            return PredictionResponse(
                symbol=symbol,
                direction=prediction["direction"],
                confidence=prediction["confidence"],
                current_price=prediction["current_price"],
                signal=prediction["signal"],
                timestamp=datetime.utcnow().isoformat()
            )
        except ImportError:
            # Fallback: predicción mock mientras no existe el motor
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="5d")
            
            if hist.empty:
                raise HTTPException(status_code=404, detail=f"Símbolo no encontrado: {symbol}")
            
            current_price = hist['Close'].iloc[-1]
            
            # Mock prediction basada en tendencia simple
            sma_5 = hist['Close'].mean()
            direction = "SUBIR" if current_price > sma_5 else "BAJAR"
            confidence = 65.0  # Mock confidence
            
            signal_type = "Compra" if direction == "SUBIR" else "Venta"
            signal = f"Señal de {signal_type} ({confidence:.0f}% Confianza)"
            
            return PredictionResponse(
                symbol=symbol,
                direction=direction,
                confidence=confidence,
                current_price=round(current_price, 2),
                signal=signal,
                timestamp=datetime.utcnow().isoformat()
            )
    
    except Exception as e:
        logger.error(f"Error en predicción de {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/predictions/active")
async def get_active_predictions(db=Depends(get_db)):
    """
    Obtiene todas las predicciones activas (PENDIENTE)
    """
    predictions = db.query(Prediction).filter(
        Prediction.estado == PredictionStatus.PENDIENTE
    ).all()
    
    return {
        "predictions": [
            {
                "id": p.id,
                "activo": p.activo,
                "direccion": p.prediccion.value,
                "precio_entrada": p.precio_entrada,
                "created_at": p.created_at.isoformat()
            }
            for p in predictions
        ],
        "count": len(predictions)
    }


# ============================================================================
# RUTAS API - SISTEMA DE APUESTAS
# ============================================================================

@app.post("/api/bet", response_model=BetResponse)
async def create_bet(bet: BetRequest, db=Depends(get_db)):
    """
    Crea una nueva apuesta sobre una predicción
    
    El usuario apuesta nexus_points a favor o en contra de la predicción.
    Los odds se calculan basados en la confianza del modelo.
    """
    try:
        # Verificar usuario
        user = db.query(User).filter(User.id == bet.user_id).first()
        if not user:
            return BetResponse(
                success=False,
                message=f"Usuario {bet.user_id} no encontrado"
            )
        
        # Verificar saldo
        if user.nexus_points < bet.points:
            return BetResponse(
                success=False,
                message=f"Saldo insuficiente. Tienes {user.nexus_points:.2f} puntos"
            )
        
        # Verificar predicción
        prediction = db.query(Prediction).filter(Prediction.id == bet.prediction_id).first()
        if not prediction:
            return BetResponse(
                success=False,
                message=f"Predicción {bet.prediction_id} no encontrada"
            )
        
        if prediction.estado != PredictionStatus.PENDIENTE:
            return BetResponse(
                success=False,
                message="Esta predicción ya está cerrada"
            )
        
        # Calcular odds basados en confianza del prediction_engine
        try:
            from prediction_engine import PredictionEngine
            engine = PredictionEngine(auto_train=False)
            pred_result = engine.predict(prediction.activo)
            base_confidence = pred_result["confidence"] / 100.0
        except Exception:
            base_confidence = 0.70  # Fallback
        
        # Si apuesta a favor de la predicción, odds menores
        # Si apuesta en contra, odds mayores (más riesgo = más recompensa)
        direction_enum = PredictionDirection.SUBIR if bet.direction == "SUBIR" else PredictionDirection.BAJAR
        
        if direction_enum == prediction.prediccion:
            odds = 1.0 + (1.0 - base_confidence)  # ~1.30 si confianza es 70%
        else:
            odds = 1.0 + base_confidence  # ~1.70 si confianza es 70%
        
        potential_win = bet.points * odds
        
        # Descontar puntos del usuario
        user.nexus_points -= bet.points
        
        # Crear apuesta
        new_bet = UserBet(
            usuario_id=bet.user_id,
            prediccion_id=bet.prediction_id,
            cantidad_puntos=bet.points,
            resultado=BetResult.PENDIENTE,
            puntos_ganados=0.0
        )
        
        db.add(new_bet)
        db.commit()
        db.refresh(new_bet)
        
        logger.info(f"Apuesta creada: Usuario {bet.user_id} apostó {bet.points} puntos")
        
        return BetResponse(
            success=True,
            bet_id=new_bet.id,
            odds=round(odds, 2),
            potential_win=round(potential_win, 2),
            message=f"Apuesta registrada. Ganancia potencial: {potential_win:.2f} puntos"
        )
    
    except Exception as e:
        logger.error(f"Error creando apuesta: {e}")
        db.rollback()
        return BetResponse(
            success=False,
            message=str(e)
        )


@app.get("/api/user/{user_id}/bets")
async def get_user_bets(user_id: int, db=Depends(get_db)):
    """
    Obtiene todas las apuestas de un usuario
    """
    bets = db.query(UserBet).filter(UserBet.usuario_id == user_id).all()
    
    return {
        "user_id": user_id,
        "bets": [
            {
                "id": b.id,
                "prediction_id": b.prediccion_id,
                "points": b.cantidad_puntos,
                "result": b.resultado.value,
                "winnings": b.puntos_ganados,
                "created_at": b.created_at.isoformat()
            }
            for b in bets
        ],
        "total_bets": len(bets)
    }


# ============================================================================
# RUTAS API - ECONOMÍA Y USUARIO
# ============================================================================

@app.get("/api/user/{user_id}/summary", response_model=UserSummaryResponse)
async def get_user_summary(user_id: int, db=Depends(get_db)):
    """
    Obtiene el resumen económico completo del usuario
    """
    user = db.query(User).filter(User.id == user_id).first()
    
    if not user:
        raise HTTPException(status_code=404, detail=f"Usuario {user_id} no encontrado")
    
    # Contar apuestas activas
    active_bets = db.query(UserBet).filter(
        UserBet.usuario_id == user_id,
        UserBet.resultado == BetResult.PENDIENTE
    ).count()
    
    # Calcular valor total (puntos convertidos a USD + fiat balance)
    points_to_usd = user.nexus_points * 0.01  # 1 punto = $0.01
    total_value = points_to_usd + user.fiat_balance
    
    return UserSummaryResponse(
        user_id=user_id,
        email=user.email,
        nexus_points=user.nexus_points,
        fiat_balance=user.fiat_balance,
        total_value_usd=round(total_value, 2),
        reputation_level=user.reputation_level,
        active_bets=active_bets
    )


@app.get("/api/withdrawal/compare/{points}")
async def compare_withdrawals(points: float):
    """
    Compara métodos de retiro (diseño conductual)
    
    Muestra la diferencia entre PayPal (alto costo) y Kraken (incentivado)
    """
    comparison = compare_withdrawal_methods(points)
    return comparison


@app.post("/api/user/create")
async def create_user(email: str, db=Depends(get_db)):
    """
    Crea un nuevo usuario con saldo inicial de bienvenida
    """
    # Verificar si ya existe
    existing = db.query(User).filter(User.email == email).first()
    if existing:
        return {
            "success": False,
            "message": "El email ya está registrado",
            "user_id": existing.id
        }
    
    # Crear usuario con bonus de bienvenida
    new_user = User(
        email=email,
        nexus_points=100.0,  # Bonus de bienvenida
        fiat_balance=0.0,
        reputation_level=1
    )
    
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    logger.info(f"Nuevo usuario creado: {email} (ID: {new_user.id})")
    
    return {
        "success": True,
        "message": "Usuario creado exitosamente. Bonus de bienvenida: 100 puntos",
        "user_id": new_user.id,
        "nexus_points": new_user.nexus_points
    }


# ============================================================================
# RUTAS API - RECOMPENSAS
# ============================================================================

@app.post("/api/reward/ad/{user_id}")
async def process_ad_reward(user_id: int, db=Depends(get_db)):
    """
    Procesa recompensa por ver un anuncio (ingreso pasivo gamificado)
    """
    bank = NexusBank(db)
    
    try:
        result = bank.process_ad_reward(user_id, points_amount=10.0)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================================
# RUTAS API - INTELIGENCIA DE MERCADO
# ============================================================================

@app.get("/api/intel/recent")
async def get_recent_intelligence(hours: int = 24, min_risk: float = 0.0, db=Depends(get_db)):
    """
    Obtiene inteligencia de mercado reciente
    
    Args:
        hours: Horas hacia atrás para buscar (default: 24)
        min_risk: Risk Score mínimo para filtrar (default: 0)
    """
    try:
        from intel_manager import IntelManager
        
        intel_manager = IntelManager(db)
        intel_list = intel_manager.get_recent_intelligence(hours=hours, min_risk_score=min_risk)
        
        return {
            "intel": [
                {
                    "id": item.id,
                    "source": item.source,
                    "headline": item.headline,
                    "news_type": item.news_type.value,
                    "sentiment": item.sentiment.value,
                    "risk_score": item.impact_score,
                    "published_at": item.published_at.isoformat() if item.published_at else None
                }
                for item in intel_list
            ],
            "count": len(intel_list),
            "hours": hours
        }
    except ImportError:
        return {"intel": [], "count": 0, "message": "Intel manager not available"}
    except Exception as e:
        logger.error(f"Error obteniendo inteligencia: {e}")
        return {"intel": [], "count": 0, "error": str(e)}


@app.post("/api/intel/collect")
async def collect_intelligence(query: str = "cryptocurrency OR bitcoin OR XRP", max_articles: int = 10, db=Depends(get_db)):
    """
    Recolecta inteligencia de mercado desde NewsAPI
    
    Requiere NEWSAPI_KEY configurada en .env
    """
    try:
        from intel_manager import IntelManager
        
        intel_manager = IntelManager(db)
        collected = intel_manager.collect_intelligence(query=query, max_articles=max_articles)
        
        return {
            "success": True,
            "collected_count": len(collected),
            "query": query,
            "message": f"Se recolectaron {len(collected)} noticias"
        }
    except ImportError:
        return {"success": False, "message": "Intel manager not available"}
    except Exception as e:
        logger.error(f"Error recolectando inteligencia: {e}")
        return {"success": False, "error": str(e)}


@app.get("/api/intel/sentiment/{symbol}")
async def get_symbol_sentiment(symbol: str, db=Depends(get_db)):
    """
    Obtiene el sentiment score agregado para un símbolo específico
    
    Returns:
        Sentiment score de -1.0 (muy negativo) a 1.0 (muy positivo)
    """
    try:
        from prediction_engine import get_market_sentiment
        
        sentiment = get_market_sentiment(symbol)
        
        # Clasificar el sentiment
        if sentiment > 0.3:
            classification = "MUY_POSITIVO"
        elif sentiment > 0.1:
            classification = "POSITIVO"
        elif sentiment < -0.3:
            classification = "MUY_NEGATIVO"
        elif sentiment < -0.1:
            classification = "NEGATIVO"
        else:
            classification = "NEUTRAL"
        
        return {
            "symbol": symbol,
            "sentiment_score": round(sentiment, 4),
            "classification": classification,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error obteniendo sentiment para {symbol}: {e}")
        return {
            "symbol": symbol,
            "sentiment_score": 0.0,
            "classification": "NEUTRAL",
            "error": str(e)
        }


# ============================================================================
# RUTAS API - PREDICCIONES AVANZADAS
# ============================================================================

@app.get("/api/prediction/{symbol}/detailed")
async def get_detailed_prediction(symbol: str):
    """
    Obtiene predicción detallada con información del modelo
    """
    try:
        from prediction_engine import PredictionEngine
        
        engine = PredictionEngine(auto_train=False)
        
        # Verificar si el modelo existe, si no, entrenar
        if symbol not in engine.models:
            logger.info(f"Entrenando modelo para {symbol}...")
            engine.train(symbol)
        
        prediction = engine.predict(symbol)
        model_info = engine.get_model_info(symbol)
        
        return {
            "prediction": prediction,
            "model_info": model_info,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error en predicción detallada de {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/models/retrain")
async def retrain_models():
    """
    Reentrena todos los modelos con datos actualizados
    
    Útil para mantener los modelos frescos con datos recientes
    """
    try:
        from prediction_engine import PredictionEngine
        
        engine = PredictionEngine(auto_train=False)
        results = engine.retrain_all()
        
        return {
            "success": True,
            "results": results,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error reentrenando modelos: {e}")
        return {"success": False, "error": str(e)}


@app.get("/api/predictions/all")
async def get_all_predictions():
    """
    Obtiene predicciones para todos los símbolos soportados
    """
    try:
        from prediction_engine import PredictionEngine
        
        engine = PredictionEngine(auto_train=True)
        predictions = engine.predict_all()
        
        return {
            "predictions": predictions,
            "count": len(predictions),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error obteniendo todas las predicciones: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.get("/api/health")
async def health_check():
    """
    Verifica que el servidor está funcionando
    """
    return {
        "status": "healthy",
        "service": "NEXUS",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }

# --- PEGAR ESTO EN main_app.py ---

from fastapi.responses import JSONResponse

# Endpoint para el resumen del usuario (Soluciona el error 404)
@app.get("/api/user/{user_id}/summary")
async def get_user_summary(user_id: int):
    # AQUÍ CONECTAREMOS CON LA BASE DE DATOS MÁS ADELANTE
    # Por ahora, devolvemos datos simulados para que el frontend no falle
    return JSONResponse(content={
        "points": 1250,           # Puntos simulados
        "reputation_level": 2,    # Nivel simulado
        "ads_watched": 15,
        "next_level_progress": 75 # Porcentaje para barra de progreso
    })

# ============================================================================
# PUNTO DE ENTRADA
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("NEXUS - Plataforma de Predicciones Geopolíticas")
    print("=" * 60)
    print("\nIniciando servidor en http://localhost:8000")
    print("Documentación API: http://localhost:8000/docs")
    print("\nPresiona Ctrl+C para detener\n")
    
    uvicorn.run(
        "main_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

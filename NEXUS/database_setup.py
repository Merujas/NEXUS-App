"""
NEXUS - Database Setup Module
Lead Software Architect & Behavioral Economist
==============================================

Este módulo define la estructura completa de la base de datos SQLite usando SQLAlchemy.
Implementa el modelo de datos para el sistema de "Learn-to-Earn" con predicciones de mercado.

Arquitectura:
- users: Gestión de usuarios y economía virtual (nexus_points) vs real (fiat_balance)
- market_intel: Inteligencia de mercado desde fuentes externas (NewsAPI/YFinance)
- predictions: Predicciones de precios generadas por IA
- user_bets: Apuestas de usuarios sobre predicciones (gamificación)
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Enum, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import enum

# Base para los modelos SQLAlchemy
Base = declarative_base()


# ============================================================================
# ENUMS para Tipos y Estados
# ============================================================================

class NewsType(enum.Enum):
    """Tipos de noticias geopolíticas que impactan mercados"""
    GUERRA = "Guerra"
    LEY = "Ley"
    BANCO = "Banco"
    REGULACION = "Regulacion"
    CRISIS = "Crisis"
    ACUERDO = "Acuerdo"


class SentimentType(enum.Enum):
    """Sentimiento básico de la noticia"""
    POSITIVO = "POSITIVO"
    NEUTRAL = "NEUTRAL"
    NEGATIVO = "NEGATIVO"


class PredictionDirection(enum.Enum):
    """Dirección de la predicción de precio"""
    SUBIR = "SUBIR"
    BAJAR = "BAJAR"


class PredictionStatus(enum.Enum):
    """Estado del ciclo de vida de una predicción"""
    PENDIENTE = "PENDIENTE"
    CERRADA = "CERRADA"
    CANCELADA = "CANCELADA"


class BetResult(enum.Enum):
    """Resultado de una apuesta del usuario"""
    GANO = "GANO"
    PERDIO = "PERDIO"
    PENDIENTE = "PENDIENTE"


class WithdrawalMethod(enum.Enum):
    """Métodos de retiro disponibles (parte del diseño conductual)"""
    PAYPAL = "PAYPAL"
    KRAKEN_PARTNER = "KRAKEN_PARTNER"


# ============================================================================
# MODELOS DE BASE DE DATOS
# ============================================================================

class User(Base):
    """
    Tabla de Usuarios - El corazón del modelo Learn-to-Earn
    
    Diseño Conductual:
    - nexus_points: Moneda virtual gamificada (baja fricción psicológica)
    - fiat_balance: Saldo real acumulado (alta fricción psicológica)
    - reputation_level: Sistema de niveles que incentiva participación continua
    """
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    nexus_points = Column(Float, default=0.0, nullable=False)  # Moneda virtual
    fiat_balance = Column(Float, default=0.0, nullable=False)  # Saldo real acumulado
    reputation_level = Column(Integer, default=1, nullable=False)  # Nivel de reputación (1-10)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relaciones
    bets = relationship("UserBet", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User(id={self.id}, email={self.email}, points={self.nexus_points}, fiat={self.fiat_balance}, level={self.reputation_level})>"


class MarketIntel(Base):
    """
    Tabla de Inteligencia de Mercado - Fuente de datos para predicciones
    
    Diseño de Análisis:
    - source: Identifica la fuente (NewsAPI, YFinance, etc.)
    - headline: Titular procesado
    - news_type: Categorización geopolítica
    - sentiment: Análisis básico de sentimiento
    - impact_score: Score del 1-10 que mide impacto potencial en mercados
    """
    __tablename__ = 'market_intel'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    source = Column(String(100), nullable=False)  # 'NewsAPI', 'YFinance', etc.
    headline = Column(String(500), nullable=False)
    news_type = Column(Enum(NewsType), nullable=False)
    sentiment = Column(Enum(SentimentType), nullable=False)
    impact_score = Column(Float, nullable=False)  # 1.0 - 10.0
    raw_content = Column(String(2000))  # Contenido completo para análisis futuro
    published_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relaciones
    predictions = relationship("Prediction", back_populates="intel_source")
    
    def __repr__(self):
        return f"<MarketIntel(id={self.id}, type={self.news_type.value}, impact={self.impact_score}, sentiment={self.sentiment.value})>"


class Prediction(Base):
    """
    Tabla de Predicciones - El núcleo del sistema de IA
    
    Ciclo de Vida:
    1. PENDIENTE: Predicción activa, usuarios pueden apostar
    2. CERRADA: Se resolvió el precio real, se calculan ganadores/perdedores
    3. CANCELADA: Evento externo invalida la predicción
    
    Diseño de Precisión:
    - precio_entrada: Precio del activo al momento de la predicción
    - precio_salida: Precio real cuando se cierra la predicción
    - resultado_real: Calculado automáticamente comparando entrada vs salida
    """
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    activo = Column(String(50), nullable=False)  # 'XRP', 'BTC', 'ETH', etc.
    precio_entrada = Column(Float, nullable=False)
    prediccion = Column(Enum(PredictionDirection), nullable=False)  # SUBIR o BAJAR
    precio_salida = Column(Float, nullable=True)  # Se completa cuando se cierra
    resultado_real = Column(Enum(PredictionDirection), nullable=True)  # Calculado automáticamente
    estado = Column(Enum(PredictionStatus), default=PredictionStatus.PENDIENTE, nullable=False)
    fecha_cierre = Column(DateTime, nullable=True)  # Cuándo se cerró la predicción
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relación con la fuente de inteligencia
    intel_source_id = Column(Integer, ForeignKey('market_intel.id'), nullable=True)
    intel_source = relationship("MarketIntel", back_populates="predictions")
    
    # Relaciones
    bets = relationship("UserBet", back_populates="prediction", cascade="all, delete-orphan")
    
    def calcular_resultado(self):
        """Calcula automáticamente el resultado real basado en precios"""
        if self.precio_salida is None:
            return None
        
        if self.precio_salida > self.precio_entrada:
            self.resultado_real = PredictionDirection.SUBIR
        elif self.precio_salida < self.precio_entrada:
            self.resultado_real = PredictionDirection.BAJAR
        else:
            # Empate técnico (muy raro)
            self.resultado_real = None
        
        return self.resultado_real
    
    def __repr__(self):
        return f"<Prediction(id={self.id}, activo={self.activo}, direccion={self.prediccion.value}, estado={self.estado.value})>"


class UserBet(Base):
    """
    Tabla de Apuestas de Usuarios - La capa de gamificación
    
    Modelo Learn-to-Earn:
    - Usuario apuesta nexus_points (moneda virtual)
    - Si gana: Recibe puntos multiplicados según odds
    - Si pierde: Pierde los puntos apostados
    - Los puntos ganados pueden convertirse a fiat_balance (con comisiones estratégicas)
    """
    __tablename__ = 'user_bets'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    usuario_id = Column(Integer, ForeignKey('users.id'), nullable=False, index=True)
    prediccion_id = Column(Integer, ForeignKey('predictions.id'), nullable=False, index=True)
    cantidad_puntos = Column(Float, nullable=False)  # Cuántos nexus_points apostó
    resultado = Column(Enum(BetResult), default=BetResult.PENDIENTE, nullable=False)
    puntos_ganados = Column(Float, default=0.0, nullable=False)  # Puntos ganados si acertó
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    resolved_at = Column(DateTime, nullable=True)  # Cuándo se resolvió la apuesta
    
    # Relaciones
    user = relationship("User", back_populates="bets")
    prediction = relationship("Prediction", back_populates="bets")
    
    def __repr__(self):
        return f"<UserBet(id={self.id}, user={self.usuario_id}, prediction={self.prediccion_id}, puntos={self.cantidad_puntos}, resultado={self.resultado.value})>"


# ============================================================================
# FUNCIONES DE INICIALIZACIÓN
# ============================================================================

def init_database(db_path: str = "nexus_core.sqlite"):
    """
    Inicializa la base de datos SQLite y crea todas las tablas
    
    Args:
        db_path: Ruta al archivo de base de datos SQLite
    
    Returns:
        engine: Motor de SQLAlchemy
        SessionLocal: Clase de sesión para crear sesiones de BD
    """
    # Crear engine con configuración optimizada para SQLite
    engine = create_engine(
        f'sqlite:///{db_path}',
        connect_args={'check_same_thread': False},  # Necesario para SQLite en aplicaciones multi-thread
        echo=False  # Cambiar a True para ver queries SQL en desarrollo
    )
    
    # Crear todas las tablas
    Base.metadata.create_all(engine)
    
    # Crear clase de sesión
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    return engine, SessionLocal


def get_db_session(SessionLocal):
    """
    Generador de sesiones de base de datos (patrón de contexto)
    
    Uso:
        SessionLocal = init_database()[1]
        db = next(get_db_session(SessionLocal))
        # usar db...
        db.close()
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ============================================================================
# SCRIPT DE PRUEBA / INICIALIZACIÓN
# ============================================================================

if __name__ == "__main__":
    print("🚀 Inicializando base de datos NEXUS...")
    
    engine, SessionLocal = init_database()
    
    print("✅ Base de datos creada exitosamente: nexus_core.sqlite")
    print("\n📊 Tablas creadas:")
    print("   - users")
    print("   - market_intel")
    print("   - predictions")
    print("   - user_bets")
    
    # Crear una sesión de prueba
    db = SessionLocal()
    
    try:
        # Verificar que las tablas existen
        from sqlalchemy import inspect
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        print(f"\n✅ Verificación: {len(tables)} tablas encontradas")
        for table in tables:
            print(f"   ✓ {table}")
    finally:
        db.close()
    
    print("\n✨ Base de datos lista para usar!")

"""
NEXUS - Intel Manager Module
Lead Software Architect & Behavioral Economist
==============================================

Este módulo gestiona la recolección y análisis de inteligencia de mercado desde
fuentes externas (NewsAPI, YFinance) y aplica análisis de sentimiento profundo
usando TextBlob y conceptos de economía conductual.

Diseño de Análisis:
- Busca conceptos profundos: "Liquidity", "SEC Ruling", "Central Bank", "Conflict"
- Categoriza noticias no solo como positivo/negativo, sino con Risk Score (1-10)
- Conecta eventos geopolíticos con impacto en mercados financieros
"""

import os
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from textblob import TextBlob
from sqlalchemy.orm import Session
from database_setup import MarketIntel, NewsType, SentimentType
from dotenv import load_dotenv
import logging

# Cargar variables de entorno
load_dotenv()

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntelManager:
    """
    Gestor de Inteligencia de Mercado
    
    Responsabilidades:
    1. Recolectar noticias desde NewsAPI
    2. Analizar sentimiento usando TextBlob
    3. Detectar conceptos profundos (liquidez, regulación, conflictos)
    4. Calcular Risk Score (1-10) basado en impacto potencial
    5. Guardar inteligencia en base de datos
    """
    
    # Conceptos clave para análisis profundo
    LIQUIDITY_KEYWORDS = [
        "liquidity", "liquidez", "cash flow", "flujo de efectivo",
        "money supply", "oferta monetaria", "quantitative easing", "QE"
    ]
    
    REGULATION_KEYWORDS = [
        "sec ruling", "sec", "regulation", "regulación", "regulatory",
        "compliance", "cumplimiento", "legislation", "legislación",
        "ban", "prohibición", "approval", "aprobación"
    ]
    
    CENTRAL_BANK_KEYWORDS = [
        "central bank", "banco central", "federal reserve", "fed",
        "ecb", "european central bank", "banco de inglaterra",
        "monetary policy", "política monetaria", "interest rate", "tasa de interés"
    ]
    
    CONFLICT_KEYWORDS = [
        "conflict", "conflicto", "war", "guerra", "sanctions", "sanciones",
        "tension", "tensión", "crisis", "crisis", "attack", "ataque"
    ]
    
    def __init__(self, db_session: Session):
        """
        Inicializa el IntelManager
        
        Args:
            db_session: Sesión activa de SQLAlchemy
        """
        self.db = db_session
        self.newsapi_key = os.getenv("NEWS_API_KEY") or os.getenv("NEWSAPI_KEY")
        
        if not self.newsapi_key:
            logger.warning("⚠️ NEWSAPI_KEY no encontrada en .env. Algunas funciones estarán limitadas.")
    
    def fetch_news_from_newsapi(
        self, 
        query: str = "cryptocurrency OR bitcoin OR XRP OR SEC",
        language: str = "en",
        sort_by: str = "publishedAt",
        page_size: int = 20
    ) -> List[Dict]:
        """
        Obtiene noticias desde NewsAPI
        
        Args:
            query: Query de búsqueda
            language: Idioma (en, es, etc.)
            sort_by: Ordenamiento (publishedAt, relevancy, popularity)
            page_size: Cantidad de resultados (máx 100)
        
        Returns:
            Lista de artículos de noticias
        """
        if not self.newsapi_key:
            logger.error("❌ NEWSAPI_KEY no configurada")
            return []
        
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "language": language,
            "sortBy": sort_by,
            "pageSize": page_size,
            "apiKey": self.newsapi_key
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data.get("status") == "ok":
                articles = data.get("articles", [])
                logger.info(f"✅ Obtenidas {len(articles)} noticias desde NewsAPI")
                return articles
            else:
                logger.error(f"❌ Error en NewsAPI: {data.get('message', 'Unknown error')}")
                return []
        
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Error al conectar con NewsAPI: {e}")
            return []
    
    def detect_concept_keywords(self, text: str) -> Dict[str, bool]:
        """
        Detecta conceptos clave en el texto
        
        Args:
            text: Texto a analizar
        
        Returns:
            Dict con flags de detección por concepto
        """
        text_lower = text.lower()
        
        return {
            "has_liquidity": any(keyword in text_lower for keyword in self.LIQUIDITY_KEYWORDS),
            "has_regulation": any(keyword in text_lower for keyword in self.REGULATION_KEYWORDS),
            "has_central_bank": any(keyword in text_lower for keyword in self.CENTRAL_BANK_KEYWORDS),
            "has_conflict": any(keyword in text_lower for keyword in self.CONFLICT_KEYWORDS)
        }
    
    def categorize_news_type(self, text: str, concepts: Dict[str, bool]) -> NewsType:
        """
        Categoriza el tipo de noticia basado en conceptos detectados
        
        Prioridad:
        1. CONFLICT (guerra, sanciones) - Mayor impacto emocional
        2. LEY (regulación, SEC) - Impacto directo en mercados
        3. BANCO (banco central, política monetaria) - Impacto macro
        4. REGULACION (fallback para regulaciones menores)
        
        Args:
            text: Texto de la noticia
            concepts: Dict con conceptos detectados
        
        Returns:
            NewsType enum
        """
        if concepts["has_conflict"]:
            return NewsType.GUERRA
        
        if concepts["has_regulation"]:
            # Distinguir entre ley importante (SEC) y regulación general
            text_lower = text.lower()
            if "sec" in text_lower or "ruling" in text_lower:
                return NewsType.LEY
            return NewsType.REGULACION
        
        if concepts["has_central_bank"]:
            return NewsType.BANCO
        
        # Default: REGULACION (más seguro que CRISIS)
        return NewsType.REGULACION
    
    def analyze_sentiment(self, text: str) -> SentimentType:
        """
        Analiza el sentimiento básico usando TextBlob
        
        Args:
            text: Texto a analizar
        
        Returns:
            SentimentType enum
        """
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        if polarity > 0.1:
            return SentimentType.POSITIVO
        elif polarity < -0.1:
            return SentimentType.NEGATIVO
        else:
            return SentimentType.NEUTRAL
    
    def calculate_risk_score(
        self, 
        news_type: NewsType, 
        sentiment: SentimentType,
        concepts: Dict[str, bool],
        headline: str
    ) -> float:
        """
        Calcula el Risk Score (1-10) basado en múltiples factores
        
        Factores:
        - Tipo de noticia (GUERRA = alto riesgo, BANCO = medio)
        - Sentimiento (NEGATIVO = alto riesgo)
        - Conceptos detectados (múltiples conceptos = mayor impacto)
        - Palabras clave de urgencia
        
        Args:
            news_type: Tipo de noticia
            sentiment: Sentimiento
            concepts: Conceptos detectados
            headline: Titular de la noticia
        
        Returns:
            Risk Score de 1.0 a 10.0
        """
        score = 1.0  # Base
        
        # Factor 1: Tipo de noticia (peso: 40%)
        type_scores = {
            NewsType.GUERRA: 4.0,
            NewsType.CRISIS: 3.5,
            NewsType.LEY: 3.0,
            NewsType.REGULACION: 2.5,
            NewsType.BANCO: 2.0,
            NewsType.ACUERDO: 1.5
        }
        score += type_scores.get(news_type, 2.0)
        
        # Factor 2: Sentimiento (peso: 30%)
        sentiment_scores = {
            SentimentType.NEGATIVO: 2.5,
            SentimentType.NEUTRAL: 1.0,
            SentimentType.POSITIVO: 0.5
        }
        score += sentiment_scores.get(sentiment, 1.0)
        
        # Factor 3: Conceptos múltiples (peso: 20%)
        concept_count = sum(concepts.values())
        score += min(concept_count * 0.5, 2.0)  # Máximo 2.0 puntos
        
        # Factor 4: Palabras de urgencia (peso: 10%)
        urgency_keywords = ["urgent", "urgente", "breaking", "breaking news", "alert", "alerta"]
        headline_lower = headline.lower()
        if any(keyword in headline_lower for keyword in urgency_keywords):
            score += 1.0
        
        # Normalizar a rango 1-10
        return min(max(score, 1.0), 10.0)
    
    def process_article(self, article: Dict) -> Optional[MarketIntel]:
        """
        Procesa un artículo completo: análisis + guardado en BD
        
        Args:
            article: Dict con datos del artículo de NewsAPI
        
        Returns:
            MarketIntel object o None si hay error
        """
        try:
            title = article.get("title", "")
            description = article.get("description", "") or ""
            content = article.get("content", "") or ""
            published_at_str = article.get("publishedAt", "")
            source_name = article.get("source", {}).get("name", "NewsAPI")
            
            # Combinar texto para análisis
            full_text = f"{title} {description} {content}"
            
            # Detectar conceptos
            concepts = self.detect_concept_keywords(full_text)
            
            # Categorizar tipo
            news_type = self.categorize_news_type(full_text, concepts)
            
            # Analizar sentimiento
            sentiment = self.analyze_sentiment(full_text)
            
            # Calcular Risk Score
            risk_score = self.calculate_risk_score(
                news_type, sentiment, concepts, title
            )
            
            # Parsear fecha
            try:
                published_at = datetime.fromisoformat(published_at_str.replace("Z", "+00:00"))
            except:
                published_at = datetime.utcnow()
            
            # Crear objeto MarketIntel
            market_intel = MarketIntel(
                source=source_name,
                headline=title[:500],  # Limitar longitud
                news_type=news_type,
                sentiment=sentiment,
                impact_score=risk_score,
                raw_content=full_text[:2000],  # Limitar longitud
                published_at=published_at
            )
            
            # Guardar en BD
            self.db.add(market_intel)
            self.db.commit()
            self.db.refresh(market_intel)
            
            logger.info(
                f"✅ Inteligencia procesada: {news_type.value} | "
                f"Sentimiento: {sentiment.value} | Risk Score: {risk_score:.1f}"
            )
            
            return market_intel
        
        except Exception as e:
            logger.error(f"❌ Error procesando artículo: {e}")
            self.db.rollback()
            return None
    
    def collect_intelligence(
        self, 
        query: str = "cryptocurrency OR bitcoin OR XRP OR SEC",
        max_articles: int = 10
    ) -> List[MarketIntel]:
        """
        Recolecta y procesa inteligencia completa desde NewsAPI
        
        Args:
            query: Query de búsqueda
            max_articles: Máximo de artículos a procesar
        
        Returns:
            Lista de MarketIntel objects procesados
        """
        logger.info(f"🔍 Iniciando recolección de inteligencia: '{query}'")
        
        # Obtener noticias
        articles = self.fetch_news_from_newsapi(query=query, page_size=max_articles)
        
        if not articles:
            logger.warning("⚠️ No se obtuvieron artículos")
            return []
        
        # Procesar cada artículo
        processed_intel = []
        for article in articles[:max_articles]:
            intel = self.process_article(article)
            if intel:
                processed_intel.append(intel)
        
        logger.info(f"✅ Procesadas {len(processed_intel)} noticias exitosamente")
        return processed_intel
    
    def get_recent_intelligence(
        self, 
        hours: int = 24,
        min_risk_score: float = 5.0
    ) -> List[MarketIntel]:
        """
        Obtiene inteligencia reciente de la BD filtrada por Risk Score
        
        Args:
            hours: Horas hacia atrás para buscar
            min_risk_score: Risk Score mínimo para incluir
        
        Returns:
            Lista de MarketIntel objects
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        intel_list = self.db.query(MarketIntel).filter(
            MarketIntel.created_at >= cutoff_time,
            MarketIntel.impact_score >= min_risk_score
        ).order_by(
            MarketIntel.impact_score.desc()
        ).all()
        
        return intel_list


# ============================================================================
# SCRIPT DE PRUEBA
# ============================================================================

if __name__ == "__main__":
    print("🔍 NEXUS Intel Manager - Prueba")
    print("=" * 50)
    
    # Verificar configuración
    newsapi_key = os.getenv("NEWSAPI_KEY")
    if not newsapi_key:
        print("⚠️ ADVERTENCIA: NEWSAPI_KEY no encontrada en .env")
        print("   Crea un archivo .env con: NEWSAPI_KEY=tu_api_key")
        print("   Obtén tu API key en: https://newsapi.org/")
        print("\n   Continuando con prueba limitada...\n")
    
    # Inicializar base de datos
    from database_setup import init_database
    
    print("📦 Inicializando base de datos...")
    engine, SessionLocal = init_database("nexus_intel_demo.sqlite")
    db = SessionLocal()
    
    try:
        # Crear IntelManager
        print("🔧 Creando IntelManager...")
        intel_manager = IntelManager(db)
        
        # Probar análisis de texto de ejemplo
        print("\n📊 Prueba de análisis de texto:")
        print("-" * 50)
        
        test_headline = "SEC Approves Bitcoin ETF, Central Bank Warns of Liquidity Crisis"
        print(f"Titular: {test_headline}")
        
        concepts = intel_manager.detect_concept_keywords(test_headline)
        print(f"Conceptos detectados: {concepts}")
        
        news_type = intel_manager.categorize_news_type(test_headline, concepts)
        print(f"Tipo de noticia: {news_type.value}")
        
        sentiment = intel_manager.analyze_sentiment(test_headline)
        print(f"Sentimiento: {sentiment.value}")
        
        risk_score = intel_manager.calculate_risk_score(
            news_type, sentiment, concepts, test_headline
        )
        print(f"Risk Score: {risk_score:.1f}/10.0")
        
        # Si hay API key, intentar recolección real
        if newsapi_key:
            print("\n🌐 Intentando recolección real desde NewsAPI...")
            intel_list = intel_manager.collect_intelligence(
                query="cryptocurrency OR bitcoin",
                max_articles=3
            )
            
            if intel_list:
                print(f"\n✅ {len(intel_list)} noticias procesadas:")
                for intel in intel_list:
                    print(f"   - {intel.headline[:60]}...")
                    print(f"     Tipo: {intel.news_type.value} | Risk: {intel.impact_score:.1f}")
        else:
            print("\n⏭️  Saltando recolección real (sin API key)")
        
    finally:
        db.close()
    
    print("\n✨ Prueba completada!")

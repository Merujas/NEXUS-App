"""
NEXUS - Economy Engine Module
Lead Software Architect & Behavioral Economist
==============================================

Este módulo implementa el "NexusBank" - el motor económico que gestiona la conversión
de puntos virtuales (nexus_points) a dinero real (fiat_balance) bajo un modelo de
incentivos conductuales.

Diseño Conductual Clave:
------------------------
El sistema está diseñado para DESINCENTIVAR retiros a PayPal (alta comisión, mínimo alto)
y FOMENTAR retiros a Kraken Partner (sin comisión, bonus, mínimo bajo).

Esto crea un "embudo" psicológico donde:
1. Los usuarios acumulan puntos fácilmente (baja fricción)
2. La conversión a fiat tiene fricción estratégica (alta fricción)
3. Kraken se convierte en la opción "obvia" (sesgo de anclaje + efecto de default)
"""

from sqlalchemy.orm import Session
from database_setup import User, BetResult, WithdrawalMethod
from typing import Tuple, Optional, Dict
from datetime import datetime
from dotenv import load_dotenv
import logging
import os

# Cargar variables de entorno
load_dotenv()

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NexusBank:
    """
    Motor Económico de NEXUS
    
    Responsabilidades:
    1. Calcular comisiones de retiro según método (diseño conductual)
    2. Procesar recompensas publicitarias (ingreso pasivo gamificado)
    3. Gestionar conversión puntos -> fiat con incentivos estratégicos
    """
    
    # Constantes de configuración económica
    PAYPAL_MIN_WITHDRAWAL = 5.0
    PAYPAL_FEE_PERCENTAGE = 0.10  # 10% de comisión
    
    KRAKEN_MIN_WITHDRAWAL = 2.0
    KRAKEN_FEE_PERCENTAGE = 0.0  # 0% de comisión (incentivo)
    KRAKEN_BONUS_PERCENTAGE = 0.05  # 5% de bonus en puntos (incentivo adicional)
    
    # Tasa de conversión puntos -> fiat (1 punto = 0.01 USD)
    POINTS_TO_FIAT_RATE = 0.01
    
    def __init__(self, db_session: Session):
        """
        Inicializa el NexusBank con una sesión de base de datos
        
        Args:
            db_session: Sesión activa de SQLAlchemy
        """
        self.db = db_session
    
    def calculate_withdrawal_fee(
        self, 
        method: WithdrawalMethod, 
        amount: float
    ) -> Tuple[float, float, Optional[float]]:
        """
        Calcula la comisión de retiro según el método seleccionado
        
        Diseño Conductual:
        - PayPal: Alta fricción (mínimo alto + comisión alta) = DESINCENTIVO
        - Kraken: Baja fricción (mínimo bajo + sin comisión + bonus) = INCENTIVO
        
        Args:
            method: Método de retiro (PAYPAL o KRAKEN_PARTNER)
            amount: Cantidad en puntos a retirar
        
        Returns:
            Tuple con:
            - fee_amount: Comisión en puntos
            - net_amount: Cantidad neta después de comisión (en puntos)
            - bonus_amount: Bonus adicional (solo para Kraken, None para PayPal)
        
        Raises:
            ValueError: Si el monto es menor al mínimo requerido
        """
        # Convertir puntos a fiat para validación
        fiat_amount = amount * self.POINTS_TO_FIAT_RATE
        
        if method == WithdrawalMethod.PAYPAL:
            # DESINCENTIVO: Validar mínimo alto
            if fiat_amount < self.PAYPAL_MIN_WITHDRAWAL:
                raise ValueError(
                    f"[ERROR] Retiro minimo a PayPal: ${self.PAYPAL_MIN_WITHDRAWAL:.2f} USD. "
                    f"Intentaste retirar: ${fiat_amount:.2f} USD"
                )
            
            # Calcular comisión del 10%
            fee_amount = amount * self.PAYPAL_FEE_PERCENTAGE
            net_amount = amount - fee_amount
            
            logger.info(
                f"Retiro PayPal: {amount:.2f} puntos -> "
                f"${fiat_amount:.2f} USD | Comision: {fee_amount:.2f} puntos (10%) | "
                f"Neto: {net_amount:.2f} puntos"
            )
            
            return fee_amount, net_amount, None
        
        elif method == WithdrawalMethod.KRAKEN_PARTNER:
            # INCENTIVO: Validar mínimo bajo
            if fiat_amount < self.KRAKEN_MIN_WITHDRAWAL:
                raise ValueError(
                    f"[ERROR] Retiro minimo a Kraken: ${self.KRAKEN_MIN_WITHDRAWAL:.2f} USD. "
                    f"Intentaste retirar: ${fiat_amount:.2f} USD"
                )
            
            # Sin comisión (incentivo principal)
            fee_amount = amount * self.KRAKEN_FEE_PERCENTAGE  # 0
            
            # Bonus del 5% en puntos (incentivo adicional)
            bonus_amount = amount * self.KRAKEN_BONUS_PERCENTAGE
            net_amount = amount + bonus_amount  # Recibe MÁS de lo que retira
            
            logger.info(
                f"[OK] Retiro Kraken: {amount:.2f} puntos -> "
                f"${fiat_amount:.2f} USD | Comision: {fee_amount:.2f} puntos (0%) | "
                f"Bonus: +{bonus_amount:.2f} puntos (5%) | "
                f"Total recibido: {net_amount:.2f} puntos"
            )
            
            return fee_amount, net_amount, bonus_amount
        
        else:
            raise ValueError(f"Método de retiro no válido: {method}")
    
    def process_withdrawal(
        self, 
        user_id: int, 
        method: WithdrawalMethod, 
        points_amount: float
    ) -> dict:
        """
        Procesa un retiro de puntos a fiat
        
        Flujo:
        1. Validar que el usuario tiene suficientes puntos
        2. Calcular comisión según método
        3. Actualizar saldos del usuario
        4. Registrar transacción
        
        Args:
            user_id: ID del usuario
            method: Método de retiro
            points_amount: Cantidad de puntos a retirar
        
        Returns:
            dict con detalles de la transacción
        
        Raises:
            ValueError: Si el usuario no tiene suficientes puntos o monto inválido
        """
        # Obtener usuario
        user = self.db.query(User).filter(User.id == user_id).first()
        if not user:
            raise ValueError(f"Usuario con ID {user_id} no encontrado")
        
        # Validar saldo suficiente
        if user.nexus_points < points_amount:
            raise ValueError(
                f"[ERROR] Saldo insuficiente. Tienes {user.nexus_points:.2f} puntos, "
                f"intentas retirar {points_amount:.2f} puntos"
            )
        
        # Calcular comisión y monto neto
        fee_amount, net_amount, bonus_amount = self.calculate_withdrawal_fee(
            method, points_amount
        )
        
        # Convertir a fiat
        fiat_to_add = net_amount * self.POINTS_TO_FIAT_RATE
        
        # Actualizar saldos del usuario
        user.nexus_points -= points_amount  # Descontar puntos retirados
        
        # Si hay bonus (Kraken), agregarlo de vuelta a puntos
        if bonus_amount:
            user.nexus_points += bonus_amount
        
        user.fiat_balance += fiat_to_add  # Agregar fiat
        
        # Guardar cambios
        self.db.commit()
        
        # Preparar respuesta
        result = {
            "success": True,
            "user_id": user_id,
            "method": method.value,
            "points_withdrawn": points_amount,
            "fee_points": fee_amount,
            "bonus_points": bonus_amount if bonus_amount else 0,
            "fiat_added": fiat_to_add,
            "remaining_points": user.nexus_points,
            "total_fiat_balance": user.fiat_balance,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        logger.info(f"[OK] Retiro procesado exitosamente para usuario {user_id}")
        return result
    
    def process_ad_reward(self, user_id: int, points_amount: float = 10.0) -> dict:
        """
        Procesa una recompensa publicitaria (ingreso pasivo gamificado)
        
        Diseño Conductual:
        - Recompensas pequeñas pero frecuentes crean "variable rewards" (Skinner Box)
        - Los usuarios ven crecimiento constante de puntos (efecto de progreso visual)
        - Baja fricción: solo ver un anuncio, recibir puntos instantáneamente
        
        Args:
            user_id: ID del usuario
            points_amount: Cantidad de puntos a otorgar (default: 10 puntos)
        
        Returns:
            dict con detalles de la recompensa
        
        Raises:
            ValueError: Si el usuario no existe
        """
        user = self.db.query(User).filter(User.id == user_id).first()
        if not user:
            raise ValueError(f"Usuario con ID {user_id} no encontrado")
        
        # Agregar puntos
        user.nexus_points += points_amount
        
        # Guardar cambios
        self.db.commit()
        
        result = {
            "success": True,
            "user_id": user_id,
            "points_awarded": points_amount,
            "new_points_balance": user.nexus_points,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        logger.info(
            f"Recompensa publicitaria: +{points_amount:.2f} puntos "
            f"para usuario {user_id} (Nuevo saldo: {user.nexus_points:.2f})"
        )
        
        return result
    
    def process_bet_winnings(
        self, 
        user_id: int, 
        bet_points: float, 
        multiplier: float = 2.0
    ) -> dict:
        """
        Procesa ganancias de una apuesta ganadora
        
        Args:
            user_id: ID del usuario
            bet_points: Puntos apostados originalmente
            multiplier: Multiplicador de ganancia (default: 2x)
        
        Returns:
            dict con detalles de las ganancias
        """
        user = self.db.query(User).filter(User.id == user_id).first()
        if not user:
            raise ValueError(f"Usuario con ID {user_id} no encontrado")
        
        # Calcular ganancias
        winnings = bet_points * multiplier
        total_points = bet_points + winnings  # Recupera lo apostado + ganancias
        
        # Agregar puntos ganados
        user.nexus_points += total_points
        
        # Guardar cambios
        self.db.commit()
        
        result = {
            "success": True,
            "user_id": user_id,
            "bet_points": bet_points,
            "winnings": winnings,
            "total_received": total_points,
            "new_points_balance": user.nexus_points,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        logger.info(
            f"Apuesta ganadora: Usuario {user_id} gano {winnings:.2f} puntos "
            f"(Nuevo saldo: {user.nexus_points:.2f})"
        )
        
        return result
    
    def get_user_economy_summary(self, user_id: int) -> dict:
        """
        Obtiene un resumen completo de la economía del usuario
        
        Args:
            user_id: ID del usuario
        
        Returns:
            dict con resumen económico
        """
        user = self.db.query(User).filter(User.id == user_id).first()
        if not user:
            raise ValueError(f"Usuario con ID {user_id} no encontrado")
        
        # Calcular valores equivalentes en fiat
        points_fiat_value = user.nexus_points * self.POINTS_TO_FIAT_RATE
        total_portfolio_value = user.fiat_balance + points_fiat_value
        
        return {
            "user_id": user_id,
            "email": user.email,
            "nexus_points": user.nexus_points,
            "points_fiat_value": points_fiat_value,
            "fiat_balance": user.fiat_balance,
            "total_portfolio_value": total_portfolio_value,
            "reputation_level": user.reputation_level,
            "conversion_rate": self.POINTS_TO_FIAT_RATE
        }


# ============================================================================
# FUNCIONES DE UTILIDAD
# ============================================================================

def compare_withdrawal_methods(amount_points: float) -> dict:
    """
    Compara los dos métodos de retiro para mostrar al usuario las diferencias
    
    Esta función es clave para el diseño conductual: muestra claramente
    por qué Kraken es la mejor opción (anclaje + comparación directa)
    
    Args:
        amount_points: Cantidad de puntos a comparar
    
    Returns:
        dict con comparación detallada
    """
    bank = NexusBank(None)  # No necesita DB para cálculos
    
    try:
        paypal_fee, paypal_net, _ = bank.calculate_withdrawal_fee(
            WithdrawalMethod.PAYPAL, amount_points
        )
        paypal_fiat = paypal_net * bank.POINTS_TO_FIAT_RATE
    except ValueError as e:
        paypal_fee = None
        paypal_net = None
        paypal_fiat = None
        paypal_error = str(e)
    else:
        paypal_error = None
    
    try:
        kraken_fee, kraken_net, kraken_bonus = bank.calculate_withdrawal_fee(
            WithdrawalMethod.KRAKEN_PARTNER, amount_points
        )
        kraken_fiat = kraken_net * bank.POINTS_TO_FIAT_RATE
    except ValueError as e:
        kraken_fee = None
        kraken_net = None
        kraken_bonus = None
        kraken_fiat = None
        kraken_error = str(e)
    else:
        kraken_error = None
    
    return {
        "amount_points": amount_points,
        "paypal": {
            "available": paypal_error is None,
            "error": paypal_error,
            "fee_points": paypal_fee,
            "net_points": paypal_net,
            "fiat_received": paypal_fiat,
            "min_withdrawal": bank.PAYPAL_MIN_WITHDRAWAL
        },
        "kraken": {
            "available": kraken_error is None,
            "error": kraken_error,
            "fee_points": kraken_fee,
            "bonus_points": kraken_bonus,
            "net_points": kraken_net,
            "fiat_received": kraken_fiat,
            "min_withdrawal": bank.KRAKEN_MIN_WITHDRAWAL
        },
        "recommendation": "KRAKEN_PARTNER" if kraken_error is None else None
    }


def generate_withdrawal_options(amount_points: float) -> Dict:
    """
    Genera opciones de retiro con copy persuasivo para maximizar conversiones a Kraken
    
    Diseño Conductual:
    - PayPal se presenta como opción "estándar" con fricción visible (comisión 10%)
    - Kraken se presenta como opción "RECOMENDADA" con beneficios destacados
    - El enlace de referido se carga desde .env para tracking
    
    Args:
        amount_points: Cantidad de puntos a retirar
    
    Returns:
        Dict con opciones formateadas para UI
    """
    bank = NexusBank(None)
    kraken_ref_link = os.getenv('KRAKEN_REF_LINK', 'https://www.kraken.com/')
    
    # Calcular valores
    fiat_amount = amount_points * bank.POINTS_TO_FIAT_RATE
    paypal_fee = amount_points * bank.PAYPAL_FEE_PERCENTAGE
    paypal_net = amount_points - paypal_fee
    paypal_fiat = paypal_net * bank.POINTS_TO_FIAT_RATE
    
    kraken_bonus = amount_points * bank.KRAKEN_BONUS_PERCENTAGE
    kraken_net = amount_points + kraken_bonus
    kraken_fiat = kraken_net * bank.POINTS_TO_FIAT_RATE
    
    return {
        "amount_points": amount_points,
        "fiat_equivalent": fiat_amount,
        
        "standard": {
            "name": "PayPal",
            "type": "STANDARD",
            "available": fiat_amount >= bank.PAYPAL_MIN_WITHDRAWAL,
            "min_withdrawal_usd": bank.PAYPAL_MIN_WITHDRAWAL,
            "fee_percentage": 10,
            "fee_points": paypal_fee,
            "net_points": paypal_net,
            "fiat_received": paypal_fiat,
            "message": "Recibe tu dinero en 24h. Se aplica comisión del 10%.",
            "link": None,
            "badge": None
        },
        
        "partner": {
            "name": "Kraken",
            "type": "PARTNER",
            "recommended": True,
            "available": fiat_amount >= bank.KRAKEN_MIN_WITHDRAWAL,
            "min_withdrawal_usd": bank.KRAKEN_MIN_WITHDRAWAL,
            "fee_percentage": 0,
            "fee_points": 0,
            "bonus_percentage": 5,
            "bonus_points": kraken_bonus,
            "net_points": kraken_net,
            "fiat_received": kraken_fiat,
            "message": (
                "Maximiza tu ganancia. Retira el 100% de tus puntos SIN COMISIÓN. "
                "Además, recibe un bono de bienvenida de hasta 1.125€ según condiciones de Kraken."
            ),
            "bonus_message": f"¡Te regalamos {int(kraken_bonus)} Puntos NEXUS extra si usas este enlace!",
            "link": kraken_ref_link,
            "badge": "RECOMENDADO",
            "benefits": [
                "0% comisión en retiros",
                f"+{int(kraken_bonus)} puntos NEXUS de regalo",
                "Bono de bienvenida hasta 1.125€",
                "Retiros procesados en minutos",
                "Exchange regulado y seguro"
            ]
        },
        
        "comparison": {
            "savings_points": paypal_fee + kraken_bonus,
            "savings_percentage": ((kraken_net - paypal_net) / paypal_net) * 100 if paypal_net > 0 else 0,
            "recommendation_text": (
                f"Con Kraken Partner ahorras {paypal_fee:.0f} puntos en comisiones "
                f"y ganas {kraken_bonus:.0f} puntos extra. "
                f"¡Eso es {paypal_fee + kraken_bonus:.0f} puntos más en tu bolsillo!"
            )
        }
    }


# ============================================================================
# SCRIPT DE PRUEBA
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("NEXUS Bank - Motor Economico")
    print("=" * 60)
    
    # Mostrar nueva función generate_withdrawal_options
    print("\n[TEST] generate_withdrawal_options(500 puntos):")
    print("-" * 60)
    
    options = generate_withdrawal_options(500.0)
    
    print(f"\nMonto: {options['amount_points']} puntos = ${options['fiat_equivalent']:.2f} USD")
    
    print(f"\n--- OPCION ESTANDAR: {options['standard']['name']} ---")
    print(f"   Comision: {options['standard']['fee_percentage']}%")
    print(f"   Descuento: -{options['standard']['fee_points']:.0f} puntos")
    print(f"   Recibes: {options['standard']['net_points']:.0f} puntos (${options['standard']['fiat_received']:.2f})")
    print(f"   Mensaje: {options['standard']['message']}")
    
    print(f"\n--- OPCION PARTNER: {options['partner']['name']} [{options['partner']['badge']}] ---")
    print(f"   Comision: {options['partner']['fee_percentage']}% (GRATIS)")
    print(f"   Bonus: +{options['partner']['bonus_points']:.0f} puntos ({options['partner']['bonus_percentage']}%)")
    print(f"   Recibes: {options['partner']['net_points']:.0f} puntos (${options['partner']['fiat_received']:.2f})")
    print(f"   Mensaje: {options['partner']['message']}")
    print(f"   Bonus Msg: {options['partner']['bonus_message']}")
    print(f"   Enlace: {options['partner']['link']}")
    print(f"\n   Beneficios:")
    for benefit in options['partner']['benefits']:
        print(f"      [OK] {benefit}")
    
    print(f"\n--- COMPARACION ---")
    print(f"   Ahorro total: {options['comparison']['savings_points']:.0f} puntos")
    print(f"   Mejora: +{options['comparison']['savings_percentage']:.1f}%")
    print(f"   {options['comparison']['recommendation_text']}")
    
    # Comparación clásica
    print("\n" + "=" * 60)
    print("Comparacion clasica (compare_withdrawal_methods):")
    print("-" * 60)
    
    comparison = compare_withdrawal_methods(500.0)
    
    print(f"\nPayPal:")
    if comparison["paypal"]["available"]:
        print(f"   Comision: {comparison['paypal']['fee_points']:.2f} puntos (10%)")
        print(f"   Neto recibido: {comparison['paypal']['net_points']:.2f} puntos")
        print(f"   Fiat recibido: ${comparison['paypal']['fiat_received']:.2f} USD")
    else:
        print(f"   [ERROR] {comparison['paypal']['error']}")
    
    print(f"\nKraken Partner:")
    if comparison["kraken"]["available"]:
        print(f"   Comision: {comparison['kraken']['fee_points']:.2f} puntos (0%)")
        print(f"   Bonus: +{comparison['kraken']['bonus_points']:.2f} puntos (5%)")
        print(f"   Total recibido: {comparison['kraken']['net_points']:.2f} puntos")
        print(f"   Fiat recibido: ${comparison['kraken']['fiat_received']:.2f} USD")
    else:
        print(f"   [ERROR] {comparison['kraken']['error']}")
    
    print(f"\nRecomendacion: {comparison['recommendation']}")
    print("\n[OK] Diseno Conductual: Kraken ofrece mejor valor (sin comision + bonus)")

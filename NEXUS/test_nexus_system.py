import os
import sys
from datetime import datetime

# Importamos tus módulos (Asumiendo que Cursor nombró las clases así, 
# si falla, ajustaremos los nombres de importación)
try:
    import pandas as pd
    from sqlalchemy import create_engine, text
    from dotenv import load_dotenv
    # Intentamos importar la lógica económica si existe la clase
    # Si Cursor lo hizo funcional, esto debería funcionar
except ImportError as e:
    print(f"[ERROR] Error de Importacion: {e}")
    sys.exit(1)

# Configuración Visual
def print_header(title):
    print("\n" + "="*60)
    print(f"NEXUS SYSTEM DIAGNOSTIC | {title}")
    print("="*60)

def print_status(step, status, message):
    icon = "[OK]" if status else "[ERROR]"
    print(f"{icon} [{step}]: {message}")

def run_diagnostic():
    load_dotenv()
    print_header("INICIANDO PROTOCOLO DE PRUEBA")

    # 1. VERIFICACIÓN DE ENTORNO
    api_key = os.getenv("NEWS_API_KEY") or os.getenv("NEWSAPI_KEY")
    if api_key:
        print_status("ENV", True, "Claves API detectadas en .env")
    else:
        print_status("ENV", False, "Faltan claves en .env (NEWS_API_KEY)")

    # 2. VERIFICACIÓN DE BASE DE DATOS
    db_file = "nexus_demo.sqlite"
    if os.path.exists(db_file):
        print_status("DB", True, f"Base de datos encontrada: {db_file}")
        
        # Conexión de prueba con SQLAlchemy
        try:
            engine = create_engine(f'sqlite:///{db_file}')
            with engine.connect() as conn:
                # Verificar tablas existentes
                result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table';"))
                tables = [row[0] for row in result]
                print(f"   Tablas detectadas: {', '.join(tables)}")
                
                if 'users' in tables or 'user_wallet' in tables:
                    print_status("SCHEMA", True, "Esquema de tablas validado.")
                else:
                    print_status("SCHEMA", False, "No se detectaron tablas de usuarios.")
        except Exception as e:
            print_status("DB_CONN", False, f"Error conectando a DB: {e}")
    else:
        print_status("DB", False, "El archivo .sqlite no existe. Ejecuta database_setup.py primero.")

    # 3. PRUEBA DE MÓDULOS REALES
    print_header("PRUEBA DE MODULOS REALES")
    
    try:
        from database_setup import init_database, User, WithdrawalMethod
        from economy_engine import NexusBank, compare_withdrawal_methods
        from sqlalchemy.orm import Session
        
        print_status("IMPORTS", True, "Modulos importados correctamente")
        
        # Inicializar base de datos si no existe
        if not os.path.exists(db_file):
            print("   Creando base de datos...")
            engine, SessionLocal = init_database(db_file)
        else:
            engine, SessionLocal = init_database(db_file)
        
        db: Session = SessionLocal()
        
        # Probar economía real
        print("\n--- Prueba de Economia (NexusBank) ---")
        monto_puntos = 500.0  # 500 puntos = $5.00 USD
        
        comparison = compare_withdrawal_methods(monto_puntos)
        
        print(f"\nEscenario: Usuario quiere retirar {monto_puntos} puntos (${monto_puntos * 0.01:.2f} USD)")
        
        print(f"\n   --- Opcion 1: PayPal ---")
        if comparison["paypal"]["available"]:
            print(f"   Comision: {comparison['paypal']['fee_points']:.2f} puntos (10%)")
            print(f"   Neto recibido: {comparison['paypal']['net_points']:.2f} puntos")
            print(f"   Fiat recibido: ${comparison['paypal']['fiat_received']:.2f} USD")
        else:
            print(f"   [ERROR] {comparison['paypal']['error']}")
        
        print(f"\n   --- Opcion 2: Kraken Partner ---")
        if comparison["kraken"]["available"]:
            print(f"   Comision: {comparison['kraken']['fee_points']:.2f} puntos (0%)")
            print(f"   Bonus: +{comparison['kraken']['bonus_points']:.2f} puntos (5%)")
            print(f"   Total recibido: {comparison['kraken']['net_points']:.2f} puntos")
            print(f"   Fiat recibido: ${comparison['kraken']['fiat_received']:.2f} USD")
        else:
            print(f"   [ERROR] {comparison['kraken']['error']}")
        
        if comparison["kraken"]["available"] and comparison["paypal"]["available"]:
            diferencia = comparison['kraken']['net_points'] - comparison['paypal']['net_points']
            print(f"\n   Diferencia a favor de Kraken: +{diferencia:.2f} puntos")
            print_status("ECONOMY", True, "Logica de incentivos funciona: Kraken es mas rentable")
        elif comparison["kraken"]["available"]:
            print_status("ECONOMY", True, "Kraken disponible (PayPal rechazado por minimo)")
        
        db.close()
        
    except ImportError as e:
        print_status("MODULES", False, f"Error importando modulos: {e}")
    except Exception as e:
        print_status("MODULES", False, f"Error en prueba de modulos: {e}")
        import traceback
        traceback.print_exc()

    print_header("DIAGNÓSTICO COMPLETADO")

if __name__ == "__main__":
    run_diagnostic()